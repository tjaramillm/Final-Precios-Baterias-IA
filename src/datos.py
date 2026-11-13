"""
Descarga y carga de datos reales.

Fuentes:
  - XM (operador del mercado eléctrico colombiano), API pública Sinergox:
      https://servapibi.xm.com.co/hourly   métrica PrecBolsNaci (COP/kWh)
    La misma API que envuelve la librería pydataxm; aquí se llama con requests
    para no agregar dependencias.
  - CREG, gestor normativo Alejandría (resoluciones en HTML):
      https://gestornormativo.creg.gov.co/gestor/entorno/docs/

Plan B: si la API falla, se usa el CSV ya descargado en data/raw/.
"""
import re
import sys
import html
from pathlib import Path
from datetime import date, timedelta
from html.parser import HTMLParser

import pandas as pd
import requests

# La consola de Windows usa cp1252 por defecto: forzar UTF-8 para tildes y emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DOCS_DIR = RAW_DIR / "docs"
PRICES_CSV = RAW_DIR / "precio_bolsa_xm.csv"

XM_URL = "https://servapibi.xm.com.co/hourly"
CREG_URL = "https://gestornormativo.creg.gov.co/gestor/entorno/docs/{}.htm"

# Corpus RAG: resoluciones CREG relacionadas con baterías, autogeneración y bolsa
CREG_DOCS = {
    "resolucion_creg_101-113_2026": "Resolución CREG 101 113 de 2026 (SAEB en el SIN)",
    "resolucion_creg_0098_2019": "Resolución CREG 098 de 2019 (SAEB, derogada)",
    "resolucion_creg_0174_2021": "Resolución CREG 174 de 2021 (autogeneración a pequeña escala)",
    "resolucion_creg_0024_1995": "Resolución CREG 024 de 1995 (reglas de la bolsa de energía)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Precio de bolsa (XM)
# ─────────────────────────────────────────────────────────────────────────────
def _xm_request(start: date, end: date, metric: str = "PrecBolsNaci") -> pd.DataFrame:
    """Una consulta a la API de XM (máximo 31 días por solicitud)."""
    body = {
        "MetricId": metric,
        "StartDate": start.isoformat(),
        "EndDate": end.isoformat(),
        "Entity": "Sistema",
        "Filter": [],
    }
    r = requests.post(XM_URL, json=body, timeout=60)
    r.raise_for_status()

    rows = []
    for item in r.json()["Items"]:
        values = item["HourlyEntities"][0]["Values"]
        for h in range(1, 25):
            v = values.get(f"Hour{h:02d}")
            if v not in (None, ""):
                # Hour01 corresponde a la hora 0:00-1:00
                rows.append({"fecha": item["Date"], "hora": h - 1, "precio": float(v)})
    return pd.DataFrame(rows)


def download_prices(start: str = "2024-01-01", end: str = "2026-10-31") -> pd.DataFrame:
    """Descarga el precio de bolsa horario por bloques de 30 días y lo guarda en CSV."""
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    parts = []
    cur = start_d
    while cur <= end_d:
        block_end = min(cur + timedelta(days=29), end_d)
        print(f"  XM {cur} → {block_end}")
        parts.append(_xm_request(cur, block_end))
        cur = block_end + timedelta(days=1)

    df = pd.concat(parts, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRICES_CSV, index=False)
    print(f"✅ {len(df)} registros guardados en {PRICES_CSV}")
    return df


def load_prices() -> pd.DataFrame:
    """
    Carga los precios (plan B: CSV local). Si el CSV no existe, lo descarga.
    Solo conserva días completos (24 horas).
    """
    if not PRICES_CSV.exists():
        download_prices()
    df = pd.read_csv(PRICES_CSV)
    df["fecha"] = pd.to_datetime(df["fecha"])
    complete = df.groupby("fecha")["hora"].transform("count") == 24
    df = df[complete].sort_values(["fecha", "hora"]).reset_index(drop=True)
    return df


def price_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: una fila por día, 24 columnas (horas). Facilita simular por episodios."""
    return df.pivot(index="fecha", columns="hora", values="precio")


# ─────────────────────────────────────────────────────────────────────────────
# Documentos CREG para el RAG
# ─────────────────────────────────────────────────────────────────────────────
class _TextExtractor(HTMLParser):
    """Extrae el texto visible de una página HTML (sin librerías externas)."""

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head", "nav"):
            self._skip += 1
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head", "nav") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw_html: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw_html)
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def download_docs() -> None:
    """Descarga las resoluciones CREG y guarda el texto plano en data/raw/docs/."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id in CREG_DOCS:
        r = requests.get(CREG_URL.format(doc_id), timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        r.encoding = "latin-1"
        text = html_to_text(r.text)
        # Quitar el menú del sitio: el contenido inicia en "RESOLUCIÓN <número>"
        m = re.search(r"^RESOLUCI[ÓO]N (No\. )?\d", text, flags=re.MULTILINE)
        if m:
            text = text[m.start():]
        (DOCS_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        print(f"✅ {doc_id}: {len(text):,} caracteres")


def load_documents() -> list[dict]:
    """Retorna [{id, title, content, url}] con el formato usado en clase (lecture10)."""
    if not all((DOCS_DIR / f"{d}.txt").exists() for d in CREG_DOCS):
        download_docs()
    return [
        {
            "id": doc_id,
            "title": title,
            "content": (DOCS_DIR / f"{doc_id}.txt").read_text(encoding="utf-8"),
            "url": CREG_URL.format(doc_id),
        }
        for doc_id, title in CREG_DOCS.items()
    ]


if __name__ == "__main__":
    download_prices()
    download_docs()
