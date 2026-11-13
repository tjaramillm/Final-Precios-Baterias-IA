"""
Evaluación del agente con preguntas de prueba FIJAS.

Para cada pregunta se verifica automáticamente:
  - herramienta_ok : el agente llamó a todas las herramientas esperadas
  - cita_ok        : la respuesta incluye la cita esperada ([Resolución …], [XM], [Simulación MDP])

Ejecutar:  python src/evaluacion_agente.py
Salida:    resultados/evaluacion_agente.csv y resultados/evaluacion_agente.md
"""
import re
import sys
import time

import pandas as pd

from agente import run_agent
from datos import ROOT

CITE = {
    "XM": r"\[[^\]]*XM[^\]]*\]",
    "MDP": r"\[[^\]]*Simulaci[oó]n MDP[^\]]*\]",
    "CREG": r"\[[^\]]*(Resoluci[oó]n|CREG)[^\]]*\]",
}

# (pregunta, herramientas esperadas, citas esperadas, tipo)
QUESTIONS = [
    ("¿Cuál fue el precio promedio de bolsa en marzo de 2026?",
     {"consultar_precios"}, ["XM"], "datos"),
    ("¿A qué hora del día fue más caro, en promedio, el precio de bolsa durante 2025?",
     {"consultar_precios"}, ["XM"], "datos"),
    ("Compara el precio promedio de bolsa de enero de 2024 con el de enero de 2026.",
     {"consultar_precios"}, ["XM"], "datos"),
    ("¿Cuánto ahorra la política de Q-learning entre 2026-01-01 y 2026-10-31 con la batería por defecto?",
     {"simular"}, ["MDP"], "simulación"),
    ("¿Es mejor Q-learning que el horario fijo en el periodo de prueba de 2026? Justifica con cifras.",
     {"simular"}, ["MDP"], "simulación"),
    ("Simula iteración de valor de junio a agosto de 2026 con una batería de 20 kWh y 5 kW.",
     {"simular"}, ["MDP"], "simulación"),
    ("¿En qué horas carga y en qué horas descarga la política aprendida por Q-learning en 2026?",
     {"simular"}, ["MDP"], "explicación"),
    ("¿Qué es un SAEB según la regulación de la CREG?",
     {"buscar_documentos"}, ["CREG"], "RAG"),
    ("Según la resolución CREG 101 113 de 2026, ¿cómo se liquida la energía de los SAEB que son activos de red?",
     {"buscar_documentos"}, ["CREG"], "RAG"),
    ("¿Qué es un autogenerador a pequeña escala según la CREG 174 de 2021?",
     {"buscar_documentos"}, ["CREG"], "RAG"),
    ("¿La regulación contempla que las baterías hagan arbitraje de energía? ¿Y cuánto habría ahorrado "
     "Q-learning en septiembre de 2026?",
     {"buscar_documentos", "simular"}, ["CREG", "MDP"], "multi-herramienta"),
    ("¿Cuál será el precio de bolsa en enero de 2030?",
     {"consultar_precios"}, [], "caso límite"),
    ("¿Cuál es la tarifa de energía de EPM para estrato 3 este mes?",
     set(), [], "fuera de alcance"),
]


def evaluate(pause_s: float = 2.0) -> pd.DataFrame:
    rows = []
    for i, (q, tools, cites, kind) in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q}")
        t0 = time.time()
        out = run_agent(q)
        called = [t["tool"] for t in out["trace"]]
        answer = out["answer"]
        tool_ok = tools.issubset(called) if tools else not called or set(called) <= {"buscar_documentos"}
        cite_ok = all(re.search(CITE[c], answer, flags=re.IGNORECASE) for c in cites) if cites else True
        rows.append({
            "id": i,
            "tipo": kind,
            "pregunta": q,
            "herramientas_esperadas": ", ".join(sorted(tools)) or "(ninguna)",
            "herramientas_llamadas": ", ".join(called) or "(ninguna)",
            "herramienta_ok": tool_ok,
            "cita_esperada": ", ".join(cites) or "-",
            "cita_ok": cite_ok,
            "iteraciones": out["iterations"],
            "segundos": round(time.time() - t0, 1),
            "respuesta": answer,
        })
        print(f"   tools={called} tool_ok={tool_ok} cita_ok={cite_ok}")
        time.sleep(pause_s)   # respetar límites de la capa gratuita
    return pd.DataFrame(rows)


def save(df: pd.DataFrame):
    out = ROOT / "resultados"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "evaluacion_agente.csv", index=False, encoding="utf-8")

    lines = [
        "# Evaluación del agente",
        "",
        f"- Herramienta correcta: **{df['herramienta_ok'].sum()}/{len(df)}** ({df['herramienta_ok'].mean():.0%})",
        f"- Cita correcta: **{df['cita_ok'].sum()}/{len(df)}** ({df['cita_ok'].mean():.0%})",
        "",
        "| # | Tipo | Pregunta | Esperadas | Llamadas | Tool OK | Cita OK |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['id']} | {r['tipo']} | {r['pregunta']} | {r['herramientas_esperadas']} | "
                     f"{r['herramientas_llamadas']} | {'✅' if r['herramienta_ok'] else '❌'} | "
                     f"{'✅' if r['cita_ok'] else '❌'} |")
    lines += ["", "## Respuestas", ""]
    for _, r in df.iterrows():
        lines += [f"**{r['id']}. {r['pregunta']}**", "", r["respuesta"], ""]
    (out / "evaluacion_agente.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Resultados en {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    save(evaluate())
