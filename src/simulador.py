"""
Capa de servicio usada por el agente (tools) y por el dashboard.

  - Entrenamiento:  2024-01-01 → 2025-12-31  (precio de bolsa real XM)
  - Prueba:         2026-01-01 → 2026-10-31  (datos no vistos)

Las políticas se entrenan una vez por combinación de parámetros de batería y
se guardan en memoria (caché), así cambiar parámetros en el dashboard solo
cuesta unos segundos la primera vez.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

import datos
import politicas
from entorno import BatteryEnv

TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"

POLICY_KEYS = {
    "q_learning": "Q-learning",
    "iteracion_valor": "Iteración de valor",
    "horario_fijo": "Horario fijo (ingenua)",
    "aleatoria": "Aleatoria",
    "oraculo": "Oráculo (info. perfecta)",
}


@lru_cache(maxsize=1)
def get_prices() -> pd.DataFrame:
    return datos.price_matrix(datos.load_prices())


def data_range() -> tuple[str, str]:
    m = get_prices()
    return m.index.min().strftime("%Y-%m-%d"), m.index.max().strftime("%Y-%m-%d")


@lru_cache(maxsize=16)
def get_policies(capacity_kwh=10.0, power_kw=2.5, efficiency=0.9):
    """Entrena las políticas con el periodo de entrenamiento (con caché)."""
    m = get_prices()
    env_train = BatteryEnv(m[m.index <= TRAIN_END], capacity_kwh, power_kw, efficiency)
    policies, q_history = politicas.build_policies(env_train)
    return policies, q_history


def _validate(capacity_kwh, power_kw, efficiency):
    if not (1 <= capacity_kwh <= 1000):
        raise ValueError("capacidad_kwh debe estar entre 1 y 1000")
    if not (0.1 <= power_kw <= capacity_kwh):
        raise ValueError("potencia_kw debe estar entre 0.1 y la capacidad")
    if not (0.5 <= efficiency <= 1.0):
        raise ValueError("eficiencia debe estar entre 0.5 y 1.0")
    if capacity_kwh / power_kw > 12:
        raise ValueError("capacidad/potencia no puede superar 12 horas")


def _slice(start: str, end: str) -> pd.DataFrame:
    m = get_prices()
    lo, hi = data_range()
    try:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
    except Exception:
        raise ValueError("Fechas inválidas: use el formato AAAA-MM-DD")
    if s > e:
        raise ValueError("fecha_inicio debe ser anterior o igual a fecha_fin")
    sub = m[(m.index >= s) & (m.index <= e)]
    if sub.empty:
        raise ValueError(f"No hay datos entre {start} y {end}. Rango disponible: {lo} a {hi}")
    return sub


def simulate_range(policy_key: str, start: str, end: str,
                   capacity_kwh=10.0, power_kw=2.5, efficiency=0.9) -> tuple[pd.DataFrame, dict]:
    """Simula TODAS las políticas en el rango.
    Retorna (tabla_resumen, {clave_politica: detalle_horario})."""
    if policy_key not in POLICY_KEYS:
        raise ValueError(f"Política desconocida '{policy_key}'. Opciones: {list(POLICY_KEYS)}")
    _validate(capacity_kwh, power_kw, efficiency)
    sub = _slice(start, end)
    policies, _ = get_policies(float(capacity_kwh), float(power_kw), float(efficiency))
    env = BatteryEnv(sub, capacity_kwh, power_kw, efficiency)
    return politicas.compare_policies(policies, env)


def hourly_pattern(detail: pd.DataFrame) -> pd.DataFrame:
    """% de días en que la política cargó / descargó en cada hora."""
    pat = pd.crosstab(detail["hora"], detail["accion"], normalize="index") * 100
    for col in ("cargar", "descargar", "esperar"):
        if col not in pat:
            pat[col] = 0.0
    return pat[["cargar", "descargar", "esperar"]].round(1)


def price_stats(start: str, end: str) -> dict:
    sub = _slice(start, end)
    profile = sub.mean(axis=0)
    flat = sub.to_numpy().ravel()
    return {
        "fecha_inicio": sub.index.min().strftime("%Y-%m-%d"),
        "fecha_fin": sub.index.max().strftime("%Y-%m-%d"),
        "dias": int(len(sub)),
        "precio_promedio_cop_kwh": round(float(flat.mean()), 2),
        "precio_min_cop_kwh": round(float(flat.min()), 2),
        "precio_max_cop_kwh": round(float(flat.max()), 2),
        "hora_mas_barata_promedio": int(profile.idxmin()),
        "hora_mas_cara_promedio": int(profile.idxmax()),
        "diferencial_promedio_diario_cop_kwh": round(float((sub.max(axis=1) - sub.min(axis=1)).mean()), 2),
        "perfil_horario_promedio": {int(h): round(float(v), 1) for h, v in profile.items()},
        "fuente": "XM - Precio de Bolsa Nacional (PrecBolsNaci), API servapibi.xm.com.co",
    }
