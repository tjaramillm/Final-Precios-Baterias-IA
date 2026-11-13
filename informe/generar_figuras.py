"""
Genera las figuras del informe (informe/figuras/*.pdf).

Ejecutar desde la raíz del proyecto:
    python informe/generar_figuras.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import app          # reutiliza las funciones de gráficos del dashboard
import simulador

OUT = ROOT / "informe" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9})

# Periodo de prueba, batería por defecto (10 kWh / 2.5 kW / η = 0.9)
table, details = simulador.simulate_range("q_learning", simulador.TEST_START, "2026-10-31")
named = {simulador.POLICY_KEYS[k]: v for k, v in details.items()}

figs = {
    "precios_xm.pdf": app.plot_prices(),
    "ahorro_acumulado.pdf": app.plot_cumulative(named),
    "decisiones_semana.pdf": app.plot_decisions(details["q_learning"], "2026-04-06", "Q-learning"),
    "patron_horario.pdf": app.plot_pattern(simulador.hourly_pattern(details["q_learning"]), "Q-learning"),
}
for name, fig in figs.items():
    fig.savefig(OUT / name, bbox_inches="tight")
    print("✅", OUT / name)

print(table[["politica", "ahorro_total_cop", "ciclos_equivalentes", "pct_del_oraculo"]].round(1))
