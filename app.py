"""
Dashboard: Despacho de batería con RL + asistente sobre regulación CREG.

Ejecutar:
    python app.py        → abre http://127.0.0.1:7860
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))
import simulador                      # noqa: E402
from agente import run_agent          # noqa: E402

COLORS = {
    "Q-learning": "#1f77b4",
    "Iteración de valor": "#2ca02c",
    "Horario fijo (ingenua)": "#ff7f0e",
    "Aleatoria": "#7f7f7f",
    "Oráculo (info. perfecta)": "#9467bd",
}
POLICY_CHOICES = [(v, k) for k, v in simulador.POLICY_KEYS.items()]
LO, HI = simulador.data_range()


# ─────────────────────────────────────────────────────────────────────────────
# Visualizaciones
# ─────────────────────────────────────────────────────────────────────────────
def plot_decisions(detail: pd.DataFrame, week_start: str, title: str):
    """Precio de bolsa vs. decisiones de la batería durante 7 días."""
    start = pd.Timestamp(week_start)
    d = detail[(detail["fecha"] >= start) & (detail["fecha"] < start + pd.Timedelta(days=7))].copy()
    if d.empty:
        d = detail.head(24 * 7).copy()
    d["t"] = d["fecha"] + pd.to_timedelta(d["hora"], unit="h")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d["t"], d["precio"], color="#444", lw=1.2, label="Precio bolsa (COP/kWh)")
    for acc, color, marker in [("cargar", "#2ca02c", "^"), ("descargar", "#d62728", "v")]:
        m = d[d["accion"] == acc]
        ax.scatter(m["t"], m["precio"], color=color, marker=marker, s=45, zorder=3, label=acc.capitalize())
    ax.set_ylabel("COP/kWh")
    ax2 = ax.twinx()
    ax2.fill_between(d["t"], d["soc_kwh"], step="post", alpha=0.15, color="#1f77b4", label="Carga batería (kWh)")
    ax2.set_ylabel("kWh almacenados")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=4)
    ax.set_title(f"{title}: precio vs. decisiones (semana desde {d['fecha'].min():%Y-%m-%d})")
    fig.tight_layout()
    return fig


def plot_cumulative(details: dict):
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, df in details.items():
        daily = df.groupby("fecha")["recompensa"].sum().cumsum() / 1000
        ax.plot(daily.index, daily.values, label=name, color=COLORS.get(name), lw=2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Ahorro acumulado (miles de COP)")
    ax.set_title("Ahorro acumulado frente a no tener batería")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pattern(pattern: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(pattern.index, pattern["cargar"], color="#2ca02c", label="% días cargando")
    ax.bar(pattern.index, -pattern["descargar"], color="#d62728", label="% días descargando")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("% de días")
    ax.set_title(f"{title}: patrón horario aprendido")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_prices():
    m = simulador.get_prices()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    daily = m.mean(axis=1)
    axes[0].plot(daily.index, daily.values, lw=0.8, color="#444")
    axes[0].plot(daily.rolling(30).mean(), lw=2, color="#d62728", label="Media móvil 30 días")
    axes[0].axvline(pd.Timestamp(simulador.TEST_START), ls="--", color="#1f77b4", label="Inicio prueba")
    axes[0].set_title("Precio de bolsa promedio diario (XM)")
    axes[0].set_ylabel("COP/kWh")
    axes[0].legend(fontsize=8)
    for year, g in m.groupby(m.index.year):
        axes[1].plot(range(24), g.mean(axis=0).values, marker="o", ms=3, label=str(year))
    axes[1].set_title("Perfil horario promedio por año")
    axes[1].set_xlabel("Hora")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
def run_dashboard(policy_key, start, end, capacity, power, efficiency, week_start):
    try:
        table, details = simulador.simulate_range(policy_key, start, end, capacity, power, efficiency)
    except ValueError as e:
        raise gr.Error(str(e))
    name = simulador.POLICY_KEYS[policy_key]
    detail = details[policy_key]
    details = {simulador.POLICY_KEYS[k]: v for k, v in details.items()}

    shown = table[["politica", "ahorro_total_cop", "ahorro_diario_cop", "ciclos_equivalentes", "pct_del_oraculo"]].copy()
    shown.columns = ["Política", "Ahorro total (COP)", "Ahorro diario (COP)", "Ciclos equivalentes", "% del oráculo"]
    shown = shown.round(1).reset_index(drop=True)

    note = ("⚠️ El rango incluye datos de entrenamiento (2024-2025): resultados optimistas."
            if start <= simulador.TRAIN_END else
            "✅ Rango dentro del periodo de prueba (no visto en entrenamiento).")
    return (
        shown,
        plot_decisions(detail, week_start, name),
        plot_cumulative(details),
        plot_pattern(simulador.hourly_pattern(detail), name),
        note,
    )


def format_trace(trace: list) -> str:
    if not trace:
        return "_El agente respondió sin usar herramientas._"
    lines = []
    for i, t in enumerate(trace, 1):
        lines.append(f"**{i}. `{t['tool']}`** `{json.dumps(t['args'], ensure_ascii=False)}`")
        if t["tool"] == "buscar_documentos" and t["result"].startswith("["):
            for h in json.loads(t["result"]):
                lines.append(f"  - 📄 [{h['fuente']}]({h['url']}) (score {h['score']})")
        else:
            preview = t["result"][:300] + ("…" if len(t["result"]) > 300 else "")
            lines.append(f"  - ↳ {preview}")
    return "\n".join(lines)


def get_text(content):
    """Extrae texto de los formatos de contenido usados por Gradio (Lecture 9)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        return first.get("text", "") if isinstance(first, dict) else str(first)
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def chat(message, history):
    # Historial de Gradio → mensajes OpenAI (solo texto, últimos 6 mensajes)
    past = [{"role": m["role"], "content": get_text(m["content"])} for m in history[-6:]]
    out = run_agent(message, history=past)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": out["answer"]},
    ]
    return history, "", format_trace(out["trace"])


# ─────────────────────────────────────────────────────────────────────────────
# Interfaz
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Despacho de batería con RL") as demo:
    gr.Markdown(
        "# 🔋 Despacho de batería con RL y asistente sobre regulación CREG\n"
        "Decide cuándo **cargar o descargar** una batería según el **precio de bolsa** real de XM. "
        f"Entrenamiento: 2024-2025 · Prueba: 2026 · Datos disponibles: {LO} a {HI}.\n\n"
        "_Herramienta académica: no constituye asesoría financiera, legal ni de seguridad._"
    )

    with gr.Tab("Simulación"):
        with gr.Row():
            with gr.Column(scale=1):
                policy = gr.Dropdown(POLICY_CHOICES, value="q_learning", label="Política")
                start = gr.Textbox(value=simulador.TEST_START, label="Fecha inicio (AAAA-MM-DD)")
                end = gr.Textbox(value=HI, label="Fecha fin (AAAA-MM-DD)")
                week = gr.Textbox(value="2026-04-06", label="Semana a visualizar (inicio)")
            with gr.Column(scale=1):
                capacity = gr.Slider(2, 50, value=10, step=1, label="Capacidad (kWh)")
                power = gr.Slider(0.5, 25, value=2.5, step=0.5, label="Potencia (kW)")
                efficiency = gr.Slider(0.7, 1.0, value=0.9, step=0.01, label="Eficiencia ida y vuelta")
                btn = gr.Button("Simular", variant="primary")
        note = gr.Markdown()
        table = gr.Dataframe(label="Comparación de políticas", interactive=False)
        fig_dec = gr.Plot(label="Precio vs. decisiones")
        fig_cum = gr.Plot(label="Ahorro acumulado")
        fig_pat = gr.Plot(label="Patrón horario")
        inputs = [policy, start, end, capacity, power, efficiency, week]
        outputs = [table, fig_dec, fig_cum, fig_pat, note]
        btn.click(run_dashboard, inputs, outputs)

    with gr.Tab("Datos XM"):
        gr.Markdown("Fuente: XM, *Precio de Bolsa Nacional* horario (API pública `servapibi.xm.com.co`).")
        fig_prices = gr.Plot()

    with gr.Tab("Asistente"):
        gr.Markdown(
            "Pregunta sobre precios, ahorros de las políticas o regulación CREG de baterías. "
            "Abajo se muestran las herramientas que usó el agente y las fuentes."
        )
        chatbot = gr.Chatbot(height=420, label="Agente")
        msg = gr.Textbox(placeholder="Ej: ¿Cuánto ahorra Q-learning en 2026 con una batería de 20 kWh y 5 kW?",
                         label="Mensaje")
        gr.Examples(
            [
                "¿Cuál fue el precio promedio de bolsa en marzo de 2026 y a qué hora fue más caro?",
                "¿Cuánto ahorra Q-learning frente al horario fijo entre 2026-01-01 y 2026-10-31?",
                "¿Qué es un SAEB según la CREG?",
                "¿Cómo se liquida la energía de un SAEB según la resolución 101 113 de 2026?",
            ],
            msg,
        )
        trace_md = gr.Markdown(label="Traza")
        msg.submit(chat, [msg, chatbot], [chatbot, msg, trace_md])

    demo.load(run_dashboard, inputs, outputs)
    demo.load(plot_prices, None, fig_prices)


if __name__ == "__main__":
    print("Entrenando políticas con los parámetros por defecto (~30 s)...")
    simulador.get_policies(10.0, 2.5, 0.9)
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
