"""
Agente LLM con RAG y herramientas (loop manual, adaptado de Lecture11/part01).

Herramientas:
  1. consultar_precios(fecha_inicio, fecha_fin)      → datos reales XM
  2. simular(politica, fecha_inicio, fecha_fin, ...)  → ejecuta el MDP (Parte 1)
  3. buscar_documentos(consulta)                      → RAG sobre resoluciones CREG

Buenas prácticas del enunciado:
  - T_MAX iteraciones en el loop.
  - Los errores de las herramientas se devuelven al modelo como observación.
  - El agente debe citar la fuente: [Resolución …] para documentos, [XM] o
    [Simulación MDP] para datos.
"""
import json

from config import MODEL, get_chat_client
import simulador

T_MAX = 6   # máximo de iteraciones del loop agéntico

_store = None


def _get_store():
    """Carga perezosa del índice vectorial (solo cuando el agente lo necesita)."""
    global _store
    if _store is None:
        from rag import get_vector_store
        _store = get_vector_store()
    return _store


# ── Implementaciones reales ──────────────────────────────────────────────────
def consultar_precios(fecha_inicio: str, fecha_fin: str) -> str:
    """Estadísticas del precio de bolsa horario (XM) en un rango de fechas."""
    stats = simulador.price_stats(fecha_inicio, fecha_fin)
    return json.dumps(stats, ensure_ascii=False)


def simular(politica: str, fecha_inicio: str, fecha_fin: str,
            capacidad_kwh: float = 10.0, potencia_kw: float = 2.5, eficiencia: float = 0.9) -> str:
    """Simula la política sobre precios reales y la compara con las demás."""
    table, details = simulador.simulate_range(
        politica, fecha_inicio, fecha_fin, capacidad_kwh, potencia_kw, eficiencia
    )
    pattern = simulador.hourly_pattern(details[politica])
    result = {
        "politica": simulador.POLICY_KEYS[politica],
        "parametros": {"capacidad_kwh": capacidad_kwh, "potencia_kw": potencia_kw, "eficiencia": eficiencia},
        "periodo": {"inicio": fecha_inicio, "fin": fecha_fin, "dias": int(table.iloc[0]["dias"])},
        "advertencia": (
            "El periodo incluye datos de entrenamiento (2024-2025): el resultado es optimista."
            if fecha_inicio <= simulador.TRAIN_END else "Periodo de prueba (no visto en entrenamiento)."
        ),
        "comparacion": {
            row["politica"]: {
                "ahorro_total_cop": round(row["ahorro_total_cop"]),
                "pct_del_oraculo": round(row["pct_del_oraculo"], 1),
                "ciclos_equivalentes": round(row["ciclos_equivalentes"], 1),
            }
            for _, row in table.iterrows()
        },
        "horas_donde_mas_carga": pattern["cargar"].nlargest(3).index.tolist(),
        "horas_donde_mas_descarga": pattern["descargar"].nlargest(3).index.tolist(),
        "fuente": "Simulación MDP con precio de bolsa real de XM",
    }
    return json.dumps(result, ensure_ascii=False)


def buscar_documentos(consulta: str, k: int = 4) -> str:
    """Busca fragmentos relevantes en las resoluciones CREG (RAG)."""
    hits = _get_store().search(consulta, k=k)
    if not hits:
        return "No se encontraron fragmentos relevantes en los documentos CREG."
    return json.dumps(
        [{"fuente": h["title"], "url": h["url"], "score": round(h["score"], 3),
          "texto": h["content"]} for h in hits],
        ensure_ascii=False,
    )


# ── Registro: nombre → función Python (el "router" del loop) ─────────────────
TOOL_REGISTRY = {
    "consultar_precios": consultar_precios,
    "simular": simular,
    "buscar_documentos": buscar_documentos,
}

# ── JSON schemas: lo que el modelo ve ────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "consultar_precios",
            "description": (
                "Consulta estadísticas reales del precio de bolsa de energía en Colombia (XM, COP/kWh) "
                "en un rango de fechas: promedio, mínimo, máximo, hora más barata y más cara, "
                "y perfil horario. Datos disponibles de 2024-01-01 a 2026-10-31."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fecha_inicio": {"type": "string", "description": "Fecha inicial AAAA-MM-DD"},
                    "fecha_fin": {"type": "string", "description": "Fecha final AAAA-MM-DD"},
                },
                "required": ["fecha_inicio", "fecha_fin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simular",
            "description": (
                "Simula una política de carga/descarga de una batería con precios de bolsa reales y "
                "devuelve el ahorro en COP comparado con todas las políticas (Q-learning, iteración de "
                "valor, horario fijo, aleatoria y oráculo) y las horas en que más carga/descarga. "
                "Úsala para preguntas de ahorro, escenarios o para explicar decisiones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "politica": {
                        "type": "string",
                        "enum": list(simulador.POLICY_KEYS),
                        "description": "Política a simular.",
                    },
                    "fecha_inicio": {"type": "string", "description": "Fecha inicial AAAA-MM-DD"},
                    "fecha_fin": {"type": "string", "description": "Fecha final AAAA-MM-DD"},
                    "capacidad_kwh": {"type": "number", "description": "Capacidad de la batería en kWh (defecto 10)"},
                    "potencia_kw": {"type": "number", "description": "Potencia de carga/descarga en kW (defecto 2.5)"},
                    "eficiencia": {"type": "number", "description": "Eficiencia ida y vuelta entre 0.5 y 1 (defecto 0.9)"},
                },
                "required": ["politica", "fecha_inicio", "fecha_fin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_documentos",
            "description": (
                "Busca en resoluciones de la CREG (baterías SAEB 101 113 de 2026 y 098 de 2019, "
                "autogeneración 174 de 2021, reglas de bolsa 024 de 1995). Úsala para preguntas "
                "de regulación, definiciones o normas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {"type": "string", "description": "Pregunta o términos de búsqueda."},
                },
                "required": ["consulta"],
            },
        },
    },
]

SYSTEM_PROMPT = """Eres un asistente sobre despacho de baterías y regulación del mercado eléctrico colombiano.

Herramientas:
- consultar_precios: datos reales del precio de bolsa (XM).
- simular: resultados de las políticas de la batería (MDP / Q-learning).
- buscar_documentos: resoluciones de la CREG.

Reglas:
1. Para cifras de precios o ahorros usa SIEMPRE una herramienta; nunca inventes números.
2. Para regulación usa buscar_documentos y responde SOLO con lo que digan los fragmentos.
   Si no hay información suficiente, dilo explícitamente.
3. Cita la fuente al final de cada afirmación: [nombre de la resolución], [XM] o [Simulación MDP].
4. Fechas en formato AAAA-MM-DD. Los datos van de 2024-01-01 a 2026-10-31.
   El periodo de prueba (no visto en entrenamiento) es 2026-01-01 a 2026-10-31.
5. Si una herramienta devuelve un error, explica el problema al usuario o corrige los argumentos.
6. Esto es una herramienta analítica: no es asesoría financiera, legal ni de seguridad.
Responde en español, de forma breve y clara."""


def _execute_tool(name: str, raw_args: str) -> tuple[dict, str]:
    """Ejecuta una herramienta. Cualquier error se devuelve como observación."""
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        return {}, f"Error: argumentos JSON inválidos ({e})."
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return args, f"Error: herramienta '{name}' no encontrada."
    try:
        return args, fn(**args)
    except Exception as e:  # el error vuelve al LLM como observación
        return args, f"Error en {name}: {e}"


def run_agent(user_message: str, history: list | None = None, verbose: bool = False) -> dict:
    """
    Ejecuta el agentic loop.

    Returns:
        {"answer": str, "trace": [{"tool", "args", "result"}], "iterations": int}
    """
    client = get_chat_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": user_message})
    trace = []

    for iteration in range(1, T_MAX + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2500,
            )
        except Exception as e:
            return {"answer": f"Error al contactar el LLM: {e}", "trace": trace, "iterations": iteration}

        choice = response.choices[0]
        msg = choice.message
        if verbose:
            print(f"  iter={iteration} finish_reason={choice.finish_reason}")

        # ── ¿Hay tool calls? ──────────────────────────────────────────────
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args, result = _execute_tool(tc.function.name, tc.function.arguments)
                trace.append({"tool": tc.function.name, "args": args, "result": result})
                if verbose:
                    print(f"  🔧 {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
                    print(f"     → {result[:150]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        # ── Respuesta final ───────────────────────────────────────────────
        return {"answer": (msg.content or "").strip(), "trace": trace, "iterations": iteration}

    return {
        "answer": f"[Agente detenido: se alcanzó el límite de {T_MAX} iteraciones]",
        "trace": trace,
        "iterations": T_MAX,
    }
