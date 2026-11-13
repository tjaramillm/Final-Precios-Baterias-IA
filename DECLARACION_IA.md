# Declaración de uso de IA generativa

Conforme al pacto pedagógico de SI3003, el equipo declara el uso de las siguientes herramientas de IA generativa:

| Herramienta | Propósito | Alcance |
|---|---|---|
| **Claude Code (Anthropic, modelo Claude Opus)** | Asistente de programación: lectura del enunciado y de los notebooks del curso, propuesta de la arquitectura, escritura de código (módulos `src/`, `app.py`, notebooks), depuración, ajuste de hiperparámetros de Q-learning y redacción del README, el informe y esta declaración | Todo el código fue revisado, ejecutado y validado por el equipo. Las métricas provienen de ejecuciones reales. |
| **NVIDIA NIM – `nemotron-3.5-super-vl-120b-a12b`** | LLM del agente dentro de la aplicación (componente del producto) | Responde preguntas del usuario y llama herramientas |
| **NVIDIA NIM – `nemotron-3-embed-1b`** | Embeddings para el RAG (componente del producto) | Indexación y búsqueda en resoluciones CREG |

## Decisiones tomadas y verificadas por el equipo

- Formulación del MDP (estado, acciones, recompensa, episodios diarios) y elección de las líneas base.
- Ajustes de Q-learning respaldados por experimentos documentados en `notebooks/02`:
  tasa de aprendizaje $1/N(s,a)^{0.7}$, recompensa relativa al entrenar y 200 épocas.
- Verificación manual de que las cifras que entrega el agente coinciden con los datos (`notebooks/04`, sección 3).

## Fuentes de código

El estilo y la estructura se adaptaron de los notebooks del curso
([EAFIT-IA/si3003-artificial-intelligence](https://github.com/EAFIT-IA/si3003-artificial-intelligence)):
`lecture4/01_value_iteration_gridworld`, `lecture5/01_cheeseworld_q_learning`,
`lecture10/01_rag_nvidia_nim`, `Lecture11/part01_from_prompt_to_action` y `Lecture 9/04_nvidia_gradio_lab`.

_El equipo entiende y puede explicar todo el código entregado._
