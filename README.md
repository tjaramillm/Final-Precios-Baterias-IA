# 🔋 Despacho de batería con RL y asistente sobre regulación CREG

**SI3003 – Inteligencia Artificial · Universidad EAFIT · 2026-1 · Proyecto final (idea 3.3)**

**Integrantes:** Tomás Jaramillo, Mateo Zapata

## Problema

¿Cuándo conviene **cargar o descargar una batería** según el **precio de bolsa** horario del mercado
eléctrico colombiano? La solución combina:

| Parte | Técnica | Dónde |
|---|---|---|
| **Parte 1 (clásica)** | MDP resuelto con **Q-learning** (clase 5) e **iteración de valor** con modelo estimado (clase 4), comparado con una política ingenua (horario fijo), una aleatoria y un oráculo de referencia | `src/entorno.py`, `src/politicas.py`, `notebooks/02` |
| **Parte 2 (moderna)** | **Agente LLM** con loop manual, **RAG** (embeddings NVIDIA + FAISS) sobre resoluciones CREG y **3 herramientas** | `src/agente.py`, `src/rag.py`, `notebooks/03` |
| **Dashboard** | Gradio: parámetros de la batería, política, fechas, 4 visualizaciones y chat con el agente | `app.py` |

La integración no es trivial: el agente **ejecuta el MDP** con la herramienta `simular`, explica las
decisiones aprendidas (horas de carga y descarga, ahorro frente a las líneas base) y las contrasta con la
regulación de baterías (SAEB) recuperada por RAG.

## Datos (reales, citados)

| Fuente | Contenido | Acceso |
|---|---|---|
| **XM** – Precio de Bolsa Nacional (`PrecBolsNaci`) | Horario, COP/kWh, 2024-01-01 → 2026-10-31 (1.035 días) | API pública `https://servapibi.xm.com.co/hourly` (la misma que usa `pydataxm`) |
| **CREG** – Resolución 101 113 de 2026 | SAEB en el SIN (vigente) | [gestor normativo](https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_101-113_2026.htm) |
| **CREG** – Resolución 098 de 2019 | SAEB (derogada, contexto histórico) | [gestor normativo](https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0098_2019.htm) |
| **CREG** – Resolución 174 de 2021 | Autogeneración a pequeña escala | [gestor normativo](https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0174_2021.htm) |
| **CREG** – Resolución 024 de 1995 | Reglas de la bolsa de energía | [gestor normativo](https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0024_1995.htm) |

**Plan B:** los datos ya descargados están en `data/raw/` y el índice vectorial en `data/index/`. Si las
fuentes fallan, la aplicación funciona con estos archivos. Para volver a descargar todo: `python src/datos.py`.
Solo se usan datos públicos y agregados; no hay datos personales.

## Cómo ejecutar

Requisitos: Python 3.11+ y una API key gratuita de NVIDIA (https://build.nvidia.com/).

```bash
# 1. Entorno
python -m venv .venv
.venv\Scripts\activate          # Windows   |   source .venv/bin/activate  (Linux/Mac)
pip install -r requirements.txt

# 2. Clave
copy .env.example .env          # Windows   |   cp .env.example .env
#    editar .env → NVIDIA_API_KEY=nvapi-...

# 3. Dashboard
python app.py                   # abre http://127.0.0.1:7860
```

El primer arranque entrena las políticas para la batería por defecto (~30 s). Si se cambian la capacidad,
la potencia o la eficiencia, se reentrena una vez (~30 s) y el resultado queda en caché.

**Notebooks** (ejecutar en orden desde `notebooks/`; ya incluyen sus salidas):

| Notebook | Contenido | Requiere API key |
|---|---|---|
| `01_datos_eda.ipynb` | Descarga, calidad de datos, serie de tiempo, perfil horario, división temporal | No |
| `02_mdp_qlearning.ipynb` | MDP, Q-learning, iteración de valor, líneas base, métricas, robustez | No |
| `03_agente_rag.ipynb` | Índice FAISS, herramientas, manejo de errores, trazas del agente, límite T_max | Sí |
| `04_evaluacion_agente.ipynb` | 13 preguntas fijas: herramienta correcta y citas | Sí |

Evaluación del agente por consola: `python src/evaluacion_agente.py` → `resultados/evaluacion_agente.{csv,md}`.

**Informe:** [`informe/informe.pdf`](informe/informe.pdf).

## Estructura

```
├── app.py                     Dashboard (Gradio)
├── src/
│   ├── datos.py               Descarga XM + CREG (requests, html.parser)
│   ├── entorno.py             MDP de la batería (estado, acciones, recompensa)
│   ├── politicas.py           Q-learning, iteración de valor, horario fijo, aleatoria, oráculo
│   ├── simulador.py           Entrenamiento con caché y simulación por rango de fechas
│   ├── rag.py                 Chunking, embeddings NVIDIA NIM, FAISS
│   ├── agente.py              Herramientas + loop agéntico (T_MAX = 6)
│   ├── config.py              Backend del LLM (NVIDIA / Groq / Ollama) leído de .env
│   └── evaluacion_agente.py   Preguntas fijas y métricas del agente
├── notebooks/                 01–04 (ejecutados)
├── data/raw/                  precio_bolsa_xm.csv, docs/*.txt (plan B)
├── data/index/                Índice FAISS + chunks
├── resultados/                Métricas de políticas y evaluación del agente
├── informe/informe.pdf        Informe breve (3 páginas)
└── DECLARACION_IA.md          Declaración de uso de IA generativa
```

## Parte 1: MDP de la batería

- **Estado** $s=(\text{hora},\ \text{nivel SOC},\ \text{nivel de precio})$. El nivel de precio es bajo, medio o alto según
  $p_t/\bar p_{\text{día anterior}}$ (umbrales 0.85 y 1.15). Es causal: no usa información futura.
- **Acciones:** descargar / esperar / cargar, una hora a potencia nominal.
- **Recompensa (COP):** cargar $-p_t e$, descargar $+p_t e \eta$, esperar 0. La suma es el ahorro frente a no tener batería.
- **Episodio:** un día (24 h), batería inicia vacía, $\gamma=1$.
- **Entrenamiento:** 2024–2025 (731 días). **Prueba:** 2026-01-01 → 2026-10-31 (304 días no vistos).

**Resultados en prueba, como % del ahorro del oráculo** (`resultados/robustez_politicas.csv`; Q-learning con 5 semillas):

| Batería | Q-learning | Iteración de valor | Horario fijo (ingenua) | Aleatoria |
|---|---|---|---|---|
| 10 kWh / 2.5 kW | **77.7 ± 1.7** | 75.8 | 72.3 | −146.6 |
| 10 kWh / 5 kW | **76.8 ± 3.3** | 76.6 | 70.6 | −126.1 |
| 10 kWh / 1.25 kW | **81.3 ± 2.1** | 80.5 | 73.0 | −148.5 |

En la batería por defecto (10 kWh / 2.5 kW), el oráculo ahorra 649.756 COP en 2026, Q-learning (semilla 42)
484.607 COP y el horario fijo 469.599 COP.

## Parte 2: Agente

| Herramienta | Descripción |
|---|---|
| `consultar_precios(fecha_inicio, fecha_fin)` | Estadísticas reales del precio de bolsa (XM) |
| `simular(politica, fecha_inicio, fecha_fin, capacidad_kwh, potencia_kw, eficiencia)` | Ejecuta el MDP y compara todas las políticas |
| `buscar_documentos(consulta)` | RAG sobre las 4 resoluciones CREG (960 chunks, umbral coseno 0.25) |

- Loop manual (Lecture 11) con **T_MAX = 6** iteraciones. Al alcanzarlo, el agente responde
  `[Agente detenido: se alcanzó el límite …]`.
- Los **errores de las herramientas se devuelven como observación**: fechas fuera de rango, política
  inexistente, JSON inválido o parámetros de batería fuera de límites.
- El system prompt exige **citar**: `[Resolución …]`, `[XM]` o `[Simulación MDP]`, y decir explícitamente
  cuando el contexto no alcanza.
- LLM: `nvidia/nemotron-3.5-super-vl-120b-a12b`. Embeddings: `nvidia/nemotron-3-embed-1b` (NVIDIA NIM, capa gratuita).

**Evaluación (13 preguntas fijas, `resultados/evaluacion_agente.md`):** herramienta correcta **12/13 (92 %)**,
cita correcta **13/13 (100 %)**. Las cifras se verificaron contra los datos originales. El único fallo es el caso
límite "precio en 2030": el agente respondió correctamente que no hay datos, pero sin llamar la herramienta.

## Limitaciones

- El MDP discretiza el precio en 3 niveles y la batería en pasos de 1 hora. No modela degradación,
  autodescarga ni la tarifa real de un usuario regulado, que no paga el precio de bolsa directamente.
- Se supone que el precio de la hora se conoce al decidir. En realidad el precio de bolsa se publica ex-post.
- La ventaja del RL sobre el horario fijo es moderada (+3 a +8 puntos) porque el perfil diario del precio
  en Colombia es muy regular (pico 18–21 h).
- No se usó radiación solar (IDEAM). Sería el siguiente paso, para batería + paneles.
- El agente depende de un LLM no determinista y de la disponibilidad de la API gratuita. La verificación de
  citas es sintáctica.
- **Herramienta académica: no constituye asesoría financiera, legal ni de seguridad.**

## Uso de IA generativa

Ver [`DECLARACION_IA.md`](DECLARACION_IA.md).
