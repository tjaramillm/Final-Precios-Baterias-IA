# Despacho de batería con aprendizaje por refuerzo y asistente sobre regulación CREG

**SI3003 Inteligencia Artificial – EAFIT 2026-1 · Integrantes:** Tomás Jaramillo, Mateo Zapata

## 1. Problema

Con el despliegue de sistemas de almacenamiento con baterías (SAEB), que la CREG regula desde la
Resolución 101 113 de 2026, surge una pregunta operativa: **¿en qué horas cargar y en cuáles descargar
para aprovechar las diferencias del precio de bolsa?** Se construyó un MVP que (i) aprende una política
de despacho con aprendizaje por refuerzo sobre precios reales, (ii) la compara con líneas base y (iii) la
explica mediante un agente LLM que también responde preguntas regulatorias con RAG.

## 2. Datos

- **Precio de Bolsa Nacional horario (XM)**, COP/kWh, 2024-01-01 → 2026-10-31: 1.035 días completos y sin nulos.
  Se descargó de la API pública de XM (`servapibi.xm.com.co`) en bloques de 30 días y se guardó como CSV (plan B).
  El precio de 2024 refleja El Niño: media de 558,7 COP/kWh en enero de 2024, frente a 212,4 en enero de 2026.
  El perfil diario es muy regular, con pico a las 18–21 h. El diferencial diario promedio es de 328 COP/kWh.
- **Corpus RAG:** Resoluciones CREG 101 113 de 2026 (SAEB), 098 de 2019 (SAEB, derogada), 174 de 2021
  (autogeneración) y 024 de 1995 (reglas de la bolsa), descargadas del gestor normativo de la CREG.
  Son ~700.000 caracteres que se dividen en 960 fragmentos.
- **División temporal:** entrenamiento 2024–2025 (731 días) y prueba 2026 (304 días no vistos).

## 3. Parte 1: MDP y aprendizaje por refuerzo

**Formulación.** Estado $s=(h, k, \ell)$, donde $h$ es la hora, $k\in\{0..N\}$ es el nivel de carga
($N$ = capacidad/potencia) y $\ell$ es el nivel de precio: bajo, medio o alto según $p_t/\bar p_{d-1}$ con umbrales
0.85 y 1.15. Es una variable causal y relativa, para que la política no dependa del nivel absoluto de precios.
Las acciones son descargar, esperar o cargar. La recompensa es $-p_t e$ al cargar y $+p_t e\eta$ al descargar
($\eta=0.9$ de ida y vuelta). Cada episodio es un día de 24 pasos que empieza con la batería vacía, con $\gamma=1$.
La recompensa acumulada es el ahorro frente a no tener batería.

**Métodos implementados por el equipo** (`src/politicas.py`):

1. **Q-learning** (clase 5) con ε-greedy, donde ε decrece de 1 a 0.05 en 200 épocas sobre los 731 días.
   Hubo dos ajustes motivados por experimentos. (a) Con $\alpha$ fijo, la Q-table oscilaba por el ruido de
   los precios: con 30 épocas se capturaba solo entre 57 % y 67 % del oráculo. Se usó $\alpha=1/N(s,a)^{0.7}$.
   (b) Al entrenar, la recompensa se divide por $\bar p_{d-1}$, porque los días de El Niño dominaban las actualizaciones.
2. **Iteración de valor con modelo estimado** (clase 4). Se estiman $P(\ell'|h,\ell)$ y $E[p|h,\ell]$ con
   los datos de entrenamiento y se resuelve Bellman de horizonte finito por inducción hacia atrás.
3. **Líneas base:** *horario fijo* (ingenua: carga en las $N$ horas más baratas del perfil promedio y
   descarga en las $N$ más caras) y *aleatoria*.
4. **Referencia: oráculo** con información perfecta del día, resuelto con programación dinámica. Es la cota superior.

**Métrica:** ahorro en COP en el periodo de prueba y porcentaje del ahorro del oráculo.

| Batería (prueba 2026) | Q-learning (5 semillas) | Iter. de valor | Horario fijo | Aleatoria |
|---|---|---|---|---|
| 10 kWh / 2.5 kW | **77.7 ± 1.7 %** | 75.8 % | 72.3 % | −146.6 % |
| 10 kWh / 5 kW | **76.8 ± 3.3 %** | 76.6 % | 70.6 % | −126.1 % |
| 10 kWh / 1.25 kW | **81.3 ± 2.1 %** | 80.5 % | 73.0 % | −148.5 % |

En la configuración por defecto, el oráculo ahorra 649.756 COP en 304 días, Q-learning 484.607 COP
(semilla 42) y el horario fijo 469.599 COP. Ambas políticas aprendidas superan a la ingenua en las tres
configuraciones y capturan ~¾ de la cota superior. La política aprendida carga de madrugada (0–3 h), descarga
entre las 18 y las 21 h y, a diferencia del horario fijo, **reacciona al precio del día**. No carga de madrugada
si el nivel de precio es alto, y aprovecha un segundo ciclo parcial por la tarde: descarga a las 15–16 h y
recarga a las 17 h, antes del pico. Por eso hace más ciclos equivalentes (346 frente a 304).

## 4. Parte 2: Agente LLM con RAG

- **Loop agéntico manual** (Lecture 11) sobre un endpoint compatible con OpenAI (NVIDIA NIM,
  `nemotron-3.5-super-vl-120b-a12b`), con $T_{max}=6$ iteraciones. Los errores de las herramientas
  (fechas fuera de rango, política inexistente, JSON inválido) se devuelven al modelo como observación.
- **RAG:** chunking semántico (≤ 900 caracteres), embeddings `nemotron-3-embed-1b` normalizados y FAISS
  `IndexFlatIP` (coseno) con umbral 0.25. Las preguntas fuera de dominio no recuperan fragmentos.
- **Herramientas:** `consultar_precios` (datos XM), `simular` (ejecuta el MDP de la Parte 1 y compara
  todas las políticas) y `buscar_documentos` (RAG). El system prompt exige usar herramientas para cualquier
  cifra y citar `[Resolución …]`, `[XM]` o `[Simulación MDP]`.

**Evaluación con 13 preguntas fijas** (datos, simulación, explicación, RAG, multi-herramienta, caso límite y
fuera de alcance): herramienta correcta **12/13 (92 %)**, cita correcta **13/13 (100 %)**, 1.8 iteraciones
en promedio. Todas las cifras reportadas coinciden con los valores calculados directamente desde los datos.
El único fallo es el caso límite "precio en 2030": el agente respondió bien que no hay datos, pero sin llamar
la herramienta, porque el system prompt ya le informa el rango. En la pregunta sobre arbitraje, el agente
distinguió lo que el fragmento afirma (la política de MinEnergía faculta a la CREG para reglar el arbitraje)
de lo que no confirma.

## 5. Dashboard

Aplicación Gradio (`python app.py`):

- **Simulación:** el usuario elige la política, el rango de fechas, la capacidad, la potencia y la eficiencia.
  Hay una tabla comparativa y tres gráficos: precio vs. decisiones con el estado de carga, ahorro acumulado
  de las 5 políticas y patrón horario aprendido.
- **Datos XM:** serie diaria con media móvil y perfil horario por año.
- **Asistente:** chat con traza visible de las herramientas llamadas, sus argumentos y las fuentes con enlace
  a la CREG.

## 6. Limitaciones y trabajo futuro

- **Modelo simplificado:** 3 niveles de precio, pasos de 1 hora, sin degradación ni autodescarga, y batería
  vacía al inicio de cada día. Un usuario regulado no paga el precio de bolsa directamente, así que el
  "ahorro" es un arbitraje teórico.
- **Información:** el precio de bolsa se conoce ex-post. En el MDP se supone observable en la hora de decisión,
  aunque el nivel de precio solo usa el día anterior como referencia.
- **Margen sobre el horario fijo:** es moderado (+3 a +8 pp) porque el perfil horario colombiano es muy
  regular. Variables como el nivel de embalses o la demanda (también disponibles en XM) podrían mejorarlo.
- **Radiación solar:** no se incorporó la del IDEAM (escenario batería + paneles).
- **Agente:** el LLM no es determinista, la verificación de citas es sintáctica y el corpus tiene solo 4 normas.
- **Uso:** es una herramienta académica y no constituye asesoría financiera, legal ni de seguridad.
