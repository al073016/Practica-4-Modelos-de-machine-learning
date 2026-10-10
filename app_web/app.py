"""Interfaz ejecutable de la Práctica 4 para Streamlit Community Cloud."""

import gc
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import streamlit as st
import tensorflow as tf

from modelos import TRAINERS, predict_one


st.set_page_config(page_title="Práctica 4 · Redes neuronales", page_icon="🧠", layout="wide")

REPO = "https://github.com/al073016/Practica-4-Modelos-de-machine-learning"
CASES = {
    "Clasificación bancaria": {
        "key": "bank", "tag": "PATTERN RECOGNITION · CLASIFICACIÓN",
        "question": "¿Contratará un depósito?",
        "description": "MLP que estima la probabilidad de suscripción a un depósito con datos de campañas bancarias. Se excluye la duración de la llamada para evitar usar información futura.",
        "dataset": "[Bank Marketing · UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing)",
        "notebook": "01_Clasificacion_Bancaria.ipynb", "reference": "ROC-AUC ≈ 0.792 · PR-AUC ≈ 0.438",
    },
    "Regresión energética": {
        "key": "energy", "tag": "FITTING · REGRESIÓN",
        "question": "¿Cuánta calefacción necesita?",
        "description": "MLP que estima la carga de calefacción de un diseño de edificio a partir de ocho características físicas. Los datos corresponden a simulaciones.",
        "dataset": "[Energy Efficiency · UCI](https://archive.ics.uci.edu/dataset/242/energy+efficiency)",
        "notebook": "02_Regresion_Energetica.ipynb", "reference": "MAE ≈ 0.493 · R² ≈ 0.9959",
    },
    "Series de tiempo FordA": {
        "key": "forda", "tag": "SERIES DE TIEMPO · CLASIFICACIÓN",
        "question": "¿Qué clase tiene esta señal?",
        "description": "CNN 1D que procesa 500 mediciones consecutivas de una señal de motor y asigna clase 0 o 1. No se atribuye significado físico no documentado a las etiquetas.",
        "dataset": "[FordA · UCR/UEA](https://www.timeseriesclassification.com/description.php?Dataset=FordA)",
        "notebook": "03_Series_Tiempo_FordA.ipynb", "reference": "Exactitud ≈ 89.9 % · F1 ≈ 0.895",
    },
}

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1180px;}
  h1, h2, h3 {letter-spacing: -.035em;}
  .hero {background: linear-gradient(115deg,#102d2b,#1b6650); color: white;
         padding: 32px 38px; border-radius: 18px; margin-bottom: 25px;}
  .hero .eyebrow {font-size: .76rem; letter-spacing: .16em; color: #b4e8c8; font-weight: 700;}
  .hero h1 {color: white; margin: 9px 0 12px; font-size: 2.5rem; line-height: 1.12;}
  .hero p {color: #e1f1e8; max-width: 790px; margin-bottom: 0;}
  .step {background: #f4f8f4; border: 1px solid #dce8dc; border-radius: 12px;
         padding: 16px 20px; height: 100%;}
  .step b {display:block; color:#087455; margin-bottom:5px;}
</style>
<div class="hero"><div class="eyebrow">PRÁCTICA 4 · PYTHON · EJECUCIÓN REAL</div>
<h1>Tres redes neuronales,<br>un laboratorio interactivo.</h1>
<p>Elige un problema, entrena la red con datos públicos y calcula una predicción nueva sin instalar nada en tu computadora.</p></div>
""", unsafe_allow_html=True)

st.markdown("### Cómo usar esta página")
cols = st.columns(3)
for column, number, heading, copy in zip(
    cols, ["01", "02", "03"], ["Elige una tarea", "Pulsa Entrenar", "Prueba un índice"],
    ["Selecciona banca, energía o FordA.", "Se descargan los datos, se comparan dos redes y se calculan métricas de prueba.",
     "Introduce cualquier índice válido y pulsa Predecir. La red entrenada calcula el resultado en ese momento."],
):
    with column:
        st.markdown(f'<div class="step"><b>{number} · {heading}</b>{copy}</div>', unsafe_allow_html=True)

st.divider()
choice = st.selectbox("Problema", list(CASES), label_visibility="collapsed")
case = CASES[choice]
kind = case["key"]
st.caption(case["tag"])
st.header(case["question"])
st.write(case["description"])
st.markdown(f'**Datos:** {case["dataset"]} · **Código reproducible:** [abrir notebook en Colab](https://colab.research.google.com/github/al073016/Practica-4-Modelos-de-machine-learning/blob/main/{case["notebook"]})')

run = st.session_state.get("run")
if run is not None and run["kind"] != kind:
    run = None

st.info("El entrenamiento se ejecuta cuando pulses el botón. Puede tardar varios minutos, especialmente FordA. Las cifras de referencia del notebook son solo para comparar; los resultados de abajo se calculan de nuevo.")
st.caption(f"Referencia medida antes: {case['reference']}")

if st.button("Repetir entrenamiento" if run else "Entrenar red neuronal", type="primary", width="stretch"):
    st.session_state.pop("last_prediction", None)
    if "run" in st.session_state:
        del st.session_state["run"]
        run = None
        tf.keras.backend.clear_session()
        gc.collect()

    with st.status("Descargando datos y entrenando…", expanded=True) as status:
        bar = st.progress(0.0, text="Preparando el dataset público…")
        progress_state = {"phase": -1, "architecture": None}

        def report(architecture: str, epoch: int, maximum: int, logs: dict):
            if architecture != progress_state["architecture"]:
                progress_state["phase"] += 1
                progress_state["architecture"] = architecture
            value = min((progress_state["phase"] + epoch / maximum) / 2, 1.0)
            loss = logs.get("val_loss")
            label = f"Arquitectura {architecture} · época {epoch}/{maximum}"
            if loss is not None:
                label += f" · pérdida de validación {loss:.4f}"
            bar.progress(value, text=label)

        try:
            trained = TRAINERS[kind](report=report)
        except Exception as error:
            status.update(label="El entrenamiento no pudo completarse", state="error")
            st.error(f"Error: {error}. Comprueba la conexión y vuelve a pulsar Entrenar.")
            st.stop()
        st.session_state["run"] = trained
        run = trained
        bar.progress(1.0, text="Modelo evaluado con datos de prueba")
        status.update(label="Entrenamiento y evaluación completados", state="complete")

if run is None:
    st.stop()

st.divider()
st.subheader("Resultados de esta ejecución")
st.caption("Arquitectura elegida con validación: " + " → ".join(map(str, run["configuration"])))
with st.expander("Parámetros de entrenamiento y selección"):
    details = {
        "bank": "MLP · capas densas 32→16 y 64→32 · dropout 0.15 · Adam (0.001) · lote 128 · máximo 50 épocas · selección por ROC-AUC en validación · umbral por F1 en validación · parada temprana (paciencia 7)",
        "energy": "MLP · capas densas 32→16 y 64→32 · Adam (0.001) · lote 16 · máximo 300 épocas · selección por MAE en validación · parada temprana (paciencia 25)",
        "forda": "CNN 1D · filtros 32→64 y 64→64, kernel 5 · Adam (0.001) · lote 64 · máximo 100 épocas · selección por F1 en validación · parada temprana (paciencia 12)",
    }
    st.write(details[kind])
training, validation, test = run["sizes"]
st.caption(f"Datos separados: {training:,} entrenamiento · {validation:,} validación · {test:,} prueba".replace(",", " "))
metric_columns = st.columns(len(run["metrics"]))
for column, (label, value) in zip(metric_columns, run["metrics"].items()):
    column.metric(label, f"{value:.4f}" if label == "R²" else f"{value:.3f}")

left, right = st.columns([1.1, 0.9])
with left:
    st.markdown("**Pérdida durante el entrenamiento**")
    st.line_chart(pd.DataFrame({name: pd.Series(values) for name, values in run["history"].items()}))
with right:
    if kind in ("bank", "forda"):
        labels = ["No", "Sí"] if kind == "bank" else ["Clase 0", "Clase 1"]
        st.markdown("**Matriz de confusión · prueba**")
        st.dataframe(pd.DataFrame(run["confusion"], index=[f"Real: {v}" for v in labels],
                                  columns=[f"Predicho: {v}" for v in labels]), width="stretch")
        if kind == "bank":
            st.caption(f"Umbral elegido con validación: {run['threshold']:.2f}. Referencia PR-AUC por prevalencia: {run['baseline']:.3f}.")
        else:
            st.caption(f"Adivinar siempre la clase mayoritaria daría {run['baseline']:.1%} de exactitud.")
    else:
        st.markdown("**Real frente a estimado · prueba**")
        st.scatter_chart(pd.DataFrame({"Valor real": run["actual"], "Estimación": run["pred_test"]}),
                         x="Valor real", y="Estimación")
        st.caption(f"MAE de referencia usando la media de entrenamiento: {run['baseline']:.3f}.")

st.divider()
st.subheader("Probar una predicción nueva")
st.write(f"Elige un índice entre **0 y {test - 1}**. Al pulsar **Predecir**, el modelo entrenado en esta sesión procesa ese caso de prueba otra vez.")
index = st.number_input("Índice del conjunto de prueba", min_value=0, max_value=test - 1, value=0, step=1)
if st.button("Predecir este caso", width="stretch"):
    st.session_state["last_prediction"] = {"kind": kind, "index": int(index),
                                           "result": predict_one(run, int(index))}

last = st.session_state.get("last_prediction")
if last is not None and last["kind"] == kind:
    result = last["result"]
    st.caption(f"Inferencia recién calculada para el índice {last['index']}")
    if kind == "energy":
        one, two, three = st.columns(3)
        one.metric("Carga estimada", f"{result['prediction']:.2f}")
        two.metric("Valor real simulado", f"{result['actual']:.2f}")
        three.metric("Error absoluto", f"{abs(result['prediction'] - result['actual']):.2f}")
        st.dataframe(result["features"], width="stretch")
    else:
        label = (lambda value: "Sí contrata" if value else "No contrata") if kind == "bank" else (lambda value: f"Clase {value}")
        one, two, three = st.columns(3)
        one.metric("Predicción", label(result["prediction"]))
        two.metric("Clase real", label(result["actual"]))
        three.metric("Probabilidad clase 1", f"{result['probability']:.1%}")
        if kind == "bank":
            st.dataframe(result["features"], width="stretch")
            st.caption("La probabilidad es una estimación estadística, no una decisión financiera automática.")
        else:
            st.line_chart(pd.DataFrame({"Señal": result["signal"]}))
            st.caption("Las etiquetas 0 y 1 no se interpretan como normal o falla sin documentación adicional.")

st.divider()
st.caption(f"[Código y notebooks en GitHub]({REPO}) · El dataset de cada modelo se descarga de la fuente enlazada arriba. Código desarrollado con ayuda de Codex; consulta PROMPTS_IA.md en el repositorio.")
