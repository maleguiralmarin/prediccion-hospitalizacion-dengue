"""Aplicación Streamlit: Predicción temprana de hospitalización por dengue.

Proyecto Integrador CRISP-DM (Fase 6: Despliegue).
Carga el pipeline completo serializado (imputación + escalado + codificación + modelo)
y predice si un paciente con dengue requerirá hospitalización a partir de los datos
disponibles en la primera consulta. Incluye además el reporte exploratorio de los datos.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
REPORTE = BASE / "reports" / "reporte_profiling.html"

st.set_page_config(page_title="Riesgo de hospitalización por dengue", layout="centered")


@st.cache_resource
def cargar_modelo():
    pipeline = joblib.load(BASE / "pipeline_dengue.joblib")
    meta = json.loads((BASE / "model_metadata.json").read_text(encoding="utf-8"))
    return pipeline, meta


@st.cache_data
def cargar_reporte():
    return REPORTE.read_text(encoding="utf-8") if REPORTE.exists() else None


pipeline, meta = cargar_modelo()


def pct(x):
    """Porcentaje con coma decimal, como se escribe en español (6,7 %)."""
    return f"{x * 100:.1f}".replace(".", ",") + " %"


def dec(x):
    """Número con coma decimal (0,845)."""
    return f"{x:.3f}".replace(".", ",")

SINTOMAS_LEVES = {
    "fiebre": "Fiebre", "cefalea": "Cefalea (dolor de cabeza)", "dolrretroo": "Dolor retroocular",
    "malgias": "Mialgias (dolor muscular)", "artralgia": "Artralgias (dolor articular)", "erupcionr": "Erupción o rash",
}
SIGNOS_ALARMA = {
    "dolor_abdo": "Dolor abdominal", "vomito": "Vómito", "diarrea": "Diarrea",
    "somnolenci": "Somnolencia o irritabilidad", "hipotensio": "Hipotensión", "hepatomeg": "Hepatomegalia",
    "hem_mucosa": "Hemorragia en mucosas", "hipotermia": "Hipotermia", "aum_hemato": "Aumento del hematocrito",
    "caida_plaq": "Caída de plaquetas (< 100.000)", "acum_liqui": "Acumulación de líquidos",
}
CONTEXTO = {
    "desplazami": "Se desplazó en los últimos 15 días", "famantdngu": "Familiar/conviviente con síntomas de dengue (15 días)",
    "gp_gestan": "Paciente gestante", "gp_migrant": "Población migrante",
}
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
         "Septiembre", "Octubre", "Noviembre", "Diciembre"]

st.title("Riesgo de hospitalización por dengue")
st.caption("Proyecto Integrador CRISP-DM, Datos SIVIGILA Bucaramanga 2015 a 2025 (Datos Abiertos Colombia)")

tab_pred, tab_rep = st.tabs(["Predicción", "Reporte exploratorio"])

with tab_pred:
    st.info("Herramienta académica de apoyo al triaje. **No reemplaza el criterio médico** ni la guía clínica del Ministerio de Salud.")

    with st.form("paciente"):
        st.subheader("1. Datos del paciente")
        c1, c2 = st.columns(2)
        grupo = c1.selectbox("Grupo de edad", list(meta["edad_grupos"].keys()), index=5)
        sexo = c2.radio("Sexo", ["F", "M"], horizontal=True, format_func=lambda s: "Femenino" if s == "F" else "Masculino")
        regimen = c1.selectbox("Régimen de afiliación", list(meta["regimen"].keys()), format_func=lambda k: meta["regimen"][k])
        estrato_opt = c2.selectbox("Estrato socioeconómico", ["Desconocido", 1, 2, 3, 4, 5, 6], index=3)

        st.subheader("2. Evolución")
        c3, c4 = st.columns(2)
        dias = c3.number_input("Días desde el inicio de síntomas hasta la consulta", min_value=0,
                               max_value=meta["reglas"]["dias_sintomas"][1], value=3, step=1)
        mes = c4.selectbox("Mes de inicio de síntomas", range(1, 13), index=5, format_func=lambda m: MESES[m - 1])

        st.subheader("3. Síntomas")
        cols = st.columns(2)
        valores = {}
        for i, (k, v) in enumerate(SINTOMAS_LEVES.items()):
            valores[k] = cols[i % 2].checkbox(v, value=(k == "fiebre"))
        st.markdown("**Signos de alarma**")
        cols = st.columns(2)
        for i, (k, v) in enumerate(SIGNOS_ALARMA.items()):
            valores[k] = cols[i % 2].checkbox(v)
        st.markdown("**Contexto**")
        cols = st.columns(2)
        for i, (k, v) in enumerate(CONTEXTO.items()):
            valores[k] = cols[i % 2].checkbox(v, disabled=(k == "gp_gestan" and sexo == "M"))

        enviar = st.form_submit_button("Predecir", type="primary")

    if enviar:
        registro = {
            "edad_aprox": meta["edad_grupos"][grupo],
            "dias_sintomas": dias,
            "estrato": None if estrato_opt == "Desconocido" else estrato_opt,
            "mes": mes,
            **{k: int(v) for k, v in valores.items()},
            "sexo": sexo,
            "regimen": regimen,
        }
        if sexo == "M":
            registro["gp_gestan"] = 0
        entrada = pd.DataFrame([registro])[meta["features"]].astype({"estrato": "float"})
        prob = float(pipeline.predict_proba(entrada)[0, 1])
        pred = int(pipeline.predict(entrada)[0])

        st.subheader("Resultado")
        if pred == 1:
            st.error(f"**Probable hospitalización.** Probabilidad estimada: **{pct(prob)}**")
        else:
            st.success(f"**Manejo ambulatorio probable.** Probabilidad de hospitalización: **{pct(prob)}**")
        st.progress(prob)
        if prob >= 0.7:
            st.write("Riesgo **alto**: priorizar valoración médica para hospitalización.")
        elif prob >= 0.5:
            st.write("Riesgo **moderado-alto**: vigilancia estrecha.")
        elif prob >= 0.3:
            st.write("Riesgo **moderado**: manejo ambulatorio con control a las 24 a 48 horas.")
        else:
            st.write("Riesgo **bajo**: manejo ambulatorio; explicar signos de alarma al paciente.")
        with st.expander("Ver datos enviados al modelo"):
            st.dataframe(entrada.T.rename(columns={0: "valor"}).astype(str))

    with st.expander("Acerca del modelo"):
        m = meta["metricas_test"]
        n_test = f"{meta['n_test']:,}".replace(",", ".")
        st.markdown(f"""
- **Modelo:** {meta['modelo']}
- **Desempeño en el conjunto de prueba (30 %, {n_test} casos):** F1-macro **{dec(m['F1'])}**, Accuracy {dec(m['Accuracy'])},
  Recall de hospitalizados {dec(m['Recall Hosp.'])}, ROC-AUC {dec(m['ROC-AUC'])}
- **Pipeline:** imputación (KNN / moda), luego escalado, luego one-hot, luego modelo (SMOTE usado solo en entrenamiento)
- **Datos:** [SIVIGILA Bucaramanga en datos.gov.co]({meta['dataset']})
- **Versiones de entrenamiento:** scikit-learn {meta['versiones']['scikit-learn']}, xgboost {meta['versiones']['xgboost']}
""")

with tab_rep:
    st.subheader("Reporte exploratorio de los datos")
    st.write("Reporte generado con ydata-profiling en la Fase 2 de CRISP-DM (Entendimiento de los datos), "
             "sobre las variables seleccionadas. La interfaz del reporte está en inglés porque así la genera la librería.")
    html = cargar_reporte()
    if html is None:
        st.warning("No se encontró el archivo reports/reporte_profiling.html.")
    else:
        st.download_button("Descargar reporte (HTML)", html, file_name="reporte_profiling.html", mime="text/html")
        st.iframe(html, height=900)
