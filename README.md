# Predicción temprana de hospitalización por dengue: Proyecto Integrador CRISP-DM

Proyecto de analítica predictiva (clasificación binaria) desarrollado con la metodología **CRISP-DM** (6 fases), a partir de datos abiertos del **SIVIGILA de la Alcaldía de Bucaramanga** publicados en Datos Abiertos Colombia.

**Pregunta de negocio:** con la información disponible en la *primera consulta* (datos sociodemográficos, síntomas, signos de alarma y días de evolución), ¿el paciente con dengue requerirá hospitalización?

| | |
|---|---|
| **Dataset** | [Dengue, dengue grave y mortalidad por dengue: Bucaramanga (datos.gov.co, `qzc7-jbg3`)](https://www.datos.gov.co/d/qzc7-jbg3), 28.626 registros x 76 variables (2015 a 2025) |
| **Variable objetivo** | `pac_hos_`: hospitalizado (31,6 %) / no hospitalizado (68,4 %) |
| **Métrica principal** | F1-score macro |
| **App en Streamlit** | `<URL pública de Streamlit.io>` |
| **Notebook en Colab** | `<enlace público de Google Colab>` |

## Estructura del repositorio

- `app.py`: aplicación web Streamlit.
- `pipeline_dengue.joblib`: pipeline completo serializado (preprocesamiento y modelo).
- `model_metadata.json`: variables, reglas de calidad, métricas y versiones.
- `requirements.txt`: dependencias con versiones fijadas.
- `README.md`: este archivo.
- `data/dengue_bucaramanga.csv`: dataset utilizado.
- `notebooks/proyecto_crispdm_dengue.ipynb`: notebook documentado con las 6 fases de CRISP-DM.
- `reports/reporte_profiling.html`: reporte exploratorio de ydata-profiling.
- `reports/figuras/` y `reports/tablas/`: gráficos y tablas de resultados generados por el notebook.
- `scripts/`: generadores del notebook y del documento de entrega.

## Metodología (CRISP-DM)

1. **Entendimiento del negocio:** triaje y planeación de camas en temporada de brote.
2. **Entendimiento de los datos:** diccionario de las 76 variables, reglas de calidad de negocio, distribución del target y sesgos (temporal, de brote, geográfico, de grupos minoritarios y de subregistro).
3. **Preparación:** eliminación de la fuga de información (`fec_hos_`, `conducta`, `clasfinal`, desenlace, entre otras), limpieza (centinela `-89`, fechas en formatos mixtos, categorías inconsistentes), imputación (KNN / moda), one-hot, escalado, análisis de relaciones lineales y no lineales (Pearson, Spearman, información mutua), PCA (evaluado y descartado) y SMOTE solo en entrenamiento.
4. **Modelamiento:** 5 modelos clásicos (Regresión Logística, KNN, Árbol de Decisión, SVM, MLP) y ensambles (Votación, Bagging, Random Forest, XGBoost). Ajuste de hiperparámetros con validación cruzada estratificada de 5 particiones sobre el 70 % de entrenamiento.
5. **Evaluación:** métricas en test (30 %), prueba t pareada corregida, prueba de McNemar e intervalos de confianza bootstrap.
6. **Despliegue:** pipeline serializado con `joblib` y aplicación Streamlit.

## Ejecución local

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py              # http://localhost:8501
```

Para reentrenar, ejecuta `notebooks/proyecto_crispdm_dengue.ipynb` (requiere además `ydata-profiling`, `seaborn`, `matplotlib` y `scipy`).

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub como **público** y haz *commit* de todos los archivos (en especial `pipeline_dengue.joblib` y `model_metadata.json`).
2. Entra a <https://share.streamlit.io> e inicia sesión con GitHub.
3. Elige **Create app** y luego **Deploy a public app from GitHub**, selecciona el repositorio, la rama `main` y el archivo `app.py`.
4. En **Advanced settings** selecciona **Python 3.12**, la misma versión con la que se entrenó el modelo.
5. Haz clic en **Deploy**. Cada *commit* a `main` vuelve a desplegar la app automáticamente.

> Las versiones de `requirements.txt` son las mismas del entrenamiento. Si cambia la versión de `scikit-learn`, el `.joblib` puede fallar al cargarse (`UnpicklingError` o `AttributeError`).

## Aviso

Este es un proyecto académico de apoyo a la decisión: **no reemplaza el criterio médico** ni la guía clínica de manejo del dengue del Ministerio de Salud.
