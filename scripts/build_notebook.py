"""Genera el notebook del proyecto integrador CRISP-DM (dengue Bucaramanga).

Uso:  .venv/Scripts/python scripts/build_notebook.py
Crea notebooks/proyecto_crispdm_dengue.ipynb (sin ejecutar).
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DESC = json.loads((ROOT / "scripts" / "descripciones.json").read_text(encoding="utf8"))

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
# PORTADA
# ---------------------------------------------------------------------------
md(r"""
# Proyecto Integrador: Metodología CRISP-DM
## Predicción temprana de hospitalización en pacientes con dengue (Bucaramanga, SIVIGILA 2015 a 2025)

| Campo | Valor |
|---|---|
| **Tipo de problema** | Clasificación binaria (aprendizaje supervisado) |
| **Fuente del dataset** | Datos Abiertos Colombia: Alcaldía de Bucaramanga (SIVIGILA) |
| **Enlace al dataset** | https://www.datos.gov.co/d/qzc7-jbg3 |
| **Variable objetivo** | `pac_hos_`: ¿el paciente fue hospitalizado? (1 = Sí, 2 = No) |
| **Métrica principal** | F1-score macro |
| **Modalidad** | Individual |

Este cuaderno sigue **estrictamente las seis fases de CRISP-DM** y el orden del formato de entrega *U4_0 Formato de Entrega Trabajo Integrador*:

1. **Entendimiento del negocio**: contexto, problema, objetivos y diseño de la solución.
2. **Entendimiento de los datos**: diccionario, reglas de calidad, variable objetivo y sesgos.
3. **Preparación de los datos**: selección, ydata-profiling, limpieza, nulos, codificación, relaciones, reducción de dimensiones y balanceo.
4. **Modelamiento**: 5 modelos clásicos del curso, ensambles (votación, bagging, boosting) y ajuste de hiperparámetros con validación cruzada sobre el 70 %.
5. **Evaluación**: métricas en el conjunto de prueba (30 %) y selección justificada técnica y estadísticamente.
6. **Despliegue**: serialización del pipeline completo, predicción con datos nuevos y aplicación Streamlit.

> Cada bloque de código va precedido de su **propósito** y seguido de su **interpretación**.
""")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
md(r"""
## 0. Configuración del entorno

En Google Colab se instalan las **mismas versiones** de librerías con las que se entrena y se despliega el modelo. Esto es indispensable: si la versión de `scikit-learn` del entrenamiento y la de la aplicación Streamlit no coinciden, al cargar el `.joblib` se puede producir un `UnpicklingError` (ver documento de despliegue del curso).
""")

code(r"""
import sys, os
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q scikit-learn==1.9.1 imbalanced-learn==0.14.2 xgboost==3.4.1 ydata-profiling==4.18.4 "setuptools<81"
    # Colab: se clona el repositorio para tener la misma estructura de carpetas (opcional)
print("Ejecutando en Colab:", IN_COLAB)
""")

code(r"""
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sklearn
import imblearn
import xgboost
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_validate,
                                     GridSearchCV, RandomizedSearchCV, cross_val_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import joblib
from sklearn.base import clone

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", palette="deep")

RANDOM_STATE = 42
N_JOBS = 4          # procesos en paralelo (limitado para no agotar la memoria RAM)
np.random.seed(RANDOM_STATE)

# Estructura de carpetas del proyecto
if Path.cwd().name == "notebooks":
    os.chdir("..")
for d in ["data", "reports/figuras", "reports/tablas", "models"]:
    Path(d).mkdir(parents=True, exist_ok=True)
FIG, TAB = Path("reports/figuras"), Path("reports/tablas")

def guardar_fig(nombre):
    plt.tight_layout()
    plt.savefig(FIG / f"{nombre}.png", dpi=130, bbox_inches="tight")
    plt.show()

def guardar_tabla(df_, nombre, index=False):
    df_.to_csv(TAB / f"{nombre}.csv", index=index, encoding="utf-8-sig")
    return df_

print("Python", sys.version.split()[0])
for m in [pd, np, sklearn, imblearn, xgboost, joblib, matplotlib]:
    print(f"{m.__name__:12s} {m.__version__}")
""")

# ---------------------------------------------------------------------------
# FASE 1
# ---------------------------------------------------------------------------
md(r"""
---
# FASE 1. Entendimiento del negocio

### Descripción del negocio
El **dengue** es la arbovirosis de mayor impacto en salud pública en Colombia. Es transmitida por el mosquito *Aedes aegypti* y Bucaramanga (Santander) es zona hiperendémica. Cada caso es notificado por las instituciones de salud (UPGD) al **Sistema Nacional de Vigilancia en Salud Pública (SIVIGILA)**. La Alcaldía de Bucaramanga publica en Datos Abiertos Colombia el seguimiento de los eventos *Dengue* (cód. 210), *Dengue grave* (220) y *Mortalidad por dengue* (580), con variables sociodemográficas, de afiliación al sistema de salud, de pertenencia a grupos poblacionales, síntomas, signos de alarma, fechas clave y el desenlace del caso.

La red hospitalaria de la ciudad enfrenta **picos de demanda** durante los brotes (el año 2024 concentra el 40 % de los casos del periodo). La decisión de hospitalizar a un paciente es costosa en ambos sentidos: hospitalizar a quien no lo necesita ocupa camas escasas; enviar a casa a quien sí lo necesita puede derivar en dengue grave o muerte.

### Descripción del problema
**¿Es posible predecir, con la información disponible en la primera consulta (datos sociodemográficos, síntomas y signos de alarma, tiempo de evolución), si un paciente con dengue requerirá hospitalización?**

Un modelo de este tipo apoyaría el **triaje** y la **planeación de camas** de la Secretaría de Salud y de las IPS, y complementaría (no reemplazaría) el criterio médico y la guía clínica del Ministerio de Salud.

### Objetivos de la minería de datos
**Objetivo general:** construir un modelo de clasificación supervisada que estime si un caso notificado de dengue requerirá hospitalización, a partir de variables disponibles en la primera atención.

**Objetivos específicos (medibles):**
1. Construir un conjunto de datos limpio y libre de fuga de información (*data leakage*), aplicando reglas de calidad de negocio.
2. Comparar al menos **5 modelos clásicos** (los vistos en el curso) y **3 modelos de ensamble** con validación cruzada estratificada de 5 particiones sobre el 70 % de entrenamiento.
3. Alcanzar un **F1-macro mayor o igual a 0,85** en el conjunto de prueba (30 %) con el modelo final, con un **Recall de la clase "Hospitalizado" mayor o igual a 0,80**.
4. Desplegar el pipeline completo en una aplicación web pública en Streamlit.io que permita predecir casos nuevos.

**Criterio de éxito de negocio:** el modelo identifica correctamente al menos 8 de cada 10 pacientes que terminan hospitalizados, con una tasa de falsas alarmas razonable para apoyar el triaje.
""")

md(r"""
### Diseño de la solución

| Tipo de análisis | Tipo de aprendizaje | Modelos propuestos | Métrica principal |
|---|---|---|---|
| Predictivo | Supervisado (clasificación binaria) | **Clásicos (curso):** Regresión Logística, KNN, Árbol de Decisión, SVM, Red Neuronal (MLP). **Ensambles:** Votación (suave), Bagging (árboles), Random Forest, Boosting (XGBoost) | **F1-score macro** |

**Justificación técnica de la métrica.** La clase positiva (*hospitalizado*) representa el 31,6 % de los casos, por lo que la *Accuracy* es engañosa: un modelo que siempre dijera "no hospitalizar" tendría 68 % de exactitud y sería inútil. El **F1-macro** promedia el F1 de ambas clases, equilibra precisión y sensibilidad y **da el mismo peso a los falsos positivos y a los falsos negativos de cada clase**, que es lo que se requiere cuando ambos errores tienen costo (camas ocupadas sin necesidad vs. pacientes graves enviados a casa). Como métrica de apoyo se reporta el **Recall de la clase Hospitalizado**, por su relevancia clínica.

**Flujo de la solución:** datos SIVIGILA, luego limpieza y reglas de calidad, luego selección de variables sin fuga, luego *pipeline* (imputación + escalado + codificación + SMOTE + modelo), luego validación cruzada y ajuste, luego evaluación en el conjunto de prueba, luego serialización (`joblib`), luego app Streamlit.
""")

# ---------------------------------------------------------------------------
# FASE 2
# ---------------------------------------------------------------------------
md(r"""
---
# FASE 2. Entendimiento de los datos

### 2.0 Carga del dataset
Se lee el archivo local si existe; de lo contrario se descarga directamente desde el portal oficial de Datos Abiertos Colombia. Todas las columnas se leen como texto (`dtype=str`) para **no perder información** (por ejemplo, códigos con ceros a la izquierda o el centinela `-89`) y controlar explícitamente cada conversión de tipo.
""")

code(r"""
URL = "https://www.datos.gov.co/api/views/qzc7-jbg3/rows.csv?accessType=DOWNLOAD"
LOCAL = Path("data/dengue_bucaramanga.csv")
if LOCAL.exists():
    df_raw = pd.read_csv(LOCAL, dtype=str)
else:
    df_raw = pd.read_csv(URL, dtype=str)
    df_raw.to_csv(LOCAL, index=False)
print("Dimensiones:", df_raw.shape)
df_raw.head()
""")

code(r"""
df_raw.info(verbose=False)
print("\nRegistros por evento:")
print(df_raw["nom_eve"].value_counts())
print("\nRegistros por año:")
print(df_raw["año"].value_counts().sort_index())
""")

md(r"""
### 2.1 Diccionario de datos
El diccionario documenta las **76 variables originales**: descripción oficial (metadatos del portal), tipo (numérica / categórica / fecha), rol en el modelo (entrada, salida o no usada) y, si no se usa, el motivo. Las variables marcadas como *Entrada* se transforman en la Fase 3 (por ejemplo `grupo_etario` se convierte en `edad_aprox`, y las fechas se convierten en `dias_sintomas` y `mes`).
""")

code("DESCRIPCIONES = " + json.dumps(DESC, ensure_ascii=False, indent=1))

code(r"""
SINTOMAS = ["fiebre", "cefalea", "dolrretroo", "malgias", "artralgia", "erupcionr",
            "dolor_abdo", "vomito", "diarrea", "somnolenci", "hipotensio", "hepatomeg",
            "hem_mucosa", "hipotermia", "aum_hemato", "caida_plaq", "acum_liqui"]
OTRAS_BIN = ["desplazami", "famantdngu", "gp_gestan", "gp_migrant"]
SIGNOS_GRAVE = ["extravasac", "hemorr_hem", "choque", "daño_organ"]
MUESTRAS = ["muesttejid", "mueshigado", "muesbazo", "muespulmon", "muescerebr",
            "muesmiocar", "muesmedula", "muesriñon"]

ROL = {}  # variable -> (rol, motivo)
for v in SINTOMAS + OTRAS_BIN + ["sexo_", "tip_ss_", "estrato_"]:
    ROL[v] = ("Entrada", "")
ROL["grupo_etario"] = ("Entrada", "Se transforma en edad_aprox (numérica)")
ROL["fec_con_"] = ("Entrada", "Se usa para calcular dias_sintomas")
ROL["ini_sin_"] = ("Entrada", "Se usa para calcular dias_sintomas y mes")
ROL["pac_hos_"] = ("Salida", "Variable objetivo")
fuga = {"fec_hos_": "Fuga: fecha de hospitalización (revela el target)",
        "conducta": "Fuga: decisión médica final (ambulatoria/hospitalización)",
        "clasfinal": "Fuga: clasificación final posterior a la atención",
        "nom_eve": "Fuga: evento final (dengue grave/mortalidad) definido después",
        "cod_eve": "Fuga: código del evento final", "tip_cas_": "Fuga: confirmación posterior (laboratorio)",
        "ajuste_": "Fuga: ajuste posterior del caso", "con_fin_": "Fuga: desenlace (vivo/muerto)",
        "fec_def_": "Fuga: fecha de defunción"}
for v in SIGNOS_GRAVE: fuga[v] = "Fuga: signo de dengue grave registrado durante la hospitalización"
for v in MUESTRAS: fuga[v] = "Fuga: muestra de autopsia (solo casos fallecidos)"
for v, m in fuga.items(): ROL[v] = ("No usada", m)
otros = {"orden": "Identificador consecutivo", "grupo": "Constante (un solo valor)",
         "fec_not": "Fecha administrativa de notificación", "semana": "Redundante con la fecha de inicio de síntomas",
         "año": "No es serie de tiempo; además presenta deriva (ver sesgos)",
         "clasif_edad": "Redundante con grupo_etario", "def_clas_edad": "Redundante con grupo_etario",
         "ocupacion_": "855 códigos, sin catálogo y 45 % códigos genéricos",
         "cod_ase_": "Alta cardinalidad (119); su información la resume tip_ss_",
         "aseguradora": "Alta cardinalidad (134 nombres inconsistentes)",
         "per_etn_": "Casi constante (99,3 % = 6 'otro')", "sem_ges_": "99,6 % sin dato",
         "fuente_": "Variable del proceso de notificación, no del paciente",
         "cod_pais_r": "Casi constante / 15 % sin dato", "cod_dpto_r": "Casi constante (98,9 % Santander)",
         "cod_mun_r": "Casi constante (96 % Bucaramanga)", "version": "Versión del software SIVIGILA",
         "gp_otros": "Categoría residual sin significado clínico"}
for v in ["gp_discapa", "gp_desplaz", "gp_carcela", "gp_indigen", "gp_pobicbf", "gp_mad_com",
          "gp_desmovi", "gp_psiquia", "gp_vic_vio"]:
    otros[v] = "Casi constante (< 0,3 % de casos positivos)"
for v, m in otros.items(): ROL[v] = ("No usada", m)

NUMERICAS_ORIG = {"orden", "semana", "año", "estrato_", "sem_ges_"}
FECHAS_ORIG = {"fec_not", "fec_con_", "ini_sin_", "fec_hos_", "fec_def_"}
def tipo(v):
    if v in FECHAS_ORIG: return "Fecha"
    return "Num" if v in NUMERICAS_ORIG else "Cat"

diccionario = pd.DataFrame([{
    "Variable": v,
    "Descripción": DESCRIPCIONES.get(v, ""),
    "Tipo (Num/Cat)": tipo(v),
    "Entrada/Salida": ROL.get(v, ("No usada", ""))[0],
    "Observación": ROL.get(v, ("", ""))[1],
} for v in df_raw.columns])
guardar_tabla(diccionario, "2_1_diccionario")
print(diccionario["Entrada/Salida"].value_counts())
diccionario
""")

md(r"""
**Interpretación.** De las 76 variables, 25 son candidatas de entrada (17 síntomas/signos de alarma, 4 binarias de contexto, 3 sociodemográficas y la edad), dos fechas se usan solo para construir variables derivadas y **una es la salida** (`pac_hos_`). La mayoría de las variables descartadas lo son por **fuga de información** (describen lo que pasó *después* de decidir hospitalizar) o por ser casi constantes. La justificación detallada está en la sección 3.1.
""")

md(r"""
### 2.2 Reglas de calidad
Las reglas se definen **desde el conocimiento del negocio y de los metadatos oficiales**, no a partir del mínimo y máximo observados (eso sería un error: el dataset puede contener errores, como 341 días de síntomas). Para cada regla se cuantifica cuántos registros la incumplen en los datos crudos.

| Variable | Regla | Fuente de la regla |
|---|---|---|
| `edad` (por `grupo_etario`) | 0 a 110 años; 11 grupos válidos | Clasificación del Minsalud |
| `dias_sintomas` = consulta - inicio síntomas | 0 a 30 días | La fase febril del dengue dura 2 a 7 días; más de 30 días no corresponde a dengue agudo |
| `estrato_` | 1 a 6 | Estratificación socioeconómica colombiana |
| `mes` | 1 a 12 | Calendario |
| `año` | 2015 a 2025 | Periodo publicado |
| `semana` | 1 a 53 | Semana epidemiológica |
| Síntomas, signos y grupos (desde `fiebre` hasta `gp_migrant`) | {1 = Sí, 2 = No} | Ficha de notificación SIVIGILA (metadatos) |
| `famantdngu` | {1, 2} (3 = desconocido, se trata como faltante) | Metadatos |
| `sexo_` | {M, F} | Metadatos |
| `tip_ss_` | {C, S, P, E, N, I} | Metadatos |
| `pac_hos_` | {1, 2} | Metadatos |
""")

code(r"""
def parse_fecha(s):
    # El portal mezcla formatos ISO (2024-06-11T00:00:00.000) y dd/mm/aaaa
    iso = pd.to_datetime(s, format="ISO8601", errors="coerce")
    dmy = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(dmy)

REGIMEN = {"C": "C", "CONTRIBUTIVO": "C", "S": "S", "SUBSIDIADO": "S", "P": "P", "EXCEPCIÓN": "P",
           "EXCEPCIÃ“N": "P", "E": "E", "ESPECIAL": "E", "N": "N", "NO ASEGURADO": "N",
           "I": "I", "INDETERMINADO": "I"}
GRUPOS_VALIDOS = ["MENOR DE 1", "1 A 4", "5 A 9", "10 A 14", "15 A 19", "20 A 29",
                  "30 A 39", "40 A 49", "50 A 59", "60 A 69", "70 Y MAS"]
_dias = (parse_fecha(df_raw["fec_con_"]) - parse_fecha(df_raw["ini_sin_"])).dt.days

def num(s): return pd.to_numeric(s, errors="coerce")

reglas_num = [
    ("dias_sintomas", "0 a 30", _dias, 0, 30),
    ("estrato_", "1 a 6", num(df_raw["estrato_"]), 1, 6),
    ("mes (inicio síntomas)", "1 a 12", parse_fecha(df_raw["ini_sin_"]).dt.month, 1, 12),
    ("año", "2015 a 2025", num(df_raw["año"]), 2015, 2025),
    ("semana", "1 a 53", num(df_raw["semana"]), 1, 53),
]
filas = []
for var, rango, serie, lo, hi in reglas_num:
    fuera = ((serie < lo) | (serie > hi)).sum()
    nulos = serie.isna().sum()
    filas.append({"Variable": var, "Rango esperado": rango, "Mín. observado": serie.min(),
                  "Máx. observado": serie.max(), "Fuera de rango": int(fuera),
                  "Sin dato (-89/vacío)": int(nulos), "% incumple": round(100 * (fuera + nulos) / len(serie), 2)})
calidad_num = guardar_tabla(pd.DataFrame(filas), "2_2_reglas_numericas")
calidad_num
""")

code(r"""
reglas_cat = {v: ["1", "2"] for v in SINTOMAS + OTRAS_BIN}
reglas_cat.update({"sexo_": ["M", "F"], "tip_ss_": ["C", "S", "P", "E", "N", "I"],
                   "pac_hos_": ["1", "2"], "grupo_etario": GRUPOS_VALIDOS})
filas = []
for var, validas in reglas_cat.items():
    s = df_raw[var]
    invalidos = (~s.isin(validas)).sum()
    ejemplos = s[~s.isin(validas)].value_counts().head(4).to_dict()
    ejemplos = {("EXCEPCIÓN (texto dañado)" if "Ã" in str(k) else k): v for k, v in ejemplos.items()}
    filas.append({"Variable": var, "Categorías válidas": ", ".join(validas),
                  "Registros que incumplen": int(invalidos),
                  "% incumple": round(100 * invalidos / len(s), 2),
                  "Valores inválidos más frecuentes": ejemplos})
calidad_cat = guardar_tabla(pd.DataFrame(filas), "2_2_reglas_categoricas")
calidad_cat
""")

md(r"""
**Interpretación.** Se confirman problemas de calidad que deben tratarse en la Fase 3:
- **Centinela `-89` (sin dato)**: afecta sobre todo a `estrato_` (aproximadamente 20 %) y a `famantdngu`; además `famantdngu` usa `3 = desconocido`.
- **Categorías inconsistentes**: `grupo_etario` aparece en minúsculas y mayúsculas (`"20 a 29"` / `"20 A 29"`); `tip_ss_` mezcla códigos y nombres (`"C"` / `"CONTRIBUTIVO"`) e incluso presenta **errores de codificación de caracteres** (la palabra EXCEPCIÓN aparece con caracteres dañados).
- **Valores fuera de dominio**: síntomas con valor `3`; `gp_gestan` con valores 3 y 4 (no definidos en los metadatos).
- **Valores atípicos imposibles**: hasta 341 días entre inicio de síntomas y consulta.
- **Formatos de fecha mixtos** (ISO y dd/mm/aaaa), que se resolvieron con un parser doble.
""")

md(r"""
### 2.3 Distribución de la variable objetivo
""")

code(r"""
y_raw = df_raw["pac_hos_"].map({"1": "Hospitalizado", "2": "No hospitalizado"})
dist = y_raw.value_counts().to_frame("Casos")
dist["%"] = (100 * dist["Casos"] / dist["Casos"].sum()).round(2)
guardar_tabla(dist.reset_index().rename(columns={"pac_hos_": "Clase"}), "2_3_target")
print(dist)
print("Razón de desbalance (mayoritaria/minoritaria): %.2f : 1" % (dist["Casos"].max() / dist["Casos"].min()))

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=y_raw, order=["No hospitalizado", "Hospitalizado"], ax=ax, hue=y_raw, legend=False)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}\n({100*p.get_height()/len(y_raw):.1f} %)",
                (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
ax.set_title("Distribución de la variable objetivo (pac_hos_)"); ax.set_xlabel(""); ax.set_ylabel("Casos")
ax.set_ylim(0, dist["Casos"].max() * 1.2)
guardar_fig("2_3_distribucion_target")
""")

md(r"""
**Interpretación.** El problema está **moderadamente desbalanceado** (aproximadamente 2,17 : 1). No es un desbalance extremo, pero sí suficiente para que un clasificador ingenuo favorezca la clase mayoritaria y para que la *Accuracy* no sea una métrica adecuada. Esto justifica (i) usar **F1-macro** como métrica principal, (ii) particionar de forma **estratificada** y (iii) aplicar una técnica de **balanceo solo sobre los datos de entrenamiento** (sección 3.7).
""")

md(r"""
### 2.4 Identificación de posibles sesgos
""")

code(r"""
tmp = df_raw.assign(y=(df_raw["pac_hos_"] == "1").astype(int), año=num(df_raw["año"]))
por_año = tmp.groupby("año").agg(casos=("y", "size"), tasa_hosp=("y", "mean"))
por_año["tasa_hosp"] = (100 * por_año["tasa_hosp"]).round(1)
guardar_tabla(por_año.reset_index(), "2_4_sesgo_temporal")

fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.bar(por_año.index, por_año["casos"], color="#9ecae1", label="Casos notificados")
ax1.set_ylabel("Casos notificados")
ax2 = ax1.twinx()
ax2.plot(por_año.index, por_año["tasa_hosp"], color="#d62728", marker="o", label="% hospitalizados")
ax2.set_ylabel("% hospitalizados"); ax2.grid(False)
ax1.set_title("Sesgo temporal: casos y tasa de hospitalización por año")
fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.9))
guardar_fig("2_4_sesgo_temporal")
por_año.T
""")

code(r"""
sesgos = []
sesgos.append(("Temporal / deriva", f"Tasa de hospitalización entre {por_año.tasa_hosp.min()} % y {por_año.tasa_hosp.max()} % según el año"))
sesgos.append(("Concentración en brote", f"El año 2024 aporta el {100*(tmp['año']==2024).mean():.1f} % de los registros"))
sesgos.append(("Geográfico", f"{100*(df_raw['cod_mun_r'].isin(['1','001'])&(df_raw['cod_dpto_r']=='68')).mean():.1f} % reside en Bucaramanga"))
for g, nombre in [("gp_migrant", "Migrantes"), ("gp_gestan", "Gestantes"), ("gp_indigen", "Indígenas")]:
    n = (df_raw[g] == "1").sum()
    sesgos.append(("Grupos minoritarios", f"{nombre}: {n} casos ({100*n/len(df_raw):.2f} %)"))
reg = tmp.assign(regimen=df_raw["tip_ss_"].str.upper().str.strip().map(REGIMEN)).groupby("regimen")["y"].agg(casos="size", tasa_hosp="mean").round(3)
sesgos.append(("Subregistro", "Solo casos notificados al SIVIGILA (los leves que no consultan no aparecen)"))
tabla_sesgos = guardar_tabla(pd.DataFrame(sesgos, columns=["Tipo de sesgo", "Evidencia"]), "2_4_sesgos")
display(tabla_sesgos)
guardar_tabla(reg.reset_index(), "2_4_sesgo_regimen")
print("Tasa de hospitalización por régimen (tip_ss_ normalizado):"); print(reg)
""")

md(r"""
**Interpretación de sesgos.**
- **Sesgo temporal (deriva del concepto):** la proporción de hospitalizados varía de forma drástica entre años (de aproximadamente 12 % en 2015 y 2016 a cerca de 60 % en 2022), lo que refleja cambios en protocolos clínicos, en la virulencia de los serotipos circulantes y en la forma de notificar. El modelo aprenderá principalmente el comportamiento reciente (2023 a 2025 suman el 62 % de los datos), por lo que **debe monitorearse y reentrenarse** periódicamente. No se usa `año` como predictor porque el objetivo no es pronosticar series de tiempo y porque amarraría el modelo a periodos pasados.
- **Sesgo de brote:** 2024 concentra el 40 % de los casos; el perfil de ese brote domina el aprendizaje.
- **Sesgo geográfico:** casi todos los casos son de Bucaramanga; el modelo **no debe generalizarse** a otras ciudades sin validación.
- **Grupos minoritarios** (migrantes, gestantes, indígenas) con muy pocos casos: el desempeño en esos subgrupos es incierto (riesgo de equidad).
- **Sesgo de selección / subregistro:** solo existen los casos que consultaron y fueron notificados.
- **Sesgo por régimen de afiliación:** la tasa de hospitalización difiere entre regímenes (contributivo, subsidiado, excepción), lo que puede reflejar acceso al servicio más que gravedad clínica.
""")

# ---------------------------------------------------------------------------
# FASE 3
# ---------------------------------------------------------------------------
md(r"""
---
# FASE 3. Preparación de los datos

### 3.1 Selección de variables
Criterios aplicados, en este orden:
1. **Eliminar fuga de información (*data leakage*)**: todo lo que se conoce *después* de decidir la hospitalización. Igual que en el caso de estudio del curso se eliminó la *Nota Final* para predecir *Aprobó*, aquí se eliminan `fec_hos_`, `conducta`, `clasfinal`, el evento final, la confirmación de laboratorio, el desenlace y las muestras de autopsia.
2. **Eliminar identificadores y variables administrativas** (`orden`, `version`, `fuente_`, `fec_not`).
3. **Eliminar variables casi constantes o de altísima cardinalidad** sin valor predictivo generalizable.
4. **Construir variables derivadas** con significado clínico: `edad_aprox` (punto medio del grupo etario), `dias_sintomas` (tiempo de evolución al consultar) y `mes` (estacionalidad).

Primero se **verifica empíricamente la fuga** de las variables más sospechosas:
""")

code(r"""
print("fec_hos_ vacía (1900-01-01) vs target:")
print(pd.crosstab(df_raw["fec_hos_"].str.startswith("1900"), df_raw["pac_hos_"], margins=True))
cond = df_raw["conducta"].replace({"AMBULATORIA": "1", "HOSPITALIZACION PISO": "2", "UNIDAD DE CUIDADOS INTENSIVOS": "3"})
print("\nconducta (1=ambulatoria, 2=hosp. piso, 3=UCI) vs target:")
print(pd.crosstab(cond, df_raw["pac_hos_"]).loc[["1", "2", "3"]])
""")

md(r"""
**Interpretación.** `fec_hos_` coincide con el target en el **99,99 %** de los casos y `conducta` es literalmente la decisión de hospitalizar: incluirlas daría un modelo "perfecto" en el notebook pero **inútil en la práctica**, porque esa información no existe en el momento de la consulta.

**Iteración CRISP-DM (de la Fase 3 a la Fase 1).** Los signos de *dengue grave* (`extravasac`, `hemorr_hem`, `choque`, `daño_organ`) generaron una discusión: son hallazgos clínicos, pero en la práctica se documentan casi siempre **durante la hospitalización**. Se realizó un experimento (abajo) que muestra cuánto "ayudan" al modelo; con base en él **se volvió a la Fase 1 para precisar el problema**: *predecir en la primera consulta*, con lo cual estos signos quedan excluidos.
""")

code(r"""
# Experimento: ¿cuánto aportan los signos de dengue grave? (modelo rápido, solo para la decisión)
_b = lambda c: df_raw[c].map({"1": 1, "2": 0})
X_exp = pd.DataFrame({c: _b(c) for c in SINTOMAS}).fillna(0)
y_exp = (df_raw["pac_hos_"] == "1").astype(int)
cv_exp = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
f1_sin = cross_val_score(XGBClassifier(n_estimators=200, verbosity=0, n_jobs=N_JOBS), X_exp, y_exp, cv=cv_exp, scoring="f1_macro").mean()
X_exp2 = X_exp.join(pd.DataFrame({c: _b(c) for c in SIGNOS_GRAVE}).fillna(0))
f1_con = cross_val_score(XGBClassifier(n_estimators=200, verbosity=0, n_jobs=N_JOBS), X_exp2, y_exp, cv=cv_exp, scoring="f1_macro").mean()
print(f"F1-macro solo síntomas: {f1_sin:.4f} | con signos de dengue grave: {f1_con:.4f} | diferencia: {f1_con-f1_sin:+.4f}")
""")

code(r"""
TARGET = "hospitalizado"
NUM_FEATS = ["edad_aprox", "dias_sintomas", "estrato", "mes"]
BIN_FEATS = SINTOMAS + OTRAS_BIN
CAT_FEATS = ["sexo", "regimen"]
FEATURES = NUM_FEATS + BIN_FEATS + CAT_FEATS

JUSTIF = {
    "edad_aprox": "Derivada de grupo_etario. Niños y adultos mayores tienen más riesgo de complicación",
    "dias_sintomas": "Derivada (consulta - inicio). La fase crítica del dengue ocurre entre los días 3 y 7",
    "estrato": "Proxy socioeconómico (acceso a servicios, condiciones de vivienda)",
    "mes": "Derivada de ini_sin_. Captura la estacionalidad de los brotes",
    "sexo": "Variable demográfica básica", "regimen": "Régimen de afiliación (tip_ss_ normalizado)",
    "desplazami": "Desplazamiento reciente (exposición)", "famantdngu": "Nexo epidemiológico familiar",
    "gp_gestan": "Gestantes: grupo de riesgo según guía clínica", "gp_migrant": "Migrantes: posible barrera de acceso",
}
for s in SINTOMAS[:6]: JUSTIF[s] = "Síntoma de dengue sin signos de alarma (guía clínica)"
for s in SINTOMAS[6:]: JUSTIF[s] = "Signo de alarma de dengue: criterio clínico de hospitalización"
seleccion = guardar_tabla(pd.DataFrame({"Variable seleccionada": FEATURES,
                                        "Justificación": [JUSTIF[f] for f in FEATURES]}), "3_1_seleccion")
print(f"Variables de entrada: {len(FEATURES)} ({len(NUM_FEATS)} numéricas, {len(BIN_FEATS)} binarias, {len(CAT_FEATS)} categóricas)")
seleccion
""")

md(r"""
### 3.2 Descripción estadística y reporte ydata-profiling
Se construye el dataset con las variables seleccionadas **tal como vienen** (antes de limpiar), de modo que el reporte exploratorio evidencie los problemas de calidad que luego se corrigen. El reporte HTML se guarda en `reports/reporte_profiling.html` y se adjunta a la entrega.
""")

code(r"""
df_sel = pd.DataFrame({
    "grupo_etario": df_raw["grupo_etario"], "dias_sintomas": _dias,
    "estrato": df_raw["estrato_"], "mes": parse_fecha(df_raw["ini_sin_"]).dt.month,
    **{c: df_raw[c] for c in BIN_FEATS},
    "sexo": df_raw["sexo_"], "regimen": df_raw["tip_ss_"], "pac_hos_": df_raw["pac_hos_"],
})
desc = pd.DataFrame({"Métrica": ["Número de registros", "Número de variables (dataset original)",
                                 "Número de variables de entrada seleccionadas", "Variables numéricas",
                                 "Variables categóricas (21 binarias + 2 nominales)", "Variable objetivo",
                                 "Registros duplicados exactos en la fuente"],
                     "Resultado": [len(df_raw), df_raw.shape[1], len(FEATURES), len(NUM_FEATS),
                                   len(BIN_FEATS) + len(CAT_FEATS), "1 (pac_hos_, renombrada como hospitalizado)",
                                   int(df_raw.drop(columns="orden").duplicated().sum())]})
guardar_tabla(desc, "3_2_descripcion")
display(desc)
df_sel.describe(include="all").T.head(30)
""")

code(r"""
from ydata_profiling import ProfileReport
t0 = time.time()
profile = ProfileReport(df_sel, title="Dengue Bucaramanga: Reporte exploratorio (variables seleccionadas)",
                        explorative=True, progress_bar=False)
profile.config.interactions.continuous = False   # evita miles de gráficos de dispersión
profile.to_file("reports/reporte_profiling.html")
print(f"Reporte generado en {time.time()-t0:.0f} s -> reports/reporte_profiling.html")
""")

md(r"""
**Interpretación del reporte.** El reporte ydata-profiling confirma las alertas detectadas con las reglas de calidad: valores `-89` en `estrato`, categorías duplicadas por mayúsculas en `grupo_etario` y `regimen`, valores `3` en las binarias y una distribución muy asimétrica de `dias_sintomas` (mediana 3 días con cola hasta 341). También muestra que varias binarias están muy desbalanceadas (p. ej. `hipotermia`, `hipotensio`), lo que es esperable en signos de alarma poco frecuentes.
""")

md(r"""
### 3.3 Limpieza de errores
La limpieza de errores es **determinística** (reglas fijas que no "aprenden" de los datos), por eso puede aplicarse a todo el dataset antes de la partición sin generar fuga. Lo que sí aprende de los datos (imputación, escalado, codificación, SMOTE) se ajusta solo con entrenamiento dentro del *pipeline*.
""")

code(r"""
EDAD_GRUPO = {"MENOR DE 1": 0.5, "1 A 4": 2.5, "5 A 9": 7, "10 A 14": 12, "15 A 19": 17,
              "20 A 29": 24.5, "30 A 39": 34.5, "40 A 49": 44.5, "50 A 59": 54.5,
              "60 A 69": 64.5, "70 Y MAS": 75}

registro_limpieza = []
def log(tipo, accion, n):
    registro_limpieza.append({"Tipo de error": tipo, "Acción realizada": accion, "Registros afectados": int(n)})

def limpiar(df_in):
    out = pd.DataFrame(index=df_in.index)
    # Edad: normalizar mayúsculas y mapear a punto medio
    ge = df_in["grupo_etario"].str.upper().str.strip()
    log("Categorías inconsistentes (mayúsculas/minúsculas) en grupo_etario",
        "Normalización a mayúsculas y conversión a edad_aprox (punto medio del grupo)",
        (df_in["grupo_etario"] != ge).sum())
    out["edad_aprox"] = ge.map(EDAD_GRUPO)
    # Fechas mixtas y días de síntomas
    fc, fi = parse_fecha(df_in["fec_con_"]), parse_fecha(df_in["ini_sin_"])
    log("Formatos de fecha mixtos (ISO y dd/mm/aaaa)", "Parseo con ambos formatos",
        df_in["fec_con_"].str.contains("/").sum())
    d = (fc - fi).dt.days
    fuera = (d < 0) | (d > 30)
    log("dias_sintomas fuera de la regla [0, 30]", "Se convierte en NaN (se imputa luego)", fuera.sum())
    out["dias_sintomas"] = d.where(~fuera)
    out["mes"] = fi.dt.month
    # Estrato
    e = pd.to_numeric(df_in["estrato_"], errors="coerce")
    log("Centinela -89 o vacío en estrato_", "Se convierte en NaN", (~e.between(1, 6)).sum())
    out["estrato"] = e.where(e.between(1, 6))
    # Binarias: 1 -> 1, 2 -> 0, resto (-89, 3, 4) -> NaN
    n_inv = 0
    for c in BIN_FEATS:
        n_inv += (~df_in[c].isin(["1", "2"])).sum()
        out[c] = df_in[c].map({"1": 1, "2": 0})
    log("Binarias con -89, 3 (desconocido) o 4 (fuera de dominio)",
        "Recodificación: 1 se mantiene como 1, 2 pasa a 0 y los valores inválidos pasan a NaN", n_inv)
    # Categóricas
    out["sexo"] = df_in["sexo_"].where(df_in["sexo_"].isin(["M", "F"]))
    reg = df_in["tip_ss_"].str.upper().str.strip()
    log("tip_ss_ mezcla códigos y nombres, con error de codificación (la palabra EXCEPCIÓN aparece con caracteres dañados)",
        "Mapeo a códigos oficiales C/S/P/E/N/I", (reg.str.len() > 1).sum())
    out["regimen"] = reg.map(REGIMEN)
    # Target
    out[TARGET] = df_in["pac_hos_"].map({"1": 1, "2": 0})
    return out

mask_dup = df_raw.drop(columns="orden").duplicated()
df = limpiar(df_raw.loc[~mask_dup])
log("Registros duplicados exactos (todas las columnas excepto el consecutivo 'orden')",
    "Se eliminan (misma notificación registrada dos veces)", mask_dup.sum())
tabla_limpieza = guardar_tabla(pd.DataFrame(registro_limpieza), "3_3_limpieza")
display(tabla_limpieza)
print("Dataset limpio:", df.shape); df.head()
""")

md(r"""
**Interpretación.** Tras la limpieza todas las variables cumplen su regla de calidad: los valores que no cumplen se transformaron en **faltantes explícitos (NaN)** en lugar de eliminar registros, para no perder casos y dejar que el *pipeline* los impute de forma controlada.

### Partición 70 / 30 (estratificada)
Se separa el **30 % como conjunto de prueba** que no se toca hasta la Fase 5. Todo el ajuste (imputación, escalado, SMOTE, validación cruzada e hiperparámetros) se hace **solo con el 70 % de entrenamiento**.
""")

code(r"""
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
particion = pd.DataFrame({"Conjunto": ["Entrenamiento (70 %)", "Prueba (30 %)"],
                          "Registros": [len(X_train), len(X_test)],
                          "% Hospitalizado": [round(100 * y_train.mean(), 2), round(100 * y_test.mean(), 2)]})
guardar_tabla(particion, "3_particion"); particion
""")

md(r"""
### 3.4 Limpieza de valores nulos
""")

code(r"""
nulos = pd.DataFrame({"% Nulos (total)": (100 * df[FEATURES].isna().mean()).round(2),
                      "% Nulos (entrenamiento)": (100 * X_train.isna().mean()).round(2)})
nulos["Método aplicado"] = np.where(nulos.index.isin(NUM_FEATS),
                                    "KNNImputer (k=5) dentro del pipeline",
                                    "SimpleImputer (moda) dentro del pipeline")
nulos = nulos[nulos["% Nulos (total)"] > 0].sort_values("% Nulos (total)", ascending=False)
guardar_tabla(nulos.reset_index().rename(columns={"index": "Variable"}), "3_4_nulos")
nulos
""")

md(r"""
**Justificación de los métodos.**
- **Numéricas (`estrato`, `dias_sintomas`): `KNNImputer(n_neighbors=5)`**: estima el valor a partir de los 5 casos más parecidos en las demás numéricas (edad, mes, días), conservando relaciones entre variables mejor que la media. Es el mismo criterio aplicado en el caso de estudio del curso.
- **Binarias y categóricas: `SimpleImputer(strategy="most_frequent")`**: la moda es la imputación natural para variables nominales; los porcentajes faltantes son bajos, salvo `famantdngu` (~19 %), donde la moda ("No") es también la respuesta clínicamente más probable.
- Los imputadores se ajustan **solo con los datos de entrenamiento de cada partición** (están dentro del *pipeline*), con lo que se evita la fuga de información hacia la validación y el conjunto de prueba.
""")

md(r"""
### 3.5 Codificación de variables categóricas y escalado
""")

code(r"""
codif = pd.DataFrame([
    ("sexo", "OneHotEncoder(handle_unknown='ignore')", "Nominal sin orden"),
    ("regimen", "OneHotEncoder(handle_unknown='ignore')", "Nominal sin orden (6 categorías)"),
    ("17 síntomas/signos + 4 binarias", "Codificación binaria 1/0 (1 = Sí; el 2 original pasa a 0 = No)", "Dicotómicas: un solo indicador basta"),
    ("grupo_etario (convertida en edad_aprox)", "Codificación ordinal numérica (punto medio del rango)", "Conserva el orden y la magnitud de la edad"),
    ("estrato", "Ordinal numérica (1 a 6)", "El estrato tiene orden natural"),
    ("Numéricas (edad_aprox, dias_sintomas, estrato, mes)", "StandardScaler", "LR, SVM, KNN y MLP son sensibles a la escala"),
    ("hospitalizado (target)", "Codificación de etiquetas 1 = Hospitalizado, 0 = No", "La clase positiva es la de interés clínico"),
], columns=["Variable", "Método de codificación", "Justificación"])
guardar_tabla(codif, "3_5_codificacion"); codif
""")

code(r"""
def crear_preprocesador():
    num = Pipeline([("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())])
    binp = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_FEATS), ("bin", binp, BIN_FEATS), ("cat", cat, CAT_FEATS)])

prep_demo = crear_preprocesador().fit(X_train)
Xtr_prep = pd.DataFrame(prep_demo.transform(X_train), columns=prep_demo.get_feature_names_out(), index=X_train.index)
print("Dimensión tras preprocesar:", Xtr_prep.shape)
Xtr_prep.head()
""")

md(r"""
**Interpretación.** El preprocesamiento produce 33 columnas: 4 numéricas escaladas, 21 binarias y 8 indicadoras one-hot (2 de sexo + 6 de régimen). `handle_unknown="ignore"` garantiza que una categoría nueva en producción no rompa la aplicación (problema común de despliegue señalado en el material del curso).

### Relaciones lineales y no lineales
- **Lineal:** correlación de **Pearson**.
- **No lineal (monótona):** correlación de **Spearman**.
- **No lineal (general):** **Información mutua**, que detecta cualquier dependencia (no solo monótona) entre cada variable y el target.
""")

code(r"""
corr_df = Xtr_prep.copy(); corr_df[TARGET] = y_train.values
corr_df.columns = [c.split("__")[-1] for c in corr_df.columns]
pear = corr_df.corr(method="pearson"); spear = corr_df.corr(method="spearman")
fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
for ax, m, t in [(axes[0], pear, "Pearson (lineal)"), (axes[1], spear, "Spearman (monótona no lineal)")]:
    sns.heatmap(m, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": .6})
    ax.set_title(f"Correlación de {t}")
guardar_fig("3_6_correlaciones")
""")

code(r"""
mi = mutual_info_classif(Xtr_prep, y_train, discrete_features=[not c.startswith("num__") for c in Xtr_prep.columns],
                         random_state=RANDOM_STATE)
rel = pd.DataFrame({"Pearson con target": pear[TARGET].drop(TARGET),
                    "Spearman con target": spear[TARGET].drop(TARGET),
                    "Información mutua": pd.Series(mi, index=pear.columns.drop(TARGET))}).round(4)
rel = rel.sort_values("Información mutua", ascending=False)
guardar_tabla(rel.reset_index().rename(columns={"index": "Variable"}), "3_6_relaciones")
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
rel["Pearson con target"].sort_values().plot.barh(ax=ax[0], color="#4c72b0"); ax[0].set_title("Correlación de Pearson con el target")
rel["Información mutua"].sort_values().plot.barh(ax=ax[1], color="#dd8452"); ax[1].set_title("Información mutua con el target")
guardar_fig("3_6_relacion_target")
pares = pear.where(np.triu(np.ones(pear.shape, dtype=bool), 1)).stack().sort_values(key=abs, ascending=False)
print("Pares de variables más correlacionados (Pearson):"); print(pares.head(8).round(3))
rel.head(12)
""")

md(r"""
**Interpretación.** *(se completa con los resultados: ver la sección de interpretación de resultados al final de la fase)*.
""")

md(r"""
### 3.6 Reducción de dimensiones
Se evalúa **PCA** sobre los datos preprocesados: cuántos componentes se requieren para retener el 90 % / 95 % de la varianza y si mejora el desempeño de un modelo base (Regresión Logística) en validación cruzada.
""")

code(r"""
pca = PCA(random_state=RANDOM_STATE).fit(Xtr_prep)
var_acum = np.cumsum(pca.explained_variance_ratio_)
k90, k95 = int(np.argmax(var_acum >= .90) + 1), int(np.argmax(var_acum >= .95) + 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(var_acum) + 1), var_acum, marker="o", ms=3)
for k, c in [(k90, "orange"), (k95, "red")]:
    ax.axvline(k, ls="--", color=c, label=f"{int(var_acum[k-1]*100)} % con {k} componentes")
ax.set_xlabel("Número de componentes"); ax.set_ylabel("Varianza explicada acumulada"); ax.legend()
ax.set_title("PCA: varianza explicada acumulada")
guardar_fig("3_7_pca")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
base_lr = LogisticRegression(max_iter=2000)
f1_sin_pca = cross_val_score(ImbPipeline([("prep", crear_preprocesador()), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", base_lr)]),
                             X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=N_JOBS)
f1_con_pca = cross_val_score(ImbPipeline([("prep", crear_preprocesador()), ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                                          ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", base_lr)]),
                             X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=N_JOBS)
pca_res = pd.DataFrame({"Configuración": ["Sin PCA (33 variables)", f"Con PCA 95 % ({k95} componentes)"],
                        "F1-macro CV (media)": [f1_sin_pca.mean(), f1_con_pca.mean()],
                        "F1-macro CV (desv. est.)": [f1_sin_pca.std(), f1_con_pca.std()]}).round(4)
guardar_tabla(pca_res, "3_7_pca_resultado"); pca_res
""")

md(r"""
### 3.7 Balanceo de clases
Se comparan en validación cruzada (Regresión Logística como modelo de referencia) cuatro estrategias. **El balanceo se aplica dentro del pipeline de `imblearn`**, de modo que solo se remuestrea cada partición de entrenamiento: **las particiones de validación y el conjunto de prueba conservan la distribución real** (balancear el conjunto de prueba daría métricas irreales).
""")

code(r"""
estrategias = {
    "Sin balanceo": ImbPipeline([("prep", crear_preprocesador()), ("model", LogisticRegression(max_iter=2000))]),
    "class_weight='balanced'": ImbPipeline([("prep", crear_preprocesador()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "SMOTE (sobremuestreo sintético)": ImbPipeline([("prep", crear_preprocesador()), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", LogisticRegression(max_iter=2000))]),
    "RandomUnderSampler (submuestreo)": ImbPipeline([("prep", crear_preprocesador()), ("rus", RandomUnderSampler(random_state=RANDOM_STATE)), ("model", LogisticRegression(max_iter=2000))]),
}
filas = []
for nombre, pipe in estrategias.items():
    r = cross_validate(pipe, X_train, y_train, cv=skf, n_jobs=N_JOBS,
                       scoring={"f1_macro": "f1_macro", "recall_hosp": "recall", "precision_hosp": "precision"})
    filas.append({"Técnica": nombre, "F1-macro": r["test_f1_macro"].mean(),
                  "Recall (Hospitalizado)": r["test_recall_hosp"].mean(),
                  "Precision (Hospitalizado)": r["test_precision_hosp"].mean()})
balanceo = guardar_tabla(pd.DataFrame(filas).round(4), "3_8_balanceo"); balanceo
""")

md(r"""
**Decisión.** Se adopta **SMOTE dentro del pipeline** para todos los modelos. Aunque `class_weight` produce un efecto similar en la Regresión Logística, **no está disponible en KNN ni en MLP**, por lo que SMOTE permite tratar a los 10 modelos de forma homogénea y comparable. SMOTE genera ejemplos sintéticos de la clase minoritaria interpolando vecinos, en lugar de duplicar registros (sobreajuste) o descartar datos (pérdida de información, como el submuestreo).
""")

# ---------------------------------------------------------------------------
# FASE 4
# ---------------------------------------------------------------------------
md(r"""
---
# FASE 4. Modelamiento

Todos los modelos usan **exactamente el mismo pipeline** (`preprocesador, luego SMOTE, luego modelo`) y **las mismas 5 particiones** de validación cruzada estratificada (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) **sobre el 70 % de entrenamiento**. Así las diferencias de desempeño se deben solo al algoritmo, y los puntajes por partición son **pareados**, lo que permite pruebas estadísticas en la Fase 5.

### 4.1 Modelos clásicos (modelos vistos en el curso)
""")

code(r"""
def crear_pipeline(modelo):
    return ImbPipeline([("prep", crear_preprocesador()), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", modelo)])

SCORING = {"f1_macro": "f1_macro", "accuracy": "accuracy", "recall_hosp": "recall", "roc_auc": "roc_auc"}

clasicos = {
    "Regresión Logística": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    "Red Neuronal (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
}
HIP_INI = {"Regresión Logística": "C=1.0, penalty='l2', solver='lbfgs'", "KNN": "n_neighbors=5, weights='uniform', p=2",
           "Árbol de Decisión": "max_depth=None, min_samples_leaf=1, criterion='gini'", "SVM": "kernel='rbf', C=1.0, gamma='scale'",
           "Red Neuronal (MLP)": "hidden_layer_sizes=(100,), activation='relu', alpha=0.0001, early_stopping=True"}

cv_scores = {}   # puntajes F1 por partición (para pruebas estadísticas)
def evaluar_cv(nombre, modelo):
    t0 = time.time()
    r = cross_validate(crear_pipeline(modelo), X_train, y_train, cv=skf, scoring=SCORING, n_jobs=N_JOBS, return_train_score=True)
    cv_scores[nombre] = r["test_f1_macro"]
    return {"Modelo": nombre, "CV Mean (F1-macro)": r["test_f1_macro"].mean(), "CV Std": r["test_f1_macro"].std(),
            "F1 entrenamiento": r["train_f1_macro"].mean(), "Recall Hosp.": r["test_recall_hosp"].mean(),
            "ROC-AUC": r["test_roc_auc"].mean(), "Tiempo (s)": time.time() - t0}

res_clasicos = pd.DataFrame([evaluar_cv(n, m) for n, m in clasicos.items()])
res_clasicos.insert(1, "Hiperparámetros iniciales", res_clasicos["Modelo"].map(HIP_INI))
res_clasicos.insert(2, "Métrica", "F1-macro")
res_clasicos = res_clasicos.round(4).sort_values("CV Mean (F1-macro)", ascending=False)
guardar_tabla(res_clasicos, "4_1_clasicos"); res_clasicos
""")

md(r"""
### 4.2 Modelos de ensamble (actividad investigativa)

| Ensamble | Idea | Qué reduce |
|---|---|---|
| **Votación (Voting, suave)** | Combina modelos **distintos** promediando sus probabilidades. Funciona mejor cuando los modelos cometen errores diferentes (diversidad). | Varianza y error de modelo individual |
| **Bagging** (*Bootstrap Aggregating*, Breiman 1996) | Entrena muchos árboles sobre **muestras bootstrap** (con reemplazo) y vota. Promediar modelos inestables reduce su variabilidad. | **Varianza** |
| **Random Forest** (Breiman 2001) | Bagging + selección aleatoria de variables en cada división, lo que **descorrelaciona** los árboles. | Varianza (más que bagging) |
| **Boosting con XGBoost** (Chen & Guestrin 2016) | Entrena árboles **secuencialmente**; cada uno corrige los residuos (gradiente) del anterior, con regularización L1/L2. | **Sesgo** (y varianza por regularización) |

La **votación** usa los 3 mejores modelos clásicos según la validación cruzada inicial (por diversidad y desempeño).
""")

code(r"""
top3 = res_clasicos["Modelo"].head(3).tolist()
def con_proba(nombre):
    m = clone(clasicos[nombre])
    if isinstance(m, SVC): m.set_params(probability=True)
    return m
print("Modelos de la votación:", top3)

ensambles = {
    "Votación (suave)": VotingClassifier([(n, con_proba(n)) for n in top3], voting="soft"),
    "Bagging (árboles)": BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
    "Boosting (XGBoost)": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, eval_metric="logloss",
                                        random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
}
TIPO = {"Votación (suave)": "Votación", "Bagging (árboles)": "Bagging", "Random Forest": "Bagging", "Boosting (XGBoost)": "Boosting"}
res_ens = pd.DataFrame([evaluar_cv(n, m) for n, m in ensambles.items()])
mejor_clasico = res_clasicos.iloc[0]
res_ens.insert(1, "Tipo", res_ens["Modelo"].map(TIPO))
res_ens["Mejora vs mejor clásico (pts F1)"] = res_ens["CV Mean (F1-macro)"] - mejor_clasico["CV Mean (F1-macro)"]
res_ens["Mejora (%)"] = 100 * res_ens["Mejora vs mejor clásico (pts F1)"] / mejor_clasico["CV Mean (F1-macro)"]
res_ens = res_ens.round(4).sort_values("CV Mean (F1-macro)", ascending=False)
print(f"Mejor clásico: {mejor_clasico['Modelo']} (F1-macro CV = {mejor_clasico['CV Mean (F1-macro)']:.4f})")
guardar_tabla(res_ens, "4_2_ensambles"); res_ens
""")

code(r"""
cv_ini = pd.DataFrame(cv_scores)
orden = cv_ini.mean().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=cv_ini[orden], ax=ax, color="#9ecae1")
sns.stripplot(data=cv_ini[orden], ax=ax, color="black", size=4)
ax.set_ylabel("F1-macro por partición (CV de 5 particiones)"); ax.set_title("Comparación inicial: modelos clásicos y ensambles (70 % entrenamiento)")
plt.xticks(rotation=25, ha="right")
guardar_fig("4_comparacion_inicial")
cv_ini_guardado = cv_ini.copy()
""")

md(r"""
### 4.3 Ajuste de hiperparámetros
- **Validación cruzada:** `GridSearchCV` (o `RandomizedSearchCV` cuando el espacio es grande) con las **mismas 5 particiones estratificadas sobre el 70 % de entrenamiento**.
- **Métrica de optimización:** `f1_macro`, coherente con el objetivo de negocio (equilibrar falsos positivos y falsos negativos con desbalance de clases).
- Se ajustan los **5 modelos clásicos y los 3 ensambles de árboles**; la votación se reconstruye con los modelos ya ajustados.
- Para cada hiperparámetro se reporta **qué controla, por qué se eligió el rango y cómo afectó el desempeño** (efecto marginal calculado a partir de `cv_results_`).
""")

code(r"""
P = "model__"
espacios = {
    "Regresión Logística": (LogisticRegression(max_iter=3000, solver="liblinear", random_state=RANDOM_STATE),
        {P+"C": [0.001, 0.01, 0.1, 1, 10, 100], P+"penalty": ["l1", "l2"]}, "grid"),
    "KNN": (KNeighborsClassifier(),
        {P+"n_neighbors": [5, 15, 31, 51, 101], P+"weights": ["uniform", "distance"], P+"p": [1, 2]}, "grid"),
    "Árbol de Decisión": (DecisionTreeClassifier(random_state=RANDOM_STATE),
        {P+"max_depth": [3, 5, 8, 12, 20, None], P+"min_samples_leaf": [1, 10, 50, 100], P+"criterion": ["gini", "entropy"]}, "grid"),
    "SVM": (SVC(kernel="rbf", random_state=RANDOM_STATE),
        {P+"C": [0.1, 1, 10], P+"gamma": ["scale", 0.01, 0.1]}, "grid"),
    "Red Neuronal (MLP)": (MLPClassifier(max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
        {P+"hidden_layer_sizes": [(50,), (100,), (50, 50)], P+"alpha": [0.0001, 0.01, 0.1], P+"learning_rate_init": [0.001, 0.01]}, "grid"),
    "Bagging (árboles)": (BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE), random_state=RANDOM_STATE, n_jobs=1),
        {P+"n_estimators": [50, 150], P+"max_samples": [0.5, 0.8], P+"estimator__max_depth": [10, 20, None]}, "grid"),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        {P+"n_estimators": [200], P+"max_depth": [12, 20, None], P+"min_samples_leaf": [1, 5, 20], P+"max_features": ["sqrt", 0.5]}, "grid"),
    "Boosting (XGBoost)": (XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
        {P+"n_estimators": [100, 200, 400, 600], P+"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
         P+"max_depth": [3, 4, 6, 8], P+"subsample": [0.7, 0.85, 1.0], P+"colsample_bytree": [0.6, 0.8, 1.0]}, "random"),
}

busquedas, tiempos = {}, {}
for nombre, (modelo, grid, tipo_b) in espacios.items():
    t0 = time.time()
    if tipo_b == "grid":
        gs = GridSearchCV(crear_pipeline(modelo), grid, cv=skf, scoring="f1_macro", n_jobs=N_JOBS, return_train_score=True)
    else:
        gs = RandomizedSearchCV(crear_pipeline(modelo), grid, n_iter=40, cv=skf, scoring="f1_macro", n_jobs=N_JOBS,
                                random_state=RANDOM_STATE, return_train_score=True)
    gs.fit(X_train, y_train)
    busquedas[nombre], tiempos[nombre] = gs, time.time() - t0
    print(f"{nombre:22s} mejor F1-macro CV = {gs.best_score_:.4f} | {time.time()-t0:6.0f} s | {gs.best_params_}")
""")

code(r"""
# Votación reconstruida con los 3 mejores modelos ajustados (sus hiperparámetros óptimos)
top3_aj = sorted([n for n in clasicos], key=lambda n: busquedas[n].best_score_, reverse=True)[:3]
def modelo_ajustado(nombre, proba=False):
    m = clone(busquedas[nombre].best_estimator_.named_steps["model"])
    if proba and isinstance(m, SVC): m.set_params(probability=True)
    return m
voting_aj = VotingClassifier([(n, modelo_ajustado(n, proba=True)) for n in top3_aj], voting="soft")
r_vot = cross_validate(crear_pipeline(voting_aj), X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=N_JOBS)
print("Votación ajustada con:", top3_aj, "-> F1-macro CV = %.4f" % r_vot["test_score"].mean())

antes = {**dict(zip(res_clasicos["Modelo"], res_clasicos["CV Mean (F1-macro)"])),
         **dict(zip(res_ens["Modelo"], res_ens["CV Mean (F1-macro)"]))}
despues = {n: busquedas[n].best_score_ for n in busquedas}; despues["Votación (suave)"] = r_vot["test_score"].mean()
comp = pd.DataFrame({"F1-macro CV antes": antes, "F1-macro CV después": despues})
comp["Ganancia"] = comp["F1-macro CV después"] - comp["F1-macro CV antes"]
comp = comp.round(4).sort_values("F1-macro CV después", ascending=False)
guardar_tabla(comp.reset_index().rename(columns={"index": "Modelo"}), "4_3_antes_despues"); comp
""")

code(r"""
QUE_CONTROLA = {
    "C": "Inverso de la regularización: C pequeño = modelo más simple (más sesgo); C grande = se ajusta más a los datos (más varianza)",
    "penalty": "Tipo de regularización: L1 (lasso) anula coeficientes y selecciona variables; L2 (ridge) los encoge suavemente",
    "n_neighbors": "Número de vecinos que votan: pocos = frontera irregular (sobreajuste); muchos = frontera suave (subajuste)",
    "weights": "Peso del voto: 'uniform' igual para todos; 'distance' los vecinos más cercanos pesan más",
    "p": "Métrica de distancia de Minkowski: p=1 Manhattan, p=2 Euclidiana",
    "max_depth": "Profundidad máxima del árbol: controla la complejidad y el sobreajuste",
    "min_samples_leaf": "Mínimo de casos por hoja: valores altos suavizan el modelo y evitan reglas basadas en pocos casos",
    "criterion": "Medida de impureza para escoger divisiones: Gini o entropía (ganancia de información)",
    "gamma": "Alcance del kernel RBF: gamma alto = cada punto influye poco (frontera muy flexible); bajo = frontera suave",
    "hidden_layer_sizes": "Arquitectura: número de capas ocultas y neuronas (capacidad del modelo)",
    "alpha": "Regularización L2 de los pesos de la red (evita sobreajuste)",
    "learning_rate_init": "Tamaño del paso inicial del optimizador Adam",
    "n_estimators": "Número de modelos (árboles) del ensamble",
    "max_samples": "Fracción de la muestra bootstrap usada por cada árbol (diversidad)",
    "estimator__max_depth": "Profundidad de cada árbol base del bagging",
    "max_features": "Variables candidatas en cada división (descorrelaciona los árboles)",
    "learning_rate": "Contribución de cada árbol nuevo (contracción): bajo = aprendizaje lento y más robusto",
    "subsample": "Fracción de registros usada por cada árbol (regulariza, reduce varianza)",
    "colsample_bytree": "Fracción de variables usada por cada árbol",
}
POR_QUE_RANGO = {
    "C": "Escala logarítmica de 1e-3 a 1e2: cubre desde regularización muy fuerte hasta casi nula",
    "penalty": "Las dos regularizaciones soportadas por liblinear (como en el caso de estudio del curso)",
    "n_neighbors": "De 5 (local) a 101 (aproximadamente raíz de n); valores impares para evitar empates",
    "weights": "Las dos opciones disponibles", "p": "Las dos distancias más usadas",
    "max_depth": "Desde árboles poco profundos e interpretables (3) hasta sin límite (None)",
    "min_samples_leaf": "De 1 (sin restricción) a 100 (aproximadamente 0,5 % del entrenamiento)",
    "criterion": "Los dos criterios clásicos",
    "gamma": "'scale' (1/(n_var*var)) y valores en escala logarítmica alrededor de él",
    "hidden_layer_sizes": "Una capa pequeña/mediana y dos capas: el problema tiene solo 33 entradas",
    "alpha": "Escala logarítmica de regularización débil a fuerte",
    "learning_rate_init": "Valor por defecto de Adam y uno 10 veces mayor",
    "n_estimators": "Rango típico (50 a 600); a partir de cierto número de árboles el error se estabiliza",
    "max_samples": "Mitad de la muestra (más diversidad entre árboles) vs 80 %",
    "estimator__max_depth": "Árboles medianos a completos (bagging funciona mejor con árboles de alta varianza)",
    "max_features": "Recomendación clásica (raíz) y la mitad de las variables",
    "learning_rate": "Rango recomendado para boosting (0.01 a 0.2)",
    "subsample": "Valores habituales de submuestreo estocástico",
    "colsample_bytree": "Valores habituales de submuestreo de columnas",
}

def efecto_hiperparametros(nombre, gs):
    cvr = pd.DataFrame(gs.cv_results_)
    filas = []
    for param in [k for k in cvr.columns if k.startswith("param_")]:
        hp = param.replace("param_model__", "")
        marg = cvr.groupby(cvr[param].astype(str))["mean_test_score"].max()
        mejor = str(gs.best_params_[param.replace("param_", "")])
        filas.append({"Modelo": nombre, "Hiperparámetro": hp, "Qué controla": QUE_CONTROLA.get(hp, ""),
                      "Rango probado": ", ".join(marg.index), "Por qué el rango": POR_QUE_RANGO.get(hp, ""),
                      "Valor final": mejor,
                      "F1 con peor valor": round(marg.min(), 4), "F1 con mejor valor": round(marg.max(), 4),
                      "Efecto (pts F1)": round(marg.max() - marg.min(), 4)})
    return filas

tabla_hp = pd.DataFrame([f for n, gs in busquedas.items() for f in efecto_hiperparametros(n, gs)])
guardar_tabla(tabla_hp, "4_3_hiperparametros")
tabla_hp[["Modelo", "Hiperparámetro", "Rango probado", "Valor final", "F1 con peor valor", "F1 con mejor valor", "Efecto (pts F1)"]]
""")

code(r"""
# Curvas de validación: efecto marginal del hiperparámetro principal de cada modelo
principales = {"Regresión Logística": "C", "KNN": "n_neighbors", "Árbol de Decisión": "max_depth", "SVM": "C",
               "Red Neuronal (MLP)": "alpha", "Bagging (árboles)": "n_estimators", "Random Forest": "min_samples_leaf",
               "Boosting (XGBoost)": "learning_rate"}
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, (nombre, hp) in zip(axes.ravel(), principales.items()):
    cvr = pd.DataFrame(busquedas[nombre].cv_results_)
    col = f"param_model__{hp}"
    g = cvr.groupby(cvr[col].astype(str)).agg(test=("mean_test_score", "max"), train=("mean_train_score", "max"))
    try: g = g.loc[sorted(g.index, key=lambda v: float(v) if v not in ("None",) else 1e9)]
    except ValueError: pass
    ax.plot(g.index, g["train"], marker="o", label="entrenamiento"); ax.plot(g.index, g["test"], marker="o", label="validación")
    ax.set_title(f"{nombre}: {hp}"); ax.set_ylabel("F1-macro"); ax.tick_params(axis="x", rotation=30); ax.legend()
guardar_fig("4_3_curvas_validacion")
""")

# ---------------------------------------------------------------------------
# FASE 5
# ---------------------------------------------------------------------------
md(r"""
---
# FASE 5. Evaluación

### 5.1 Resultados en el conjunto de prueba (30 %, nunca visto)
Se evalúan **todos los modelos ajustados**. Precision, Recall y F1 se reportan en promedio **macro** (coherente con la métrica de optimización) y adicionalmente para la clase de interés (*Hospitalizado*).
""")

code(r"""
finales = {n: busquedas[n].best_estimator_ for n in busquedas}
finales["Votación (suave)"] = crear_pipeline(voting_aj).fit(X_train, y_train)

filas, preds, probas = [], {}, {}
for nombre, pipe in finales.items():
    yp = pipe.predict(X_test); preds[nombre] = yp
    score = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") and not isinstance(pipe.named_steps["model"], SVC) \
        else pipe.decision_function(X_test)
    probas[nombre] = score
    yp_tr = pipe.predict(X_train)
    filas.append({"Modelo": nombre, "Accuracy": accuracy_score(y_test, yp),
                  "Precision": precision_score(y_test, yp, average="macro"),
                  "Recall": recall_score(y_test, yp, average="macro"),
                  "F1": f1_score(y_test, yp, average="macro"),
                  "Recall Hosp.": recall_score(y_test, yp), "Precision Hosp.": precision_score(y_test, yp),
                  "ROC-AUC": roc_auc_score(y_test, score),
                  "F1 entrenamiento": f1_score(y_train, yp_tr, average="macro"),
                  "F1 CV": despues[nombre]})
res_test = pd.DataFrame(filas)
res_test["Brecha entrenamiento-prueba"] = res_test["F1 entrenamiento"] - res_test["F1"]
res_test = res_test.round(4).sort_values("F1", ascending=False)
guardar_tabla(res_test, "5_1_test"); res_test
""")

md(r"""
### 5.2 Selección del mejor modelo: justificación técnica y estadística
La selección se basa en el **desempeño en validación cruzada** (criterio de selección honesto); el conjunto de prueba se usa para **confirmar** que el desempeño se generaliza. Para saber si las diferencias entre los mejores modelos son **estadísticamente significativas** se aplican:
1. **prueba t pareada corregida (Nadeau & Bengio, 2003)** sobre los F1 de una validación cruzada repetida 5x2 (10 particiones) en entrenamiento: corrige la dependencia entre particiones que hace demasiado optimista la prueba t clásica.
2. **Prueba de McNemar** sobre las predicciones en el conjunto de prueba: compara los aciertos y errores *discordantes* de dos clasificadores sobre los mismos pacientes.
3. **Intervalo de confianza del 95 % por bootstrap** (1000 remuestreos) del F1-macro en prueba.
""")

code(r"""
from sklearn.model_selection import RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=RANDOM_STATE)
ranking_cv = sorted(despues, key=despues.get, reverse=True)
candidatos = ranking_cv[:3]
f1_rep = {}
for n in candidatos:
    pipe = finales[n] if n != "Votación (suave)" else crear_pipeline(voting_aj)
    f1_rep[n] = cross_val_score(clone(pipe), X_train, y_train, cv=rskf, scoring="f1_macro", n_jobs=N_JOBS)

def ttest_corregido(a, b, n_train, n_test):
    d = a - b; k = len(d)
    var = np.var(d, ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return t, 2 * stats.t.sf(abs(t), k - 1)

def mcnemar(y_true, p1, p2):
    a_ok, b_ok = p1 == y_true, p2 == y_true
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    return b, c, chi2, stats.chi2.sf(chi2, 1)

def bootstrap_f1(y_true, y_pred, n=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed); yt, yp = np.asarray(y_true), np.asarray(y_pred)
    vals = [f1_score(yt[i], yp[i], average="macro") for i in (rng.integers(0, len(yt), len(yt)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])

mejor = candidatos[0]
filas = []
for otro in candidatos[1:]:
    t, p_t = ttest_corregido(f1_rep[mejor], f1_rep[otro], len(X_train) / 2, len(X_train) / 2)
    b, c, chi2, p_m = mcnemar(y_test.values, preds[mejor], preds[otro])
    filas.append({"Comparación": f"{mejor} vs {otro}",
                  "Diferencia F1 CV 5x2": round(f1_rep[mejor].mean() - f1_rep[otro].mean(), 4),
                  "t corregido": round(t, 3), "p-valor prueba t": round(p_t, 4),
                  "McNemar (b, c)": f"({b}, {c})", "Chi cuadrado McNemar": round(chi2, 3), "p-valor McNemar": round(p_m, 4)})
pruebas = guardar_tabla(pd.DataFrame(filas), "5_2_pruebas_estadisticas")
ic = pd.DataFrame([{"Modelo": n, "F1 prueba": round(f1_score(y_test, preds[n], average="macro"), 4),
                    "IC 95 % inferior": round(bootstrap_f1(y_test, preds[n])[0], 4),
                    "IC 95 % superior": round(bootstrap_f1(y_test, preds[n])[1], 4),
                    "F1 CV 5x2 (media +/- desv. est.)": f"{f1_rep[n].mean():.4f} +/- {f1_rep[n].std():.4f}"} for n in candidatos])
guardar_tabla(ic, "5_2_intervalos")
print("Ranking por F1-macro CV:", [f"{n}: {despues[n]:.4f}" for n in ranking_cv])
display(pruebas); display(ic)
""")

code(r"""
MODELO_FINAL = mejor
pipe_final = finales[MODELO_FINAL]
if not hasattr(pipe_final, "predict_proba"):   # SVC sin probabilidades -> se reentrena con probability=True
    pipe_final = clone(pipe_final).set_params(model__probability=True).fit(X_train, y_train)
print("MODELO SELECCIONADO:", MODELO_FINAL)
print("Hiperparámetros:", busquedas[MODELO_FINAL].best_params_ if MODELO_FINAL in busquedas else top3_aj)
print(classification_report(y_test, preds[MODELO_FINAL], target_names=["No hospitalizado", "Hospitalizado"], digits=4))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay.from_predictions(y_test, preds[MODELO_FINAL], display_labels=["No hosp.", "Hosp."],
                                        cmap="Blues", ax=ax[0], values_format="d")
ax[0].set_title(f"Matriz de confusión (prueba): {MODELO_FINAL}")
ax[0].set_xlabel("Clase predicha"); ax[0].set_ylabel("Clase real")
for n in ranking_cv[:5]:
    RocCurveDisplay.from_predictions(y_test, probas[n], name=n, ax=ax[1])
ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].set_title("Curvas ROC en prueba (5 mejores)")
ax[1].set_xlabel("Tasa de falsos positivos"); ax[1].set_ylabel("Tasa de verdaderos positivos")
guardar_fig("5_matriz_confusion_roc")
""")

code(r"""
# Importancia de variables por permutación (en prueba): ¿qué variables usa el modelo?
pi = permutation_importance(pipe_final, X_test, y_test, scoring="f1_macro", n_repeats=5, random_state=RANDOM_STATE, n_jobs=N_JOBS)
imp = pd.DataFrame({"Variable": FEATURES, "Importancia (caída F1)": pi.importances_mean, "Std": pi.importances_std})
imp = imp.sort_values("Importancia (caída F1)", ascending=False).round(4)
guardar_tabla(imp, "5_importancia")
fig, ax = plt.subplots(figsize=(8, 7))
imp.set_index("Variable")["Importancia (caída F1)"].sort_values().plot.barh(ax=ax, xerr=imp.set_index("Variable")["Std"], color="#55a868")
ax.set_title(f"Importancia por permutación: {MODELO_FINAL}"); ax.set_xlabel("Disminución del F1-macro al permutar")
guardar_fig("5_importancia_variables")
imp.head(10)
""")

code(r"""
f1_final = f1_score(y_test, preds[MODELO_FINAL], average="macro")
rec_final = recall_score(y_test, preds[MODELO_FINAL])
objetivos = pd.DataFrame([
    ("F1-macro en prueba mayor o igual a 0,85", round(f1_final, 4), "Cumple" if f1_final >= .85 else "No cumple"),
    ("Recall Hospitalizado mayor o igual a 0,80", round(rec_final, 4), "Cumple" if rec_final >= .80 else "No cumple"),
    ("Al menos 5 clásicos y 3 ensambles comparados con CV", f"{len(clasicos)} + {len(ensambles)}", "Cumple"),
], columns=["Objetivo (Fase 1)", "Resultado", "Estado"])
guardar_tabla(objetivos, "5_objetivos"); objetivos
""")

# ---------------------------------------------------------------------------
# FASE 6
# ---------------------------------------------------------------------------
md(r"""
---
# FASE 6. Despliegue

### 6.0 Serialización del pipeline completo
Siguiendo el principio del material del curso (*"el preprocesamiento forma parte integral del modelo productivo"*), se serializa **un único objeto `Pipeline`** que contiene imputadores, escalador, codificador one-hot y el modelo. SMOTE solo actúa en `fit`, por lo que en predicción no altera los datos. Junto al modelo se guarda un archivo de **metadatos** (variables, rangos válidos, versiones y métricas) que usa la aplicación.
""")

code(r"""
joblib.dump(pipe_final, "pipeline_dengue.joblib", compress=3)
metadata = {
    "modelo": MODELO_FINAL,
    "hiperparametros": {k.replace("model__", ""): str(v) for k, v in busquedas[MODELO_FINAL].best_params_.items()} if MODELO_FINAL in busquedas else {},
    "features": FEATURES, "num_features": NUM_FEATS, "bin_features": BIN_FEATS, "cat_features": CAT_FEATS,
    "edad_grupos": EDAD_GRUPO, "regimen": {"C": "Contributivo", "S": "Subsidiado", "P": "Excepción", "E": "Especial", "N": "No asegurado", "I": "Indeterminado"},
    "reglas": {"dias_sintomas": [0, 30], "estrato": [1, 6], "mes": [1, 12]},
    "metricas_test": res_test.set_index("Modelo").loc[MODELO_FINAL, ["Accuracy", "Precision", "Recall", "F1", "Recall Hosp.", "ROC-AUC"]].to_dict(),
    "versiones": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "imbalanced-learn": imblearn.__version__,
                  "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    "dataset": "https://www.datos.gov.co/d/qzc7-jbg3", "n_train": int(len(X_train)), "n_test": int(len(X_test)),
}
Path("model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print("Guardado: pipeline_dengue.joblib (%.1f KB) y model_metadata.json" % (Path("pipeline_dengue.joblib").stat().st_size / 1024))
""")

md(r"""
### 6.1 Predicción con datos nuevos (no usados en el entrenamiento)
Se **recarga el pipeline desde disco** (como lo hará la aplicación) y se predicen **perfiles clínicos nuevos** construidos para validar la lógica del modelo, más registros reales del conjunto de prueba. Se incluye un registro con valores faltantes para comprobar que el pipeline los imputa sin fallar.
""")

code(r"""
modelo_cargado = joblib.load("pipeline_dengue.joblib")
base = {c: 0 for c in BIN_FEATS}
def paciente(**kw):
    r = {"edad_aprox": 24.5, "dias_sintomas": 3, "estrato": 3, "mes": 6, **base, "sexo": "F", "regimen": "C"}
    r.update(kw); return r

nuevos = pd.DataFrame([
    paciente(edad_aprox=24.5, dias_sintomas=2, fiebre=1, cefalea=1, malgias=1),
    paciente(edad_aprox=7, dias_sintomas=5, sexo="M", regimen="S", estrato=1, fiebre=1, dolor_abdo=1, vomito=1, caida_plaq=1),
    paciente(edad_aprox=75, dias_sintomas=4, regimen="S", estrato=2, fiebre=1, caida_plaq=1, aum_hemato=1, somnolenci=1),
    paciente(edad_aprox=34.5, dias_sintomas=3, gp_gestan=1, fiebre=1, cefalea=1, dolor_abdo=1),
    paciente(edad_aprox=17, dias_sintomas=1, sexo="M", estrato=4, fiebre=1, artralgia=1, erupcionr=1),
    paciente(edad_aprox=44.5, dias_sintomas=np.nan, estrato=np.nan, sexo="M", fiebre=1, hem_mucosa=1, caida_plaq=1, vomito=1),
])[FEATURES]
desc_casos = ["Adulto joven, 2 días, solo síntomas leves (fiebre, cefalea, mialgias)",
              "Niño de 5 a 9 años, 5 días, dolor abdominal + vómito + caída de plaquetas",
              "Adulto mayor (70+), 4 días, caída de plaquetas + aumento de hematocrito + somnolencia",
              "Gestante 30 a 39 años, 3 días, fiebre + cefalea + dolor abdominal",
              "Adolescente, 1 día, fiebre + artralgia + erupción",
              "Adulto 40 a 49, días y estrato DESCONOCIDOS, sangrado de mucosas + plaquetas + vómito"]
prob = modelo_cargado.predict_proba(nuevos)[:, 1] if hasattr(modelo_cargado, "predict_proba") else None
pred = modelo_cargado.predict(nuevos)
def interpretar(p):
    if p >= .7: return "Riesgo ALTO: priorizar valoración para hospitalización"
    if p >= .5: return "Riesgo MODERADO-ALTO: probable hospitalización, vigilancia estrecha"
    if p >= .3: return "Riesgo MODERADO: manejo ambulatorio con control a las 24 a 48 h"
    return "Riesgo BAJO: manejo ambulatorio y signos de alarma al paciente"
tabla_61 = pd.DataFrame({"Registro de prueba": desc_casos,
                         "Predicción": np.where(pred == 1, "Hospitalizado", "No hospitalizado"),
                         "Probabilidad hospitalización": np.round(prob, 3),
                         "Interpretación": [interpretar(p) for p in prob]})
guardar_tabla(tabla_61, "6_1_prediccion_nuevos"); tabla_61
""")

code(r"""
# Registros reales del conjunto de prueba (no vistos en entrenamiento)
muestra = X_test.sample(6, random_state=7)
real = pd.DataFrame({"Real": y_test.loc[muestra.index].map({1: "Hospitalizado", 0: "No hospitalizado"}),
                     "Predicho": pd.Series(modelo_cargado.predict(muestra), index=muestra.index).map({1: "Hospitalizado", 0: "No hospitalizado"}),
                     "Prob. hosp.": np.round(modelo_cargado.predict_proba(muestra)[:, 1], 3)})
guardar_tabla(muestra.join(real), "6_1_prediccion_test")
muestra.join(real)
""")

md(r"""
### 6.2 Aplicación web en Streamlit
La aplicación `app.py` (raíz del repositorio):
1. Carga `pipeline_dengue.joblib` y `model_metadata.json` con `@st.cache_resource`.
2. Recibe los datos del paciente con controles que respetan las **reglas de calidad** (rangos de días, estrato, edad por grupo etario, régimen).
3. Construye un `DataFrame` con las mismas columnas del entrenamiento y llama a `predict_proba`.
4. Muestra el resultado con `st.error` (riesgo alto) o `st.success` (riesgo bajo), la probabilidad y su interpretación.

**Despliegue:** repositorio público en GitHub con `app.py`, `pipeline_dengue.joblib`, `model_metadata.json`, `requirements.txt` (versiones fijadas) y `README.md`, luego share.streamlit.io, luego *New app*, luego `app.py`, luego *Deploy*.
""")

md(r"""
---
# Conclusiones
*(se completan con los resultados obtenidos)*
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
out = ROOT / "notebooks" / "proyecto_crispdm_dengue.ipynb"
out.parent.mkdir(exist_ok=True)
import sys
if out.exists() and "--force" not in sys.argv:
    previo = nbf.read(out, as_version=4)
    if any(c.get("outputs") for c in previo.cells if c.cell_type == "code"):
        sys.exit(f"{out} ya está ejecutado; use --force para sobrescribirlo.")
nbf.write(nb, out)
print("Notebook generado:", out, "| celdas:", len(cells))
