"""Genera el documento de entrega en la estructura del formato
'U4_0 Formato de Entrega Trabajo Integrador' (incluido en doc/Actividad.docx),
llenándolo con las tablas y figuras que produce el notebook.

Uso: .venv/Scripts/python scripts/build_word.py
Salida: doc/entrega/Entrega_Proyecto_Integrador_CRISP-DM.docx
"""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "reports" / "tablas", ROOT / "reports" / "figuras"
CFG = json.loads((ROOT / "scripts" / "entrega_config.json").read_text(encoding="utf-8"))
META = json.loads((ROOT / "model_metadata.json").read_text(encoding="utf-8"))
TXT = json.loads((ROOT / "scripts" / "textos_resultados.json").read_text(encoding="utf-8"))


def t(nombre):
    return pd.read_csv(TAB / f"{nombre}.csv", encoding="utf-8-sig")


AZUL = "1F3864"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)  # Carta
for m in ("left_margin", "right_margin"):
    setattr(sec, m, Cm(2.5))
sec.top_margin = sec.bottom_margin = Cm(2.2)

st = doc.styles
st["Normal"].font.name = "Arial"
st["Normal"].font.size = Pt(10.5)
st["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
st["Normal"].paragraph_format.space_after = Pt(6)
st["Normal"].paragraph_format.line_spacing = 1.15
for nivel, tam in [(1, 15), (2, 12.5), (3, 11)]:
    h = st[f"Heading {nivel}"]
    h.font.name, h.font.size, h.font.bold = "Arial", Pt(tam), True
    h.font.color.rgb = RGBColor.from_string(AZUL)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    h.paragraph_format.space_before = Pt(14 if nivel == 1 else 10)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True


def sombrear(celda, color):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def repetir_encabezado(fila):
    trPr = fila._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trPr.append(el)


def p(texto="", bold=False, italic=False, size=None, align=None, style=None):
    par = doc.add_paragraph(style=style)
    _runs(par, texto, bold, italic, size)
    if align: par.alignment = align
    return par


def _runs(par, texto, bold=False, italic=False, size=None):
    # **negrita** simple dentro del texto
    partes = str(texto).split("**")
    for i, parte in enumerate(partes):
        if not parte: continue
        r = par.add_run(parte)
        r.bold = bold or (i % 2 == 1); r.italic = italic
        if size: r.font.size = Pt(size)


def viñetas(items, estilo="List Bullet"):
    for it in items:
        par = doc.add_paragraph(style=estilo)
        _runs(par, it)
        par.paragraph_format.space_after = Pt(2)


import re
NUM_COMPUESTO = re.compile(r"[\d.]+ \+/- [\d.]+")   # p. ej. "0.8444 +/- 0.0018"


def coma(txt):
    """Cambia el punto decimal por coma (formato numérico en español)."""
    return re.sub(r"(?<=\d)\.(?=\d)", ",", str(txt))


def tabla(df, anchos=None, tam=8.5, encabezado_color="D9E2F3"):
    df = df.copy().fillna("")
    tb = doc.add_table(rows=1, cols=len(df.columns))
    tb.style = "Table Grid"; tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, col in enumerate(df.columns):
        c = tb.rows[0].cells[i]; c.text = ""
        _runs(c.paragraphs[0], str(col), bold=True, size=tam)
        sombrear(c, encabezado_color)
    repetir_encabezado(tb.rows[0])
    for _, fila in df.iterrows():
        cells = tb.add_row().cells
        for i, v in enumerate(fila):
            if isinstance(v, float):
                v = str(int(v)) if v.is_integer() else coma(f"{v:.4f}".rstrip("0").rstrip("."))
            elif isinstance(v, str) and NUM_COMPUESTO.fullmatch(v):
                v = coma(v)
            cells[i].text = ""
            _runs(cells[i].paragraphs[0], str(v), size=tam)
    for fila in tb.rows:
        for c in fila.cells:
            for par in c.paragraphs:
                par.paragraph_format.space_after = Pt(0); par.paragraph_format.line_spacing = 1.0
    if anchos:
        for fila in tb.rows:
            for i, w in enumerate(anchos):
                fila.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tb


def tabla_kv(pares, ancho_k=5.5, ancho_v=11):
    tb = doc.add_table(rows=0, cols=2); tb.style = "Table Grid"
    for k, v in pares:
        cells = tb.add_row().cells
        cells[0].text = ""; _runs(cells[0].paragraphs[0], k, bold=True, size=9.5); sombrear(cells[0], "D9E2F3")
        cells[1].text = ""; _runs(cells[1].paragraphs[0], v, size=9.5)
        cells[0].width, cells[1].width = Cm(ancho_k), Cm(ancho_v)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tb


def figura(nombre, ancho=15.5, pie=""):
    ruta = FIG / f"{nombre}.png"
    if not ruta.exists(): return
    doc.add_picture(str(ruta), width=Cm(ancho))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if pie:
        p(pie, italic=True, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER)


def interpretacion(texto):
    par = p(f"**Interpretación.** {texto}")
    par.paragraph_format.left_indent = Cm(0.3)
    return par


# =============================================================================
# PORTADA / INFORMACIÓN GENERAL
# =============================================================================
p("Formato de entrega proyecto integrador", bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER)
p("Experto temático: John Fernando Vargas Buitrago", size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER)
p("Aprendizaje de Máquinas: Proyecto Integrador con Metodología CRISP-DM", italic=True, size=10,
  align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_heading("Información General", 1)
tabla_kv([
    ("Título del Proyecto", "Predicción temprana de hospitalización en pacientes con dengue en Bucaramanga mediante aprendizaje supervisado (SIVIGILA 2015 a 2025)"),
    ("Integrantes", CFG["integrantes"]),
    ("Tipo de problema (Clasificación/Regresión)", "Clasificación binaria"),
    ("Fuente del Dataset", "Datos Abiertos Colombia (datos.gov.co): Alcaldía de Bucaramanga, Secretaría de Salud (SIVIGILA)"),
    ("Enlace al Dataset", "https://www.datos.gov.co/d/qzc7-jbg3"),
    ("Enlace a Google Colab", CFG["colab"]),
    ("Repositorio GitHub", CFG["github"]),
    ("URL Streamlit.io", CFG["streamlit"]),
])

# =============================================================================
# 1. ENTENDIMIENTO DEL NEGOCIO
# =============================================================================
doc.add_heading("1. Entendimiento del Negocio", 1)
doc.add_heading("Descripción del negocio", 3)
p("El dengue es la arbovirosis de mayor impacto en salud pública en Colombia. Es transmitido por el mosquito "
  "*Aedes aegypti*, y Bucaramanga (Santander) es una zona hiperendémica. Cada caso es notificado por las instituciones "
  "de salud al **Sistema Nacional de Vigilancia en Salud Pública (SIVIGILA)**. La Alcaldía de Bucaramanga publica en "
  "Datos Abiertos Colombia el seguimiento de los eventos *Dengue* (código 210), *Dengue grave* (220) y *Mortalidad por "
  "dengue* (580), con 76 variables: sociodemográficas, de afiliación al sistema de salud, grupos poblacionales, síntomas, "
  "signos de alarma, fechas clave y desenlace del caso. El conjunto contiene **28.626 notificaciones entre 2015 y 2025**.".replace("*", ""))
p("La red hospitalaria de la ciudad enfrenta picos de demanda durante los brotes: el año 2024 concentra el 40 % de los "
  "casos del periodo. Decidir si hospitalizar a un paciente es costoso en ambos sentidos: hospitalizar a quien no lo "
  "necesita ocupa camas escasas, y enviar a casa a quien sí lo necesita puede derivar en dengue grave o en la muerte.")

doc.add_heading("Descripción del problema", 3)
p("**¿Es posible predecir, con la información disponible en la primera consulta (datos sociodemográficos, síntomas y "
  "signos de alarma, y tiempo de evolución), si un paciente con dengue requerirá hospitalización?** La variable objetivo "
  "es `pac_hos_` (1 = hospitalizado, 2 = no hospitalizado). Es un problema de **clasificación binaria** con clases "
  "moderadamente desbalanceadas (31,6 % hospitalizados). Un modelo de este tipo apoya el triaje y la planeación de camas, "
  "como complemento (no reemplazo) del criterio médico y de la guía clínica del Ministerio de Salud.".replace("`", ""))

doc.add_heading("Objetivos de la minería", 3)
p("**Objetivo general:** construir un modelo de clasificación supervisada que estime si un caso notificado de dengue "
  "requerirá hospitalización, a partir de las variables disponibles en la primera atención.")
p("**Objetivos específicos (medibles):**")
viñetas([
    "Construir un conjunto de datos limpio y libre de fuga de información (data leakage), aplicando reglas de calidad de negocio.",
    "Comparar 5 modelos clásicos del curso y al menos 3 modelos de ensamble con validación cruzada estratificada de 5 particiones sobre el 70 % de entrenamiento.",
    "Alcanzar un F1-macro mayor o igual a 0,85 en el conjunto de prueba (30 %) con un Recall de la clase \"Hospitalizado\" mayor o igual a 0,80.",
    "Desplegar el pipeline completo en una aplicación web pública en Streamlit.io que permita predecir casos nuevos.",
])

doc.add_heading("Diseño de solución", 3)
p("La solución sigue las seis fases de CRISP-DM: (1) definición del problema de triaje; (2) comprensión y calidad de "
  "los datos del SIVIGILA; (3) limpieza, eliminación de la fuga de información y construcción de variables; "
  "(4) modelamiento con 5 modelos clásicos y 4 ensambles, ajustados por validación cruzada; (5) evaluación en un conjunto "
  "de prueba independiente con pruebas estadísticas; y (6) despliegue del pipeline serializado en Streamlit. El "
  "preprocesamiento (imputación, escalado, codificación) y el balanceo (SMOTE) forman parte de un único Pipeline para "
  "evitar la fuga de información y garantizar el mismo tratamiento en producción.")

doc.add_heading("Diseño de solución detallado", 3)
tabla(pd.DataFrame([{
    "Tipo de análisis": "Predictivo",
    "Tipo de aprendizaje": "Supervisado (clasificación binaria)",
    "Modelos propuestos": "Clásicos (curso): Regresión Logística, KNN, Árbol de Decisión, SVM, Red Neuronal (MLP). "
                          "Ensambles: Votación (suave), Bagging, Random Forest, Boosting (XGBoost)",
    "Métrica principal": "F1-score macro (secundaria: Recall de la clase Hospitalizado)",
}]), anchos=[2.8, 3.2, 7.0, 3.5], tam=9)
p("**Justificación de la métrica.** La clase positiva representa el 31,6 % de los casos, por lo que la Accuracy es "
  "engañosa: un modelo que siempre dijera \"no hospitalizar\" tendría 68 % de exactitud y sería inútil. El F1-macro "
  "promedia el F1 de ambas clases, equilibra la precisión y la sensibilidad, y da el mismo peso a los falsos positivos "
  "(camas ocupadas sin necesidad) y a los falsos negativos (pacientes que se complicarán y son enviados a casa).")

# =============================================================================
# 2. ENTENDIMIENTO DE LOS DATOS
# =============================================================================
doc.add_heading("2. Entendimiento de los Datos", 1)
doc.add_heading("2.1 Diccionario de datos", 2)
dic = t("2_1_diccionario")
dic["Descripción"] = dic["Descripción"].fillna("").str.slice(0, 140)
dic["Entrada/Salida"] = dic.apply(lambda r: r["Entrada/Salida"] if r["Entrada/Salida"] != "No usada"
                                  else f"No usada: {r['Observación']}", axis=1)
tabla(dic[["Variable", "Descripción", "Tipo (Num/Cat)", "Entrada/Salida"]], anchos=[2.6, 7.4, 1.6, 4.9], tam=7.5)
interpretacion("De las 76 variables originales, 27 se marcan como entrada: 17 síntomas y signos de alarma, "
               "4 binarias de contexto, sexo, régimen, estrato, grupo etario y las dos fechas a partir de las cuales se "
               "derivan dias_sintomas y mes. Una es la salida (pac_hos_). Las variables descartadas lo son principalmente por "
               "fuga de información (describen lo que pasó después de decidir la hospitalización), por ser "
               "identificadores o variables administrativas, o por ser casi constantes.")

doc.add_heading("2.2 Reglas de calidad", 2)
p("Las reglas se definen a partir del conocimiento del negocio y de los metadatos oficiales del SIVIGILA, **no** a partir "
  "del mínimo y el máximo observados en el dataset, que pueden contener errores. Para cada regla se cuantifica el "
  "incumplimiento en los datos crudos.")
p("**Variables numéricas**")
rn = t("2_2_reglas_numericas")
tabla(rn.rename(columns={"Sin dato (-89/vacío)": "Sin dato"}), tam=8.5)
p("**Variables categóricas**")
rc = t("2_2_reglas_categoricas")
rc_res = pd.concat([
    pd.DataFrame([{"Variable": "17 síntomas y signos de alarma (desde fiebre hasta acum_liqui)", "Categorías válidas": "1 = Sí, 2 = No",
                   "Registros que incumplen": "1 o 2 por variable (valores 3 y -89)", "% incumple": "menor o igual a 0,01 %"}]),
    rc[~rc["Variable"].isin(META["bin_features"][:17])][["Variable", "Categorías válidas", "Registros que incumplen", "% incumple"]],
])
tabla(rc_res, anchos=[4.5, 6.0, 3.5, 2.5], tam=8.5)
interpretacion(TXT["calidad"])

doc.add_heading("Distribución de la variable objetivo", 3)
tabla(t("2_3_target"), tam=9)
figura("2_3_distribucion_target", 9, "Figura 1. Distribución de la variable objetivo (pac_hos_).")
interpretacion("El problema está moderadamente desbalanceado (razón de 2,17 a 1). Un clasificador ingenuo favorecería a la "
               "clase mayoritaria, por lo que se adopta el F1-macro como métrica, se particiona de forma estratificada y "
               "se balancea únicamente el conjunto de entrenamiento (sección 3.7).")

doc.add_heading("Posibles sesgos en los datos", 3)
tabla(t("2_4_sesgos"), anchos=[4.5, 12], tam=9)
figura("2_4_sesgo_temporal", 14, "Figura 2. Casos notificados y porcentaje de hospitalización por año.")
viñetas([
    "**Temporal (deriva del concepto):** la proporción de hospitalizados va de 11,7 % (2015) a 60,4 % (2022), lo que refleja cambios en protocolos, serotipos circulantes y forma de notificar. Por eso `año` no se usa como predictor, y el modelo debe monitorearse y reentrenarse.".replace("`", ""),
    "**Brote 2024:** aporta el 40,3 % de los registros; ese perfil domina el aprendizaje.",
    "**Geográfico:** el 96,3 % reside en Bucaramanga; el modelo no debe generalizarse a otras regiones sin validación.",
    "**Grupos minoritarios:** migrantes (1,05 %), gestantes (0,64 %) e indígenas (0,06 %) tienen muy pocos casos, así que el desempeño en estos subgrupos es incierto (riesgo de equidad).",
    "**Régimen de afiliación:** la tasa de hospitalización es 27 % en el régimen contributivo y 43 % en el subsidiado. Esta diferencia puede reflejar acceso y prácticas institucionales, no solo gravedad clínica.",
    "**Subregistro / selección:** solo están los casos que consultaron y fueron notificados al SIVIGILA.",
])

# =============================================================================
# 3. PREPARACIÓN DE LOS DATOS
# =============================================================================
doc.add_heading("3. Preparación de los Datos", 1)
doc.add_heading("3.1 Selección de variables", 2)
p("Criterios: (1) **eliminar la fuga de información**: igual que en el caso de estudio del curso se eliminó la *Nota "
  "Final* para predecir *Aprobó*, aquí se eliminan fec_hos_ (coincide con el target en el 99,99 % de los casos), "
  "conducta (es la decisión médica de hospitalizar), clasfinal, el evento final, la confirmación por laboratorio, el "
  "desenlace y las muestras de autopsia; (2) eliminar identificadores y variables administrativas; (3) eliminar variables "
  "casi constantes o de altísima cardinalidad; y (4) construir variables derivadas con significado clínico.".replace("*", ""))
tabla(t("3_1_seleccion"), anchos=[4, 12.5], tam=8.5)
p(TXT["iteracion"])

doc.add_heading("3.2 Descripción estadística", 2)
tabla(t("3_2_descripcion"), anchos=[10, 6.5], tam=9)
p("**Reporte HTML generado con ydata-profiling:** se adjunta el archivo `reports/reporte_profiling.html` (también "
  "disponible en el repositorio de GitHub). " + TXT["profiling"])

doc.add_heading("3.3 Limpieza de errores", 2)
tabla(t("3_3_limpieza"), anchos=[6, 8, 2.5], tam=8.5)
interpretacion("Los errores se corrigen con reglas determinísticas (no aprenden de los datos), por lo que se aplican antes "
               "de la partición sin generar fuga. Los valores que incumplen una regla se convierten en faltantes "
               "explícitos (NaN), en lugar de eliminar registros, para que el pipeline los impute de forma controlada. "
               "Después se separa el 70 % para entrenamiento y el 30 % para prueba de forma estratificada (31,57 % de "
               "hospitalizados en ambos conjuntos).")

doc.add_heading("3.4 Limpieza de nulos", 2)
nul = t("3_4_nulos")
nul_res = pd.concat([nul[nul["% Nulos (total)"] >= 0.02],
                     pd.DataFrame([{"Variable": "14 síntomas/signos restantes", "% Nulos (total)": "0,01",
                                    "% Nulos (entrenamiento)": "0,00", "Método aplicado": "SimpleImputer (moda) dentro del pipeline"}])])
tabla(nul_res.rename(columns={"% Nulos (total)": "% Nulos"}).drop(columns=["% Nulos (entrenamiento)"]), anchos=[4.5, 2.5, 9.5], tam=9)
p("**Justificación.** Para las numéricas (estrato, días de síntomas) se usa **KNNImputer (k = 5)**, que estima cada valor "
  "a partir de los 5 casos más parecidos y conserva las relaciones entre variables mejor que la media (es el mismo "
  "criterio del caso de estudio del curso). Para las binarias y categóricas se usa la **moda**, que es la imputación "
  "natural para variables nominales. Los imputadores están **dentro del pipeline**, por lo que se ajustan solo con los "
  "datos de entrenamiento de cada partición y así se evita la fuga de información.")

doc.add_heading("3.5 Codificación", 2)
tabla(t("3_5_codificacion"), anchos=[5, 5.5, 6], tam=8.5)
p("**Relaciones lineales y no lineales.** La relación lineal se midió con la correlación de **Pearson**, y las no "
  "lineales con **Spearman** (monótonas) y con la **información mutua** (cualquier dependencia), sobre el conjunto de "
  "entrenamiento preprocesado.")
figura("3_6_relacion_target", 16, "Figura 3. Correlación de Pearson e información mutua de cada variable con el target.")
rel = t("3_6_relaciones").head(8)
tabla(rel, tam=8.5)
interpretacion(TXT["relaciones"])

doc.add_heading("3.6 Reducción de dimensiones (si aplica)", 2)
pca = t("3_7_pca_resultado")
tabla(pd.DataFrame([{"Técnica": "PCA (Análisis de Componentes Principales): evaluado y NO aplicado",
                     "Justificación": TXT["pca"]}]), anchos=[4.5, 12], tam=9)
tabla(pca, tam=9)
figura("3_7_pca", 11, "Figura 4. Varianza explicada acumulada por PCA.")

doc.add_heading("3.7 Balanceo (si clasificación)", 2)
tabla(pd.DataFrame([{"Técnica aplicada": "SMOTE (Synthetic Minority Over-sampling Technique) dentro de un Pipeline de imbalanced-learn",
                     "Justificación": TXT["balanceo"]}]), anchos=[4.5, 12], tam=9)
tabla(t("3_8_balanceo"), tam=9)

# =============================================================================
# 4. MODELAMIENTO
# =============================================================================
doc.add_heading("4. Modelamiento", 1)
p("Todos los modelos usan el mismo pipeline (preprocesador, luego SMOTE, luego modelo) y las **mismas 5 particiones** de "
  "validación cruzada estratificada (StratifiedKFold, shuffle, random_state = 42) **sobre el 70 % de entrenamiento** "
  "(20.031 registros). Así, las diferencias de desempeño se deben solo al algoritmo, y los puntajes por partición quedan "
  "pareados para las pruebas estadísticas.")
doc.add_heading("4.1 Modelos clásicos", 2)
c41 = t("4_1_clasicos")
tabla(c41[["Modelo", "Hiperparámetros iniciales", "Métrica", "CV Mean (F1-macro)", "CV Std"]]
      .rename(columns={"CV Mean (F1-macro)": "CV Mean"}), anchos=[3.3, 6.7, 2.0, 2.3, 2.2], tam=8.5)
p("Métricas complementarias en validación cruzada:", italic=True, size=9)
tabla(c41[["Modelo", "F1 entrenamiento", "Recall Hosp.", "ROC-AUC", "Tiempo (s)"]], tam=8.5)
interpretacion(TXT["clasicos"])

doc.add_heading("4.2 Modelos de ensamble", 2)
p("**Marco teórico (actividad investigativa).** La **votación** combina modelos distintos promediando sus probabilidades "
  "(voto *suave*). Funciona cuando los modelos cometen errores diferentes y aquí usa los 3 mejores clásicos. El **bagging** "
  "(Breiman, 1996) entrena muchos árboles sobre muestras bootstrap y los combina por votación, lo que reduce la "
  "**varianza** de modelos inestables. El **Random Forest** (Breiman, 2001) agrega una selección aleatoria de variables en "
  "cada división para descorrelacionar los árboles. El **boosting** con **XGBoost** (Chen y Guestrin, 2016) entrena árboles "
  "de forma secuencial, de modo que cada uno corrige el gradiente del error del anterior, con regularización L1/L2: "
  "reduce principalmente el **sesgo**.".replace("*", ""))
e42 = t("4_2_ensambles")
e42["Desempeño"] = e42.apply(lambda r: coma(f"F1-macro CV = {r['CV Mean (F1-macro)']:.4f} +/- {r['CV Std']:.4f}"), axis=1)
e42["Mejora vs clásico"] = e42.apply(lambda r: coma(f"{r['Mejora vs mejor clásico (pts F1)']:+.4f} puntos ({r['Mejora (%)']:+.2f} %)"), axis=1)
tabla(e42[["Modelo", "Tipo", "Desempeño", "Mejora vs clásico"]], anchos=[4, 2.5, 5.5, 4.5], tam=9)
figura("4_comparacion_inicial", 15.5, "Figura 5. F1-macro por partición de la validación cruzada (comparación inicial).")
interpretacion(TXT["ensambles"])

doc.add_heading("4.3 Ajuste de hiperparámetros", 2)
p("**Procedimiento.** Se usó GridSearchCV, y RandomizedSearchCV con 40 combinaciones para XGBoost, con las mismas 5 "
  "particiones estratificadas **sobre el 70 % de entrenamiento**, optimizando **f1_macro**. Esta métrica es coherente con "
  "el objetivo de negocio: equilibra falsos positivos y falsos negativos bajo desbalance de clases, y no premia a un "
  "modelo que favorezca a la clase mayoritaria. Se ajustaron los 5 modelos clásicos y los 3 ensambles de árboles; la "
  "votación se reconstruyó con los 3 mejores modelos ya ajustados.")
hp = t("4_3_hiperparametros")
def _orden(r):
    vals = [v.strip() for v in str(r).split(",")]
    if any(v.startswith("(") for v in vals): return r
    def k(v):
        try: return (0, float(v))
        except ValueError: return (1 if v != "None" else 2, v)
    return ", ".join(sorted(vals, key=k))
hp["Rango probado"] = hp["Rango probado"].map(_orden)
tabla(hp[["Modelo", "Hiperparámetro", "Qué controla", "Rango probado", "Valor final"]], anchos=[2.7, 2.6, 6.2, 3.3, 1.7], tam=7.5)
p("**Por qué se eligió cada rango y cómo afectó el desempeño.** El efecto se mide como la diferencia del F1-macro de "
  "validación entre el peor y el mejor valor del hiperparámetro, tomando el mejor resultado de las demás combinaciones:")
tabla(hp[["Modelo", "Hiperparámetro", "Por qué el rango", "F1 con peor valor", "F1 con mejor valor", "Efecto (pts F1)"]]
      .rename(columns={"F1 con peor valor": "F1 peor", "F1 con mejor valor": "F1 mejor", "Efecto (pts F1)": "Efecto"}),
      anchos=[2.7, 2.6, 6.5, 1.6, 1.6, 1.5], tam=7.5)
figura("4_3_curvas_validacion", 16.5, "Figura 6. Curvas de validación (entrenamiento frente a validación) del hiperparámetro principal de cada modelo.")
p("**Comparación antes y después del ajuste (F1-macro en validación cruzada):**")
tabla(t("4_3_antes_despues"), tam=9)
interpretacion(TXT["ajuste"])

# =============================================================================
# 5. EVALUACIÓN
# =============================================================================
doc.add_heading("5. Evaluación Final", 1)
doc.add_heading("5.1 Resultados en test", 2)
p("Se evaluaron todos los modelos ajustados sobre el **conjunto de prueba (30 %, 8.585 registros) que no se usó en "
  "ninguna etapa del entrenamiento**. Precision, Recall y F1 se reportan en promedio macro.")
p("**Clasificación**")
rt = t("5_1_test")
tabla(rt[["Modelo", "Accuracy", "Precision", "Recall", "F1"]], anchos=[5, 2.8, 2.8, 2.8, 2.8], tam=9)
p("Métricas complementarias (clase Hospitalizado, ROC-AUC y sobreajuste):", italic=True, size=9)
tabla(rt[["Modelo", "Recall Hosp.", "Precision Hosp.", "ROC-AUC", "F1 entrenamiento", "F1 CV", "Brecha entrenamiento-prueba"]], tam=8)
p("**Regresión:** no aplica (el problema es de clasificación).")
tabla(pd.DataFrame([{"Modelo": "No aplica", "MAE": "No aplica", "RMSE": "No aplica", "R2": "No aplica"}]), tam=9)
figura("5_matriz_confusion_roc", 16, "Figura 7. Matriz de confusión del modelo seleccionado y curvas ROC en test.")
interpretacion(TXT["test"])

doc.add_heading("5.2 Modelo seleccionado", 2)
tabla_kv([("Modelo final", TXT["modelo_final"]), ("Justificación técnica", TXT["justificacion"])], 3.5, 13)
p("**Pruebas de significancia estadística** (prueba t pareada corregida de Nadeau y Bengio sobre una validación cruzada "
  "repetida 5x2 en entrenamiento; prueba de McNemar sobre las predicciones en el conjunto de prueba):")
tabla(t("5_2_pruebas_estadisticas"), tam=8.5)
tabla(t("5_2_intervalos"), tam=8.5)
figura("5_importancia_variables", 12, "Figura 8. Importancia de las variables por permutación (test).")
interpretacion(TXT["importancia"])
p("**Cumplimiento de los objetivos de minería (Fase 1):**")
obj = t("5_objetivos"); obj["Objetivo (Fase 1)"] = obj["Objetivo (Fase 1)"].map(coma)
tabla(obj, anchos=[8, 4, 4.5], tam=9)

# =============================================================================
# 6. DESPLIEGUE
# =============================================================================
doc.add_heading("6. Despliegue", 1)
p("**Serialización.** Se serializó con joblib un **único Pipeline** (pipeline_dengue.joblib) que contiene los "
  "imputadores (KNN y moda), el StandardScaler, el OneHotEncoder(handle_unknown=\"ignore\"), SMOTE (que solo actúa durante "
  "el entrenamiento) y el modelo final. Junto con él se guarda model_metadata.json, con las variables, las reglas de calidad, "
  "las métricas y las versiones exactas de las librerías. El archivo requirements.txt fija esas mismas versiones "
  "(scikit-learn " + META["versiones"]["scikit-learn"] + ") para evitar errores al cargar el modelo en Streamlit Cloud.")
doc.add_heading("6.1 Predicción con datos nuevos", 2)
p("Se recargó el pipeline desde disco, como lo hace la aplicación, y se predijeron perfiles clínicos nuevos construidos "
  "para validar la lógica del modelo. Ninguno estuvo en el entrenamiento, y uno de ellos tiene valores faltantes para "
  "comprobar la imputación automática.")
n61 = t("6_1_prediccion_nuevos")
n61["Predicción"] = [f"{pr} (p = {coma(f'{pb:.3f}')})" for pr, pb in zip(n61["Predicción"], n61["Probabilidad hospitalización"])]
tabla(n61[["Registro de prueba", "Predicción", "Interpretación"]], anchos=[7, 4, 5.5], tam=8.5)
interpretacion(TXT["prediccion"])
p("**Predicción sobre registros reales del conjunto de prueba (no vistos en el entrenamiento):**")
pt = t("6_1_prediccion_test")
tabla(pt[["edad_aprox", "dias_sintomas", "dolor_abdo", "vomito", "caida_plaq", "regimen", "Real", "Predicho", "Prob. hosp."]], tam=8)

doc.add_heading("Información del despliegue", 3)
tabla_kv([
    ("Modelo serializado", "Sí: pipeline_dengue.joblib (pipeline completo) + model_metadata.json"),
    ("App Streamlit funcional", CFG["app_funcional"]),
    ("URL pública activa", CFG["url_activa"]),
    ("Pantallazo adjunto", CFG["pantallazo_txt"]),
], 5, 11.5)
if CFG.get("pantallazo") and (ROOT / CFG["pantallazo"]).exists():
    doc.add_picture(str(ROOT / CFG["pantallazo"]), width=Cm(15))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p("Figura 9. Aplicación desplegada en Streamlit.io.", italic=True, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_heading("Conclusiones", 1)
viñetas(TXT["conclusiones"])
doc.add_heading("Limitaciones y trabajo futuro", 2)
viñetas(TXT["limitaciones"])

doc.add_heading("Entregables", 2)
viñetas([
    "Documento completo con la estructura CRISP-DM en el formato U4_0 (este documento).",
    "Dataset utilizado: data/dengue_bucaramanga.csv (fuente: https://www.datos.gov.co/d/qzc7-jbg3).",
    f"Cuaderno de Google Colab con acceso público: {CFG['colab']}",
    f"Repositorio GitHub ({CFG['github']}) con app.py, el modelo serializado (pipeline_dengue.joblib), requirements.txt y README.md.",
    f"URL pública activa en Streamlit.io: {CFG['streamlit']}",
    "Reporte HTML generado con ydata-profiling: reports/reporte_profiling.html.",
])

doc.add_heading("Referencias", 2)
viñetas([
    "Alcaldía de Bucaramanga (2025). Dengue, Dengue grave y mortalidad por dengue municipio de Bucaramanga [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/d/qzc7-jbg3",
    "Breiman, L. (1996). Bagging predictors. Machine Learning, 24(2), 123-140.",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
    "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD, 785-794.",
    "Chapman, P., et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.",
    "Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52(3), 239-281.",
    "Ministerio de Salud y Protección Social (2010). Guía de atención clínica integral del paciente con dengue. Bogotá.",
    "Vargas Buitrago, J. (2026). Protocolo de despliegue: de modelos predictivos a aplicaciones web funcionales. UPB.",
])

out = ROOT / "doc" / "entrega" / "Entrega_Proyecto_Integrador_CRISP-DM.docx"
out.parent.mkdir(exist_ok=True)
doc.save(out)
print("Documento generado:", out)
