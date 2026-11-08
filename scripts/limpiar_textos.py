"""Deja los textos de los reportes en español y sin caracteres poco comunes
(raya, comillas tipográficas, flechas, emojis, símbolos matemáticos especiales).

Uso: .venv/Scripts/python scripts/limpiar_textos.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTEGER = '"EXCEPCIÃ“N": "P",'   # clave real del dato en el código: no se toca
TOKEN = "@@CLAVE_EXCEPCION@@"

# --- Reemplazos específicos (se aplican primero, en orden) -------------------
ESPECIFICOS = [
    # Ejemplo del error de codificación citado en el texto
    ('(`"EXCEPCIÃ“N"`)', "(la palabra EXCEPCIÓN aparece con caracteres dañados)"),
    ("('EXCEPCIÃ“N')", "(la palabra EXCEPCIÓN aparece con caracteres dañados)"),
    ("(«EXCEPCIÃ“N»)", "(la palabra EXCEPCIÓN aparece con caracteres dañados)"),
    # Flechas con significado propio
    ("`grupo_etario` → `edad_aprox`", "`grupo_etario` se convierte en `edad_aprox`"),
    ("y las fechas → `dias_sintomas` y `mes`", "y las fechas se convierten en `dias_sintomas` y `mes`"),
    ("(3 = desconocido → faltante)", "(3 = desconocido, se trata como faltante)"),
    ("(Fase 3 → Fase 1)", "(de la Fase 3 a la Fase 1)"),
    ('"1 (pac_hos_ → hospitalizado)"', '"1 (pac_hos_, renombrada como hospitalizado)"'),
    ('"Recodificación 1→1, 2→0; valores inválidos → NaN"',
     '"Recodificación: 1 se mantiene como 1, 2 pasa a 0 y los valores inválidos pasan a NaN"'),
    ("(`estrato`, `dias_sintomas`) → `KNNImputer", "(`estrato`, `dias_sintomas`): `KNNImputer"),
    ("**Binarias y categóricas → `SimpleImputer", "**Binarias y categóricas: `SimpleImputer"),
    ('"Codificación binaria 1/0 (1=Sí, 2→0=No)"', '"Codificación binaria 1/0 (1 = Sí; el 2 original pasa a 0 = No)"'),
    ('("grupo_etario → edad_aprox",', '("grupo_etario (convertida en edad_aprox)",'),
    ("`pac_hos_` → hospitalizado", "`pac_hos_`: hospitalizado"),
    ("**Create app → Deploy a public app from GitHub**", "**Create app** y luego **Deploy a public app from GitHub**"),
    # Rayas y guiones con significado propio
    ('"MAE": "—", "RMSE": "—", "R2": "—"', '"MAE": "No aplica", "RMSE": "No aplica", "R2": "No aplica"'),
    ("Salud Pública – SIVIGILA**", "Salud Pública (SIVIGILA)**"),
    ("**Boosting – XGBoost**", "**Boosting con XGBoost**"),
    ("**SIVIGILA – Alcaldía de Bucaramanga**", "**SIVIGILA de la Alcaldía de Bucaramanga**"),
    ('f"1–2 por variable (valores 3 y -89)"', '"1 o 2 por variable (valores 3 y -89)"'),
    ("123–140.", "123-140."), ("5–32.", "5-32."), ("321–357.", "321-357."), ("785–794.", "785-794."), ("239–281.", "239-281."),
    # Símbolos
    ('"≥ 5 clásicos y 3 ensambles', '"Al menos 5 clásicos y 3 ensambles'),
    ("(≈20 %)", "(aproximadamente 20 %)"),
    ("(de ~12 % en 2015–2016 a ~60 % en 2022)", "(de aproximadamente 12 % en 2015 y 2016 a cerca de 60 % en 2022)"),
    ('"Δ F1 CV 5x2"', '"Diferencia F1 CV 5x2"'),
    ('"χ² McNemar"', '"Chi cuadrado McNemar"'),
    ("(`fiebre`…`gp_migrant`)", "(desde `fiebre` hasta `gp_migrant`)"),
    ("(fiebre … acum_liqui)", "(desde fiebre hasta acum_liqui)"),
    ("desenlace…)", "desenlace, entre otras)"),
    ("(1/(n_var·var))", "(1/(n_var*var))"),
    # Emojis de la app y del README
    ('page_icon="🦟", ', ""), ('st.title("🦟 ', 'st.title("'), ('f"⚠️ **', 'f"**'), ('f"✅ **', 'f"**'),
    ('st.expander("ℹ️ ', 'st.expander("'), ("# 🦟 ", "# "),
    # Términos en inglés en la prosa y en los nombres de columnas
    ("triage", "triaje"),
    ("Votación (soft)", "Votación (suave)"), ("(Voting, soft)", "(Voting, suave)"), ("(voto *soft*)", "(voto *suave*)"),
    ("**Outliers imposibles**", "**Valores atípicos imposibles**"), ("existen outliers imposibles", "existen valores atípicos imposibles"),
    ("(shrinkage)", "(contracción)"),
    ("% del train)", "% del entrenamiento)"),
    ("F1 train", "F1 entrenamiento"),
    ('"Brecha train-test"', '"Brecha entrenamiento-prueba"'),
    ("% Nulos (train)", "% Nulos (entrenamiento)"),
    ('"F1 test"', '"F1 prueba"'),
    ('"F1-macro CV (std)"', '"F1-macro CV (desv. est.)"'),
    ("(media ± std)", "(media +/- desv. est.)"),
    ("(std 0,0033)", "(desviación estándar 0,0033)"),
    ("(std 0,0018 en 5×2 CV)", "(desviación estándar 0,0018 en la CV 5x2)"),
    ("El boxplot muestra", "El diagrama de caja muestra"), ("; boxplot;", "; diagrama de caja;"),
    ("el t-test corregido", "la prueba t corregida"), ("el t-test clásico", "la prueba t clásica"),
    ("t-test pareado corregido", "prueba t pareada corregida"), ("t-test corregido", "prueba t corregida"),
    ("p-valor t-test", "p-valor prueba t"),
    ("en el test independiente", "en el conjunto de prueba independiente"),
    ("la validación y el test conservan", "la validación y el conjunto de prueba conservan"),
    ("hacia la validación y el test.", "hacia la validación y el conjunto de prueba."),
    ("(balancear el test daría", "(balancear el conjunto de prueba daría"),
    ("el test se usa para", "el conjunto de prueba se usa para"),
    ("en train: corrige", "en entrenamiento: corrige"),
    ("sobre las predicciones en test", "sobre las predicciones en el conjunto de prueba"),
    ("del F1-macro en test.", "del F1-macro en prueba."),
    ("(70 % train)", "(70 % entrenamiento)"),
    ("(CV 5-fold)", "(CV de 5 particiones)"),
    ("Matriz de confusión (test)", "Matriz de confusión (prueba)"),
    ("Curvas ROC en test", "Curvas ROC en prueba"),
    ("(en test):", "(en prueba):"),
    ("evaluación en test", "evaluación en el conjunto de prueba"),
    ('label="train"', 'label="entrenamiento"'),
    ("Los resultados en test coinciden", "Los resultados en prueba coinciden"),
    ("brecha entre train y test", "brecha entre entrenamiento y prueba"),
    ("pacientes de test", "pacientes del conjunto de prueba"),
    ("en test es [0,837", "en prueba es [0,837"),
    ("registros reales de test", "registros reales del conjunto de prueba"),
    ("en validación cruzada y en test,", "en validación cruzada y en prueba,"),
    ("quedó a 0,005 en test", "quedó a 0,005 en prueba"),
    ("Curvas de validación (train vs validación)", "Curvas de validación (entrenamiento frente a validación)"),
    ('"F1-macro en test ≥ 0.85"', '"F1-macro en prueba mayor o igual a 0.85"'),
    ("brecha entre train-test", "brecha entre entrenamiento y prueba"),
]

# --- Reemplazos genéricos (después de los específicos) -----------------------
def genericos(s, ruta):
    if ruta.name == "PASOS_ENTREGA.md":
        s = s.replace(" → ", " > ")
    s = s.replace(" → ", ", luego ").replace("→", " a ")
    s = s.replace(" — ", ": ").replace("—", ", ")
    s = re.sub(r"\s*–\s*", " a ", s)
    s = s.replace("≥ ", "mayor o igual a ").replace("≥", "mayor o igual a ")
    s = s.replace("≤ ", "menor o igual a ").replace("≤", "menor o igual a ")
    s = s.replace("≈ ", "aproximadamente ").replace("≈", "aproximadamente ")
    s = s.replace("−", "-").replace("±", "+/-").replace("×", "x").replace("…", "...")
    s = s.replace(" · ", ", ").replace("·", ", ")
    comilla = '\\"' if ruta.suffix == ".json" else '"'
    s = s.replace("«", comilla).replace("»", comilla).replace("“", comilla).replace("”", comilla)
    return s

ARCHIVOS = ["scripts/build_notebook.py", "scripts/build_word.py", "scripts/textos_resultados.json",
            "scripts/add_interpretaciones.py", "scripts/entrega_config.json", "app.py", "README.md",
            "doc/PASOS_ENTREGA.md"]
RAROS = re.compile(r"[^\x00-\x7FáéíóúÁÉÍÓÚñÑüÜ¿¡°]")

for rel in ARCHIVOS:
    ruta = ROOT / rel
    s = ruta.read_text(encoding="utf-8").replace(PROTEGER, TOKEN)
    cambios = 0
    for a, b in ESPECIFICOS:
        n = s.count(a)
        if n:
            s = s.replace(a, b); cambios += n
    s2 = genericos(s, ruta)
    s = s2.replace(TOKEN, PROTEGER)
    restantes = sorted({c for c in s.replace(PROTEGER, "") if RAROS.match(c)})
    ruta.write_text(s, encoding="utf-8")
    print(f"{rel:36s} específicos={cambios:3d} | caracteres raros restantes: {restantes}")
