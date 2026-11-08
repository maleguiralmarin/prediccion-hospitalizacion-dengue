"""Inserta las interpretaciones de resultados (textos_resultados.json) en el notebook ya ejecutado,
sin volver a ejecutarlo. Cada texto se ubica justo después de la celda de código cuya fuente
contiene la "ancla" indicada.

Uso: .venv/Scripts/python scripts/add_interpretaciones.py
"""
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks" / "proyecto_crispdm_dengue.ipynb"
TXT = json.loads((ROOT / "scripts" / "textos_resultados.json").read_text(encoding="utf-8"))
nb = nbformat.read(nb_path, as_version=4)

# Elimina interpretaciones insertadas antes (idempotente) y los marcadores provisionales
nb.cells = [c for c in nb.cells if not (c.cell_type == "markdown" and
            (c.source.startswith("**Interpretación de resultados") or "se completa con los resultados" in c.source))]

ANCLAS = [
    ('f1_sin = cross_val_score', "Experimento signos de dengue grave (iteración CRISP-DM)", "iteracion"),
    ('rel = rel.sort_values("Información mutua"', "Relaciones lineales y no lineales", "relaciones"),
    ('pca_res = pd.DataFrame', "Reducción de dimensiones", "pca"),
    ('res_clasicos = pd.DataFrame([evaluar_cv', "Modelos clásicos", "clasicos"),
    ('cv_ini = pd.DataFrame(cv_scores)', "Ensambles", "ensambles"),
    ('comp = pd.DataFrame({"F1-macro CV antes"', "Ajuste de hiperparámetros", "ajuste"),
    ('res_test = pd.DataFrame(filas)', "Resultados en test", "test"),
    ('mejor = candidatos[0]', "Pruebas estadísticas", "estadistica"),
    ('pi = permutation_importance', "Importancia de variables", "importancia"),
    ('tabla_61 = pd.DataFrame', "Predicción con datos nuevos", "prediccion"),
]


def insertar_despues(ancla, titulo, clave):
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and ancla in c.source:
            nb.cells.insert(i + 1, nbformat.v4.new_markdown_cell(f"**Interpretación de resultados: {titulo}.** {TXT[clave]}"))
            return
    raise ValueError(f"Ancla no encontrada: {ancla}")


for a, t, k in ANCLAS:
    insertar_despues(a, t, k)

concl = "# Conclusiones\n\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(TXT["conclusiones"], 1))
concl += "\n\n### Limitaciones y trabajo futuro\n" + "\n".join(f"- {x}" for x in TXT["limitaciones"])
for c in nb.cells:
    if c.cell_type == "markdown" and c.source.lstrip("-\n ").startswith("# Conclusiones"):
        c.source = "---\n" + concl
        break
else:
    nb.cells.append(nbformat.v4.new_markdown_cell("---\n" + concl))

nbformat.write(nb, nb_path)
print("Interpretaciones insertadas en", nb_path)
