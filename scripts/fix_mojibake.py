"""Repara texto UTF-8 que fue leído como cp1252 (p. ej. 'notificaciÃ³n' -> 'notificación')
en las descripciones del diccionario de datos (no toca los datos del SIVIGILA)."""
import json, re
from pathlib import Path
import nbformat, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"[ÃÂ][\x80-\xBF\u0152-\u2122]")

def reparar(s):
    if not isinstance(s, str) or not PAT.search(s):
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        # reparar palabra por palabra
        out = []
        for w in re.split(r"(\s+)", s):
            try:
                out.append(w.encode("cp1252").decode("utf-8") if PAT.search(w) else w)
            except (UnicodeEncodeError, UnicodeDecodeError):
                out.append(w)
        return "".join(out)

# 1) JSON de descripciones
p = ROOT / "scripts" / "descripciones.json"
d = {reparar(k): reparar(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
p.write_text(json.dumps(d, ensure_ascii=False, indent=0), encoding="utf-8")

# 2) Tabla del diccionario
t = ROOT / "reports" / "tablas" / "2_1_diccionario.csv"
df = pd.read_csv(t, encoding="utf-8-sig")
df["Descripción"] = df["Descripción"].map(reparar)
df.to_csv(t, index=False, encoding="utf-8-sig")

# 3) Notebook: celda DESCRIPCIONES (fuente) y celda del diccionario (salidas)
nbp = ROOT / "notebooks" / "proyecto_crispdm_dengue.ipynb"
nb = nbformat.read(nbp, as_version=4)
n = 0
for c in nb.cells:
    if c.cell_type != "code":
        continue
    if c.source.startswith("DESCRIPCIONES = "):
        c.source = "DESCRIPCIONES = " + json.dumps(d, ensure_ascii=False, indent=1); n += 1
    if "diccionario = pd.DataFrame" in c.source:
        for o in c.outputs:
            if "data" in o:
                for k in ("text/plain", "text/html"):
                    if k in o["data"]:
                        o["data"][k] = reparar(o["data"][k])
            if "text" in o:
                o["text"] = reparar(o["text"])
        n += 1
nbformat.write(nb, nbp)
bad = [v for v in d.values() if PAT.search(v)]
print("celdas reparadas:", n, "| descripciones aún dañadas:", len(bad), bad[:3])
print(df.loc[df.Variable.isin(["fec_not", "sexo_", "tip_ss_"]), "Descripción"].tolist())
