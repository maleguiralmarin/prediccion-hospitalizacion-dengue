"""Ejecuta el notebook celda por celda, guardando después de cada celda (robusto a cortes)."""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks" / "proyecto_crispdm_dengue.ipynb"
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
t0 = time.time()
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        t = time.time()
        try:
            client.execute_cell(cell, i)
        except Exception as e:
            nbformat.write(nb, nb_path)
            print(f"ERROR en celda {i}: {type(e).__name__}: {str(e)[-1500:]}", flush=True)
            sys.exit(1)
        nbformat.write(nb, nb_path)
        print(f"celda {i} ok ({time.time()-t:.0f}s) {cell.source.splitlines()[0][:50]}", flush=True)
print(f"Terminado en {time.time()-t0:.0f} s", flush=True)
