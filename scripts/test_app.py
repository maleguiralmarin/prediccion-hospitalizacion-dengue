"""Prueba automática de la app Streamlit con AppTest (sin navegador)."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

app = str(Path(__file__).resolve().parents[1] / "app.py")

def correr(marcar=(), estrato=None):
    at = AppTest.from_file(app, default_timeout=60).run()
    assert not at.exception, at.exception
    for cb in at.checkbox:
        if cb.label in marcar:
            cb.check()
    if estrato is not None:
        at.selectbox(key=None) if False else None
    at.button[0].click().run()
    assert not at.exception, at.exception
    res = [e.value for e in at.error] + [s.value for s in at.success]
    return res

print("Caso leve  ->", correr(marcar=("Fiebre", "Cefalea (dolor de cabeza)")))
print("Caso grave ->", correr(marcar=("Fiebre", "Dolor abdominal", "Vómito", "Caída de plaquetas (< 100.000)")))
