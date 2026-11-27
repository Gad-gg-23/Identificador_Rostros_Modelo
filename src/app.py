"""
Interfaz grafica (Gradio) para probar el identificador de rostros sin usar
la linea de comandos: sube una foto o usa la camara y te dice quien es.

Uso (desde la raiz del repo):
    python src/app.py
Abre http://127.0.0.1:7860 en el navegador.
"""
import argparse

import gradio as gr
import yaml

from prediccion import cargar_modelo, predecir

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()

with open(args.config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

modelo, mtcnn, idx_a_persona, dispositivo = cargar_modelo(cfg, args.checkpoint)


def identificar(imagen):
    if imagen is None:
        return {"Sube una foto o usa la camara": 1.0}

    resultado = predecir(imagen, modelo, mtcnn, idx_a_persona, dispositivo, top=3)
    if resultado is None:
        return {"No se detecto ningun rostro": 1.0}

    return {persona: valor for persona, valor in resultado}


demo = gr.Interface(
    fn=identificar,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Foto"),
    outputs=gr.Label(num_top_classes=3, label="Prediccion"),
    title="Identificador de Rostros - ESCOM ML",
    description=(
        "Sube una foto o usa la camara. El modelo detecta el rostro y predice "
        "a cual de las personas del equipo corresponde."
    ),
)

if __name__ == "__main__":
    demo.launch()
