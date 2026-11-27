"""
Predice la identidad de una persona a partir de una foto nueva (como la que
tomara el profesor el dia del examen): detecta el rostro, lo alinea y lo
pasa por el modelo entrenado.

Uso:
    python src/inferir.py --imagen foto.jpg --checkpoint checkpoints/mejor_modelo.pt
"""
import argparse

import yaml
from PIL import Image

from prediccion import cargar_modelo, predecir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagen", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    modelo, mtcnn, idx_a_persona, dispositivo = cargar_modelo(cfg, args.checkpoint)

    imagen = Image.open(args.imagen)
    resultado = predecir(imagen, modelo, mtcnn, idx_a_persona, dispositivo, args.top)
    if resultado is None:
        raise SystemExit("No se detecto ningun rostro en la imagen.")

    print("Predicciones:")
    for persona, valor in resultado:
        print(f"  {persona:35s} {valor * 100:5.1f}%")


if __name__ == "__main__":
    main()
