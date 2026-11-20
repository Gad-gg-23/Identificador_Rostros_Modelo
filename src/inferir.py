"""
Predice la identidad de una persona a partir de una foto nueva (como la que
tomara el profesor el dia del examen): detecta el rostro, lo alinea y lo
pasa por el modelo entrenado.

Uso:
    python src/inferir.py --imagen foto.jpg --checkpoint checkpoints/mejor_modelo.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from facenet_pytorch import MTCNN
from PIL import Image

from dataset import normalizacion_facenet
from modelo import ModeloIdentificador


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagen", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    ruta_checkpoint = Path(args.checkpoint or cfg["salida"]["mejor_modelo"])
    checkpoint = torch.load(ruta_checkpoint, map_location=dispositivo)
    persona_a_idx = checkpoint["persona_a_idx"]
    idx_a_persona = {i: p for p, i in persona_a_idx.items()}

    mtcnn = MTCNN(image_size=cfg["datos"]["tamano_imagen"], margin=20, post_process=False, device=dispositivo)
    modelo = ModeloIdentificador(
        num_clases=len(persona_a_idx),
        embedding_dim=cfg["modelo"]["embedding_dim"],
        escala=cfg["modelo"]["arcface_escala"],
        margen=cfg["modelo"]["arcface_margin"],
    ).to(dispositivo)
    modelo.load_state_dict(checkpoint["modelo"])
    modelo.eval()

    imagen = Image.open(args.imagen).convert("RGB")
    rostro = mtcnn(imagen)
    if rostro is None:
        raise SystemExit("No se detecto ningun rostro en la imagen.")

    tensor = normalizacion_facenet(rostro / 255.0).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        logits = modelo(tensor)
        probabilidades = F.softmax(logits, dim=1)[0]

    top_valores, top_indices = probabilidades.topk(args.top)
    print("Predicciones:")
    for valor, idx in zip(top_valores.tolist(), top_indices.tolist()):
        print(f"  {idx_a_persona[idx]:35s} {valor * 100:5.1f}%")


if __name__ == "__main__":
    main()
