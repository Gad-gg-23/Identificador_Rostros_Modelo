"""
Detecta, recorta y alinea el rostro de cada foto cruda en data/ y lo guarda
en rostros/ con el mismo esquema persona/set_xx/archivo.jpg.

Se hace una sola vez y se cachea en disco porque correr MTCNN en cada epoca
de entrenamiento seria demasiado lento.

Uso:
    python src/preprocesar_rostros.py --config config.yaml
"""
import argparse
from pathlib import Path

import yaml
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import torch


def listar_fotos(raiz: Path):
    return sorted(raiz.glob("*/*/*.jpg"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raiz = Path(cfg["datos"]["raiz"])
    destino_raiz = Path(cfg["datos"]["rostros_raiz"])
    tamano = cfg["datos"]["tamano_imagen"]

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=tamano, margin=20, post_process=False, device=dispositivo)

    fotos = listar_fotos(raiz)
    if not fotos:
        raise SystemExit(f"No se encontraron fotos en {raiz}. Corre descargar_datos.py primero.")

    sin_rostro = []
    for foto in tqdm(fotos, desc="Detectando rostros"):
        persona, set_id, nombre = foto.parts[-3], foto.parts[-2], foto.name
        destino = destino_raiz / persona / set_id / nombre
        if destino.exists():
            continue

        imagen = Image.open(foto).convert("RGB")
        rostro = mtcnn(imagen)
        if rostro is None:
            sin_rostro.append(str(foto))
            continue

        destino.parent.mkdir(parents=True, exist_ok=True)
        rostro_img = Image.fromarray(
            rostro.permute(1, 2, 0).clamp(0, 255).byte().numpy()
        )
        rostro_img.save(destino, quality=95)

    print(f"\nListo. Rostros guardados en {destino_raiz}")
    print(f"Fotos sin rostro detectado: {len(sin_rostro)}")
    if sin_rostro:
        reporte = destino_raiz / "sin_rostro_detectado.txt"
        reporte.parent.mkdir(parents=True, exist_ok=True)
        reporte.write_text("\n".join(sin_rostro), encoding="utf-8")
        print(f"Lista guardada en {reporte}")


if __name__ == "__main__":
    main()
