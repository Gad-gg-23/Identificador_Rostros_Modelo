"""
Logica compartida para cargar el modelo entrenado y predecir la identidad de
una foto. La usan tanto inferir.py (linea de comandos) como app.py (interfaz
Gradio), para no cargar el checkpoint/MTCNN por separado en cada uno.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN

from dataset import normalizacion_facenet
from modelo import ModeloIdentificador


def cargar_modelo(cfg, ruta_checkpoint=None, dispositivo=None):
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    ruta_checkpoint = Path(ruta_checkpoint or cfg["salida"]["mejor_modelo"])
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

    return modelo, mtcnn, idx_a_persona, dispositivo


def predecir(imagen_pil, modelo, mtcnn, idx_a_persona, dispositivo, top=3):
    """Devuelve lista de (persona, probabilidad 0-1) o None si no se detecto rostro."""
    rostro = mtcnn(imagen_pil.convert("RGB"))
    if rostro is None:
        return None

    tensor = normalizacion_facenet(rostro / 255.0).unsqueeze(0).to(dispositivo)
    with torch.no_grad():
        logits = modelo(tensor)
        probabilidades = F.softmax(logits, dim=1)[0]

    top_valores, top_indices = probabilidades.topk(min(top, len(idx_a_persona)))
    return [(idx_a_persona[idx], valor) for valor, idx in zip(top_valores.tolist(), top_indices.tolist())]
