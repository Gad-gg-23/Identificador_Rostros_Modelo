"""
Indice del dataset de rostros ya preprocesados, split por set completo
(no por foto individual) para que validacion use sesiones/ropa/fondos
que el modelo nunca vio, y Dataset de PyTorch con augmentacion.
"""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def construir_indice(raiz: Path):
    """Devuelve lista de (ruta, persona, set_id) a partir de persona/set_xx/archivo.jpg."""
    indice = []
    for ruta in sorted(raiz.glob("*/*/*.jpg")):
        persona, set_id = ruta.parts[-3], ruta.parts[-2]
        indice.append((ruta, persona, set_id))
    if not indice:
        raise SystemExit(f"No se encontraron rostros en {raiz}. Corre preprocesar_rostros.py primero.")
    return indice


def split_por_sets(indice, sets_val_por_persona: int, semilla: int):
    """Para cada persona reserva N sets completos para validacion; el resto es train."""
    rng = random.Random(semilla)

    sets_por_persona = {}
    for ruta, persona, set_id in indice:
        sets_por_persona.setdefault(persona, set()).add(set_id)

    sets_val_por_persona_elegidos = {}
    for persona, sets in sets_por_persona.items():
        sets_ordenados = sorted(sets)
        rng.shuffle(sets_ordenados)
        n = min(sets_val_por_persona, len(sets_ordenados) - 1)  # siempre deja al menos 1 set para train
        sets_val_por_persona_elegidos[persona] = set(sets_ordenados[:n])

    train, val = [], []
    for ruta, persona, set_id in indice:
        if set_id in sets_val_por_persona_elegidos[persona]:
            val.append((ruta, persona))
        else:
            train.append((ruta, persona))

    clases = sorted(sets_por_persona.keys())
    persona_a_idx = {persona: i for i, persona in enumerate(clases)}
    return train, val, persona_a_idx


def normalizacion_facenet(x):
    """Mismo preprocesado que espera InceptionResnetV1 preentrenado en VGGFace2."""
    return (x - 0.5) / 0.5


def transform_entrenamiento(tamano: int):
    return transforms.Compose([
        transforms.Resize((tamano, tamano)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Lambda(normalizacion_facenet),
    ])


def transform_evaluacion(tamano: int):
    return transforms.Compose([
        transforms.Resize((tamano, tamano)),
        transforms.ToTensor(),
        transforms.Lambda(normalizacion_facenet),
    ])


class FaceDataset(Dataset):
    def __init__(self, items, persona_a_idx, transform):
        self.items = items
        self.persona_a_idx = persona_a_idx
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ruta, persona = self.items[i]
        imagen = Image.open(ruta).convert("RGB")
        return self.transform(imagen), self.persona_a_idx[persona]
