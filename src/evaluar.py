"""
Evalua el checkpoint entrenado sobre el split de validacion: accuracy global,
precision/recall/F1 por persona y matriz de confusion.

Uso:
    python src/evaluar.py --config config.yaml --checkpoint checkpoints/mejor_modelo.pt
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import FaceDataset, construir_indice, split_por_sets, transform_evaluacion
from modelo import ModeloIdentificador


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Default: config['salida']['mejor_modelo']")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    ruta_checkpoint = Path(args.checkpoint or cfg["salida"]["mejor_modelo"])
    checkpoint = torch.load(ruta_checkpoint, map_location=dispositivo)
    persona_a_idx = checkpoint["persona_a_idx"]
    idx_a_persona = {i: p for p, i in persona_a_idx.items()}

    raiz = Path(cfg["datos"]["rostros_raiz"])
    indice = construir_indice(raiz)
    _, val_items, _ = split_por_sets(
        indice, cfg["datos"]["sets_validacion_por_persona"], cfg["datos"]["semilla"],
    )
    val_ds = FaceDataset(val_items, persona_a_idx, transform_evaluacion(cfg["datos"]["tamano_imagen"]))
    val_loader = DataLoader(val_ds, batch_size=cfg["entrenamiento"]["batch_size"], shuffle=False, num_workers=2)

    modelo = ModeloIdentificador(
        num_clases=len(persona_a_idx),
        embedding_dim=cfg["modelo"]["embedding_dim"],
        escala=cfg["modelo"]["arcface_escala"],
        margen=cfg["modelo"]["arcface_margin"],
    ).to(dispositivo)
    modelo.load_state_dict(checkpoint["modelo"])
    modelo.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imagenes, etiquetas in val_loader:
            logits = modelo(imagenes.to(dispositivo))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(etiquetas.tolist())

    nombres_clases = [idx_a_persona[i] for i in range(len(idx_a_persona))]

    print(f"Accuracy guardada en checkpoint: {checkpoint['acc_val']:.4f}")
    print("\nReporte por persona:\n")
    print(classification_report(y_true, y_pred, target_names=nombres_clases, zero_division=0))

    matriz = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(matriz, display_labels=nombres_clases).plot(
        ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues",
    )
    salida_img = ruta_checkpoint.parent / "matriz_confusion.png"
    fig.tight_layout()
    fig.savefig(salida_img, dpi=150)
    print(f"\nMatriz de confusion guardada en {salida_img}")


if __name__ == "__main__":
    main()
