"""
Entrena el identificador de rostros en dos fases:
  Fase 1: backbone congelado, solo se entrena la cabeza ArcFace (rapido, estabiliza).
  Fase 2: se descongela el ultimo bloque del backbone y se afina con LR bajo.
Guarda el mejor checkpoint (segun accuracy de validacion) con early stopping.

Uso:
    python src/entrenar.py --config config.yaml
"""
import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FaceDataset, construir_indice, split_por_sets, transform_entrenamiento, transform_evaluacion
from modelo import ModeloIdentificador


def evaluar(modelo, loader, dispositivo):
    modelo.eval()
    correctos, total = 0, 0
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            logits = modelo(imagenes)
            correctos += (logits.argmax(1) == etiquetas).sum().item()
            total += etiquetas.size(0)
    return correctos / total


def entrenar_epoca(modelo, loader, optimizador, dispositivo):
    modelo.train()
    perdida_total = 0.0
    for imagenes, etiquetas in tqdm(loader, desc="  entrenando", leave=False):
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        optimizador.zero_grad()
        logits = modelo(imagenes, etiquetas)
        perdida = torch.nn.functional.cross_entropy(logits, etiquetas, label_smoothing=0.1)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * imagenes.size(0)
    return perdida_total / len(loader.dataset)


def correr_fase(modelo, nombre_fase, epocas, lr, train_loader, val_loader, dispositivo,
                paciencia, mejor_acc_global, ruta_mejor_modelo, weight_decay, persona_a_idx, historial):
    optimizador = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, modelo.parameters()), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas)

    sin_mejora = 0
    for epoca in range(1, epocas + 1):
        perdida = entrenar_epoca(modelo, train_loader, optimizador, dispositivo)
        acc_val = evaluar(modelo, val_loader, dispositivo)
        scheduler.step()
        print(f"[{nombre_fase}] epoca {epoca}/{epocas} - perdida: {perdida:.4f} - acc_val: {acc_val:.4f}")
        historial.append({"fase": nombre_fase, "epoca": epoca, "perdida": perdida, "acc_val": acc_val})

        if acc_val > mejor_acc_global:
            mejor_acc_global = acc_val
            sin_mejora = 0
            torch.save({
                "modelo": modelo.state_dict(),
                "persona_a_idx": persona_a_idx,
                "acc_val": acc_val,
            }, ruta_mejor_modelo)
            print(f"  -> nuevo mejor modelo guardado (acc_val={acc_val:.4f})")
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                print(f"  -> early stopping ({nombre_fase}, sin mejora en {paciencia} epocas)")
                break

    return mejor_acc_global


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["datos"]["semilla"])

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {dispositivo}")

    raiz = Path(cfg["datos"]["rostros_raiz"])
    tamano = cfg["datos"]["tamano_imagen"]
    indice = construir_indice(raiz)
    train_items, val_items, persona_a_idx = split_por_sets(
        indice, cfg["datos"]["sets_validacion_por_persona"], cfg["datos"]["semilla"],
    )
    print(f"Clases: {len(persona_a_idx)} | train: {len(train_items)} fotos | val: {len(val_items)} fotos")

    train_ds = FaceDataset(train_items, persona_a_idx, transform_entrenamiento(tamano))
    val_ds = FaceDataset(val_items, persona_a_idx, transform_evaluacion(tamano))

    bs = cfg["entrenamiento"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    modelo = ModeloIdentificador(
        num_clases=len(persona_a_idx),
        embedding_dim=cfg["modelo"]["embedding_dim"],
        escala=cfg["modelo"]["arcface_escala"],
        margen=cfg["modelo"]["arcface_margin"],
    ).to(dispositivo)

    salida_dir = Path(cfg["salida"]["checkpoints"])
    salida_dir.mkdir(parents=True, exist_ok=True)
    ruta_mejor_modelo = Path(cfg["salida"]["mejor_modelo"])

    historial = []
    mejor_acc = 0.0

    print("\n=== Fase 1: backbone congelado ===")
    modelo.congelar_backbone()
    mejor_acc = correr_fase(
        modelo, "fase1", cfg["entrenamiento"]["fase1_epocas"], cfg["entrenamiento"]["lr_fase1"],
        train_loader, val_loader, dispositivo, cfg["entrenamiento"]["paciencia_early_stopping"],
        mejor_acc, ruta_mejor_modelo, cfg["entrenamiento"]["weight_decay"], persona_a_idx, historial,
    )

    print("\n=== Fase 2: fine-tuning del ultimo bloque ===")
    modelo.descongelar_ultimas_capas()
    mejor_acc = correr_fase(
        modelo, "fase2", cfg["entrenamiento"]["fase2_epocas"], cfg["entrenamiento"]["lr_fase2"],
        train_loader, val_loader, dispositivo, cfg["entrenamiento"]["paciencia_early_stopping"],
        mejor_acc, ruta_mejor_modelo, cfg["entrenamiento"]["weight_decay"], persona_a_idx, historial,
    )

    with open(cfg["salida"]["log_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fase", "epoca", "perdida", "acc_val"])
        writer.writeheader()
        writer.writerows(historial)

    print(f"\nListo. Mejor accuracy de validacion: {mejor_acc:.4f}")
    print(f"Checkpoint: {ruta_mejor_modelo}")


if __name__ == "__main__":
    main()
