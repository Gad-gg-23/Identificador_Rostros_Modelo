"""
Descarga el dataset de rostros (privado) en data/.

Requiere haber hecho login antes: hf auth login (o variable HF_TOKEN),
con una cuenta agregada como miembro de la organizacion escom-identificador-rostros.

Uso:
    python descargar_datos.py --repo escom-identificador-rostros/identificador-rostros-escom
"""
import argparse
import sys
from pathlib import Path


def descargar_desde_huggingface(repo: str, version: str, destino: Path):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Error: falta huggingface_hub. Instala con: pip install huggingface_hub")

    print(f"Descargando '{repo}' (revision: {version}) desde Hugging Face...")
    ruta_local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=version,
        local_dir=destino,
    )
    print(f"OK: datos descargados en {ruta_local}")


def main():
    parser = argparse.ArgumentParser(description="Descarga el dataset a data/.")
    parser.add_argument("--repo", default="escom-identificador-rostros/identificador-rostros-escom")
    parser.add_argument("--version", default="main", help="Tag/revision a descargar (default: main)")
    parser.add_argument("--destino", default="data", help="Carpeta destino (default: data)")
    args = parser.parse_args()

    descargar_desde_huggingface(args.repo, args.version, Path(args.destino))


if __name__ == "__main__":
    main()
