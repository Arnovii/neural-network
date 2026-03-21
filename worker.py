"""
worker.py

Punto de entrada del Worker Node para ImageNet con CNN + MLP.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
    python worker.py [opciones]

Opciones:
    --server-host   IP del Parameter Server              (default: 127.0.0.1)
    --server-port   Puerto TCP del Parameter Server      (default: 9999)
    --data-dir      Directorio raíz de ImageNet          (default: Data/ImageNet)
                    Debe contener train/ y val/ en formato ImageFolder:
                        Data/ImageNet/train/n01440764/img1.JPEG ...
                        Data/ImageNet/val/n01440764/img1.JPEG   ...
    --cnn-device    Dispositivo PyTorch: cpu | cuda | mps (default: cpu)
    --cache-dir     Directorio para caché de shards y scaler
                    (default: Data/feature_cache/)
    --quiet         Suprime mensajes de progreso

──────────────────────────────────────────────────────────────────
DIFERENCIAS CON CIFAR-10
──────────────────────────────────────────────────────────────────
* Las imágenes NO se cargan en RAM. El Worker lee del disco bajo
  demanda usando DataLoader de PyTorch (ImageNet: ~150 GB).

* La extracción de features ocurre por shards de 50 000 imágenes.
  Cada shard se guarda en disco (~200 MB) y se reutiliza en sesiones
  posteriores. La primera extracción puede tardar varias horas en CPU
  o ~30-60 min en GPU.

* El Worker aplica un FeatureScaler (StandardScaler) calculado sobre
  el shard 0 antes de cada forward del MLP — mejora la convergencia
  con 1000 clases.

* La CNN y sus pesos los dicta el PS (CNN_WEIGHTS).
  Solo se usa ResNet-18 con pesos ImageNet preentrenados.

* El Worker es persistente entre sesiones de entrenamiento.

──────────────────────────────────────────────────────────────────
EJEMPLO
──────────────────────────────────────────────────────────────────
    # Worker con GPU en máquina remota:
    python worker.py --server-host 192.168.1.10 --cnn-device cuda

    # Worker en CPU local (para pruebas):
    python worker.py --data-dir /mnt/datasets/ImageNet
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Distributed.worker_node import WorkerNode
from Utils.imagenet_loader import NUM_CLASSES, get_dataset_size


def _default_data_dir() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "Data", "ImageNet")


def _default_cache_dir() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "Data", "feature_cache")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Worker Node — Algoritmo de Diego Distribuido (ImageNet CNN+MLP)"
    )
    parser.add_argument(
        "--server-host",
        type=str,
        default="127.0.0.1",
        help="IP del Parameter Server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=9999,
        help="Puerto TCP del Parameter Server (default: 9999)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directorio raíz de ImageNet con train/ y val/ (default: Data/ImageNet/)"
        ),
    )
    parser.add_argument(
        "--cnn-device",
        type=str,
        default="cpu",
        help="Dispositivo PyTorch: cpu | cuda | mps (default: cpu)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directorio para caché de shards de features (default: Data/feature_cache/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprime mensajes de progreso",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or _default_data_dir()
    cache_dir = args.cache_dir or _default_cache_dir()

    # ── Verificar estructura del dataset ──────────────────────────
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(
            f"\n✗ ERROR: No se encontró la estructura de ImageNet en '{data_dir}'.\n"
            f"  Se esperan los directorios:\n"
            f"    {train_dir}/\n"
            f"    {val_dir}/\n"
            f"\n"
            f"  Cada split debe seguir el formato ImageFolder de torchvision:\n"
            f"    train/n01440764/img1.JPEG\n"
            f"    val/n01440764/img1.JPEG\n"
            f"\n"
            f"  Usa --data-dir para especificar la ruta correcta.\n"
        )
        sys.exit(1)

    print("=" * 70)
    print("WORKER NODE — Algoritmo de Diego Distribuido (ImageNet CNN+MLP)")
    print("=" * 70)
    print(f"  Parameter Server : {args.server_host}:{args.server_port}")
    print("  ID               : asignado por el PS al conectarse")
    print(f"  CNN device       : {args.cnn_device}")
    print("  CNN arch/pesos   : resnet18 + ImageNet (recibidos del PS)")
    print(f"  Dataset          : {data_dir}")
    print(f"  Caché de shards  : {cache_dir}")
    print(f"  Clases           : {NUM_CLASSES}")
    print("=" * 70)

    # Mostrar tamaño del dataset sin cargar imágenes
    print("\nIndexando dataset ImageNet (sin cargar imágenes)...")
    try:
        n_train = get_dataset_size("train", data_dir=data_dir)
        n_val = get_dataset_size("val", data_dir=data_dir)
        print(f"  Train: {n_train:>10,} imágenes")
        print(f"  Val:   {n_val:>10,} imágenes")
    except Exception as e:
        print(f"  ⚠ No se pudo indexar el dataset: {e}")
        print("  Continuando de todas formas...")

    print()
    print("Inicializando WorkerNode...")
    worker = WorkerNode(
        data_dir=data_dir,
        server_host=args.server_host,
        server_port=args.server_port,
        device=args.cnn_device,
        verbose=not args.quiet,
        cache_dir=cache_dir,
    )

    print(f"\nConectando al PS en {args.server_host}:{args.server_port}...")
    print("(El PS debe estar escuchando antes de ejecutar este comando)\n")
    worker.run()


if __name__ == "__main__":
    main()
