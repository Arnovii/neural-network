"""
worker.py

Punto de entrada del Worker Node para ImageNet con CNN + MLP.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
    python worker.py [opciones]

MODO LOCAL (dataset en disco):
    python worker.py --data-dir /ruta/a/ImageNet

MODO STREAM (sin descarga, requiere internet):
    python worker.py --hf-token hf_xxxx
    # o bien: export HF_TOKEN=hf_xxxx && python worker.py

MODO AUTO (detecta automáticamente):
    python worker.py --data-dir /ruta/a/ImageNet --hf-token hf_xxxx
    # usa local si existe train/ y val/, si no usa stream

──────────────────────────────────────────────────────────────────
OPCIONES
──────────────────────────────────────────────────────────────────
  --server-host   IP del Parameter Server              (default: 127.0.0.1)
  --server-port   Puerto TCP del Parameter Server      (default: 9999)
  --data-dir      Directorio raíz de ImageNet          (default: Data/ImageNet)
                  Debe contener train/ y val/ en formato ImageFolder.
                  Si no existe o está vacío → modo stream automático.
  --hf-token      Token HuggingFace para modo streaming
                  También acepta variable de entorno HF_TOKEN.
  --cnn-device    Dispositivo PyTorch: cpu | cuda | mps (default: cpu)
  --cache-dir     Directorio para shards de features   (default: Data/feature_cache)
  --quiet         Suprime mensajes de progreso

──────────────────────────────────────────────────────────────────
QUÉ OCURRE AL EJECUTAR
──────────────────────────────────────────────────────────────────
Primera vez (sin caché de shards):
  1. Se conecta al PS y recibe pesos de ResNet-18 (~44 MB)
  2. Extrae features por shards de 50k imágenes (~200 MB/shard)
     Modo local:  lee del disco  → horas en CPU, ~30 min en GPU
     Modo stream: descarga de HF → más lento (red), misma extracción
  3. Guarda shards .npy en cache_dir (~2.6 GB total, nunca las imágenes)
  4. Entrena el MLP epoch por epoch

Siguientes veces (shards en caché):
  - Detecta los shards ya guardados, se salta la extracción
  - Empieza a entrenar en segundos, sin internet
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Distributed.worker_node import WorkerNode
from Utils.imagenet_loader import NUM_CLASSES, detect_data_source


def _default_data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "ImageNet")


def _default_cache_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Data", "feature_cache"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Worker Node — Algoritmo de Diego Distribuido (ImageNet CNN+MLP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
            "Directorio raíz de ImageNet con train/ y val/. "
            "Si no existe o está vacío se usa modo stream. "
            "(default: Data/ImageNet/)"
        ),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help=(
            "Token HuggingFace para modo streaming. "
            "También acepta variable de entorno HF_TOKEN. "
            "Obligatorio si no hay dataset local."
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
        help="Directorio para shards de features (default: Data/feature_cache/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprime mensajes de progreso",
    )
    parser.add_argument(
        "--feature-cache",
        action="store_true",
        default=True,
        help="Enable per-batch feature caching for ResNet18 mode (default: enabled)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or _default_data_dir()
    cache_dir = args.cache_dir or _default_cache_dir()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    # ── Detectar modo ─────────────────────────────────────────────
    source = detect_data_source(data_dir)

    print("=" * 70)
    print("WORKER NODE — Algoritmo de Diego Distribuido (ImageNet CNN+MLP)")
    print("=" * 70)
    print(f"  Parameter Server : {args.server_host}:{args.server_port}")
    print("  ID               : asignado por el PS al conectarse")
    print(f"  CNN device       : {args.cnn_device}")
    print("  CNN arch/pesos   : resnet18 + ImageNet (recibidos del PS)")
    print(f"  Caché de shards  : {cache_dir}")
    print(f"  Clases           : {NUM_CLASSES}")
    print()
    print(f"  MODO DE DATOS    : {source.upper()}")

    if source == "local":
        print(f"  Dataset local    : {data_dir}")
    else:
        if not hf_token:
            print()
            print("  ✗ ERROR: Modo stream requiere token HuggingFace.")
            print("    Opciones:")
            print("      1. python worker.py --hf-token hf_xxxx")
            print("      2. export HF_TOKEN=hf_xxxx")
            print("    Obtén tu token en: https://huggingface.co/settings/tokens")
            sys.exit(1)
        print("  HuggingFace      : ILSVRC/imagenet-1k (streaming)")
        print(
            f"  Token            : {hf_token[:8]}{'*' * (len(hf_token) - 8) if len(hf_token) > 8 else ''}"
        )
        print()
        print("  NOTA: La primera sesión descarga imágenes bajo demanda y")
        print("        guarda solo los features (~2.6 GB). Las siguientes")
        print("        sesiones cargan los features del disco sin internet.")

    print("=" * 70)
    print()

    print("Inicializando WorkerNode...")
    worker = WorkerNode(
        data_dir=data_dir if source == "local" else None,
        server_host=args.server_host,
        server_port=args.server_port,
        device=args.cnn_device,
        verbose=not args.quiet,
        cache_dir=cache_dir,
        hf_token=hf_token,
        feature_cache_enabled=args.feature_cache,
    )

    print(f"\nConectando al PS en {args.server_host}:{args.server_port}...")
    print("(El PS debe estar escuchando antes de ejecutar este comando)\n")
    worker.run()


if __name__ == "__main__":
    main()
