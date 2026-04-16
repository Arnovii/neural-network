"""
worker_imagenet.py

Worker asíncrono para entrenamiento distribuido en ImageNet.

USO:
    python worker_imagenet.py [opciones]

OPCIONES:
    --server-host     IP del Parameter Server            (default: 127.0.0.1)
    --server-port     Puerto TCP                         (default: 9999)
    --device          cpu | cuda | cuda:0 | mps           (default: auto-detect CUDA/MPS/CPU)
    --dataset         Dataset HF Hub                     (default: ILSVRC/imagenet-1k)
    --shuffle-buffer  Imágenes en buffer de shuffle      (default: 1000)
    --prefetch        Batches pre-cargados en background  (default: 4)
    --seed            Semilla RNG (None = aleatorio)     (default: None)
    --hf-token        Token HuggingFace
    --accum-steps     Batches a acumular antes de enviar  (default: 1)
    --quiet           Suprimir mensajes de progreso

NOTA: El rank y num_workers se asignan dinámicamente por el Parameter Server.
      batch_size, image_size, seed se reciben del PS mediante CONFIG.

EJEMPLO — 2 Workers en la misma máquina con GPUs distintas:
    python worker_imagenet.py --device cuda:0 &
    python worker_imagenet.py --device cuda:1 &
    (ejecutar con PS en paralelo)

EJEMPLO — Workers en máquinas distintas:
    python worker_imagenet.py --server-host 192.168.1.10 &
    python worker_imagenet.py --server-host 192.168.1.10 &
    python worker_imagenet.py --server-host 192.168.1.10 &
    (cualquier número de workers se conectará y recibirá su rank del PS)

TOKEN HF:
    ImageNet-1k requiere aceptar la licencia en:
    https://huggingface.co/datasets/ILSVRC/imagenet-1k
    y usar un token de acceso (export HF_TOKEN=hf_... o --hf-token).
    Alternativa pública: --dataset timm/imagenet-1k-wds
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Distributed.worker_node import WorkerNode


def get_default_device() -> str:
    """
    Detecta dispositivo disponible con prioridad: CUDA > MPS > CPU.

    Selecciona automáticamente el mejor dispositivo PyTorch disponible para cómputo.
    Esto asegura que workers puedan ejecutarse en hardware heterogéneo sin configuración.

    :returns: Identificador del dispositivo ('cuda' para GPU NVIDIA, 'mps' para Apple Metal,
              'cpu' como fallback)
    :rtype: str
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """
    Punto de entrada para el proceso worker asincrónico.

    Analiza argumentos de línea de comandos, muestra configuración, e inicializa
    una instancia WorkerNode para conectar con Parameter Server e iniciar
    entrenamiento distribuido de ImageNet-1k.

    :returns: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Worker asíncrono — Entrenamiento distribuido ImageNet-1k"
    )
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9999)
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument("--dataset", type=str, default="ILSVRC/imagenet-1k")
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla RNG (None = aleatorio)"
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print("=" * 68)
    print("WORKER ASÍNCRONO — ImageNet-1k Distribuido")
    print("=" * 68)
    print(f"  PS             : {args.server_host}:{args.server_port}")
    print(f"  Dataset        : {args.dataset}")
    device_str = (
        f"{args.device} (auto-detected)"
        if args.device == get_default_device()
        else args.device
    )
    print(f"  Device         : {device_str}")
    print(f"  Shuffle buffer : {args.shuffle_buffer}")
    print(f"  Prefetch       : {args.prefetch} batches")
    print(f"  Seed           : {args.seed or 'aleatorio'}")
    print(f"  Accum steps    : {args.accum_steps}")
    print(f"  HF Token       : {'✓ configurado' if hf_token else '✗ no configurado'}")
    print("=" * 68)
    print(
        "\n  ℹ rank, num_workers, batch_size, image_size se reciben del PS via CONFIG\n"
    )

    if not hf_token:
        print("\n⚠  Sin token HF — ILSVRC/imagenet-1k requiere autenticación.")
        print("   Alternativa pública: --dataset timm/imagenet-1k-wds\n")

    WorkerNode(
        server_host=args.server_host,
        server_port=args.server_port,
        dataset_name=args.dataset,
        device=args.device,
        shuffle_buffer=args.shuffle_buffer,
        prefetch_batches=args.prefetch,
        seed=args.seed,
        hf_token=hf_token,
        accum_steps=args.accum_steps,
    ).run()


if __name__ == "__main__":
    main()
