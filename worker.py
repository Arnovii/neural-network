"""
worker.py

Punto de entrada del Worker Node para CIFAR-10 con CNN + MLP.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
    python worker.py [opciones]

Opciones:
    --server-host   IP del Parameter Server              (default: 127.0.0.1)
    --server-port   Puerto TCP del Parameter Server      (default: 9999)
    --data-dir      Directorio de datos CIFAR-10         (default: Data/)
    --hidden1       Neuronas en la capa oculta 1 del MLP (default: 256)
    --hidden2       Neuronas en la capa oculta 2 del MLP (default: 128)
    --cnn-arch      Arquitectura CNN: simple | resnet18  (default: simple)
    --cnn-pretrained  Usar pesos ImageNet (solo resnet18)
    --cnn-device    Dispositivo PyTorch: cpu|cuda|mps    (default: cpu)
    --cnn-seed      Semilla para pesos CNN               (default: 42)
    --quiet         Suprime mensajes de progreso

──────────────────────────────────────────────────────────────────
NOTAS IMPORTANTES
──────────────────────────────────────────────────────────────────
* Al arrancar, el Worker carga CIFAR-10 completo (50 000 imágenes),
  extrae features con la CNN una sola vez y los almacena en RAM.
  En cada época solo se accede al subconjunto correspondiente.

* La CNN debe tener los mismos pesos en todos los Workers.
  Asegúrate de usar el mismo --cnn-arch y --cnn-seed en todos.

* El Worker es persistente: permanece activo entre sesiones de
  entrenamiento hasta recibir STOP del PS o ser interrumpido.

* Debe iniciarse DESPUÉS de que el PS esté escuchando.

Ejemplo — tres workers en terminales distintas:
    python worker.py --server-host 192.168.1.10
    python worker.py --server-host 192.168.1.10
    python worker.py --server-host 192.168.1.10
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Distributed.worker_node import WorkerNode
from Utils.cifar_loader import NUM_CLASSES, load_cifar10_train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Worker Node — Algoritmo de Diego Distribuido (CIFAR-10 CNN+MLP)"
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
        help="Directorio de datos CIFAR-10 (default: Data/)",
    )
    parser.add_argument(
        "--hidden1",
        type=int,
        default=256,
        help="Neuronas capa oculta 1 del MLP (default: 256)",
    )
    parser.add_argument(
        "--hidden2",
        type=int,
        default=128,
        help="Neuronas capa oculta 2 del MLP (default: 128)",
    )
    parser.add_argument(
        "--cnn-arch",
        type=str,
        default="simple",
        choices=["simple", "resnet18"],
        help="Arquitectura CNN: simple | resnet18 (default: simple)",
    )
    parser.add_argument(
        "--cnn-pretrained",
        action="store_true",
        help="Usar pesos ImageNet para ResNet-18 (requiere descarga)",
    )
    parser.add_argument(
        "--cnn-device",
        type=str,
        default="cpu",
        help="Dispositivo PyTorch: cpu | cuda | mps (default: cpu)",
    )
    parser.add_argument(
        "--cnn-seed",
        type=int,
        default=42,
        help="Semilla para inicialización CNN (default: 42)",
    )
    parser.add_argument(
        "--cnn-pretrain-epochs",
        type=int,
        default=0,
        help="Épocas de preentrenamiento local CNN (default: 0, el PS la distribuye)",
    )
    parser.add_argument(
        "--cnn-pretrain-lr",
        type=float,
        default=1e-3,
        help="Learning rate del preentrenamiento CNN (default: 0.001)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suprime mensajes de progreso"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WORKER NODE — Algoritmo de Diego Distribuido (CIFAR-10 CNN+MLP)")
    print("=" * 70)
    print(f"  Parameter Server : {args.server_host}:{args.server_port}")
    print(f"  ID               : asignado por el PS al conectarse")
    print(
        f"  CNN arch         : {args.cnn_arch}"
        + (" (pretrained)" if args.cnn_pretrained else "")
    )
    print(f"  CNN device       : {args.cnn_device}")
    print(f"  CNN seed         : {args.cnn_seed}")
    print(f"  MLP hidden       : {args.hidden1} → {args.hidden2} → {NUM_CLASSES}")
    if args.cnn_arch == "simple":
        if args.cnn_pretrain_epochs > 0:
            print(
                f"  CNN pretrain     : local {args.cnn_pretrain_epochs} épocas (el PS sobreescribirá con la suya)"
            )
        else:
            print(f"  CNN pretrain     : ninguno (recibirá CNN del PS al conectarse)")
    print("=" * 70)

    # Carga CIFAR-10 completo en formato NCHW (3, 32, 32) listo para la CNN
    # El PS decide qué índices usa cada Worker en cada época.
    print("\nCargando CIFAR-10 (50 000 imágenes)...")
    X_train, Y_train = load_cifar10_train(
        data_dir=args.data_dir,
        download_if_missing=True,
        verbose=not args.quiet,
    )
    # Garantizar tipos correctos para PyTorch (float32) y NumPy (int32)
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.int32)
    print(
        f"Dataset listo: {X_train.shape[0]} imágenes  shape por imagen: {X_train.shape[1:]}\n"
    )

    worker = WorkerNode(
        server_host=args.server_host,
        server_port=args.server_port,
        X_train=X_train,
        Y_train=Y_train,
        cnn_arch=args.cnn_arch,
        cnn_pretrained=args.cnn_pretrained,
        cnn_device=args.cnn_device,
        cnn_seed=args.cnn_seed,
        cnn_pretrain_epochs=args.cnn_pretrain_epochs,
        cnn_pretrain_lr=args.cnn_pretrain_lr,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        verbose=not args.quiet,
    )

    worker.run()


if __name__ == "__main__":
    main()
