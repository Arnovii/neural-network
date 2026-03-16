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
    --cnn-device    Dispositivo PyTorch: cpu|cuda|mps    (default: cpu)
    --cnn-seed      Semilla para pesos CNN               (default: 42)
    --quiet         Suprime mensajes de progreso

──────────────────────────────────────────────────────────────────
NOTAS IMPORTANTES
──────────────────────────────────────────────────────────────────
* Al arrancar, el Worker carga CIFAR-10 completo (50 000 imágenes),
  extrae features con la CNN una sola vez y los almacena en RAM.
  En cada época solo se accede al subconjunto correspondiente.

* La arquitectura CNN y sus pesos los dicta el PS.
  El Worker los recibe automáticamente al conectarse (CNN_WEIGHTS).
  No es necesario especificar --cnn-arch.

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
from Utils.cifar_loader import NUM_CLASSES, load_cifar10_train, load_cifar10_test


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
        "--quiet", action="store_true", help="Suprime mensajes de progreso"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WORKER NODE — Algoritmo de Diego Distribuido (CIFAR-10 CNN+MLP)")
    print("=" * 70)
    print(f"  Parameter Server : {args.server_host}:{args.server_port}")
    print("  ID               : asignado por el PS al conectarse")
    print(f"  CNN device       : {args.cnn_device}")
    print(f"  CNN seed         : {args.cnn_seed}")
    print(f"  MLP hidden       : {args.hidden1} → {args.hidden2} → {NUM_CLASSES}")
    print("  CNN arch/pesos   : recibidos del PS al conectarse")
    print("=" * 70)

    # Carga CIFAR-10 completo en formato NCHW (3, 32, 32) listo para la CNN
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

    # Cargar datos de prueba — el Worker 0 los usará para extraer
    # features con su CNN/GPU y enviarlos al PS, evitando que el PS
    # tenga que hacer el forward pass en CPU.
    print("\nCargando CIFAR-10 prueba (10 000 imágenes)...")
    X_test, Y_test = load_cifar10_test(
        download_if_missing=True,
        verbose=not args.quiet,
    )
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.int32)

    worker = WorkerNode(
        server_host=args.server_host,
        server_port=args.server_port,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        cnn_device=args.cnn_device,
        cnn_seed=args.cnn_seed,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        verbose=not args.quiet,
    )

    worker.run()


if __name__ == "__main__":
    main()
