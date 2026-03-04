"""
worker.py

Punto de entrada del Worker Node.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
    python worker.py [opciones]

Opciones:
    --server-host   IP del Parameter Server              (default: 127.0.0.1)
    --server-port   Puerto TCP del Parameter Server      (default: 9999)
    --data-dir      Directorio donde está MNIST          (default: Data/)
    --quiet         Suprime mensajes de progreso

El Worker ya no necesita un ``--id``: el Parameter Server asigna
automáticamente un ID único a cada Worker que se conecta.

Ejemplo — tres workers en terminales distintas:
    python worker.py --server-host 192.168.1.10
    python worker.py --server-host 192.168.1.10
    python worker.py --server-host 192.168.1.10

──────────────────────────────────────────────────────────────────
NOTAS IMPORTANTES
──────────────────────────────────────────────────────────────────
* El Worker carga MNIST completo (60 000 imágenes de entrenamiento)
  al arrancar. El PS envía solo los índices de los ejemplos que
  debe usar cada Worker en cada época, evitando enviar cientos de
  MB por red.

* El Worker es persistente: no se desconecta al terminar una
  sesión de entrenamiento. Permanece activo esperando nuevas
  sesiones (TRAIN_START) hasta recibir STOP o ser interrumpido.

* El Worker debe iniciarse DESPUÉS de que el Parameter Server
  esté escuchando en el puerto indicado.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Distributed.worker_node import WorkerNode
from Utils.mnist_loader import load_mnist_train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Worker Node — Algoritmo de Diego Distribuido"
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
        help="Directorio de datos MNIST (default: Data/)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suprime mensajes de progreso"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WORKER NODE — Algoritmo de Diego Distribuido")
    print("=" * 70)
    print(f"  Parameter Server : {args.server_host}:{args.server_port}")
    print("  ID               : asignado por el PS al conectarse")
    print("=" * 70)

    # Carga MNIST completo localmente.
    # El PS decide qué índices usa cada Worker en cada época.
    print("\nCargando MNIST (60 000 imágenes de entrenamiento)...")
    X_raw, Y_raw = load_mnist_train(
        data_dir=args.data_dir,
        n_train=None,  # carga los 60 000
        download_if_missing=True,
        verbose=not args.quiet,
    )

    X_train = np.array(X_raw, dtype=np.float64)
    Y_train = np.array(Y_raw, dtype=np.int32)

    print(f"Dataset listo: {X_train.shape[0]} imágenes × {X_train.shape[1]} píxeles\n")

    worker = WorkerNode(
        server_host=args.server_host,
        server_port=args.server_port,
        X_train=X_train,
        Y_train=Y_train,
        verbose=not args.quiet,
    )

    worker.run()


if __name__ == "__main__":
    main()
