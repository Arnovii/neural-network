"""
worker.py

Punto de entrada del Worker Node.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
    python worker.py [opciones]

Opciones:
    --id            Identificador único de este Worker   (requerido)
    --server-host   IP del Parameter Server              (default: 127.0.0.1)
    --server-port   Puerto TCP del Parameter Server      (default: 9999)
    --hidden        Neuronas en la capa oculta           (default: 30)
    --n-train       Total de ejemplos en MNIST train     (default: 10000)
    --data-dir      Directorio donde está MNIST          (default: Data/)
    --quiet         Suprime mensajes por época

Ejemplo — tres workers en terminales distintas:
    python worker.py --id 0 --server-host 192.168.1.10
    python worker.py --id 1 --server-host 192.168.1.10
    python worker.py --id 2 --server-host 192.168.1.10

──────────────────────────────────────────────────────────────────
NOTAS IMPORTANTES
──────────────────────────────────────────────────────────────────
* Cada Worker debe tener los archivos de MNIST en Data/ (se
  descargan automáticamente si no existen).

* El Worker carga MNIST completo en memoria al arrancar y solo
  usa los índices que le asigne el Parameter Server en cada época.
  Esto evita enviar cientos de MB de imágenes por la red.

* El Worker NO actualiza sus pesos. Solo calcula gradientes y los
  envía al Parameter Server. La actualización la hace el PS.

* --n-train debe coincidir con el valor configurado en el PS,
  ya que los índices enviados pertenecen a ese rango.

* Los Workers deben iniciarse DESPUÉS de que el Parameter Server
  esté escuchando, o bien reintentar la conexión.
"""

import argparse
import os
import sys

import numpy as np

# Asegura que los módulos del proyecto sean importables
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Distributed.worker_node import WorkerNode
from Utils.mnist_loader import load_mnist_train


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Worker Node — Algoritmo de Diego Distribuido"
    )
    parser.add_argument("--id",          type=int,   required=True,
                        help="Identificador único de este Worker (requerido)")
    parser.add_argument("--server-host", type=str,   default="127.0.0.1",
                        help="IP del Parameter Server (default: 127.0.0.1)")
    parser.add_argument("--server-port", type=int,   default=9999,
                        help="Puerto TCP del Parameter Server (default: 9999)")
    parser.add_argument("--hidden",      type=int,   default=30,
                        help="Neuronas en la capa oculta (default: 30)")
    parser.add_argument("--n-train",     type=int,   default=10_000,
                        help="Total de ejemplos MNIST a cargar (default: 10000)")
    parser.add_argument("--data-dir",    type=str,   default=None,
                        help="Directorio de datos MNIST (default: Data/)")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suprime mensajes de progreso por época")
    args = parser.parse_args()

    INPUT_SIZE  = 784
    OUTPUT_SIZE = 10

    print("=" * 70)
    print(f"WORKER {args.id} — Configuración")
    print("=" * 70)
    print(f"  Parameter Server : {args.server_host}:{args.server_port}")
    print(f"  Arquitectura     : {INPUT_SIZE} → {args.hidden} → {OUTPUT_SIZE}")
    print(f"  Ejemplos train   : {args.n_train}")
    print("=" * 70)

    # Carga MNIST localmente.
    # Cada Worker tiene el dataset completo; el PS decide qué índices
    # usa cada uno en cada época, sin solapamiento entre Workers.
    print(f"\n[W{args.id}] Cargando MNIST...")
    X_raw, Y_raw = load_mnist_train(
        data_dir            = args.data_dir,
        n_train             = args.n_train,
        download_if_missing = True,
        verbose             = not args.quiet,
    )

    # Convierte a ndarray para operaciones matriciales eficientes.
    # X: (N, 784) float64 normalizado a [0, 1]
    # Y: (N,)     int32
    X_train = np.array(X_raw, dtype=np.float64)
    Y_train = np.array(Y_raw, dtype=np.int32)

    print(
        f"[W{args.id}] Dataset listo: "
        f"{X_train.shape[0]} imágenes × {X_train.shape[1]} píxeles"
    )

    worker = WorkerNode(
        worker_id   = args.id,
        server_host = args.server_host,
        server_port = args.server_port,
        X_train     = X_train,
        Y_train     = Y_train,
        input_size  = INPUT_SIZE,
        hidden_size = args.hidden,
        output_size = OUTPUT_SIZE,
        verbose     = not args.quiet,
    )

    worker.run()


if __name__ == "__main__":
    main()
