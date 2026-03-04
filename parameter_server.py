"""
parameter_server.py

Punto de entrada del Parameter Server.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
    python parameter_server.py [opciones]

Opciones:
    --host          IP en la que escucha el servidor     (default: 0.0.0.0)
    --port          Puerto TCP                           (default: 9999)
    --workers       Número de Workers esperados          (default: 2)
    --epochs        Épocas de entrenamiento              (default: 10)
    --hidden        Neuronas en la capa oculta           (default: 30)
    --lr            Tasa de aprendizaje                  (default: 0.1)
    --n-train       Total de ejemplos de entrenamiento   (default: 10000)
    --seed          Semilla aleatoria                    (default: ninguna)

Ejemplo — servidor esperando 3 workers, 20 épocas:
    python parameter_server.py --workers 3 --epochs 20

──────────────────────────────────────────────────────────────────
ARQUITECTURA
──────────────────────────────────────────────────────────────────
El PS inicializa los pesos de la red con Xavier e inmediatamente
empieza a escuchar conexiones TCP. Una vez que todos los Workers
se conectan, ejecuta el loop de entrenamiento distribuido:

    Por cada época:
        1. Dividir índices 0..n_train en N chunks disjuntos.
        2. Broadcast: enviar params + índices a cada Worker.
        3. Esperar gradientes de TODOS los Workers (barrera).
        4. Promediar gradientes: ∇θ = (1/N) * Σ ∇θL(Bᵢ)
        5. Actualizar pesos:     θ ← θ − lr * ∇θ

Al finalizar imprime el historial de precisión y pérdida por época.
"""

import argparse
import os
import sys

import numpy as np

# Asegura que los módulos del proyecto sean importables
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Distributed.parameter_server import ParameterServer
from Utils.math_utils import xavier_initialization, vector_zeros


# ================================================================
# INICIALIZACIÓN DE PARÁMETROS
# ================================================================


def _init_params(
    input_size: int,
    hidden_size: int,
    output_size: int,
    seed: int | None,
) -> dict:
    """
    Inicializa los parámetros de la red con Xavier.

    :param input_size:  Neuronas de entrada.
    :param hidden_size: Neuronas en la capa oculta.
    :param output_size: Neuronas de salida (clases).
    :param seed:        Semilla aleatoria para reproducibilidad.
    :return: Diccionario con W1, b1, W2, b2.
    """
    if seed is not None:
        np.random.seed(seed)

    return {
        "W1": xavier_initialization(input_size, hidden_size),
        "b1": vector_zeros(hidden_size),
        "W2": xavier_initialization(hidden_size, output_size),
        "b2": vector_zeros(output_size),
    }


# ================================================================
# CALLBACK DE PROGRESO
# ================================================================


def _on_epoch_end(epoch: int, total: int, accuracy: float, loss: float) -> None:
    """Muestra el resumen de la época en consola."""
    bar = "█" * int(accuracy / 5)
    print(f"  [{bar:<20}] {accuracy:5.2f}%  loss={loss:.4f}  ({epoch}/{total})")


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter Server — Algoritmo de Diego Distribuido"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="IP en la que escucha el servidor (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=9999, help="Puerto TCP (default: 9999)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Número de Workers a esperar (default: 2)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Épocas de entrenamiento (default: 10)"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=30,
        help="Neuronas en la capa oculta (default: 30)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Tasa de aprendizaje (default: 0.1)"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=10_000,
        help="Total de ejemplos de entrenamiento (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla aleatoria (default: ninguna)"
    )
    args = parser.parse_args()

    INPUT_SIZE = 784
    OUTPUT_SIZE = 10

    print("=" * 70)
    print("PARAMETER SERVER — Configuración")
    print("=" * 70)
    print(f"  Host            : {args.host}:{args.port}")
    print(f"  Workers         : {args.workers}")
    print(f"  Épocas          : {args.epochs}")
    print(f"  Arquitectura    : {INPUT_SIZE} → {args.hidden} → {OUTPUT_SIZE}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Ejemplos train  : {args.n_train}")
    print(f"  Semilla         : {args.seed if args.seed is not None else 'aleatoria'}")
    print("=" * 70)

    initial_params = _init_params(INPUT_SIZE, args.hidden, OUTPUT_SIZE, args.seed)

    server = ParameterServer(
        host=args.host,
        port=args.port,
        num_workers=args.workers,
        initial_params=initial_params,
        learning_rate=args.lr,
        n_train=args.n_train,
        on_epoch_end=_on_epoch_end,
    )

    history = server.run(epochs=args.epochs)

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN DE ENTRENAMIENTO")
    print("=" * 70)
    print(f"  Precisión final  : {history['accuracies'][-1]:.2f}%")
    print(f"  Mejor precisión  : {max(history['accuracies']):.2f}%")
    print(f"  Pérdida final       : {history['losses'][-1]:.4f}")
    print("\n  Evolución por época:")
    for i, (acc, loss) in enumerate(zip(history["accuracies"], history["losses"]), 1):
        bar = "█" * int(acc / 5)
        print(f"    Época {i:3d}: {acc:5.2f}%  loss={loss:.4f}  {bar}")


if __name__ == "__main__":
    main()
