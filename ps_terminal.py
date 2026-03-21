"""
ps_terminal.py

Punto de entrada del Parameter Server.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
    python ps_terminal.py [opciones]

Opciones:
    --host                  IP en la que escucha el servidor     (default: 0.0.0.0)
    --port                  Puerto TCP                           (default: 9999)
    --workers               Número de Workers a esperar          (default: 2)
    --epochs                Épocas de entrenamiento              (default: 10)
    --hidden1               Neuronas en la primera capa oculta   (default: 256)
    --hidden2               Neuronas en la segunda capa oculta   (default: 128)
    --lr                    Tasa de aprendizaje                  (default: 0.01)
    --n-train               Total de ejemplos de entrenamiento   (default: 50000)
    --cnn-arch              Arquitectura CNN: simple o resnet18  (default: simple)
    --cnn-device            Dispositivo PyTorch: cpu, cuda, mps  (default: cpu)
    --cnn-pretrain-samples  Imágenes para preentrenar CNN        (default: 10000)
    --seed                  Semilla aleatoria                    (default: ninguna)
    --momentum              Momentum SGD para MLP                (default: 0.9)

Ejemplo — servidor esperando 3 workers, 20 épocas, 5000 muestras para CNN:
    python ps_terminal.py --workers 3 --epochs 20 --cnn-pretrain-samples 5000

──────────────────────────────────────────────────────────────────
ARQUITECTURA
──────────────────────────────────────────────────────────────────
El PS tiene tres fases:

    1. listen()  → Abre el socket y acepta Workers en un hilo de
                   fondo. Retorna inmediatamente.

    2. Espera    → El script bloquea hasta que se conectan
                   exactamente --workers Workers.

    3. train()   → Ejecuta el loop de entrenamiento distribuido:
                       Por cada época:
                       a. Genera una semilla aleatoria de época.
                       b. Broadcast: params + semilla a cada Worker.
                       c. Cada Worker reconstruye su chunk localmente.
                       d. Esperar gradientes de TODOS los Workers.
                       e. Promediar:  ∇θ = (1/N) * Σ ∇θL(Bᵢ)
                       f. Actualizar: θ ← θ − lr * ∇θ

    4. shutdown() → Envía STOP a los Workers y cierra el servidor.

Los datos de entrenamiento (imágenes) nunca salen de cada Worker.
Al finalizar imprime el historial de precisión y pérdida por época
y exporta los resultados a ``Exports/`` vía ``Utils/results_exporter``.
"""

import argparse
import os
import sys
import threading
import time

# Asegura que los módulos del proyecto sean importables
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Distributed.parameter_server import ParameterServer
from Model.cnn_extractor import CNNExtractor
from Model.mlp import init_params
from Utils.imagenet_loader import NUM_CLASSES, get_imagenet_dataloader
from Utils.results_exporter import export_results


# ================================================================
# CALLBACK DE PROGRESO
# ================================================================


def _on_epoch_end(
    epoch: int,
    total: int,
    train_acc: float,
    train_loss: float,
    test_acc: float | None,
    test_loss: float | None,
) -> None:
    """
    Imprime en consola un resumen del estado del entrenamiento al finalizar una época.

    Muestra una barra de progreso basada en la precisión de entrenamiento, junto con
    las métricas principales de la época actual. Si se proporcionan métricas del
    conjunto de prueba, también se incluyen en la salida.

    :param epoch: Número de la época actual.
    :type epoch: int

    :param total: Número total de épocas del entrenamiento.
    :type total: int

    :param train_acc: Precisión del modelo en el conjunto de entrenamiento (porcentaje).
    :type train_acc: float

    :param train_loss: Valor de la función de pérdida en entrenamiento.
    :type train_loss: float

    :param test_acc: Precisión en el conjunto de prueba en porcentaje. Si es ``None``,
                     no se muestra en la salida.
    :type test_acc: float | None

    :param test_loss: Valor de la función de pérdida en el conjunto de prueba. Solo se
                      utiliza cuando ``test_acc`` no es ``None``.
    :type test_loss: float | None

    :return: No retorna ningún valor; solo imprime información en consola.
    :rtype: None
    """
    bar = "█" * int(train_acc / 5)
    test_str = (
        f"  | precisión_prueba={test_acc:.2f}%  pérdida_prueba={test_loss:.4f}"
        if test_acc is not None
        else ""
    )
    print(
        f"  [{bar:<20}] {train_acc:5.2f}%  pérdida={train_loss:.4f}{test_str}  ({epoch}/{total})"
    )


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
        "--hidden1",
        type=int,
        default=1024,
        help="Neuronas en la primera capa oculta (default: 1024 para ImageNet)",
    )
    parser.add_argument(
        "--hidden2",
        type=int,
        default=512,
        help="Neuronas en la segunda capa oculta (default: 512 para ImageNet)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Tasa de aprendizaje (default: 0.01)"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=50_000,
        help="Total de ejemplos de entrenamiento (default: 50000, máx ImageNet train)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla aleatoria (default: ninguna)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum SGD para el MLP (0.0=SGD puro, 0.9=default)",
    )
    parser.add_argument(
        "--cnn-arch",
        type=str,
        default="resnet18",
        choices=["resnet18"],
        help="Arquitectura CNN: simple (preentrenada local) | resnet18 (pesos ImageNet, default: simple)",
    )
    parser.add_argument(
        "--cnn-device",
        type=str,
        default="cpu",
        help="Dispositivo PyTorch para la CNN: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--cnn-pretrain-samples",
        type=int,
        default=10000,
        help="Número de imágenes para preentrenar la CNN simple (default: 10000)",
    )
    args = parser.parse_args()

    # resnet18 siempre usa pesos ImageNet — es la única configuración útil.
    cnn_pretrained = args.cnn_arch == "resnet18"

    OUTPUT_SIZE = NUM_CLASSES

    print("=" * 70)
    print("PARAMETER SERVER — Configuración (ImageNet)")
    print("=" * 70)
    print(f"  Host            : {args.host}:{args.port}")
    print(f"  Workers         : {args.workers}")
    print(f"  Épocas          : {args.epochs}")
    print(
        f"  CNN arch        : {args.cnn_arch}"
        + (" (pesos ImageNet)" if cnn_pretrained else " (preentrenada localmente)")
    )
    print(
        f"  MLP arquitectura: features → {args.hidden1} → {args.hidden2} → {OUTPUT_SIZE}"
    )
    print(f"  Learning rate   : {args.lr}")
    print(f"  Ejemplos train  : {args.n_train}")
    print(f"  Semilla         : {args.seed if args.seed is not None else 'aleatoria'}")
    print("=" * 70)

    # Espera hasta que se conecten los N Workers requeridos
    ready_event = threading.Event()
    connected_count = [0]

    def _on_worker_connected(worker_id: int, addr: str) -> None:
        connected_count[0] += 1
        print(
            f"  [+] Worker {worker_id} conectado desde {addr} "
            f"({connected_count[0]}/{args.workers})"
        )
        if connected_count[0] >= args.workers:
            ready_event.set()

    def _on_worker_disconnected(worker_id: int) -> None:
        print(f"  [-] Worker {worker_id} desconectado inesperadamente.")

    def _on_gradients_received(
        worker_id: int, epoch: int, loss: float, accuracy: float
    ) -> None:
        print(
            f"  [↓] Gradientes de Worker {worker_id}  "
            f"precisión={accuracy:.2f}%  pérdida={loss:.4f}"
        )

    # Creación del servidor
    server = ParameterServer(
        host=args.host,
        port=args.port,
        on_worker_connected=_on_worker_connected,
        on_worker_disconnected=_on_worker_disconnected,
        on_gradients_received=_on_gradients_received,
        on_epoch_end=_on_epoch_end,
    )

    server.listen()

    print(f"\n  Esperando {args.workers} worker(s)...\n")
    ready_event.wait()
    print()

    # Construye el extractor CNN con la misma semilla que usarán los Workers,
    # garantizando que todos partan de los mismos pesos convolucionales.
    print("\nConstruyendo extractor CNN...")
    # La CNN siempre usa seed=42 — independiente de la semilla MLP.
    # Mezclarlas haría que --seed invalide la caché CNN.
    cnn = CNNExtractor(
        arch=args.cnn_arch,
        pretrained=cnn_pretrained,
        device=args.cnn_device,
        seed=42,
    )
    feature_dim = cnn.feature_dim
    print(f"CNN lista — arch={args.cnn_arch}  feature_dim={feature_dim}\n")

    # Y_test: solo etiquetas. Los features de prueba los extrae el Worker
    # y los envía al PS con REQUEST_TEST_FEATURES tras la barrera CNN_READY.
    # ImageNet test: los features los extraerá el Worker vía REQUEST_TEST_FEATURES
    # El PS no carga las imágenes de test — son 50k × 224×224, demasiado para RAM
    Y_test = None
    server.set_cnn(cnn)

    initial_params = init_params(
        feature_dim, args.hidden1, args.hidden2, OUTPUT_SIZE, args.seed
    )

    t_start = time.perf_counter()

    history = server.train(
        epochs=args.epochs,
        initial_params=initial_params,
        learning_rate=args.lr,
        n_train=args.n_train,
        Y_test=Y_test,  # etiquetas de prueba (50k int32)
        momentum=args.momentum,
        seed=args.seed,
    )

    elapsed = time.perf_counter() - t_start

    server.shutdown()

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN DE ENTRENAMIENTO")
    print("=" * 70)
    print(f"  Precisión final de entrenamiento  : {history['accuracies'][-1]:.2f}%")
    print(f"  Mejor precisión de entrenamiento  : {max(history['accuracies']):.2f}%")
    print(f"  Pérdida final de entrenamiento    : {history['losses'][-1]:.4f}")
    if history["test_accuracies"]:
        print(f"  Precisión final de prueba   : {history['test_accuracies'][-1]:.2f}%")
        print(f"  Mejor precisión de prueba   : {max(history['test_accuracies']):.2f}%")
        print(f"  Pérdida final de prueba     : {history['test_losses'][-1]:.4f}")

    minutes, seconds = divmod(elapsed, 60)
    print(
        f"\n  Tiempo de ejecución         : {int(minutes)}m {seconds:.2f}s ({elapsed:.2f}s)"
    )

    print("\n  Evolución por época:")
    has_test = bool(history["test_accuracies"])
    for i, (acc, loss) in enumerate(zip(history["accuracies"], history["losses"]), 1):
        bar = "█" * int(acc / 5)
        test_str = ""
        if has_test:
            t_acc = history["test_accuracies"][i - 1]
            t_loss = history["test_losses"][i - 1]
            test_str = f"  precisión_prueba={t_acc:.2f}%  pérdida_prueba={t_loss:.4f}"
        print(
            f"    Época {i:3d}: precisión={acc:5.2f}%  pérdida={loss:.4f}{test_str}  {bar}"
        )

    # Exportar resultados
    config = {
        "epochs": args.epochs,
        "cnn_arch": args.cnn_arch,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "n_train": args.n_train,
        "workers": args.workers,
        "seed": args.seed,
    }
    json_path = export_results(history, config, elapsed)
    print(f"\n  Resultados exportados a: {json_path}")


if __name__ == "__main__":
    main()
