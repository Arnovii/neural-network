"""
ps_imagenet.py

Parameter Server asíncrono para ImageNet — punto de entrada terminal.

USO:
    python ps_imagenet.py [opciones]

OPCIONES:
    --host              IP de escucha                          (default: 0.0.0.0)
    --port              Puerto TCP                             (default: 9999)
    --lr                Learning rate para SGD local           (default: 0.001)
    --staleness-lambda  Factor de corrección de staleness      (default: 0.1)
    --hidden1           Neuronas capa oculta 1 del MLP         (default: 1024)
    --hidden2           Neuronas capa oculta 2 del MLP         (default: 512)
    --batch-size        Batch size (enviado a todos Workers)   (default: 64)
    --image-size        Resolución imágenes (enviado a Workers)(default: 224)
    --cnn-arch          resnet18 | simple                      (default: resnet18)
    --seed              Semilla RNG (default: None = aleatorio)
    --steps-per-report  Steps entre reportes de métricas       (default: 500)
    --max-steps         Detener tras N steps (0 = indefinido)  (default: 0)
    --metrics-window    Tamaño ventana deslizante de métricas  (default: 200)
    --export-dir        Directorio para exportar resultados    (default: ./Exports)
    --hf-token          Token HuggingFace (o usar HF_TOKEN env)

NOTA: El número de workers es dinámico. Los workers se conectan y desconectan
      libremente. El PS asigna ranks automáticamente para garantizar sharding sin solapamientos.

EJEMPLO — Entrenamiento con workers dinámicos:
    Terminal 1: python ps_imagenet.py --max-steps 50000
    Terminal 2: python worker_imagenet.py &
    Terminal 3: python worker_imagenet.py --device cuda:1 &
    Terminal 4: python worker_imagenet.py (en otra máquina) &

EJEMPLO — ResNet-18 + GPU Workers:
    python ps_imagenet.py --lr 0.001 --staleness-lambda 0.05 --max-steps 100000

EJEMPLO — Exportar a directorio personalizado:
    python ps_imagenet.py --export-dir ./results_exp1 --max-steps 10000
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Distributed.parameter_server import ParameterServer
from Model.cnn_extractor import CNNExtractor
from Model.mlp_pytorch import MLPPyTorch
from Utils.constants import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_LR,
    DEFAULT_STALENESS_LAMBDA,
    HIDDEN1_DEFAULT,
    HIDDEN2_DEFAULT,
    DEFAULT_BATCH_SIZE,
    IMAGE_SIZE,
    STEPS_PER_REPORT_DEFAULT,
    METRICS_WINDOW_DEFAULT,
    DEFAULT_SEED,
    EXPORT_DIR_DEFAULT,
    MAX_STEPS_UNLIMITED,
    NUM_CLASSES,
)


def main() -> None:
    """
    Punto de entrada para el Parameter Server en modo terminal.

    Orquesta el flujo completo del servidor PS asíncrono:

    1. **Parsing de argumentos CLI**: Lee parámetros de línea de comandos
       (host, puerto, learning rates, arquitectura CNN, seed, etc.)
    2. **Resolución de HF Token**: Prioridad: argumento CLI > variable HF_TOKEN
    3. **Instanciación de modelos**:
       - CNNExtractor (resnet18 preentrenado o simple CNN)
       - MLPPyTorch (feature_dim → hidden1 → hidden2 → 1000 clases)
    4. **Creación de ParameterServer**:
       - Establece callbacks para eventos (step, report, connected, disconnected)
       - Configura hiperparámetros Async-SGD (lr, staleness_lambda, METRICS_WINDOW_DEFAULT)
    5. **Apertura de servidor TCP**: ps.listen() inicia socket de escucha
    6. **Inicio de entrenamiento**: Loop sin esperar número específico de workers
    7. **Loop de entrenamiento**:
       - Ejecuta indefinidamente o hasta max_steps
       - Registra métricas cada 50 steps (throughput en steps/sec)
       - Reporta visualmente cada --steps-per-report steps
       - Acepta workers dinámicamente (conectar/desconectar en cualquier momento)
    8. **Cleanup**: ps.stop() cierra conexiones y libera recursos
    9. **Salida**: Imprime resumen de entrenamiento (steps totales, loss final, acc final)

    Callbacks internos:
    - on_connected(): Incrementa contador de workers, registra en terminal
    - on_disconnected(): Registra desconexión en terminal
    - on_step(): Muestreo cada 50 steps, cálculo de throughput, chequeo de max_steps
    - on_report(): Resporte visual cada --steps-per-report steps

    Exceptions:
    - KeyboardInterrupt (Ctrl+C): Detiene entrenamiento gracefully y ejecuta cleanup
    - Otras excepciones en ParameterServer se propagan y terminan el proceso

    :returns: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Parameter Server asíncrono — ImageNet-1k"
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--staleness-lambda", type=float, default=DEFAULT_STALENESS_LAMBDA
    )
    parser.add_argument("--hidden1", type=int, default=HIDDEN1_DEFAULT)
    parser.add_argument("--hidden2", type=int, default=HIDDEN2_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--cnn-arch", type=str, default="resnet18", choices=["resnet18", "simple"]
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--steps-per-report", type=int, default=STEPS_PER_REPORT_DEFAULT
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_UNLIMITED)
    parser.add_argument("--metrics-window", type=int, default=METRICS_WINDOW_DEFAULT)
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Token HuggingFace (alternativa: variable HF_TOKEN)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=EXPORT_DIR_DEFAULT,
        help="Directorio para exportar resultados",
    )
    args = parser.parse_args()

    # HF token: argumento CLI tiene prioridad sobre variable de entorno
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print("=" * 68)
    print("PARAMETER SERVER ASÍNCRONO — ImageNet-1k")
    print("=" * 68)
    print(f"  Host              : {args.host}:{args.port}")
    print(f"  CNN               : {args.cnn_arch}")
    print(
        f"  MLP               : feature_dim → {args.hidden1} → {args.hidden2} → {NUM_CLASSES}"
    )
    print(f"  Batch size        : {args.batch_size}  (enviado a Workers)")
    print(f"  Image size        : {args.image_size}  (enviado a Workers)")
    print(
        f"  Semilla           : {args.seed if args.seed is not None else 'aleatorio'}"
    )
    print(f"  LR                : {args.lr}")
    print(f"  Staleness λ       : {args.staleness_lambda}")
    print(f"  Steps/reporte     : {args.STEPS_PER_REPORT_DEFAULT}")
    print(f"  Max steps         : {args.max_steps or '∞'}")
    print(f"  Export dir        : {args.export_dir}")
    print(
        f"  HF Token          : {'✓ configurado' if hf_token else '✗ no configurado'}"
    )
    print("=" * 68)
    print("  ℹ Esperando workers dinámicamente (sin límite)...\n")

    # ── Evento de conexión (dinámico) ──
    connected = [0]
    stop = threading.Event()

    def on_connected(wid: int, addr: str) -> None:
        """
        Callback de PS: se ejecuta cuando un Worker se conecta exitosamente.

        Acciones:
        1. Incrementa contador de workers conectados
        2. Imprime mensaje visual con ID del worker y dirección de red
        3. No tiene "barrera" de espera — el entrenamiento continúa dinámicamente

        Este callback se registra en ParameterServer.on_worker_connected.

        :param wid: ID único del Worker asignado por el PS (= rank para sharding).
        :type wid: int

        :param addr: Dirección de red del Worker (formato "IP:puerto").
        :type addr: str

        :returns: None
        :rtype: None
        """
        connected[0] += 1
        print(f"  [+] Worker {wid} conectado desde {addr} (total: {connected[0]})")

    def on_disconnected(wid: int) -> None:
        """
        Callback de PS: se ejecuta cuando un Worker se desconecta.

        Simplemente registra en terminal que el Worker se desconectó.
        No modifica el comportamiento del entrenamiento (PS sigue funcionando
        con los workers restantes en modo asíncrono).

        Este callback se registra en ParameterServer.on_worker_disconnected.

        :param wid: ID único del Worker desconectado.
        :type wid: int

        :returns: None
        :rtype: None
        """
        print(f"  [-] Worker {wid} desconectado.")

    def on_step(
        step: int, loss: float, acc: float, staleness: int, elapsed: float
    ) -> None:
        """
        Callback de PS: se ejecuta después de cada step de entrenamiento.

        Realiza muestreo frecuente de métricas:
        1. Cada 50 steps:
           - Imprime: step, loss, accuracy, staleness, elapsed
        2. Si max_steps > 0: chequea si se alcanzó límite y establece stop event

        Este callback se registra en ParameterServer.on_step.

        :param step: Número del step global de entrenamiento.
        :type step: int

        :param loss: Valor de pérdida (loss) en el batch actual.
        :type loss: float

        :param acc: Precisión en porcentaje (0-100) en el batch actual.
        :type acc: float

        :param staleness: Máximo número de versiones de atrazo observadas.
        :type staleness: int

        :param elapsed: Tiempo elapsed en segundos desde inicio.
        :type elapsed: float

        :returns: None
        :rtype: None
        """
        # Debug: mostrar el primer step
        if step == 1:
            print(f"\n✓ PRIMER STEP RECIBIDO (step={step})\n")

        if step % 50 == 0:
            print(
                f"  step={step:6,d} | loss={loss:.4f} | acc={acc:.2f}% | "
                f"staleness={staleness} | {elapsed:.0f}s"
            )
        if args.max_steps > 0 and step >= args.max_steps:
            stop.set()

    def on_report(step: int, loss: float, acc: float, elapsed: float) -> None:
        """
        Callback de PS: se ejecuta cada --steps-per-report steps.

        Imprime un reporte visual formateado con métricas agregadas en ventana deslizante.
        Proporciona feedback visual periódico del progreso del entrenamiento.

        Formato:
        ──────────────────────────────────────────────────────────────────
          Reporte | step=XXXX | loss=X.XXXX | acc=XX.XX% | elapsed=XXXs
        ──────────────────────────────────────────────────────────────────

        Tiempo medido desde que se envió START al primer worker (punto de inicio real del entrenamiento).
        Incluye latencia de red, pero excluye inicialización del servidor.

        Este callback se registra en ParameterServer.on_report.

        :param step: Número del step global en este reporte.
        :type step: int

        :param loss: Pérdida promedio en ventana de métricas.
        :type loss: float

        :param acc: Precisión promedio en ventana de métricas (%).
        :type acc: float

        :param elapsed: Tiempo elapsed en segundos desde envío de START al primer worker.
        :type elapsed: float

        :returns: None
        :rtype: None
        """
        print(f"\n{'─' * 60}")
        print(
            f"  Reporte | step={step:,} | loss={loss:.4f} | acc={acc:.2f}% | {elapsed:.0f}s"
        )
        print(f"{'─' * 60}\n")

    # ── Crear PS ──
    ps = ParameterServer(
        host=args.host,
        port=args.port,
        learning_rate=args.lr,
        staleness_lambda=args.staleness_lambda,
        steps_per_report=args.steps_per_report,
        metrics_window=args.metrics_window,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        hf_token=hf_token,
        export_dir=args.export_dir,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        on_step=on_step,
        on_report=on_report,
        on_worker_connected=on_connected,
        on_worker_disconnected=on_disconnected,
    )

    # ── Cargar CNN ──
    print(f"\nCargando CNN {args.cnn_arch}...")
    cnn = CNNExtractor(
        arch=args.cnn_arch,
        device="cpu",
        seed=args.seed,
    )
    ps.set_cnn(cnn)

    # ── Inicializar MLP ──
    mlp = MLPPyTorch(
        feature_dim=cnn.feature_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        n_classes=NUM_CLASSES,
    )
    ps.set_mlp(mlp.state_dict_numpy())
    print(f"Modelo listo: CNN={args.cnn_arch} | feature_dim={cnn.feature_dim}\n")

    # ── Abrir servidor ──
    ps.listen()
    print(f"Servidor escuchando. Esperando workers...\n")

    try:
        # Esperar un poco para que se conecte al menos 1 worker, pero no es bloqueante
        time.sleep(2)  # Dar tiempo para que workers se conecten
    except KeyboardInterrupt:
        ps.stop()
        return

    print(f"✓ Entrenamiento asíncrono activo.")
    print("  (Ctrl+C para detener)\n")

    try:
        if args.max_steps > 0:
            stop.wait()
            print(f"\nAlcanzado max_steps={args.max_steps}. Deteniendo...")
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrumpido.")

    # ── Resumen ──
    h = ps.history
    print("\n" + "=" * 68)
    print("RESUMEN FINAL")
    print("=" * 68)
    print(f"  Steps totales : {ps.current_version:,}")
    if h["losses"]:
        print(f"  Loss final    : {h['losses'][-1]:.4f}")
        print(f"  Acc final     : {h['accuracies'][-1]:.2f}%")
    ps.stop()


if __name__ == "__main__":
    main()
