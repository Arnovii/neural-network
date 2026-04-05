"""
ps_imagenet.py

Parameter Server asíncrono para ImageNet — punto de entrada terminal.

USO:
    python ps_imagenet.py [opciones]

OPCIONES:
    --host              IP de escucha                          (default: 0.0.0.0)
    --port              Puerto TCP                             (default: 9999)
    --wait-workers      Workers a esperar antes de empezar     (default: 1)
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
    --hf-token          Token HuggingFace (o usar HF_TOKEN env)

EJEMPLO — 2 Workers, parar tras 50k steps:
    python ps_imagenet.py --wait-workers 2 --max-steps 50000

EJEMPLO — ResNet-18 + GPU Workers:
    python ps_imagenet.py --wait-workers 3 --lr 0.001 --staleness-lambda 0.05
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


def main() -> None:
    """
    Punto de entrada para el Parameter Server en modo terminal.

    Analiza configuración de línea de comandos, instancia modelos CNN+MLP,
    crea ParameterServer, espera a que Workers se conecten, y ejecuta
    el loop de entrenamiento distribuido hasta max_steps o interrupción por teclado.

    Registra métricas periódicamente y muestra throughput (steps/sec).
    Los resultados se exportan a JSON al completar o parar el servidor.

    :returns: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Parameter Server asíncrono — ImageNet-1k"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--wait-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--staleness-lambda", type=float, default=0.1)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--cnn-arch", type=str, default="resnet18", choices=["resnet18", "simple"]
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps-per-report", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--metrics-window", type=int, default=200)
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Token HuggingFace (alternativa: variable HF_TOKEN)",
    )
    args = parser.parse_args()

    # HF token: argumento CLI tiene prioridad sobre variable de entorno
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print("=" * 68)
    print("PARAMETER SERVER ASÍNCRONO — ImageNet-1k")
    print("=" * 68)
    print(f"  Host              : {args.host}:{args.port}")
    print(f"  Esperando Workers : {args.wait_workers}")
    print(f"  CNN               : {args.cnn_arch}")
    print(f"  MLP               : feature_dim → {args.hidden1} → {args.hidden2} → 1000")
    print(f"  Batch size        : {args.batch_size}  (enviado a Workers)")
    print(f"  Image size        : {args.image_size}  (enviado a Workers)")
    print(
        f"  Semilla           : {args.seed if args.seed is not None else 'aleatorio'}"
    )
    print(f"  LR                : {args.lr}")
    print(f"  Staleness λ       : {args.staleness_lambda}")
    print(f"  Steps/reporte     : {args.steps_per_report}")
    print(f"  Max steps         : {args.max_steps or '∞'}")
    print(
        f"  HF Token          : {'✓ configurado' if hf_token else '✗ no configurado'}"
    )
    print("=" * 68)

    # ── Evento de conexión ──
    ready = threading.Event()
    connected = [0]
    stop = threading.Event()

    def on_connected(wid, addr):
        connected[0] += 1
        print(f"  [+] Worker {wid} desde {addr} ({connected[0]}/{args.wait_workers})")
        if connected[0] >= args.wait_workers:
            ready.set()

    def on_disconnected(wid):
        print(f"  [-] Worker {wid} desconectado.")

    step_ts: list = []

    def on_step(step, loss, acc, staleness):
        step_ts.append(time.perf_counter())
        if step % 50 == 0:
            sps = (
                len(step_ts) / (step_ts[-1] - step_ts[0]) if len(step_ts) >= 2 else 0.0
            )
            print(
                f"  step={step:6,d} | loss={loss:.4f} | acc={acc:.2f}% | "
                f"staleness={staleness} | {sps:.1f} steps/s"
            )
        if args.max_steps > 0 and step >= args.max_steps:
            stop.set()

    def on_report(step, loss, acc):
        print(f"\n{'─' * 60}")
        print(f"  Reporte | step={step:,} | loss={loss:.4f} | acc={acc:.2f}%")
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
        on_step=on_step,
        on_report=on_report,
        on_worker_connected=on_connected,
        on_worker_disconnected=on_disconnected,
    )

    # ── Cargar CNN ──
    print(f"\nCargando CNN {args.cnn_arch}...")
    cnn = CNNExtractor(
        arch=args.cnn_arch,
        pretrained=(args.cnn_arch == "resnet18"),
        device="cpu",
        seed=args.seed,
    )
    ps.set_cnn(cnn)

    # ── Inicializar MLP ──
    mlp = MLPPyTorch(
        feature_dim=cnn.feature_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        n_classes=1000,
    )
    ps.set_mlp(mlp.state_dict_numpy())
    print(f"Modelo listo: CNN={args.cnn_arch} | feature_dim={cnn.feature_dim}\n")

    # ── Abrir servidor ──
    ps.listen()
    print(f"Esperando {args.wait_workers} Worker(s)...\n")

    try:
        ready.wait()
    except KeyboardInterrupt:
        ps.stop()
        return

    print(f"\n✓ {connected[0]} Worker(s) conectados. Entrenamiento asíncrono activo.")
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
