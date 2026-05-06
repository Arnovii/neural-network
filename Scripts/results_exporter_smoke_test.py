from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.results_exporter import ResultsExporter  # noqa: E402


def simulate_step(step: int, total_steps: int) -> dict[str, Any]:
    """Simula un paso de entrenamiento con metricas realistas.

    Genera valores de loss, accuracy, staleness y alpha que simulan
    el comportamiento de un entrenamiento distribuido con convergencia.

    :param step: Numero de paso actual (1-indexed).
    :type step: int

    :param total_steps: Numero total de pasos simulados.
    :type total_steps: int

    :returns: Diccionario con metricas simuladas del paso.
    :rtype: dict[str, Any]
    """
    progress = step / max(total_steps, 1)
    wave = math.sin(step / 37.0) * 0.08 + math.cos(step / 91.0) * 0.04
    trend = max(0.0, 2.7 - 1.9 * progress)
    loss = max(0.03, trend + wave + random.uniform(-0.03, 0.03))  # noqa: S311 (simulation data, not cryptographic)
    accuracy = min(100.0, max(0.0, 4.0 + 94.0 * progress + math.sin(step / 53.0) * 1.5))
    workers = 1 + (step // 250) % 4
    elapsed = step * 0.42 + math.sin(step / 17.0) * 0.15
    loss_std = 0.08 + 0.03 * (1.0 - progress) + abs(math.sin(step / 23.0)) * 0.015
    acc_std = 0.35 + 0.10 * (1.0 - progress) + abs(math.cos(step / 29.0)) * 0.04
    staleness = step % 5
    alpha = 1.0 / (1.0 + 0.1 * staleness)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "workers": workers,
        "elapsed": elapsed,
        "loss_std": loss_std,
        "acc_std": acc_std,
        "staleness": staleness,
        "alpha": alpha,
    }


def run_smoke_test(total_steps: int, export_dir: Path, seed: int) -> Path:
    """Ejecuta un smoke test completo del ResultsExporter.

    Simula ``total_steps`` de entrenamiento, genera todas las graficas
    y verifica que los archivos de exportacion se creen correctamente.

    :param total_steps: Numero de pasos simulados (minimo 2000).
    :type total_steps: int

    :param export_dir: Directorio base para exportar resultados.
    :type export_dir: Path

    :param seed: Semilla aleatoria para reproducibilidad.
    :type seed: int

    :returns: Path al directorio de sesion creado.
    :rtype: Path
    :raises RuntimeError: Si faltan archivos de exportacion o las metricas son incorrectas.
    """
    random.seed(seed)

    exporter = ResultsExporter(
        config={
            "lr": 0.001,
            "lr_cnn": 0.001,
            "staleness_lambda": 0.1,
            "batch_size": 64,
            "image_size": 224,
            "seed": seed,
            "cnn_arch": "simple",
            "description": "Smoke test for full-history export",
            "metrics_window": 256,
        },
        export_dir=str(export_dir),
        metrics_window=256,
    )

    exporter.record_log("Smoke test started")
    for step in range(1, total_steps + 1):
        values = simulate_step(step, total_steps)
        exporter.record_metric(
            step=step,
            loss=values["loss"],
            accuracy=values["accuracy"],
            num_workers=values["workers"],
            elapsed=values["elapsed"],
            loss_std=values["loss_std"],
            acc_std=values["acc_std"],
            staleness=values["staleness"],
            alpha=values["alpha"],
        )

        if step % 500 == 0:
            exporter.record_worker_event(
                step, "connected", step // 500, f"127.0.0.{(step // 500) + 1}"
            )
        if step % 750 == 0:
            exporter.record_worker_event(
                step, "disconnected", step // 750, f"127.0.0.{(step // 750) + 1}"
            )

    exporter.tcp_request_count = total_steps * 3
    exporter.nan_rejected_count = 0
    exporter.record_log("Smoke test finished")

    session_dir = exporter.finalize()

    expected_files = [
        "config.json",
        "metrics.csv",
        "worker_events.csv",
        "ps_logs.txt",
        "metadata.json",
        "plot_3panels.png",
        "plot_loss.png",
        "plot_accuracy.png",
        "plot_workers.png",
        "plot_band_loss.png",
        "plot_band_acc.png",
        "plot_staleness.png",
        "plot_std.png",
    ]
    missing = [name for name in expected_files if not (session_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing exported files: {missing}")

    metrics_path = session_dir / "metrics.csv"
    metrics_lines = metrics_path.read_text(encoding="utf-8").splitlines()
    if len(metrics_lines) != total_steps + 1:
        raise RuntimeError(
            f"metrics.csv line count mismatch: expected {total_steps + 1}, got {len(metrics_lines)}"
        )

    first_data = metrics_lines[1].split(",")
    last_data = metrics_lines[-1].split(",")
    if first_data[0] != "1":
        raise RuntimeError(f"metrics.csv does not start at step 1: got {first_data[0]}")
    if last_data[0] != str(total_steps):
        raise RuntimeError(
            f"metrics.csv does not reach the final step: got {last_data[0]}, expected {total_steps}"
        )

    print(f"Smoke test OK: {session_dir}")
    print(f"Verified {total_steps} simulated steps from step 1 to step {total_steps}.")
    return session_dir


def main() -> int:
    """Punto de entrada CLI para ejecutar el smoke test.

    Parsea parámetros de línea de comandos y delega en
    ``run_smoke_test`` con los parametros especificados.

    :returns: Codigo de salida (0 para exito).
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Run a full-history smoke test for Utils/results_exporter.py."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of simulated steps (minimum 2000).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=REPO_ROOT / "Exports" / "smoke_tests",
        help="Base export directory where the timestamped session folder will be created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for deterministic simulation.",
    )
    args = parser.parse_args()

    total_steps = max(args.steps, 2000)
    run_smoke_test(total_steps=total_steps, export_dir=args.export_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
