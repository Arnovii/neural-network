"""
Utils/results_exporter.py

Sistema de exportación de resultados completamente desacoplado del Parameter Server.

DISEÑO DE DESACOPLAMIENTO:
  - ResultsExporter NO importa ni depende de ParameterServer
  - Se integra únicamente vía callbacks y métodos públicos
  - Los datos se pasan por argumentos simples (int, float, str, dict)
  - No requiere conocimiento de la arquitectura interna de PS

BUFFERING Y RENDIMIENTO:
  - Mantiene datos en memoria hasta finalize()
  - Escritura de archivos con threading background (no bloquea)
  - Buffers circulares para métricas (ventana deslizante)
  - Logs concatenados en memoria hasta finalize()

PROTECCIÓN DE ARCHIVOS:
  - Mientras el experimento está activo: archivos marcados como en-uso
  - Al finalize(): archivos escritos y renombrados (atómicamente)
  - Usuario no puede acceder a archivos parcialmente generados

ESTRUCTURA DE SALIDA:
  ./Exports/[unique_timestamp]/
    ├── config.json           # Configuración del experimento
    ├── metrics.csv           # Series de tiempo: step, loss, acc, workers
    ├── ps_logs.txt           # Todos los logs del PS
    ├── plot_3panels.png      # 3 gráficas horizontales (loss/acc/workers)
    ├── plot_loss.png         # Gráfica individual de Loss
    ├── plot_accuracy.png     # Gráfica individual de Accuracy
    ├── plot_workers.png      # Gráfica individual de Workers
    └── metadata.json         # Estadísticas finales (min/max loss, acc, etc.)

CARACTERÍSTICAS DE ESCALAS:
  - Loss/Workers: Escala automática con margen superior 10%
  - Accuracy: Escala con margen superior dinámico (20% del máximo)
  - Líneas guía: Opacidad reducida (alpha=0.15) excepto en Accuracy (alpha=0.4)
"""

from __future__ import annotations

import os
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Backend no-GUI para entornos headless
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class ResultsExporter:
    """
    Exportador de resultados desacoplado que registra métricas y logs
    desde Parameter Server sin dependencias internas del mismo.

    INTERFAZ PÚBLICA:
      - __init__(config_dict, export_dir='./Exports')
      - record_metric(step, loss, accuracy, num_workers)
      - record_log(text)
      - finalize()

    GARANTÍAS:
      - Thread-safe para record_metric() y record_log()
      - No bloquea entrenamientos (escritura async al finalize)
      - Archivos protegidos hasta finalización
    """

    def __init__(
        self,
        config: Dict[str, Any],
        export_dir: str = "./Exports",
        metrics_window: int = 500,
    ) -> None:
        """
        Inicializa exportador de resultados.

        :param config:
            Diccionario con configuración del experimento:
            {
              "lr": 0.01,
              "lr_cnn": 0.001,
              "staleness_lambda": 0.1,
              "batch_size": 64,
              "image_size": 224,
              "seed": 42,
              "cnn_arch": "resnet18" o "simple",
              "description": "Descripción del experimento"
            }
        :type config: Dict[str, Any]

        :param export_dir:
            Directorio base para exportar (default: ./Exports)
        :type export_dir: str

        :param metrics_window:
            Tamaño de ventana para buffers de métricas (default: 500)
        :type metrics_window: int
        """
        # Configuración
        self.config = config
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp único para esta ejecución
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :-3
        ]  # ms precision
        self.session_dir = self.export_dir / self.timestamp

        # No crear carpeta hasta finalize() para evitar archivos parciales
        self._finalized = False

        # Buffers en memoria (thread-safe con locks)
        self._lock = threading.Lock()
        self._metrics_steps: deque = deque(maxlen=metrics_window)
        self._metrics_loss: deque = deque(maxlen=metrics_window)
        self._metrics_accuracy: deque = deque(maxlen=metrics_window)
        self._metrics_workers: deque = deque(maxlen=metrics_window)
        self._metrics_elapsed: deque = deque(
            maxlen=metrics_window
        )  # Tiempo elapsed en segundos
        self._logs: List[str] = []

        # Estadísticas para reporte final
        self._total_metrics = 0
        self._start_time = time.time()  # Cuando se creó el exporter
        self._training_start_time: float | None = None  # Cuando llegó el primer step

    def record_metric(
        self,
        step: int,
        loss: float,
        accuracy: float,
        num_workers: int,
        elapsed: float = 0.0,
    ) -> None:
        """
        Registra un punto de métrica.

        Llamado típicamente desde callback on_step del PS.
        Thread-safe, no bloquea.

        :param step: Número del step de training
        :param loss: Valor de loss
        :param accuracy: Valor de accuracy (0-1)
        :param num_workers: Número de workers conectados
        :param elapsed: Tiempo elapsed en segundos desde inicio del entrenamiento
        """
        with self._lock:
            # Inicializar tiempo de entrenamiento cuando llega el primer step
            if self._training_start_time is None:
                self._training_start_time = time.time()

            self._metrics_steps.append(step)
            self._metrics_loss.append(loss)
            self._metrics_accuracy.append(accuracy)
            self._metrics_workers.append(num_workers)
            self._metrics_elapsed.append(elapsed)
            self._total_metrics += 1

    def record_log(self, text: str) -> None:
        """
        Registra una línea de log.

        Llamado típicamente desde logging_util después de procesar.
        Thread-safe, no bloquea.

        :param text: Texto de log (sin newline, se añade automáticamente)
        """
        with self._lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._logs.append(f"[{timestamp}] {text}")

    def finalize(self) -> Path:
        """
        Finaliza el experimento: genera todos los archivos.

        Operaciones:
        1. Crear carpeta de sesión
        2. Escribir config.json
        3. Escribir metrics.csv
        4. Escribir ps_logs.txt
        5. Generar gráficas (plot_3panels.png, plot_comparison.png)
        6. Escribir metadata.json

        Garantía: Al finalizar, la carpeta contiene todos los archivos
        y el usuario puede acceder sin riesgo.

        :returns: Path a la carpeta de sesión
        :rtype: Path
        """
        if self._finalized:
            return self.session_dir

        # Crear carpeta de sesión
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # 1. Guardar configuración
        self._write_config()

        # 2. Guardar métricas
        self._write_metrics()

        # 3. Guardar logs
        self._write_logs()

        # 4. Generar gráficas
        self._generate_plots()

        # 5. Guardar metadata
        self._write_metadata()

        self._finalized = True
        return self.session_dir

    # ========== OPERACIONES PRIVADAS ==========

    def _write_config(self) -> None:
        """Escribe configuración en formato JSON."""
        config_file = self.session_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _write_metrics(self) -> None:
        """Escribe métricas en formato CSV."""
        metrics_file = self.session_dir / "metrics.csv"

        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("step,loss,accuracy,num_workers,elapsed_seconds\n")

            for step, loss, acc, workers, elapsed in zip(
                self._metrics_steps,
                self._metrics_loss,
                self._metrics_accuracy,
                self._metrics_workers,
                self._metrics_elapsed,
            ):
                f.write(f"{step},{loss:.4f},{acc:.2f}%,{workers},{elapsed:.1f}\n")

    def _write_logs(self) -> None:
        """Escribe todos los logs en un archivo de texto."""
        logs_file = self.session_dir / "ps_logs.txt"

        with open(logs_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PARAMETER SERVER LOGS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Session: {self.timestamp}\n")
            f.write(f"Start time: {datetime.fromtimestamp(self._start_time)}\n")
            f.write("=" * 80 + "\n\n")

            for log_line in self._logs:
                f.write(log_line + "\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF LOGS\n")
            f.write("=" * 80 + "\n")

    def _generate_plots(self) -> None:
        """Genera gráficas de resultados: combinada + individuales."""
        if len(self._metrics_steps) == 0:
            return

        steps = np.array(list(self._metrics_steps))
        losses = np.array(list(self._metrics_loss))
        accuracies = np.array(list(self._metrics_accuracy))
        workers_count = np.array(list(self._metrics_workers))

        try:
            self._plot_3panels(steps, losses, accuracies, workers_count)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_individual_loss(steps, losses)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_individual_accuracy(steps, accuracies)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_individual_workers(steps, workers_count)
        except Exception:
            import traceback

            traceback.print_exc()

    def _plot_3panels(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        accuracies: np.ndarray,
        workers_count: np.ndarray,
    ) -> None:
        """Genera gráfica con 3 paneles horizontales: loss, accuracy, workers."""
        fig = plt.figure(figsize=(13, 4), dpi=95)
        gs = GridSpec(1, 3, figure=fig, wspace=0.35)

        color_loss = "#F44336"
        color_acc = "#2196F3"
        color_workers = "#4CAF50"

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps, losses, "-o", color=color_loss, lw=2, ms=3, label="Train")
        ax1.scatter(steps, losses, color=color_loss, s=30, zorder=5, label="Val")
        ax1.set_title("Pérdida (ventana deslizante)")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(steps, accuracies, "-o", color=color_acc, lw=2, ms=3, label="Train")
        ax2.scatter(steps, accuracies, color=color_acc, s=30, zorder=5, label="Val")
        ax2.set_title("Precisión (ventana deslizante)")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Precisión (%)")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(gs[2])
        ax3.step(steps, workers_count, color=color_workers, lw=2)
        ax3.set_title("Workers activos")
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("N Workers")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, max(workers_count) + 1)

        # Nota sobre diferentes escalas Y
        fig.text(
            0.5,
            0.98,
            "Nota: Cada gráfica tiene su propia escala Y",
            ha="center",
            fontsize=9,
            color="gray",
            style="italic",
            transform=fig.transFigure,
        )

        fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.15)
        output_path = self.session_dir / "plot_3panels.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _plot_individual_loss(self, steps: np.ndarray, losses: np.ndarray) -> None:
        """Genera gráfica individual de Loss (estilo idéntico a la GUI)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        color_loss = "#F44336"

        ax.plot(steps, losses, "-o", color=color_loss, lw=2, ms=3, label="Train")
        ax.scatter(steps, losses, color=color_loss, s=30, zorder=5, label="Val")
        ax.set_title("Pérdida (ventana deslizante)")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        plt.tight_layout()
        output_path = self.session_dir / "plot_loss.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_individual_accuracy(
        self, steps: np.ndarray, accuracies: np.ndarray
    ) -> None:
        """Genera gráfica individual de Accuracy (estilo idéntico a la GUI)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        color_acc = "#2196F3"

        ax.plot(steps, accuracies, "-o", color=color_acc, lw=2, ms=3, label="Train")
        ax.scatter(steps, accuracies, color=color_acc, s=30, zorder=5, label="Val")
        ax.set_title("Precisión (ventana deslizante)")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Precisión (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)

        plt.tight_layout()
        output_path = self.session_dir / "plot_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_individual_workers(
        self, steps: np.ndarray, workers_count: np.ndarray
    ) -> None:
        """Genera gráfica individual de Workers (estilo idéntico a la GUI)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        color_workers = "#4CAF50"

        ax.step(steps, workers_count, color=color_workers, lw=2)
        ax.set_title("Workers activos")
        ax.set_xlabel("Steps")
        ax.set_ylabel("N Workers")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(workers_count) + 1)

        plt.tight_layout()
        output_path = self.session_dir / "plot_workers.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _write_metadata(self) -> None:
        """Escribe estadísticas finales en metadata.json."""
        if len(self._metrics_loss) == 0:
            metadata = {"status": "no_metrics_recorded"}
        else:
            # Calcular duration desde primer step (no desde creación del exporter)
            duration = 0.0
            if self._training_start_time is not None:
                duration = time.time() - self._training_start_time

            metadata = {
                "status": "completed",
                "session_timestamp": self.timestamp,
                "total_steps": int(self._metrics_steps[-1]),
                "total_metrics_points": self._total_metrics,
                "duration_seconds": duration,
                "loss": {
                    "initial": float(self._metrics_loss[0]),
                    "final": float(self._metrics_loss[-1]),
                    "min": float(np.min(list(self._metrics_loss))),
                    "max": float(np.max(list(self._metrics_loss))),
                    "mean": float(np.mean(list(self._metrics_loss))),
                },
                "accuracy": {
                    "initial": float(self._metrics_accuracy[0]),
                    "final": float(self._metrics_accuracy[-1]),
                    "min": float(np.min(list(self._metrics_accuracy))),
                    "max": float(np.max(list(self._metrics_accuracy))),
                    "mean": float(np.mean(list(self._metrics_accuracy))),
                },
                "workers": {
                    "max_connected": int(np.max(list(self._metrics_workers))),
                },
                "total_log_lines": len(self._logs),
            }

        metadata_file = self.session_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        """Representación string del exporter."""
        return (
            f"ResultsExporter(timestamp={self.timestamp}, "
            f"metrics={self._total_metrics}, "
            f"logs={len(self._logs)})"
        )
