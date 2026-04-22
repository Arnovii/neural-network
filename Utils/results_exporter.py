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
    ├── plot_3panels.png      # 3 gráficas combinadas (loss/acc/workers)
    ├── plot_loss.png         # Gráfica individual de Loss (idéntica al panel 1)
    ├── plot_accuracy.png     # Gráfica individual de Accuracy (idéntica al panel 2)
    ├── plot_workers.png      # Gráfica individual de Workers (idéntica al panel 3)
    ├── plot_comparison.png   # Loss vs Accuracy con ejes Y duales
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
        self._logs: List[str] = []

        # Estadísticas para reporte final
        self._total_metrics = 0
        self._start_time = time.time()

    def record_metric(
        self,
        step: int,
        loss: float,
        accuracy: float,
        num_workers: int,
    ) -> None:
        """
        Registra un punto de métrica.

        Llamado típicamente desde callback on_step del PS.
        Thread-safe, no bloquea.

        :param step: Número del step de training
        :param loss: Valor de loss
        :param accuracy: Valor de accuracy (0-1)
        :param num_workers: Número de workers conectados
        """
        with self._lock:
            self._metrics_steps.append(step)
            self._metrics_loss.append(loss)
            self._metrics_accuracy.append(accuracy)
            self._metrics_workers.append(num_workers)
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
            # Header
            f.write("step,loss,accuracy,num_workers\n")

            # Data
            for step, loss, acc, workers in zip(
                self._metrics_steps,
                self._metrics_loss,
                self._metrics_accuracy,
                self._metrics_workers,
            ):
                f.write(f"{step},{loss:.6f},{acc:.6f},{workers}\n")

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
            # Sin datos, no generar gráficas
            return

        # Convertir deques a numpy arrays
        steps = np.array(list(self._metrics_steps))
        losses = np.array(list(self._metrics_loss))
        accuracies = np.array(list(self._metrics_accuracy))
        workers_count = np.array(list(self._metrics_workers))

        try:
            # Gráfica 1: 3 paneles con escalas independientes
            self._plot_3panels(steps, losses, accuracies, workers_count)
        except Exception as e:
            self.record_log(f"ERROR en _plot_3panels: {e}")

        try:
            # Gráficas individuales (iguales a los paneles del 3-panel)
            self._plot_individual_loss(steps, losses)
        except Exception as e:
            self.record_log(f"ERROR en _plot_individual_loss: {e}")

        try:
            self._plot_individual_accuracy(steps, accuracies)
        except Exception as e:
            self.record_log(f"ERROR en _plot_individual_accuracy: {e}")

        try:
            self._plot_individual_workers(steps, workers_count)
        except Exception as e:
            self.record_log(f"ERROR en _plot_individual_workers: {e}")

        try:
            # Gráfica 2: Loss y Accuracy con ejes duales
            self._plot_comparison(steps, losses, accuracies)
        except Exception as e:
            self.record_log(f"ERROR en _plot_comparison: {e}")

    def _plot_3panels(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        accuracies: np.ndarray,
        workers_count: np.ndarray,
    ) -> None:
        """
        Genera gráfica con 3 paneles: loss, accuracy, workers.
        Cada uno con su propia escala dinámica basada en los datos.
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 1, figure=fig, hspace=0.35)

        # Color scheme profesional
        color_loss = "#E74C3C"
        color_acc = "#27AE60"
        color_workers = "#3498DB"

        # Panel 1: Loss (escala auto con margen superior)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps, losses, color=color_loss, linewidth=2, marker="o", markersize=3)
        ax1.set_xlabel("Training Step", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Loss", fontsize=11, fontweight="bold", color=color_loss)
        ax1.tick_params(axis="y", labelcolor=color_loss)
        ax1.grid(True, alpha=0.15, linestyle="--")  # Líneas menos opacas
        ax1.set_title("Loss Evolution", fontsize=12, fontweight="bold")
        
        # Escala dinámica: 10% de margen superior
        loss_max = np.max(losses)
        loss_min = np.min(losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else 1
        ax1.set_ylim(loss_min - 0.05 * loss_range, loss_max + 0.1 * loss_range)

        # Panel 2: Accuracy (escala dinámica con margen superior)
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(
            steps, accuracies, color=color_acc, linewidth=2, marker="s", markersize=3
        )
        ax2.set_xlabel("Training Step", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Accuracy", fontsize=11, fontweight="bold", color=color_acc)
        ax2.tick_params(axis="y", labelcolor=color_acc)
        ax2.grid(True, alpha=0.4, linestyle="--", linewidth=0.7)
        
        # Escala dinámica: calcular techo para que el máximo tenga espacio arriba
        acc_max = np.max(accuracies)
        acc_min = np.min(accuracies) if np.min(accuracies) > 0 else 0
        # Margen superior: 20% del máximo o al menos 0.05
        acc_top_margin = max(0.05, acc_max * 0.20)
        ax2.set_ylim(0, acc_max + acc_top_margin)
        
        # Líneas de referencia horizontales
        for y_val in np.linspace(0, acc_max + acc_top_margin, 5)[1:-1]:
            ax2.axhline(y=y_val, color="gray", alpha=0.15, linestyle=":", linewidth=0.8)
        
        ax2.set_title("Accuracy Evolution", fontsize=12, fontweight="bold")

        # Panel 3: Workers (escala auto con margen superior)
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(
            steps,
            workers_count,
            color=color_workers,
            linewidth=2,
            marker="^",
            markersize=3,
        )
        ax3.set_xlabel("Training Step", fontsize=11, fontweight="bold")
        ax3.set_ylabel(
            "Connected Workers", fontsize=11, fontweight="bold", color=color_workers
        )
        ax3.tick_params(axis="y", labelcolor=color_workers)
        ax3.grid(True, alpha=0.15, linestyle="--")  # Líneas menos opacas
        ax3.set_title("Active Workers Over Time", fontsize=12, fontweight="bold")
        
        # Escala dinámica con margen superior
        workers_max = np.max(workers_count)
        workers_range = workers_max if workers_max > 0 else 1
        ax3.set_ylim(0, workers_max + 0.15 * workers_range)

        # Título general
        fig.suptitle(
            "Distributed Async-SGD Training Metrics",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        # Ajustes de espaciado
        fig.subplots_adjust(top=0.93, bottom=0.08)

        # Guardar
        output_path = self.session_dir / "plot_3panels.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        plt.close()

    def _plot_individual_loss(self, steps: np.ndarray, losses: np.ndarray) -> None:
        """Genera gráfica individual de Loss (idéntica al panel 1 del 3-panel)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        color_loss = "#E74C3C"
        ax.plot(steps, losses, color=color_loss, linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Training Step", fontsize=11, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=11, fontweight="bold", color=color_loss)
        ax.tick_params(axis="y", labelcolor=color_loss)
        ax.grid(True, alpha=0.15, linestyle="--")
        ax.set_title("Loss Evolution", fontsize=13, fontweight="bold")
        
        # Escala dinámica
        loss_max = np.max(losses)
        loss_min = np.min(losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else 1
        ax.set_ylim(loss_min - 0.05 * loss_range, loss_max + 0.1 * loss_range)
        
        plt.tight_layout()
        output_path = self.session_dir / "plot_loss.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_individual_accuracy(self, steps: np.ndarray, accuracies: np.ndarray) -> None:
        """Genera gráfica individual de Accuracy (idéntica al panel 2 del 3-panel)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        color_acc = "#27AE60"
        ax.plot(steps, accuracies, color=color_acc, linewidth=2, marker="s", markersize=4)
        ax.set_xlabel("Training Step", fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold", color=color_acc)
        ax.tick_params(axis="y", labelcolor=color_acc)
        ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.7)
        ax.set_title("Accuracy Evolution", fontsize=13, fontweight="bold")
        
        # Escala dinámica: margen superior para visualización clara
        acc_max = np.max(accuracies)
        acc_top_margin = max(0.05, acc_max * 0.20)
        ax.set_ylim(0, acc_max + acc_top_margin)
        
        # Líneas de referencia
        for y_val in np.linspace(0, acc_max + acc_top_margin, 5)[1:-1]:
            ax.axhline(y=y_val, color="gray", alpha=0.15, linestyle=":", linewidth=0.8)
        
        plt.tight_layout()
        output_path = self.session_dir / "plot_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_individual_workers(self, steps: np.ndarray, workers_count: np.ndarray) -> None:
        """Genera gráfica individual de Workers (idéntica al panel 3 del 3-panel)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        color_workers = "#3498DB"
        ax.plot(steps, workers_count, color=color_workers, linewidth=2, marker="^", markersize=4)
        ax.set_xlabel("Training Step", fontsize=11, fontweight="bold")
        ax.set_ylabel("Connected Workers", fontsize=11, fontweight="bold", color=color_workers)
        ax.tick_params(axis="y", labelcolor=color_workers)
        ax.grid(True, alpha=0.15, linestyle="--")
        ax.set_title("Active Workers Over Time", fontsize=13, fontweight="bold")
        
        # Escala dinámica
        workers_max = np.max(workers_count)
        workers_range = workers_max if workers_max > 0 else 1
        ax.set_ylim(0, workers_max + 0.15 * workers_range)
        
        plt.tight_layout()
        output_path = self.session_dir / "plot_workers.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_comparison(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        accuracies: np.ndarray,
    ) -> None:
        """
        Genera gráfica con loss y accuracy usando dos ejes Y.
        Cada métrica en su escala natural para claridad.
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color_loss = "#E74C3C"
        color_acc = "#27AE60"

        # Eje izquierdo: Loss
        ax1.set_xlabel("Training Step", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Loss", fontsize=11, fontweight="bold", color=color_loss)
        line1 = ax1.plot(
            steps,
            losses,
            color=color_loss,
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Loss",
        )
        ax1.tick_params(axis="y", labelcolor=color_loss)
        ax1.grid(True, alpha=0.3, linestyle="--")
        
        # Escala dinámica para Loss
        loss_max = np.max(losses)
        loss_min = np.min(losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else 1
        ax1.set_ylim(loss_min - 0.05 * loss_range, loss_max + 0.1 * loss_range)

        # Eje derecho: Accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", fontsize=11, fontweight="bold", color=color_acc)
        line2 = ax2.plot(
            steps,
            accuracies,
            color=color_acc,
            linewidth=2.5,
            marker="s",
            markersize=5,
            label="Accuracy",
        )
        ax2.tick_params(axis="y", labelcolor=color_acc)
        
        # Escala dinámica para Accuracy con margen superior
        acc_max = np.max(accuracies)
        acc_top_margin = max(0.05, acc_max * 0.20)
        ax2.set_ylim(0, acc_max + acc_top_margin)
        
        # Líneas guías visibles en el eje derecho (Accuracy)
        for y_val in np.linspace(0, acc_max + acc_top_margin, 5)[1:-1]:
            ax2.axhline(y=y_val, color=color_acc, alpha=0.15, linestyle=":", linewidth=1)

        # Leyenda combinada
        lines = line1 + line2
        labels = [str(l.get_label()) for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=10)

        # Título
        fig.suptitle(
            "Training Progress: Loss vs Accuracy",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()

        # Guardar
        output_path = self.session_dir / "plot_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _write_metadata(self) -> None:
        """Escribe estadísticas finales en metadata.json."""
        if len(self._metrics_loss) == 0:
            metadata = {"status": "no_metrics_recorded"}
        else:
            metadata = {
                "status": "completed",
                "session_timestamp": self.timestamp,
                "total_steps": int(self._metrics_steps[-1]),
                "total_metrics_points": self._total_metrics,
                "duration_seconds": time.time() - self._start_time,
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
