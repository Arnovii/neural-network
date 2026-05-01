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
from Utils.constants import COLORS, EXPORT_DIR_DEFAULT
from matplotlib.ticker import MaxNLocator


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
        export_dir: str = EXPORT_DIR_DEFAULT,
        metrics_window: int = 500,
    ) -> None:
        """
        Inicializa el ResultsExporter.

        :param config: Diccionario de configuración del experimento.
        :param export_dir: Directorio base para exportar los resultados.
        :param metrics_window: Tamaño de la ventana para buffers de métricas.
        """
        # Configuración
        self.config = config
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_window = metrics_window
        self._config_metrics_window = int(config.get("metrics_window", metrics_window))

        # Timestamp único para esta ejecución
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.session_dir = self.export_dir / self.timestamp

        # No crear carpeta hasta finalize() para evitar archivos parciales
        self._finalized = False

        # Buffers en memoria (thread-safe con locks)
        self._lock = threading.Lock()
        self._metrics_steps: deque = deque(maxlen=metrics_window)
        self._metrics_loss: deque = deque(maxlen=metrics_window)
        self._metrics_accuracy: deque = deque(maxlen=metrics_window)
        self._metrics_workers: deque = deque(maxlen=metrics_window)
        self._metrics_elapsed: deque = deque(maxlen=metrics_window)
        # Nuevos buffers para std, staleness y alpha
        self._metrics_loss_std: deque = deque(maxlen=metrics_window)
        self._metrics_acc_std: deque = deque(maxlen=metrics_window)
        self._metrics_staleness: deque = deque(maxlen=metrics_window)
        self._metrics_alpha: deque = deque(maxlen=metrics_window)
        self._logs: List[str] = []

        # worker events
        self._worker_events: List[dict] = []

        # Full-history lists (sin límite) — usadas sólo para exportar gráficas completas
        self._full_steps: list = []
        self._full_losses: list = []
        self._full_accuracies: list = []
        self._full_workers: list = []
        self._full_elapsed: list = []
        self._full_loss_std: list = []
        self._full_acc_std: list = []
        self._full_staleness: list = []
        self._full_alpha: list = []

        # Contadores auxiliares expuestos por el PS antes de finalize()
        self.tcp_request_count: int = 0
        self.nan_rejected_count: int = 0

        # Estadísticas para reporte final
        self._total_metrics = 0

    def record_metric(
        self,
        step: int,
        loss: float,
        accuracy: float,
        num_workers: int,
        elapsed: float = 0.0,
        loss_std: float = 0.0,
        acc_std: float = 0.0,
        staleness: int = 0,
        alpha: float = 1.0,
    ) -> None:
        """Registra un punto de métrica.

        Llamado típicamente desde callback on_step del PS.
        Thread-safe, no bloquea el entrenamiento.

        Args:
            step: Número del step de training (global).
            loss: Valor de loss del batch actual.
            accuracy: Valor de accuracy (0-1) del batch actual.
            num_workers: Número de workers conectados actualmente.
            elapsed: Tiempo acumulado en segundos desde inicio.
            loss_std: Desviación estándar del loss en la ventana.
            acc_std: Desviación estándar de accuracy en la ventana.
            staleness: Valor de staleness del update.
            alpha: Factor de corrección alpha = 1/(1+λ·s).

        Returns:
            None. Actualiza el estado interno thread-safe.

        Note:
            Los datos se almacenan en dos lugares:
            - Deques con window fijo (para promedios recientes)
            - Listas sin límite (para historial completo)

        Example:
            # Llamado desde PS on_step callback
            exporter.record_metric(
                step=1000,
                loss=2.5,
                accuracy=0.45,
                num_workers=3,
                elapsed=125.5,
                loss_std=0.3,
                acc_std=0.05,
                staleness=2,
                alpha=0.83
            )
        """
        with self._lock:
            self._metrics_steps.append(step)
            self._metrics_loss.append(loss)
            self._metrics_accuracy.append(accuracy)
            self._metrics_workers.append(num_workers)
            self._metrics_elapsed.append(elapsed)
            self._metrics_loss_std.append(loss_std)
            self._metrics_acc_std.append(acc_std)
            self._metrics_staleness.append(staleness)
            self._metrics_alpha.append(alpha)
            self._total_metrics += 1
            # También acumular en los historiales completos (sin límite)
            try:
                self._full_steps.append(step)
                self._full_losses.append(loss)
                self._full_accuracies.append(accuracy)
                self._full_workers.append(num_workers)
                self._full_elapsed.append(elapsed)
                self._full_loss_std.append(loss_std)
                self._full_acc_std.append(acc_std)
                self._full_staleness.append(staleness)
                self._full_alpha.append(alpha)
            except Exception:
                # Seguridad: nunca propagar errores de escritura de historial
                pass

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

    def record_worker_event(
        self,
        step: int,
        event_type: str,
        worker_id: int,
        worker_addr: str,
    ) -> None:
        """
        Registra un evento de worker (conexión/desconexión).

        Llamado desde ParameterServer al conectar/desconectar workers.
        Thread-safe.

        :param step: Step actual del entrenamiento
        :param event_type: "connected" o "disconnected"
        :param worker_id: ID del worker
        :param worker_addr: Dirección IP del worker
        """
        with self._lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._worker_events.append(
                {
                    "timestamp": timestamp,
                    "step": step,
                    "event_type": event_type,
                    "worker_id": worker_id,
                    "worker_addr": worker_addr,
                }
            )

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

        # 4. Guardar eventos de workers
        self._write_worker_events()

        # 5. Generar gráficas
        self._generate_plots()

        # 6. Guardar metadata
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

        # Exportar SIEMPRE historial completo cuando exista; fallback a ventana
        # para compatibilidad con sesiones antiguas.
        with self._lock:
            if len(self._full_steps) > 0:
                steps_hist = list(self._full_steps)
                loss_hist = list(self._full_losses)
                loss_std_hist = list(self._full_loss_std)
                acc_hist = list(self._full_accuracies)
                acc_std_hist = list(self._full_acc_std)
                workers_hist = list(self._full_workers)
                elapsed_hist = list(self._full_elapsed)
                staleness_hist = list(self._full_staleness)
                alpha_hist = list(self._full_alpha)
            else:
                steps_hist = list(self._metrics_steps)
                loss_hist = list(self._metrics_loss)
                loss_std_hist = list(self._metrics_loss_std)
                acc_hist = list(self._metrics_accuracy)
                acc_std_hist = list(self._metrics_acc_std)
                workers_hist = list(self._metrics_workers)
                elapsed_hist = list(self._metrics_elapsed)
                staleness_hist = list(self._metrics_staleness)
                alpha_hist = list(self._metrics_alpha)

        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write(
                "step,loss,loss_std,accuracy,acc_std,num_workers,elapsed_seconds,staleness,alpha\n"
            )

            for i, step in enumerate(steps_hist):
                loss = loss_hist[i] if i < len(loss_hist) else float("nan")
                loss_std = loss_std_hist[i] if i < len(loss_std_hist) else 0.0
                acc = acc_hist[i] if i < len(acc_hist) else float("nan")
                acc_std = acc_std_hist[i] if i < len(acc_std_hist) else 0.0
                workers = workers_hist[i] if i < len(workers_hist) else 0
                elapsed = elapsed_hist[i] if i < len(elapsed_hist) else 0.0
                staleness = staleness_hist[i] if i < len(staleness_hist) else 0
                alpha = alpha_hist[i] if i < len(alpha_hist) else 1.0
                f.write(
                    f"{step},{loss:.4f},{loss_std:.4f},{acc:.2f},{acc_std:.2f},"
                    f"{workers},{elapsed:.1f},{staleness},{alpha:.4f}\n"
                )

    def _write_logs(self) -> None:
        """Escribe todos los logs en un archivo de texto."""
        logs_file = self.session_dir / "ps_logs.txt"

        with open(logs_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PARAMETER SERVER LOGS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Session: {self.timestamp}\n")
            f.write("=" * 80 + "\n\n")

            for log_line in self._logs:
                f.write(log_line + "\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF LOGS\n")
            f.write("=" * 80 + "\n")

    def _write_worker_events(self) -> None:
        """Escribe eventos de workers en CSV."""
        events_file = self.session_dir / "worker_events.csv"
        with open(events_file, "w", encoding="utf-8") as f:
            f.write("timestamp,step,event_type,worker_id,worker_addr\n")
            for ev in self._worker_events:
                f.write(
                    f"{ev['timestamp']},{ev['step']},{ev['event_type']},"
                    f"{ev['worker_id']},{ev['worker_addr']}\n"
                )

    def _generate_plots(self) -> None:
        """Genera gráficas de resultados: combinada + individuales."""
        if len(self._full_steps) == 0 and len(self._metrics_steps) == 0:
            return

        with self._lock:
            if len(self._full_steps) > 0:
                steps = np.array(list(self._full_steps))
                losses = np.array(list(self._full_losses))
                accuracies = np.array(list(self._full_accuracies))
                workers_count = np.array(list(self._full_workers))
                loss_std = np.array(list(self._full_loss_std))
                acc_std = np.array(list(self._full_acc_std))
                staleness = np.array(list(self._full_staleness))
                alpha = np.array(list(self._full_alpha))
            else:
                steps = np.array(list(self._metrics_steps))
                losses = np.array(list(self._metrics_loss))
                accuracies = np.array(list(self._metrics_accuracy))
                workers_count = np.array(list(self._metrics_workers))
                loss_std = np.array(list(self._metrics_loss_std))
                acc_std = np.array(list(self._metrics_acc_std))
                staleness = np.array(list(self._metrics_staleness))
                alpha = np.array(list(self._metrics_alpha))

        event_steps = [ev["step"] for ev in self._worker_events]
        event_types = [ev["event_type"] for ev in self._worker_events]

        loss_xlim, loss_ylim = self._compute_line_limits(steps, losses)
        acc_xlim, acc_ylim = self._compute_accuracy_limits(steps, accuracies)

        try:
            self._plot_3panels(steps, losses, accuracies, workers_count)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_individual_loss(steps, losses, loss_xlim, loss_ylim)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_band_loss(steps, losses, loss_std, loss_xlim, loss_ylim)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_individual_accuracy(steps, accuracies, acc_xlim, acc_ylim)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_band_accuracy(steps, accuracies, acc_std, acc_xlim, acc_ylim)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_staleness(steps, staleness, alpha, event_steps, event_types)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_individual_workers(steps, workers_count)
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            self._plot_std(steps, loss_std, acc_std, event_steps, event_types)
        except Exception:
            import traceback

            traceback.print_exc()

    def _compute_line_limits(
        self, steps: np.ndarray, values: np.ndarray
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if len(steps) == 0:
            return (0.0, 1.0), (0.0, 1.0)

        xlim = (float(steps[0] - 1), float(steps[-1] + 1))
        if len(values) == 1:
            delta = max(1.0, abs(float(values[0])) * 0.1)
            ylim = (float(values[0] - delta), float(values[0] + delta))
        else:
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            span = vmax - vmin
            pad = max(span * 0.05, 1e-6)
            ylim = (vmin - pad, vmax + pad)
        return xlim, ylim

    def _compute_accuracy_limits(
        self, steps: np.ndarray, values: np.ndarray
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if len(steps) == 0:
            return (0.0, 1.0), (0.0, 100.0)
        return (float(steps[0] - 1), float(steps[-1] + 1)), (0.0, 100.0)

    def _add_event_markers(
        self,
        axes,
        event_steps: List[int],
        event_types: List[str],
    ) -> None:
        if not event_steps:
            return

        for ax in axes:
            labels_added: set[str] = set()
            for step, event_type in zip(event_steps, event_types):
                color = "#4CAF50" if event_type == "connected" else "#F44336"
                label = event_type if event_type not in labels_added else None
                ax.axvline(
                    step,
                    color=color,
                    linestyle=":",
                    alpha=0.5,
                    linewidth=1.2,
                    label=label,
                )
                if label is not None:
                    labels_added.add(event_type)

    def _plot_3panels(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        accuracies: np.ndarray,
        workers_count: np.ndarray,
    ) -> None:
        """Genera gráfica con 3 paneles horizontales: loss, accuracy, workers."""
        fig = plt.figure(figsize=(20, 5.5), dpi=95)
        gs = GridSpec(1, 3, figure=fig, wspace=0.35)

        color_loss = COLORS["loss"]
        color_acc = COLORS["accuracy"]
        color_workers = COLORS["workers"]

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps, losses, "-o", color=color_loss, lw=2, ms=3, label="Train")
        ax1.scatter(steps, losses, color=color_loss, s=30, zorder=5, label="Val")
        ax1.set_title("Pérdida (ventana deslizante)")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss (nats)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Establecer límites de ejes para que coincidan con GUI
        ax1.set_xlim(steps[0] - 1, steps[-1] + 1)
        ax1.margins(y=0.05)  # 5% de margen en Y

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(steps, accuracies, "-o", color=color_acc, lw=2, ms=3, label="Train")
        ax2.scatter(steps, accuracies, color=color_acc, s=30, zorder=5, label="Val")
        ax2.set_title("Precisión (ventana deslizante)")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Precisión (%)")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=8)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Establecer límites de ejes para que coincidan con GUI
        ax2.set_xlim(steps[0] - 1, steps[-1] + 1)

        ax3 = fig.add_subplot(gs[2])
        ax3.step(steps, workers_count, color=color_workers, lw=2)
        ax3.set_title("Workers activos")
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("N Workers")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, max(workers_count) + 1)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Establecer límites de ejes para que coincidan con GUI
        ax3.set_xlim(steps[0] - 1, steps[-1] + 1)

        # Nota sobre diferentes escalas Y
        fig.text(
            0.5,
            0.95,
            "Nota: Cada gráfica tiene su propia escala Y",
            ha="center",
            fontsize=12,
            color="gray",
            style="italic",
            transform=fig.transFigure,
        )

        fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12)
        output_path = self.session_dir / "plot_3panels.png"
        plt.savefig(output_path, dpi=300, pad_inches=0.4)
        plt.close()

    def _plot_individual_loss(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
    ) -> None:
        """Genera gráfica individual de Loss (estilo idéntico a la GUI)."""
        fig, ax = plt.subplots(figsize=(10, 6.5))

        color_loss = COLORS["loss"]
        x_margin = (steps[-1] - steps[0]) * 0.03 if len(steps) > 1 else 0.0

        ax.plot(steps, losses, "-o", color=color_loss, lw=2, ms=3, label="Train")
        ax.scatter(steps, losses, color=color_loss, s=30, zorder=5, label="Val")
        if len(steps) > 0:
            ax.scatter(
                steps[-1],
                losses[-1],
                s=40,
                color="white",
                edgecolors=color_loss,
                linewidths=1.5,
                zorder=7,
            )
            ax.annotate(
                f"{losses[-1]:.4f}",
                xy=(steps[-1], losses[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=8,
                color=color_loss,
                va="center",
            )
        ax.set_title("Pérdida (ventana deslizante)")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(steps) > 0:
            ax.set_xlim(float(steps[0]), float(steps[-1] + x_margin))
        else:
            ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        plt.tight_layout()
        output_path = self.session_dir / "plot_loss.png"
        plt.savefig(output_path, dpi=300, pad_inches=0.4)
        plt.close()

    def _plot_band_loss(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        loss_std: np.ndarray,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        color_loss = COLORS["loss"]
        x_margin = (steps[-1] - steps[0]) * 0.03 if len(steps) > 1 else 0.0

        lower = losses - loss_std
        upper = losses + loss_std

        ax.plot(steps, losses, "-o", color=color_loss, lw=2, ms=3, label="Train")
        ax.fill_between(steps, lower, upper, color=color_loss, alpha=0.2, label="±1σ")
        if len(steps) > 0:
            ax.scatter(
                steps[-1],
                losses[-1],
                s=40,
                color="white",
                edgecolors=color_loss,
                linewidths=1.5,
                zorder=7,
            )
            ax.annotate(
                f"{losses[-1]:.4f}",
                xy=(steps[-1], losses[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=8,
                color=color_loss,
                va="center",
            )

        ax.set_title(
            f"Pérdida con Banda de Confianza ±1σ (ventana de {self._config_metrics_window} pasos)"
        )
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Calcular ylim seguro según restricción del usuario
        try:
            ymin = float(np.min(lower)) * 0.998
            ymax = float(np.max(upper)) * 1.002
            ax.set_ylim(ymin, ymax)
        except Exception:
            ax.set_ylim(*ylim)

        # xlim exacto para la banda
        try:
            ax.set_xlim(float(steps[0]), float(steps[-1]))
        except Exception:
            ax.set_xlim(*xlim)

        # Anotación de estadísticas sobre el historial completo
        try:
            loss_std_arr = (
                np.array(self._full_loss_std)
                if len(self._full_loss_std) > 0
                else loss_std
            )
            loss_std_mean = (
                float(np.mean(loss_std_arr)) if len(loss_std_arr) > 0 else 0.0
            )
            loss_std_final = float(loss_std_arr[-1]) if len(loss_std_arr) > 0 else 0.0
            text_box = f"σ medio: {loss_std_mean:.4f} (adim.)\nσ final: {loss_std_final:.4f} (adim.)"
            # decidir esquina: usar superior derecha por defecto
            ax.text(
                0.98,
                0.98,
                text_box,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
            )
        except Exception:
            pass

        # Líneas de referencia del rango final ±1σ y anotaciones
        try:
            if len(loss_std) > 0 and not np.allclose(loss_std, 0.0):
                y_upper = float(losses[-1] + loss_std[-1])
                y_lower = float(losses[-1] - loss_std[-1])
                ax.axhline(
                    y_upper, color="gray", linestyle=":", linewidth=0.8, alpha=0.6
                )
                ax.axhline(
                    y_lower, color="gray", linestyle=":", linewidth=0.8, alpha=0.6
                )
                # Anotar valores al final (derecha)
                try:
                    # Calcular si las etiquetas están muy cercanas (< 3% del rango Y)
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    too_close = abs(y_upper - y_lower) < y_range * 0.03

                    if too_close:
                        # Si están muy cercas, usar offset horizontal
                        ax.annotate(
                            f"+1σ: {y_upper:.4f}",
                            xy=(steps[-1], y_upper),
                            xytext=(-70, 10),
                            textcoords="offset points",
                            fontsize=7.5,
                            color="gray",
                            va="center",
                        )
                        ax.annotate(
                            f"-1σ: {y_lower:.4f}",
                            xy=(steps[-1], y_lower),
                            xytext=(-70, -14),
                            textcoords="offset points",
                            fontsize=7.5,
                            color="gray",
                            va="center",
                        )
                    else:
                        # Si no, usar posicionamiento a la derecha
                        ax.annotate(
                            f"{y_upper:.4f}",
                            xy=(steps[-1], y_upper),
                            xytext=(6, 0),
                            textcoords="offset points",
                            fontsize=8,
                            color="gray",
                            va="center",
                        )
                        ax.annotate(
                            f"{y_lower:.4f}",
                            xy=(steps[-1], y_lower),
                            xytext=(6, 0),
                            textcoords="offset points",
                            fontsize=8,
                            color="gray",
                            va="center",
                        )
                except Exception:
                    pass

                # Eje Y secundario para σ
                try:
                    ax2 = ax.twinx()
                    ax2.plot(
                        steps,
                        loss_std,
                        color="gray",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.6,
                        label="σ (eje der.)",
                    )
                    ax2.set_ylabel("σ Loss (adim.)", fontsize=9, color="gray")
                    ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
                    # Calcular ylim del eje secundario para evitar cruce con banda
                    try:
                        band_min = float(np.min(losses - loss_std))
                        sigma_max = float(np.max(loss_std))
                        ax_ylim_min = ax.get_ylim()[0]
                        ax2_scale = (band_min - ax_ylim_min) / sigma_max * 0.85
                        if ax2_scale > 0:
                            ax2.set_ylim(0, sigma_max / ax2_scale)
                        else:
                            ax2.set_ylim(0, sigma_max * 4.0)
                    except Exception:
                        ax2.set_ylim(0, float(np.max(loss_std)) * 2.5)
                    # Ocultar ticks del eje secundario para evitar clutter
                    ax2.set_yticks([])
                    ax2.grid(False)
                    # combinar leyendas
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
                except Exception:
                    pass

        except Exception:
            # Experimento corto: omitir líneas secundarias sin fallar
            pass

        # Marcar mínimo
        try:
            if len(losses) > 0:
                idx_min = int(np.argmin(losses))
                step_range = steps[-1] - steps[0] if len(steps) > 1 else 1
                # Si el mínimo está muy cerca del final, mover etiqueta a la izquierda
                if (steps[-1] - steps[idx_min]) < step_range * 0.05:
                    xytext_min = (-50, -20)  # arriba-izquierda
                else:
                    xytext_min = (8, -15)  # estándar

                ax.annotate(
                    f"Mín: {losses[idx_min]:.4f}",
                    xy=(steps[idx_min], losses[idx_min]),
                    xytext=xytext_min,
                    textcoords="offset points",
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
                    color="darkred",
                )
        except Exception:
            pass

        plt.tight_layout()
        output_path = self.session_dir / "plot_band_loss.png"
        plt.savefig(output_path, dpi=300, pad_inches=0.4)
        plt.close()

    def _plot_individual_accuracy(
        self,
        steps: np.ndarray,
        accuracies: np.ndarray,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
    ) -> None:
        """Genera gráfica individual de Accuracy (estilo idéntico a la GUI)."""
        fig, ax = plt.subplots(figsize=(10, 6.5))

        color_acc = COLORS["accuracy"]
        x_margin = (steps[-1] - steps[0]) * 0.03 if len(steps) > 1 else 0.0

        ax.plot(steps, accuracies, "-o", color=color_acc, lw=2, ms=3, label="Train")
        ax.scatter(steps, accuracies, color=color_acc, s=30, zorder=5, label="Val")
        if len(steps) > 0:
            ax.scatter(
                steps[-1],
                accuracies[-1],
                s=40,
                color="white",
                edgecolors=color_acc,
                linewidths=1.5,
                zorder=7,
            )
            ax.annotate(
                f"{accuracies[-1]:.2f}%",
                xy=(steps[-1], accuracies[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=8,
                color=color_acc,
                va="center",
            )
        ax.set_title("Precisión (ventana deslizante)")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Precisión (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(steps) > 0:
            ax.set_xlim(float(steps[0]), float(steps[-1] + x_margin))
        else:
            ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        plt.tight_layout()
        output_path = self.session_dir / "plot_accuracy.png"
        plt.savefig(output_path, dpi=300, pad_inches=0.4)
        plt.close()

    def _plot_band_accuracy(
        self,
        steps: np.ndarray,
        accuracies: np.ndarray,
        acc_std: np.ndarray,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        color_acc = COLORS["accuracy"]
        x_margin = (steps[-1] - steps[0]) * 0.03 if len(steps) > 1 else 0.0

        lower = accuracies - acc_std
        upper = accuracies + acc_std

        ax.plot(steps, accuracies, "-o", color=color_acc, lw=2, ms=3, label="Train")
        ax.fill_between(steps, lower, upper, color=color_acc, alpha=0.2, label="±1σ")
        if len(steps) > 0:
            ax.scatter(
                steps[-1],
                accuracies[-1],
                s=40,
                color="white",
                edgecolors=color_acc,
                linewidths=1.5,
                zorder=7,
            )
            ax.annotate(
                f"{accuracies[-1]:.2f}%",
                xy=(steps[-1], accuracies[-1]),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=8,
                color=color_acc,
                va="center",
            )

        ax.set_title(
            f"Precisión con Banda de Confianza ±1σ (ventana de {self._config_metrics_window} pasos)"
        )
        ax.set_xlabel("Steps")
        ax.set_ylabel("Precisión (%)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Calcular ylim seguro según restricción
        try:
            ymin = float(np.min(lower)) * 0.998
            ymax = float(np.max(upper)) * 1.002
            ax.set_ylim(ymin, ymax)
        except Exception:
            ax.set_ylim(*ylim)

        try:
            ax.set_xlim(float(steps[0]), float(steps[-1]))
        except Exception:
            ax.set_xlim(*xlim)

        # Estadísticas sobre historial completo
        try:
            acc_std_arr = (
                np.array(self._full_acc_std) if len(self._full_acc_std) > 0 else acc_std
            )
            acc_std_mean = float(np.mean(acc_std_arr)) if len(acc_std_arr) > 0 else 0.0
            acc_std_final = float(acc_std_arr[-1]) if len(acc_std_arr) > 0 else 0.0
            text_box = f"σ medio: {acc_std_mean:.4f}%\nσ final: {acc_std_final:.4f}%"
            # Agregar nota si la banda es menor a 1px a esta escala
            if acc_std_final < 1.0 and acc_std_final > 0:
                text_box += "\n(Banda < 1px a esta escala)"
            ax.text(
                0.02,
                0.98,
                text_box,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
            )
        except Exception:
            pass

        # Líneas ±1σ en el final y eje secundario para σ
        try:
            if len(acc_std) > 0 and not np.allclose(acc_std, 0.0):
                y_upper = float(accuracies[-1] + acc_std[-1])
                y_lower = float(accuracies[-1] - acc_std[-1])
                ax.axhline(
                    y_upper, color="gray", linestyle=":", linewidth=0.8, alpha=0.6
                )
                ax.axhline(
                    y_lower, color="gray", linestyle=":", linewidth=0.8, alpha=0.6
                )
                try:
                    # Calcular si las etiquetas están muy cercanas (< 3% del rango Y)
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    too_close = abs(y_upper - y_lower) < y_range * 0.03

                    if too_close:
                        # Si están muy cercas, usar offset horizontal
                        offsets = [(-70, -10), (-70, 10)]  # -1σ arriba, +1σ abajo
                    else:
                        # Si no, usar offset normal
                        offsets = [(-60, 4), (-60, 4)]

                    # Posicionar anotaciones ±1σ
                    for (y_val, label), (x_off, y_off) in zip(
                        [
                            (y_lower, f"-1σ: {y_lower:.2f}%"),
                            (y_upper, f"+1σ: {y_upper:.2f}%"),
                        ],
                        offsets,
                    ):
                        ax.annotate(
                            label,
                            xy=(steps[-1], y_val),
                            xytext=(x_off, y_off),
                            textcoords="offset points",
                            fontsize=7.5,
                            color="gray",
                            va="center",
                        )
                except Exception:
                    pass

                try:
                    ax2 = ax.twinx()
                    ax2.plot(
                        steps,
                        acc_std,
                        color="gray",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.6,
                        label="σ (eje der.)",
                    )
                    ax2.set_ylabel("σ Accuracy (%)", fontsize=9, color="gray")
                    ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
                    # Calcular ylim del eje secundario para evitar cruce con banda
                    try:
                        band_min = float(np.min(accuracies - acc_std))
                        sigma_max = float(np.max(acc_std))
                        ax_ylim_min = ax.get_ylim()[0]
                        ax2_scale = (band_min - ax_ylim_min) / sigma_max * 0.85
                        if ax2_scale > 0:
                            ax2.set_ylim(0, sigma_max / ax2_scale)
                        else:
                            ax2.set_ylim(0, sigma_max * 4.0)
                    except Exception:
                        ax2.set_ylim(0, float(np.max(acc_std)) * 2.5)
                    # Ocultar ticks del eje secundario para evitar clutter
                    ax2.set_yticks([])
                    ax2.grid(False)
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
                except Exception:
                    pass

        except Exception:
            pass

        if len(steps) > 0:
            ax.set_xlim(float(steps[0]), float(steps[-1] + x_margin))
        else:
            ax.set_xlim(*xlim)

        # Marcar máximo (precisión)
        try:
            if len(accuracies) > 0:
                idx_max = int(np.argmax(accuracies))
                step_range = steps[-1] - steps[0] if len(steps) > 1 else 1
                # Si el máximo está muy cerca del final, mover etiqueta a la izquierda
                if (steps[-1] - steps[idx_max]) < step_range * 0.05:
                    xytext_max = (-50, 20)  # arriba-izquierda
                else:
                    xytext_max = (8, 15)  # estándar

                ax.annotate(
                    f"Máx: {accuracies[idx_max]:.2f}%",
                    xy=(steps[idx_max], accuracies[idx_max]),
                    xytext=xytext_max,
                    textcoords="offset points",
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8),
                    color="darkgreen",
                )
        except Exception:
            pass

        plt.tight_layout()
        output_path = self.session_dir / "plot_band_acc.png"
        plt.savefig(output_path, dpi=300, pad_inches=0.4)
        plt.close()

    def _plot_individual_workers(
        self, steps: np.ndarray, workers_count: np.ndarray
    ) -> None:
        """Genera gráfica individual de Workers (estilo idéntico a la GUI)."""
        fig, ax = plt.subplots(figsize=(10, 6.5))

        color_workers = COLORS["workers"]

        ax.step(steps, workers_count, color=color_workers, lw=2)
        ax.set_title("Workers activos")
        ax.set_xlabel("Steps")
        ax.set_ylabel("N Workers")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(workers_count) + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Establecer límites de ejes para que coincidan con GUI
        ax.set_xlim(steps[0] - 1, steps[-1] + 1)

        plt.tight_layout()
        output_path = self.session_dir / "plot_workers.png"
        plt.savefig(output_path, dpi=300, pad_inches=0.4)
        plt.close()

    def _plot_staleness(
        self,
        steps: np.ndarray,
        staleness: np.ndarray,
        alpha: np.ndarray,
        event_steps: List[int],
        event_types: List[str],
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

        ax1.plot(steps, staleness, color=COLORS["info"], lw=2, label="Staleness")
        ax1.set_title("Staleness por Step")
        ax1.set_ylabel("Staleness (versiones de retraso)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, alpha, color=COLORS["warning"], lw=2, label="Alpha")
        ax2.axhline(1.0, color="#757575", linestyle=":", lw=1.5, label="alpha=1.0")
        ax2.set_title("Factor de Corrección α = 1/(1+λ·s)")
        ax2.set_ylabel("α")
        ax2.set_xlabel("Steps")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

        # Establecer xlim completo según historial
        try:
            ax1.set_xlim(float(steps[0]), float(steps[-1]))
            ax2.set_xlim(float(steps[0]), float(steps[-1]))
        except Exception:
            pass

        self._add_event_markers((ax1, ax2), event_steps, event_types)
        if event_steps:
            ax1.legend(fontsize=8)
            ax2.legend(fontsize=8)

        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        output_path = self.session_dir / "plot_staleness.png"
        plt.savefig(output_path, dpi=300, pad_inches=0.4)
        plt.close()

    def _plot_std(
        self,
        steps: np.ndarray,
        loss_std: np.ndarray,
        acc_std: np.ndarray,
        event_steps: List[int],
        event_types: List[str],
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

        ax1.plot(steps, loss_std, color=COLORS["loss"], lw=2, label="σ Loss")
        ax1.set_title(
            f"Variabilidad de Loss — σ (ventana de {self._config_metrics_window} pasos)"
        )
        ax1.set_ylabel("σ Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, acc_std, color=COLORS["accuracy"], lw=2, label="σ Accuracy")
        ax2.set_title(
            f"Variabilidad de Accuracy — σ (ventana de {self._config_metrics_window} pasos)"
        )
        ax2.set_ylabel("σ Accuracy (%)")
        ax2.set_xlabel("Steps")
        ax2.grid(True, alpha=0.3)

        # Establecer xlim completo según historial
        try:
            ax1.set_xlim(float(steps[0]), float(steps[-1]))
            ax2.set_xlim(float(steps[0]), float(steps[-1]))
        except Exception:
            pass

        self._add_event_markers((ax1, ax2), event_steps, event_types)
        if event_steps:
            ax1.legend(fontsize=8)
            ax2.legend(fontsize=8)

        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        output_path = self.session_dir / "plot_std.png"
        plt.savefig(output_path, dpi=300, pad_inches=0.4)
        plt.close()

    def _write_metadata(self) -> None:
        """Escribe estadísticas finales en metadata.json."""
        if len(self._full_losses) == 0 and len(self._metrics_loss) == 0:
            metadata = {"status": "no_metrics_recorded"}
        else:
            with self._lock:
                if len(self._full_losses) > 0:
                    loss_hist = np.array(list(self._full_losses))
                    acc_hist = np.array(list(self._full_accuracies))
                    workers_hist = np.array(list(self._full_workers))
                    elapsed_hist = np.array(list(self._full_elapsed))
                    staleness_hist = np.array(list(self._full_staleness))
                    alpha_hist = np.array(list(self._full_alpha))
                    loss_std_hist = np.array(list(self._full_loss_std))
                    acc_std_hist = np.array(list(self._full_acc_std))
                    steps_arr = np.array(list(self._full_steps))
                else:
                    loss_hist = np.array(list(self._metrics_loss))
                    acc_hist = np.array(list(self._metrics_accuracy))
                    workers_hist = np.array(list(self._metrics_workers))
                    elapsed_hist = np.array(list(self._metrics_elapsed))
                    staleness_hist = np.array(list(self._metrics_staleness))
                    alpha_hist = np.array(list(self._metrics_alpha))
                    loss_std_hist = np.array(list(self._metrics_loss_std))
                    acc_std_hist = np.array(list(self._metrics_acc_std))
                    steps_arr = np.array(list(self._metrics_steps))

            duration = float(elapsed_hist[-1]) if len(elapsed_hist) > 0 else 0.0
            batch_size = self.config.get("batch_size", 0)

            staleness_arr = staleness_hist
            alpha_arr = alpha_hist
            loss_std_arr = loss_std_hist
            acc_std_arr = acc_std_hist
            acc_arr = acc_hist
            elapsed_arr = elapsed_hist
            workers_arr = workers_hist

            staleness_analysis = {
                "mean": float(np.mean(staleness_arr))
                if len(staleness_arr) > 0
                else 0.0,
                "max": int(np.max(staleness_arr)) if len(staleness_arr) > 0 else 0,
                "std": float(np.std(staleness_arr)) if len(staleness_arr) >= 2 else 0.0,
                "alpha_mean": float(np.mean(alpha_arr)) if len(alpha_arr) > 0 else 1.0,
                "alpha_min": float(np.min(alpha_arr)) if len(alpha_arr) > 0 else 1.0,
                "alpha_std": float(np.std(alpha_arr)) if len(alpha_arr) >= 2 else 0.0,
            }

            delta_steps = np.diff(steps_arr)
            delta_time = np.diff(elapsed_arr)
            delta_time[delta_time == 0] = 1e-6
            steps_per_sec = delta_steps / delta_time
            throughput = {
                "mean_steps_per_second": float(np.mean(steps_per_sec))
                if len(steps_per_sec) > 0
                else 0.0,
                "peak_steps_per_second": float(np.max(steps_per_sec))
                if len(steps_per_sec) > 0
                else 0.0,
                "min_steps_per_second": float(np.min(steps_per_sec))
                if len(steps_per_sec) > 0
                else 0.0,
                "mean_images_per_second": float(np.mean(steps_per_sec) * batch_size)
                if len(steps_per_sec) > 0
                else 0.0,
                "peak_images_per_second": float(np.max(steps_per_sec) * batch_size)
                if len(steps_per_sec) > 0
                else 0.0,
                "samples_processed": int(steps_arr[-1] * batch_size)
                if len(steps_arr) > 0
                else 0,
            }

            accuracy_thresholds = [5, 10, 20, 30, 40, 50]
            convergence = {}
            for thresh in accuracy_thresholds:
                mask = acc_arr >= thresh
                if np.any(mask):
                    idx = np.argmax(mask)
                    convergence[f"steps_to_{thresh}_percent_accuracy"] = int(
                        steps_arr[idx]
                    )
                    convergence[f"time_to_{thresh}_percent_accuracy_seconds"] = float(
                        elapsed_arr[idx]
                    )
                else:
                    convergence[f"steps_to_{thresh}_percent_accuracy"] = None
                    convergence[f"time_to_{thresh}_percent_accuracy_seconds"] = None

            stability = {
                "loss_std_mean": float(np.mean(loss_std_arr))
                if len(loss_std_arr) > 0
                else 0.0,
                "loss_std_final": float(loss_std_arr[-1])
                if len(loss_std_arr) > 0
                else 0.0,
                "acc_std_mean": float(np.mean(acc_std_arr))
                if len(acc_std_arr) > 0
                else 0.0,
                "acc_std_final": float(acc_std_arr[-1])
                if len(acc_std_arr) > 0
                else 0.0,
            }

            max_simultaneous = int(np.max(workers_arr)) if len(workers_arr) > 0 else 0
            total_ever_connected = len(
                [e for e in self._worker_events if e["event_type"] == "connected"]
            )
            total_disconnections = len(
                [e for e in self._worker_events if e["event_type"] == "disconnected"]
            )

            workers_summary = {
                "max_simultaneous": max_simultaneous,
                "total_ever_connected": total_ever_connected,
                "total_disconnections": total_disconnections,
            }

            updates_dict = {
                "total_applied": int(steps_arr[-1]) if len(steps_arr) > 0 else 0,
                "total_rejected_nan": int(self.nan_rejected_count),
            }

            mean_tcp_requests_per_second = (
                float(self.tcp_request_count / duration) if duration > 0 else 0.0
            )

            training_summary = {
                "total_steps": int(steps_arr[-1]) if len(steps_arr) > 0 else 0,
                "total_samples_processed": int(steps_arr[-1] * batch_size)
                if len(steps_arr) > 0
                else 0,
                "total_wall_time_seconds": duration,
                "effective_training_seconds": duration,
                "mean_tcp_requests_per_second": mean_tcp_requests_per_second,
                "total_tcp_requests": int(self.tcp_request_count),
            }

            metadata = {
                "status": "completed",
                "session_timestamp": self.timestamp,
                "total_steps": int(steps_arr[-1]) if len(steps_arr) > 0 else 0,
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
                "staleness_analysis": staleness_analysis,
                "throughput": throughput,
                "convergence": convergence,
                "stability": stability,
                "updates": updates_dict,
                "workers_summary": workers_summary,
                "training_summary": training_summary,
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
