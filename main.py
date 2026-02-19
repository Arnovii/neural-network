"""
NN_practica - Punto de entrada principal

Ejecuta experimentos con el algoritmo
y genera visualizaciones estadísticas.
"""

# ================
# IMPORTACIONES
# ================

import sys
import os
import argparse
import json

# Asegura que los módulos locales se puedan importar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Analytics.experiment_runner import run_multiple_experiments
from Analytics.statistics_engine import compute_epoch_statistics
from Analytics.chart_generator import (
    prepare_accuracy_chart_data,
    prepare_partition_comparison_data,
    prepare_convergence_data,
    prepare_distribution_data,
    prepare_comparison_chart_data,
)

# ================
# MODO TÉRMINAL
# ================


def run_terminal_mode(args):
    """Ejecuta en modo terminal sin interfaz gráfica."""
    print("=" * 70)
    print("NN_PRACTICA - MODO TERMINAL")
    print("=" * 70)
    print(f"Particiones: {args.partitions}")
    print(f"Épocas: {args.epochs}")
    print(f"Experimentos: {args.experiments}")
    print(f"Neuronas ocultas: {args.hidden_neurons}")
    print(f"Tasa de aprendizaje: {args.learning_rate}")
    print(f"Ejemplos de entrenamiento: {args.n_train}")
    print("=" * 70)

    # Ejecuta experimentos
    results = run_multiple_experiments(
        num_partitions=args.partitions,
        num_epochs=args.epochs,
        num_experiments=args.experiments,
        hidden_neurons=args.hidden_neurons,
        learning_rate=args.learning_rate,
        n_train=args.n_train,
        verbose=True,
    )

    # Calcula estadísticas
    stats = compute_epoch_statistics(results["all_histories"])

    # Mostra resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS")
    print("=" * 70)
    print(f"Precisión final promedio: {stats['mean'][-1]:.2f}%")
    print(f"Desviación estándar final: {stats['std'][-1]:.2f}%")
    print(f"Mejor precisión alcanzada: {max(stats['max']):.2f}%")

    # Guarda resultados
    output_file = f"results/experiment_{results['timestamp']}.json"
    print(f"\nResultados guardados en: {output_file}")

    # Gráfico ASCII usando datos preparados por chart_generator
    acc_data = prepare_accuracy_chart_data(results["all_histories"])
    print("\nEvolución de precisión (promedio ± desv. estándar):")
    for epoch, (mean, std) in enumerate(zip(acc_data["y_mean"], acc_data["y_std"])):
        bar_length = int(mean / 2)
        bar = "█" * bar_length
        print(f"Época {epoch + 1:2d}: {mean:5.2f}% ± {std:4.2f}% {bar}")


# ====================
# MODO INTERACTIVO
# ====================


def run_interactive_mode():
    """Ejecuta la interfaz gráfica interactiva."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError as e:
        print(f"Error: No se pueden cargar las librerías gráficas: {e}")
        print("Instala las dependencias con: pip install matplotlib")
        sys.exit(1)

    class FederatedLearningApp:
        def __init__(self, root):
            self.root = root
            self.root.title("NN_practica - Análisis de  Algoritmo de Diego")
            self.root.geometry("1400x900")

            # Datos de ejecuciones previas para comparación
            self.previous_results = []
            self.colors = [
                "#2196F3",
                "#4CAF50",
                "#FF9800",
                "#9C27B0",
                "#F44336",
                "#00BCD4",
                "#FFEB3B",
                "#795548",
                "#607D8B",
                "#E91E63",
            ]
            self.color_index = 0

            self._create_ui()

        def _create_ui(self):

            # Función helper para redonder valores
            def snap_int(var):
                return lambda v: var.set(int(round(float(v))))
            
            # Panel izquierdo: Controles
            control_frame = ttk.Frame(self.root, padding="10")
            control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

            ttk.Label(
                control_frame, text="Parámetros", font=("Helvetica", 14, "bold")
            ).pack(pady=10)

            # Número de particiones
            ttk.Label(control_frame, text="Particiones:").pack(anchor=tk.W)
            self.partitions_var = tk.IntVar(value=2)
            ttk.Scale(
                control_frame,
                from_=1,
                to=10,
                orient=tk.HORIZONTAL,
                variable=self.partitions_var,
                length=200,
                command=snap_int(self.partitions_var)
            ).pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.partitions_var).pack()

            # Número de épocas
            ttk.Label(control_frame, text="Épocas:").pack(anchor=tk.W, pady=(10, 0))
            self.epochs_var = tk.IntVar(value=5)
            ttk.Scale(
                control_frame,
                from_=1,
                to=50,
                orient=tk.HORIZONTAL,
                variable=self.epochs_var,
                length=200,
                command=snap_int(self.epochs_var)
            ).pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.epochs_var).pack()

            # Número de experimentos
            ttk.Label(control_frame, text="Experimentos:").pack(
                anchor=tk.W, pady=(10, 0)
            )
            self.experiments_var = tk.IntVar(value=5)
            ttk.Scale(
                control_frame,
                from_=1,
                to=20,
                orient=tk.HORIZONTAL,
                variable=self.experiments_var,
                length=200,
                command=snap_int(self.experiments_var)
            ).pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.experiments_var).pack()

            # Neuronas ocultas
            ttk.Label(control_frame, text="Neuronas ocultas:").pack(
                anchor=tk.W, pady=(10, 0)
            )
            self.hidden_var = tk.IntVar(value=30)
            ttk.Scale(
                control_frame,
                from_=10,
                to=100,
                orient=tk.HORIZONTAL,
                variable=self.hidden_var,
                length=200,
                command=snap_int(self.hidden_var)
            ).pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.hidden_var).pack()

            # Tasa de aprendizaje
            ttk.Label(control_frame, text="Tasa de aprendizaje:").pack(
                anchor=tk.W, pady=(10, 0)
            )
            self.lr_var = tk.StringVar(value="1.0")
            ttk.Entry(control_frame, textvariable=self.lr_var, width=20).pack(pady=5)

            # Ejemplos de entrenamiento
            ttk.Label(control_frame, text="Ejemplos de entrenamiento:").pack(
                anchor=tk.W, pady=(10, 0)
            )
            self.n_train_var = tk.StringVar(value="5000")
            ttk.Entry(control_frame, textvariable=self.n_train_var, width=20).pack(
                pady=5
            )

            # Botones
            ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

            ttk.Button(
                control_frame,
                text="▶ Ejecutar Experimento",
                command=self._run_experiment,
            ).pack(fill=tk.X, pady=5)

            ttk.Button(
                control_frame,
                text="📊 Comparar Configuraciones",
                command=self._compare_configurations,
            ).pack(fill=tk.X, pady=5)

            ttk.Button(
                control_frame,
                text="🗑 Limpiar Gráficos",
                command=self._clear_plots,
            ).pack(fill=tk.X, pady=5)

            ttk.Button(
                control_frame,
                text="💾 Guardar Resultados",
                command=self._save_results,
            ).pack(fill=tk.X, pady=5)

            # Panel derecho: Gráficos
            self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
            self.fig.suptitle(
                "Análisis de Algoritmo de Diego", fontsize=14, fontweight="bold"
            )

            plot_frame = ttk.Frame(self.root)
            plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Barra de estado
            self.status_var = tk.StringVar(value="Listo")
            ttk.Label(
                self.root,
                textvariable=self.status_var,
                relief=tk.SUNKEN,
                anchor=tk.W,
            ).pack(side=tk.BOTTOM, fill=tk.X)

        # ==================
        # ACCIONES DE BOTONES
        # ==================

        def _run_experiment(self):
            """Ejecuta un experimento con los parámetros actuales."""
            try:
                self.status_var.set("Ejecutando experimentos...")
                self.root.update()

                params = {
                    "num_partitions": self.partitions_var.get(),
                    "num_epochs": self.epochs_var.get(),
                    "num_experiments": self.experiments_var.get(),
                    "hidden_neurons": self.hidden_var.get(),
                    "learning_rate": float(self.lr_var.get()),
                    "n_train": int(self.n_train_var.get()),
                    "verbose": False,
                }

                results = run_multiple_experiments(**params)

                self.current_results = results
                self.current_params = params

                self._plot_results(results, params, comparison=False)

                self.status_var.set(
                    f"Completado: Precisión final = {results['final_mean_accuracy']:.2f}%"
                )

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error ejecutando experimento:\n{str(e)}"
                )
                self.status_var.set("Error en ejecución")

        def _compare_configurations(self):
            """Superpone el resultado actual con ejecuciones anteriores."""
            if not hasattr(self, "current_results"):
                messagebox.showwarning("Advertencia", "Primero ejecuta un experimento")
                return

            # Guarda configuración actual antes de ejecutar la nueva
            self.previous_results.append(
                {
                    "results": self.current_results,
                    "params": self.current_params,
                    "color": self.colors[self.color_index % len(self.colors)],
                }
            )
            self.color_index += 1

            self._run_experiment()
            self._plot_comparison()

        def _clear_plots(self):
            """Limpia todos los gráficos."""
            for ax in self.axes.flatten():
                ax.clear()
            self.previous_results = []
            self.color_index = 0
            self.canvas.draw()
            self.status_var.set("Gráficos limpiados")

        def _save_results(self):
            """Guarda los resultados del experimento actual."""
            if not hasattr(self, "current_results"):
                messagebox.showwarning("Advertencia", "No hay resultados para guardar")
                return

            filename = f"results/experiment_{self.current_results['timestamp']}.json"
            os.makedirs("results", exist_ok=True)

            with open(filename, "w") as f:
                json.dump(
                    {
                        "parameters": self.current_params,
                        "results": {
                            "timestamp": self.current_results["timestamp"],
                            "final_mean_accuracy": self.current_results[
                                "final_mean_accuracy"
                            ],
                            "final_std_accuracy": self.current_results[
                                "final_std_accuracy"
                            ],
                            "all_histories": self.current_results["all_histories"],
                        },
                    },
                    f,
                    indent=2,
                )

            messagebox.showinfo("Éxito", f"Resultados guardados en:\n{filename}")
            self.status_var.set(f"Guardado: {filename}")

        # ==================
        # RENDERIZADO
        # ==================

        def _plot_results(self, results, params, comparison=False):
            """
            Visualiza los resultados de un experimento.

            La preparación de datos la delega completamente a chart_generator.
            Este método solo se encarga de dibujar con matplotlib.
            """
            ax1, ax2, ax3, ax4 = self.axes.flatten()
            color = self.colors[self.color_index % len(self.colors)]
            label = f"P={params['num_partitions']}, E={params['num_epochs']}"

            if not comparison:
                for ax in self.axes.flatten():
                    ax.clear()

            histories = results["all_histories"]

            # --- Panel 1: Evolución del promedio con banda de desviación estándar ---
            acc_data = prepare_accuracy_chart_data(histories)
            ax1.plot(
                acc_data["x"],
                acc_data["y_mean"],
                "o-",
                color=color,
                linewidth=2,
                label=label,
            )
            ax1.fill_between(
                acc_data["x"],
                acc_data["y_lower"],
                acc_data["y_upper"],
                alpha=0.2,
                color=color,
            )
            ax1.set_xlabel(acc_data["xlabel"])
            ax1.set_ylabel(acc_data["ylabel"])
            ax1.set_title("Evolución del Promedio (con desviación estándar)")
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)

            # --- Panel 2: Comparación por partición (último experimento) ---
            # Se filtra el último historial individualmente para ver particiones
            last_history = [histories[-1]] if histories else []
            part_data = prepare_partition_comparison_data(last_history)
            if part_data:
                for partition in part_data["partitions"]:
                    ax2.plot(
                        partition["x"],
                        partition["y"],
                        "o-",
                        label=f"Partición {partition['id']}",
                        alpha=0.7,
                    )
            ax2.set_xlabel(part_data.get("xlabel", "Época"))
            ax2.set_ylabel(part_data.get("ylabel", "Precisión (%)"))
            ax2.set_title("Comparación por Partición (último experimento)")
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)

            # --- Panel 3: Distribución de precisión final ---
            dist_data = prepare_distribution_data(histories)
            bin_centers = [
                (dist_data["bins"][i] + dist_data["bins"][i + 1]) / 2
                for i in range(len(dist_data["bins"]) - 1)
            ]
            ax3.bar(
                bin_centers,
                dist_data["counts"],
                width=(dist_data["bins"][1] - dist_data["bins"][0]) * 0.9,
                color=color,
                alpha=0.7,
                edgecolor="black",
            )
            ax3.axvline(
                dist_data["mean"],
                color="red",
                linestyle="--",
                linewidth=2,
                label="Promedio",
            )
            ax3.set_xlabel(dist_data["xlabel"])
            ax3.set_ylabel(dist_data["ylabel"])
            ax3.set_title(dist_data["title"])
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # --- Panel 4: Mejora por época ---
            conv_data = prepare_convergence_data(histories)
            ax4.bar(conv_data["x"], conv_data["y"], color=color, alpha=0.7)
            ax4.axhline(0, color="black", linestyle="-", linewidth=0.5)
            ax4.set_xlabel(conv_data["xlabel"])
            ax4.set_ylabel(conv_data["ylabel"])
            ax4.set_title(conv_data["title"])
            ax4.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw()

        def _plot_comparison(self):
            """
            Superpone curvas de precisión de múltiples configuraciones en el panel 1.

            Delega la preparación de datos a prepare_comparison_chart_data.
            """
            if not self.previous_results:
                return

            ax1 = self.axes.flatten()[0]
            ax1.clear()

            # Construye lista unificada de resultados y etiquetas para chart_generator
            all_results = [p["results"] for p in self.previous_results]
            labels = [
                f"P={p['params']['num_partitions']}, E={p['params']['num_epochs']}"
                for p in self.previous_results
            ]

            # Agrega configuración actual
            if hasattr(self, "current_results"):
                all_results.append(self.current_results)
                labels.append(
                    f"P={self.current_params['num_partitions']}, "
                    f"E={self.current_params['num_epochs']} (actual)"
                )

            comp_data = prepare_comparison_chart_data(all_results, labels)

            for i, config in enumerate(comp_data["configurations"]):
                color = self.colors[i % len(self.colors)]
                linestyle = "--" if i == len(comp_data["configurations"]) - 1 else "-"
                linewidth = 3 if linestyle == "--" else 2
                ax1.plot(
                    config["x"],
                    config["y"],
                    "o-",
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    label=f"{config['label']} ({config['final_accuracy']:.1f}%)",
                )

            ax1.set_xlabel(comp_data["xlabel"])
            ax1.set_ylabel(comp_data["ylabel"])
            ax1.set_title(comp_data["title"])
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)

            self.canvas.draw()

    # Crea y ejecuta aplicación
    root = tk.Tk()
    app = FederatedLearningApp(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="NN_practica - Análisis de Algoritmo de Diego para MNIST"
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Ejecutar en modo interactivo con interfaz gráfica",
    )
    parser.add_argument(
        "--partitions",
        "-p",
        type=int,
        default=2,
        help="Número de particiones (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=5,
        help="Número de épocas (default: 5)",
    )
    parser.add_argument(
        "--experiments",
        "-x",
        type=int,
        default=5,
        help="Número de experimentos a ejecutar (default: 5)",
    )
    parser.add_argument(
        "--hidden-neurons",
        "-n",
        type=int,
        default=30,
        help="Número de neuronas en capa oculta (default: 30)",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        type=float,
        default=1.0,
        help="Tasa de aprendizaje (default: 1.0)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=5000,
        help="Número de ejemplos de entrenamiento (default: 5000)",
    )

    args = parser.parse_args()

    # Crea directorios necesarios
    os.makedirs("Data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.interactive:
        run_interactive_mode()
    else:
        run_terminal_mode(args)


if __name__ == "__main__":
    main()
