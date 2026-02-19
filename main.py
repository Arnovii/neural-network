"""
Punto de entrada principal

Ejecuta experimentos de Algoritmo de Diego con redes neuronales
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
# MODO TERMINAL
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

    # Muestra resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS")
    print("=" * 70)
    print(f"Precisión final promedio: {stats['mean'][-1]:.2f}%")
    print(f"Desviación estándar final: {stats['std'][-1]:.2f}%")
    print(f"Mejor precisión alcanzada: {max(stats['max']):.2f}%")

    # Guarda resultados
    output_file = f"Results/experiment_{results['timestamp']}.json"
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
    
    # Dibuja los mensajes emergentes al pasar el mouse
    class ToolTip:
        def __init__(self, widget, text):
            self.widget = widget
            self.text = text
            self.tip_window = None

            widget.bind("<Enter>", self.show_tip)
            widget.bind("<Leave>", self.hide_tip)

        def show_tip(self, event=None):
            if self.tip_window or not self.text:
                return

            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + 20

            self.tip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")

            label = tk.Label(
                tw,
                text=self.text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("Helvetica", 9),
            )
            label.pack(ipadx=5, ipady=3)

        def hide_tip(self, event=None):
            if self.tip_window:
                self.tip_window.destroy()
                self.tip_window = None

    # Aplicación principal
    class FederatedLearningApp:
        def __init__(self, root):
            self.root = root

            # Obtiene la resolución de la pantalla
            ancho = root.winfo_screenwidth()
            alto = root.winfo_screenheight()

            # Establece el título
            self.root.title("NN_practica - Análisis de  Algoritmo de Diego")
            
            # Configura geometry con el formato "ancho x alto + 0 + 0"
            root.geometry(f"{ancho}x{alto}+0+0")

            # Datos de ejecuciones previas para comparación
            self.previous_results = []
            self.colors = [
                "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
                "#00BCD4", "#FFEB3B", "#795548", "#607D8B", "#E91E63",
            ]
            self.color_index = 0
            
            # Inicializa atributos para resultados de experimentos
            self.current_results = None
            self.current_params = None

            self._create_ui()

        # ================
        # CONSTRUCCIÓN UI
        # ================

        def _create_ui(self):

            # Helper: fuerza valores enteros en los sliders
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
                control_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                variable=self.partitions_var, length=200,
                command=snap_int(self.partitions_var),
            ).pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.partitions_var).pack()

            # Número de épocas
            ttk.Label(control_frame, text="Épocas:").pack(anchor=tk.W, pady=(10, 0))
            self.epochs_var = tk.IntVar(value=5)
            ttk.Scale(
                control_frame, from_=1, to=50, orient=tk.HORIZONTAL,
                variable=self.epochs_var, length=200,
                command=snap_int(self.epochs_var),
            ).pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.epochs_var).pack()

            # Número de experimentos
            ttk.Label(control_frame, text="Experimentos:").pack(anchor=tk.W, pady=(10, 0))
            self.experiments_var = tk.IntVar(value=5)
            ttk.Scale(
                control_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                variable=self.experiments_var, length=200,
                command=snap_int(self.experiments_var),
            ).pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.experiments_var).pack()

            # Neuronas ocultas
            ttk.Label(control_frame, text="Neuronas de la capa oculta:").pack(anchor=tk.W, pady=(10, 0))
            self.hidden_var = tk.IntVar(value=30)
            ttk.Scale(
                control_frame, from_=10, to=100, orient=tk.HORIZONTAL,
                variable=self.hidden_var, length=200,
                command=snap_int(self.hidden_var),
            ).pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.hidden_var).pack()

            # Tasa de aprendizaje
            ttk.Label(control_frame, text="Tasa de aprendizaje:").pack(anchor=tk.W, pady=(10, 0))
            self.lr_var = tk.StringVar(value="1.0")
            ttk.Entry(control_frame, textvariable=self.lr_var, width=20).pack(pady=5)

            # Ejemplos de entrenamiento
            ttk.Label(control_frame, text="Ejemplos de entrenamiento:").pack(anchor=tk.W, pady=(10, 0))
            self.n_train_var = tk.StringVar(value="5000")
            ttk.Entry(control_frame, textvariable=self.n_train_var, width=20).pack(pady=5)

            # Botones
            ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

            btn_run = ttk.Button(
                control_frame,
                text="Ejecutar Experimento",
                command=self._run_experiment,
            )
            btn_run.pack(fill=tk.X, pady=5)
            ToolTip(btn_run, "Ejecuta el algoritmo con los parámetros actuales")

            btn_compare = ttk.Button(
                control_frame,
                text="Comparar Configuraciones",
                command=self._compare_configurations,
            )
            btn_compare.pack(fill=tk.X, pady=5)
            ToolTip(
                btn_compare,
                "Ejecuta un nuevo experimento y lo superpone\n"
                "con los resultados actualmente mostrados."
            )

            btn_clear = ttk.Button(
                control_frame,
                text="Limpiar Gráficos",
                command=self._clear_plots,
            )
            btn_clear.pack(fill=tk.X, pady=5)
            ToolTip(btn_clear, "Borra todas las gráficas actuales")

            btn_save = ttk.Button(
                control_frame,
                text="Guardar Resultados",
                command=self._save_results,
            )
            btn_save.pack(fill=tk.X, pady=5)
            ToolTip(btn_save, "Guarda los resultados actuales en un archivo JSON")

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
                self.root, textvariable=self.status_var,
                relief=tk.SUNKEN, anchor=tk.W,
            ).pack(side=tk.BOTTOM, fill=tk.X)

        # ================================
        # VENTANA DE PROGRESO
        # ================================

        def _create_progress_window(self, num_experiments: int, num_epochs: int):
            """
            Crea una ventana modal que muestra el progreso del entrenamiento.

            La ventana se mantiene responsive porque el entrenamiento corre
            en un hilo separado y se comunica mediante una cola.

            :param num_experiments: Total de experimentos a ejecutar
            :type num_experiments: int
            :param num_epochs: Total de épocas por experimento
            :type num_epochs: int
            :return: Diccionario con los widgets de la ventana de progreso
            :rtype: dict
            """
            import tkinter as tk
            from tkinter import ttk

            win = tk.Toplevel(self.root)
            win.title("Ejecutando experimentos...")
            win.geometry("520x320")
            win.resizable(False, False)
            win.grab_set()  # Modal: bloquea interacción con la ventana principal
            win.protocol("WM_DELETE_WINDOW", lambda: None)  # Deshabilita el cierre manual

            # Título
            ttk.Label(
                win, text="Entrenamiento en progreso",
                font=("Helvetica", 13, "bold"),
            ).pack(pady=(18, 4))

            # Experimento actual
            exp_label_var = tk.StringVar(value="Inicializando...")
            ttk.Label(win, textvariable=exp_label_var, font=("Helvetica", 10)).pack(pady=2)

            # Barra de progreso general (experimentos)
            ttk.Label(win, text="Progreso general:").pack(anchor=tk.W, padx=24, pady=(10, 0))
            exp_progress = ttk.Progressbar(
                win, orient=tk.HORIZONTAL, length=470,
                mode="determinate", maximum=num_experiments,
            )
            exp_progress.pack(padx=24, pady=4)

            # Barra de épocas del experimento actual
            ttk.Label(win, text="Época actual:").pack(anchor=tk.W, padx=24, pady=(8, 0))
            epoch_progress = ttk.Progressbar(
                win, orient=tk.HORIZONTAL, length=470,
                mode="determinate", maximum=num_epochs,
            )
            epoch_progress.pack(padx=24, pady=4)

            # Último mensaje recibido
            ttk.Label(win, text="Último estado:").pack(anchor=tk.W, padx=24, pady=(8, 0))
            msg_var = tk.StringVar(value="—")
            ttk.Label(
                win, textvariable=msg_var,
                font=("Helvetica", 9), foreground="#555555",
                wraplength=470, justify=tk.LEFT,
            ).pack(anchor=tk.W, padx=24)

            return {
                "window": win,
                "exp_label_var": exp_label_var,
                "exp_progress": exp_progress,
                "epoch_progress": epoch_progress,
                "msg_var": msg_var,
            }

        def _poll_progress_queue(self, progress_queue, widgets, num_experiments, num_epochs, on_done):
            """
            Revisa la cola de progreso periódicamente desde el hilo principal.

            Tkinter no es thread-safe, por lo que el hilo de entrenamiento NUNCA
            toca los widgets directamente. Solo escribe en la cola, y este método
            — que siempre corre en el hilo principal — lee la cola y actualiza la UI.

            Tipos de mensajes en la cola:
              ('exp',   número_experimento) → avanza barra de experimentos
              ('epoch', número_época)        → avanza barra de épocas
              ('msg',   texto)               → actualiza etiqueta de estado
              ('done',  resultado)           → entrenamiento terminado exitosamente
              ('error', excepción)           → ocurrió un error en el hilo

            :param progress_queue: Cola compartida entre hilo principal e hilo de entrenamiento
            :param widgets: Diccionario con los widgets de la ventana de progreso
            :param num_experiments: Total de experimentos (para calcular porcentaje)
            :param num_epochs: Total de épocas (para calcular porcentaje)
            :param on_done: Callback llamado con el resultado cuando termina con éxito
            """
            import queue as queue_module

            try:
                # Drena todos los mensajes disponibles en este ciclo de polling
                while True:
                    msg_type, payload = progress_queue.get_nowait()

                    if msg_type == "exp":
                        # Nuevo experimento iniciado
                        widgets["exp_label_var"].set(
                            f"Experimento {payload} de {num_experiments}"
                        )
                        widgets["exp_progress"]["value"] = payload - 1
                        # Reinicia barra de épocas para el nuevo experimento
                        widgets["epoch_progress"]["value"] = 0

                    elif msg_type == "epoch":
                        # Época completada dentro del experimento actual
                        widgets["epoch_progress"]["value"] = payload

                    elif msg_type == "msg":
                        # Mensaje informativo de estado
                        widgets["msg_var"].set(payload)

                    elif msg_type == "done":
                        # Entrenamiento completado exitosamente
                        widgets["exp_progress"]["value"] = num_experiments
                        widgets["epoch_progress"]["value"] = num_epochs
                        widgets["window"].destroy()
                        on_done(payload)
                        return  # Detiene el polling

                    elif msg_type == "error":
                        widgets["window"].destroy()
                        raise payload

            except queue_module.Empty:
                pass  # No hay mensajes nuevos; es normal, seguimos esperando

            except Exception as e:
                widgets["window"].destroy()
                from tkinter import messagebox
                messagebox.showerror("Error", f"Error ejecutando experimento:\n{str(e)}")
                self.status_var.set("Error en ejecución")
                return

            # Reprograma el siguiente ciclo de polling cada 100ms
            self.root.after(100, lambda: self._poll_progress_queue(
                progress_queue, widgets, num_experiments, num_epochs, on_done
            ))

        # ==================
        # ACCIONES DE BOTONES
        # ==================

        def _run_experiment(self):
            """
            Ejecuta un experimento con los parámetros actuales.

            El entrenamiento se lanza en un hilo secundario para que la GUI
            permanezca responsive. La comunicación entre hilos se hace mediante
            queue.Queue, que es la única estructura thread-safe de Python estándar
            recomendada para interactuar con tkinter desde otro hilo.
            """
            import threading
            import queue

            try:
                params = {
                    "num_partitions": self.partitions_var.get(),
                    "num_epochs": self.epochs_var.get(),
                    "num_experiments": self.experiments_var.get(),
                    "hidden_neurons": self.hidden_var.get(),
                    "learning_rate": float(self.lr_var.get()),
                    "n_train": int(self.n_train_var.get()),
                    "verbose": False,
                }
            except ValueError as e:
                from tkinter import messagebox
                messagebox.showerror("Error de parámetros", str(e))
                return

            progress_queue = queue.Queue()
            num_experiments = params["num_experiments"]
            num_epochs = params["num_epochs"]

            # Crea la ventana de progreso antes de lanzar el hilo
            widgets = self._create_progress_window(num_experiments, num_epochs)
            self.status_var.set("Ejecutando experimentos...")

            # ── Hilo de entrenamiento ──────────────────────────────────────────
            # REGLA CLAVE: este hilo NUNCA toca widgets de tkinter.
            # Solo escribe en progress_queue con put(), que es thread-safe.
            def training_thread():
                current_exp = [0]  # Lista mutable para poder modificar en el closure

                def on_progress(msg: str):
                    """
                    Callback invocado desde experiment_runner y nn_diego.
                    Interpreta el texto para extraer el tipo de evento y lo encola.
                    """
                    # Siempre encola el mensaje de texto para mostrarlo
                    progress_queue.put(("msg", msg))

                    # Detecta inicio de nuevo experimento: "EXPERIMENTO X/Y"
                    if msg.startswith("EXPERIMENTO"):
                        try:
                            exp_num = int(msg.split()[1].split("/")[0])
                            current_exp[0] = exp_num
                            progress_queue.put(("exp", exp_num))
                        except (IndexError, ValueError):
                            pass

                    # Detecta fin de época: "[Época X/Y — Precisión: ...]"
                    elif msg.startswith("[Época"):
                        try:
                            epoch_num = int(msg.split()[1].split("/")[0])
                            progress_queue.put(("epoch", epoch_num))
                        except (IndexError, ValueError):
                            pass

                try:
                    results = run_multiple_experiments(**params, on_progress=on_progress)
                    progress_queue.put(("done", results))
                except Exception as e:
                    progress_queue.put(("error", e))

            # ── Callback al completar ──────────────────────────────────────────
            def on_training_done(results):
                self.current_results = results
                self.current_params = params
                self._plot_results(results, params, comparison=False)
                self.status_var.set(
                    f"Completado: Precisión final = {results['final_mean_accuracy']:.2f}%"
                )

            # Lanza el hilo como daemon para que no bloquee el cierre de la app
            thread = threading.Thread(target=training_thread, daemon=True)
            thread.start()

            # Inicia el ciclo de polling en el hilo principal (cada 100ms)
            self.root.after(100, lambda: self._poll_progress_queue(
                progress_queue, widgets, num_experiments, num_epochs, on_training_done
            ))

        def _compare_configurations(self):
            """Superpone el resultado actual con ejecuciones anteriores."""
            if not hasattr(self, "current_results"):
                from tkinter import messagebox
                messagebox.showwarning("Advertencia", "Primero ejecuta un experimento")
                return

            # Guarda configuración actual antes de ejecutar la nueva
            self.previous_results.append({
                "results": self.current_results,
                "params": self.current_params,
                "color": self.colors[self.color_index % len(self.colors)],
            })
            self.color_index += 1

            # Marca que al terminar se debe ejecutar _plot_comparison
            self._pending_comparison = True
            self._run_experiment()

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
            if not hasattr(self, "current_results") or self.current_results is None:
                from tkinter import messagebox
                messagebox.showwarning("Advertencia", "No hay resultados para guardar")
                return

            filename = f"Results/experiment_{self.current_results['timestamp']}.json"
            os.makedirs("Results", exist_ok=True)

            with open(filename, "w") as f:
                json.dump(
                    {
                        "parameters": self.current_params,
                        "results": {
                            "timestamp": self.current_results["timestamp"],
                            "final_mean_accuracy": self.current_results["final_mean_accuracy"],
                            "final_std_accuracy": self.current_results["final_std_accuracy"],
                            "all_histories": self.current_results["all_histories"],
                        },
                    },
                    f, indent=2,
                )

            from tkinter import messagebox
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
                acc_data["x"], acc_data["y_mean"], "o-",
                color=color, linewidth=2, label=label,
            )
            ax1.fill_between(
                acc_data["x"], acc_data["y_lower"], acc_data["y_upper"],
                alpha=0.2, color=color,
            )
            ax1.set_xlabel(acc_data["xlabel"])
            ax1.set_ylabel(acc_data["ylabel"])
            ax1.set_title("Evolución del Promedio (con desviación estándar)")
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)

            # --- Panel 2: Comparación por partición (último experimento) ---
            last_history = [histories[-1]] if histories else []
            part_data = prepare_partition_comparison_data(last_history)
            if part_data:
                for partition in part_data["partitions"]:
                    ax2.plot(
                        partition["x"], partition["y"], "o-",
                        label=f"Partición {partition['id']}", alpha=0.7,
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
                bin_centers, dist_data["counts"],
                width=(dist_data["bins"][1] - dist_data["bins"][0]) * 0.9,
                color=color, alpha=0.7, edgecolor="black",
            )
            ax3.axvline(
                dist_data["mean"], color="red", linestyle="--",
                linewidth=2, label="Promedio",
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

            # Si estaba pendiente una comparación, la ejecuta ahora que hay datos frescos
            if getattr(self, "_pending_comparison", False):
                self._pending_comparison = False
                self._plot_comparison()

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
            if hasattr(self, "current_results") and self.current_params is not None:
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
                    config["x"], config["y"], "o-",
                    color=color, linewidth=linewidth, linestyle=linestyle,
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


# ==========
# ENTRADA
# ==========


def main():
    parser = argparse.ArgumentParser(
        description="NN_practica - Análisis de Algoritmo de Diego para MNIST"
    )

    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Ejecutar en modo interactivo con interfaz gráfica",
    )
    parser.add_argument(
        "--partitions", "-p", type=int, default=2,
        help="Número de particiones (default: 2)",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=5,
        help="Número de épocas (default: 5)",
    )
    parser.add_argument(
        "--experiments", "-x", type=int, default=5,
        help="Número de experimentos a ejecutar (default: 5)",
    )
    parser.add_argument(
        "--hidden-neurons", "-n", type=int, default=30,
        help="Número de neuronas en capa oculta (default: 30)",
    )
    parser.add_argument(
        "--learning-rate", "-l", type=float, default=1.0,
        help="Tasa de aprendizaje (default: 1.0)",
    )
    parser.add_argument(
        "--n-train", type=int, default=5000,
        help="Número de ejemplos de entrenamiento (default: 5000)",
    )

    args = parser.parse_args()

    # Crea directorios necesarios
    os.makedirs("Data", exist_ok=True)
    os.makedirs("Results", exist_ok=True)

    if args.interactive:
        run_interactive_mode()
    else:
        run_terminal_mode(args)


if __name__ == "__main__":
    main()
