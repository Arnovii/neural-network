"""
main.py
Punto de entrada principal.

Ejecuta experimentos de Algoritmo de Diego con redes neuronales
y genera visualizaciones estadísticas con gráficas interactivas.

MODOS DE USO:
    * Modo terminal (sin GUI):      python main.py
    * Modo interactivo con GUI:     python main.py --interactive

──────────────────────────────────────────────────────────────────
ARQUITECTURA DE HILOS PARA LA UI
──────────────────────────────────────────────────────────────────
Tkinter no es thread-safe: ningún widget puede tocarse desde un
hilo secundario. La solución es un patrón productor/consumidor:

  ┌────────────────────────┐         ┌──────────────────────────┐
  │   HILO PRINCIPAL       │         │   HILO SECUNDARIO        │
  │   (Tkinter event loop) │         │   (Entrenamiento)        │
  │                        │         │                          │
  │  - Dibuja interfaz     │◄────────│  run_multiple_experiments│
  │  - Lee cola cada 100ms │ Queue() │                          │
  │  - Actualiza barras    │         │  q.put("epoch", n)       │
  └────────────────────────┘         │  q.put("msg",   texto)   │
                                     │  q.put("done",  result)  │
                                     └──────────────────────────┘

Tipos de mensaje en la cola:
    ("exp", n)         → avanza barra de experimentos
    ("epoch", n)       → avanza barra de épocas
    ("msg", texto)     → actualiza etiqueta de estado
    ("done", result)   → entrenamiento terminado, cierra ventana
    ("error", exc)     → excepción en el hilo secundario

Reglas fundamentales:
    1. Tkinter solo puede modificarse desde el hilo principal.
    2. El entrenamiento pesado nunca corre en el hilo principal.
    3. La comunicación entre hilos se hace exclusivamente con Queue.
"""

# ================
# IMPORTACIONES
# ================

import argparse
import json
import os
import queue
import sys
import threading
import numpy as np

# Asegura que los módulos locales se puedan importar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Analytics.experiment_runner import (
    run_multiple_experiments,
    run_benchmark_comparison,
)
from Analytics.statistics_engine import compute_epoch_statistics
from Analytics.chart_generator import (
    prepare_accuracy_chart_data,
    prepare_benchmark_data,
    prepare_comparison_chart_data,
    prepare_convergence_data,
    prepare_experiment_rsd_data,
    prepare_partition_comparison_data,
)
from Parallel.core_validator import get_physical_cores


# ================
# MODO TERMINAL
# ================


def run_terminal_mode(args: argparse.Namespace) -> None:
    """
    Ejecuta experimentos y muestra resultados en consola.

    Con ``--benchmark`` ejecuta ambos modos y muestra la comparación.
    Con ``--parallel`` ejecuta solo el modo paralelo.

    :param args: Argumentos parseados por argparse
    :type args: argparse.Namespace
    """
    print("=" * 70)
    print("RED NEURONAL PRÁCTICA — MODO TERMINAL")
    print("=" * 70)
    print(f"Particiones         : {args.partitions}")
    print(f"Épocas              : {args.epochs}")
    print(f"Experimentos        : {args.experiments}")
    print(f"Neuronas ocultas    : {args.hidden_neurons}")
    print(f"Tasa de aprendizaje : {args.learning_rate}")
    print(f"Ejemplos            : {args.n_train}")
    print(
        f"Modo                : {'BENCHMARK (ambos)' if args.benchmark else 'PARALELO' if args.parallel else 'SECUENCIAL'}"
    )
    print("=" * 70)

    base_params = dict(
        num_partitions=args.partitions,
        num_epochs=args.epochs,
        num_experiments=args.experiments,
        hidden_neurons=args.hidden_neurons,
        learning_rate=args.learning_rate,
        n_train=args.n_train,
        verbose=True,
    )

    if args.benchmark:
        # Ejecuta ambos modos y muestra comparativa
        bm = run_benchmark_comparison(**base_params)
        comp = bm["comparison"]
        print("\n" + "=" * 70)
        print("RESUMEN BENCHMARK")
        print("=" * 70)
        print(f"  Tiempo medio SECUENCIAL : {comp['seq_mean_time']:.2f}s")
        print(f"  Tiempo medio PARALELO   : {comp['par_mean_time']:.2f}s")
        print(f"  Speedup obtenido        : {comp['speedup']:.2f}×")
        print(f"  Eficiencia por núcleo   : {comp['efficiency_pct']:.1f}%")
        print(f"  Overhead de procesos    : {comp['overhead_sec']:.3f}s")
        print(f"  Precisión SECUENCIAL    : {comp['seq_mean_accuracy']:.2f}%")
        print(f"  Precisión PARALELO      : {comp['par_mean_accuracy']:.2f}%")
        print(f"  Δ Precisión (par - seq) : {comp['accuracy_delta']:+.2f}%")
        print("=" * 70)

        if comp["speedup"] >= 1.2:
            print("\n→ El modo PARALELO fue significativamente más rápido.")
        elif comp["speedup"] >= 0.9:
            print("\n→ Ambos modos tuvieron rendimiento similar.")
        else:
            print("\n→ El overhead de procesos superó el beneficio del paralelismo.")
            print("  Considera usar más datos o más épocas para amortizar el overhead.")
        return

    results = run_multiple_experiments(**base_params, parallel=args.parallel)
    bm_block = results["benchmark"]
    stats = compute_epoch_statistics(results["all_histories"])

    # Muestra resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Modo                     : {bm_block['mode']}")
    print(f"Precisión final promedio : {stats['mean'][-1]:.2f}%")
    print(f"Desviación estándar      : {stats['std'][-1]:.2f}%")
    print(f"Mejor precisión          : {max(stats['max']):.2f}%")
    print(
        f"Tiempo medio / exp.      : {bm_block['mean_time']:.2f}s ± {bm_block['std_time']:.2f}s"
    )
    print(f"Tiempo total             : {bm_block['total_time']:.2f}s")
    print(
        f"Throughput               : {bm_block['throughput_epochs_per_sec']:.1f} épocas/s"
    )

    # Gráfico ASCII usando datos preparados por chart_generator
    acc_data = prepare_accuracy_chart_data(results["all_histories"])
    print("\nEvolución de precisión (promedio ± desv. estándar):")
    for epoch, (mean, std) in enumerate(zip(acc_data["y_mean"], acc_data["y_std"])):
        bar = "█" * int(mean / 2)
        print(f"  Época {epoch + 1:2d}: {mean:5.2f}% ± {std:4.2f}%  {bar}")


# ====================
# MODO INTERACTIVO
# ====================


def run_interactive_mode() -> None:
    """Lanza la interfaz gráfica interactiva con tooltips en las gráficas."""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import mplcursors
    except ImportError as e:
        print(f"Error: no se pueden cargar las librerías gráficas: {e}")
        print("Instala las dependencias con: pip install matplotlib mplcursors")
        sys.exit(1)

    class ToolTip:
        """
        Muestra un tooltip al pasar el cursor sobre un widget.

        Usa un retardo de 500 ms antes de aparecer para evitar el parpadeo
        causado por eventos <Leave>/<Enter> espurios cuando el Toplevel se
        crea encima del widget. Se posiciona debajo del widget para que no
        tape el cursor ni provoque nuevos ciclos de show/hide.

        :param widget: Widget de Tkinter al que se asocia el tooltip.
        :type widget: tk.Widget

        :param text: Texto a mostrar en el tooltip.
        :type text: str
        """

        def __init__(self, widget, text):
            self.widget = widget
            self.text = text
            self.tip_window = None
            self._after_id = None
            widget.bind("<Enter>", self.show_tip)
            widget.bind("<Leave>", self.hide_tip)

        def show_tip(self, event=None):
            """
            Programa la aparición del tooltip con un retardo de 500 ms.

            :param event: Evento de Tkinter (normalmente <Enter>).
            :type event: tk.Event | None
            """
            if self.tip_window or not self.text:
                return
            if self._after_id:
                self.widget.after_cancel(self._after_id)
            self._after_id = self.widget.after(500, self._show)

        def _show(self):
            """Crea la ventana del tooltip tras el retardo."""
            self._after_id = None
            if self.tip_window:
                return
            x = self.widget.winfo_rootx() + 10
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            self.tip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tk.Label(
                tw,
                text=self.text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("Helvetica", 9),
            ).pack(ipadx=5, ipady=3)

        def hide_tip(self, event=None):
            """
            Cancela la aparición pendiente y destruye el tooltip si existe.

            :param event: Evento de Tkinter (normalmente <Leave>).
            :type event: tk.Event | None
            """
            if self._after_id:
                self.widget.after_cancel(self._after_id)
                self._after_id = None
            if self.tip_window:
                self.tip_window.destroy()
                self.tip_window = None

    class DiegoLearningApp:
        COLORS = [
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

        # =============================================================
        # Inicialización de la app y creación de la interfaz
        # - Panel izquierdo: controles y sliders
        # - Panel derecho: gráficos matplotlib
        # - Barra de estado
        # =============================================================

        def __init__(self, root: tk.Tk) -> None:
            self.root = root

            # Establece el título
            self.root.title("Red Neuronal — Análisis de Algoritmo de Diego")

            # Ajusta tamaño de ventana al espacio utilizable respetando la barra de tareas
            self.root.state("zoomed")

            self.root.columnconfigure(0, weight=0)  # Panel izquierdo fijo
            self.root.columnconfigure(1, weight=1)  # Panel derecho expandible
            self.root.rowconfigure(0, weight=1)

            # Datos de ejecuciones previas para comparación
            self.previous_results: list = []
            self.color_index: int = 0
            self._pending_comparison: bool = False

            # Inicializa atributos para resultados de experimento
            self.current_results: dict = {}
            self.current_params: dict = {}

            # Lista para mantener referencias a los cursores de mplcursors
            self._active_cursors: list = []

            # Construye la ventana
            self._create_ui()

        # ====================
        # CONSTRUCCIÓN UI
        # ====================

        def _create_ui(self) -> None:
            # tk.IntVar es la variable asociada al slider. s la variable asociada al slider.
            # Se ejecuta cada vez que el slider se mueve.
            def _snap_int(var: tk.IntVar):
                """Redondea al entero más cercano al mover el slider."""
                return lambda v: var.set(int(round(float(v))))

            def _make_int_validator(parent, max_digits: int):
                """
                Registra y devuelve un validatecommand que acepta solo
                dígitos con un máximo de max_digits caracteres.
                """

                def _validate(new_value):
                    return new_value == "" or (
                        len(new_value) <= max_digits and new_value.isdigit()
                    )

                return (parent.register(_validate), "%P")

            def _add_slider(parent, label, var, lo, hi):
                """
                Crea un control deslizante con entry editable sincronizado.

                El entry acepta solo dígitos; al confirmar con Enter o al
                perder el foco el valor se clipea al rango [lo, hi] y el
                slider se mueve automáticamente por compartir la misma var.
                """
                max_digits = len(str(hi))
                ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(10, 0))
                ttk.Scale(
                    parent,
                    from_=lo,
                    to=hi,
                    orient=tk.HORIZONTAL,
                    variable=var,
                    length=200,
                    command=_snap_int(var),
                ).pack(fill=tk.X, pady=5)

                vcmd = _make_int_validator(parent, max_digits)

                # Al confirmar (Enter o pérdida de foco) el valor se valida,
                # se clipea al rango [lo, hi] y se escribe en var, lo que
                # mueve el slider automáticamente al ser la misma variable.
                entry = ttk.Entry(
                    parent,
                    textvariable=var,
                    width=max_digits + 1,
                    justify="center",
                    validate="key",
                    validatecommand=vcmd,
                )
                entry.pack(pady=(0, 4))

                def _commit(event=None):
                    try:
                        val = int(round(float(var.get())))
                    except (ValueError, tk.TclError):
                        val = lo
                    var.set(max(lo, min(hi, val)))

                entry.bind("<Return>", _commit)
                entry.bind("<FocusOut>", _commit)

            def _add_integer_input(parent, label, var, lo, hi, max_digits=6):
                """Entry numérico entero sin slider, con clipping en [lo, hi]."""
                ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(10, 0))
                vcmd = _make_int_validator(parent, max_digits)
                entry = ttk.Entry(
                    parent,
                    textvariable=var,
                    width=max_digits + 1,
                    justify="center",
                    validate="key",
                    validatecommand=vcmd,
                )
                entry.pack(pady=(0, 6))

                def _commit(event=None):
                    try:
                        val = int(var.get())
                    except (ValueError, tk.TclError):
                        val = lo
                    var.set(max(lo, min(hi, val)))

                entry.bind("<Return>", _commit)
                entry.bind("<FocusOut>", _commit)

            def _add_float_input(parent, label, var, lo, hi, max_chars=8):
                """Entry numérico flotante sin slider, con clipping en [lo, hi]."""
                ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(10, 0))

                def _validate(new_value):
                    return new_value == "" or (
                        len(new_value) <= max_chars
                        and new_value.count(".") <= 1
                        and all(c in "0123456789." for c in new_value)
                    )

                vcmd = (parent.register(_validate), "%P")
                entry = ttk.Entry(
                    parent,
                    textvariable=var,
                    width=max_chars + 1,
                    justify="center",
                    validate="key",
                    validatecommand=vcmd,
                )
                entry.pack(pady=(0, 6))

                def _commit(event=None):
                    try:
                        val = float(var.get())
                    except (ValueError, tk.TclError):
                        val = lo
                    var.set(max(lo, min(hi, val)))

                entry.bind("<Return>", _commit)
                entry.bind("<FocusOut>", _commit)

            # Panel izquierdo: controles
            # Crea un frame principal que contendrá todo el panel izquierdo
            ctrl_container = ttk.Frame(self.root, width=260)

            # Configura las propiedades del panel, como su margen y que sólo se estire verticalmente
            ctrl_container.grid(row=0, column=0, sticky="ns", padx=5, pady=5)

            # Evita que el grid lo estire horizontalmente
            ctrl_container.grid_propagate(False)

            # Canvas desplazable para que el panel funcione con pantallas pequeñas
            panel_canvas = tk.Canvas(ctrl_container, width=260, highlightthickness=0)
            scrollbar = ttk.Scrollbar(
                ctrl_container, orient="vertical", command=panel_canvas.yview
            )
            scrollable_frame = ttk.Frame(panel_canvas, padding="10")
            scrollable_frame.bind(
                "<Configure>",
                lambda e: panel_canvas.configure(scrollregion=panel_canvas.bbox("all")),
            )
            panel_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            panel_canvas.configure(yscrollcommand=scrollbar.set)
            panel_canvas.pack(side="left", fill="y", expand=True)
            scrollbar.pack(side="right", fill="y")

            ctrl = scrollable_frame
            ttk.Label(ctrl, text="Parámetros", font=("Helvetica", 14, "bold")).pack(
                pady=10
            )

            # Variables ligadas a los controles
            self.partitions_var = tk.IntVar(value=2)
            self.epochs_var = tk.IntVar(value=50)
            self.experiments_var = tk.IntVar(value=5)
            self.hidden_var = tk.IntVar(value=30)
            self.lr_var = tk.StringVar(value="1.0")
            self.n_train_var = tk.StringVar(value="5000")
            self.parallel_var = tk.BooleanVar(value=False)

            # El límite de particiones en modo paralelo es el número de
            # núcleos físicos. El slider lo refleja directamente para que
            # el usuario nunca pueda configurar una combinación inválida.
            max_parts = get_physical_cores()

            _add_slider(
                ctrl,
                f"Particiones (1 - {max_parts}):",
                self.partitions_var,
                1,
                max_parts,
            )
            _add_slider(ctrl, "Épocas (50 - 1.000):", self.epochs_var, 50, 1000)
            _add_slider(ctrl, "Experimentos (1 - 20):", self.experiments_var, 1, 20)
            _add_slider(ctrl, "Neuronas ocultas (10 - 100):", self.hidden_var, 10, 100)
            _add_float_input(
                ctrl, "Tasa de aprendizaje (0.0001 - 10):", self.lr_var, 0.0001, 10.0
            )
            _add_integer_input(
                ctrl,
                "Ejemplos de entrenamiento (10 - 60000):",
                self.n_train_var,
                100,
                60000,
            )

            # Checkbox de modo paralelo.
            # Al activarlo cada partición corre en un proceso independiente
            # del SO. El slider ya está limitado a núcleos físicos, así que
            # la regla de oro siempre se cumple desde la interfaz.
            ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(20, 8))
            parallel_cb = ttk.Checkbutton(
                ctrl,
                text="Modo paralelo (1 proceso por partición)",
                variable=self.parallel_var,
            )
            parallel_cb.pack(anchor=tk.W, pady=(0, 4))
            ToolTip(
                parallel_cb,
                f"Entrena cada partición en un proceso del SO independiente.\n"
                f"Núcleos físicos disponibles: {max_parts}.\n"
                f"El slider de particiones ya está limitado a ese máximo.",
            )

            ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 20))

            buttons = [
                (
                    "Ejecutar Experimento",
                    self._run_experiment,
                    "Ejecuta el algoritmo con los parámetros actuales",
                ),
                (
                    "Benchmark Sec. vs Par.",
                    self._run_benchmark,
                    "Ejecuta ambos modos con los mismos parámetros\ny compara tiempos y precisión",
                ),
                (
                    "Comparar Configuraciones",
                    self._compare_configurations,
                    "Ejecuta un nuevo experimento y lo superpone\ncon los resultados actuales.",
                ),
                (
                    "Limpiar Gráficos",
                    self._clear_plots,
                    "Borra todas las gráficas actuales",
                ),
                (
                    "Guardar Resultados",
                    self._save_results,
                    "Guarda los resultados actuales en JSON",
                ),
            ]
            for text, cmd, tip in buttons:
                btn = ttk.Button(ctrl, text=text, command=cmd)
                btn.pack(fill=tk.X, pady=5)
                ToolTip(btn, tip)

            # Panel derecho: gráficos
            # Crea una figura de Matplotlib con una cuadrícula de 2x2 subplots
            # Esto significa que habrá 4 gráficos independientes dentro de la misma figura
            self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
            self.fig.suptitle(
                "Análisis de Algoritmo de Diego", fontsize=14, fontweight="bold"
            )

            # Crea un frame que contendrá la figura
            plot_frame = ttk.Frame(self.root)
            plot_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

            # Convierte la figura de MatplotLib en un widget de Tkinter
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.draw()

            # Hace que la figura ocupe todo el espacio del frame
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Crea la barra de estado en la parte inferior
            self.status_var = tk.StringVar(value="Listo")
            ttk.Label(
                self.root,
                textvariable=self.status_var,
                relief=tk.SUNKEN,
                anchor=tk.W,
            ).grid(row=1, column=0, columnspan=2, sticky="ew")

        # ========================
        # TOOLTIPS INTERACTIVOS
        # ========================

        def _add_cursor(self, artists, fmt_func=None):
            """
            Agrega tooltips interactivos (mplcursors) a artistas de matplotlib.

            :param artists: Lista de artistas (líneas, barras) de matplotlib.
            :param fmt_func: Callback opcional para personalizar el texto.
                             Recibe un objeto Selection de mplcursors.
            """
            # Si no hay líneas, no hace nada, para evitar errores
            if not artists:
                return

            # Si mplcursors falla, silenciosamente ignora el error
            try:
                # Si hover=True, el tooltip aparece solo al pasar el mouse
                cursor = mplcursors.cursor(artists, hover=True)

                # Conecta un callback al evento "add"
                # El evento "add" ocurre cuando se crea una anotación nueva
                # Al ejecutar fmt_func(sel), se permite personalizar completamente el texto
                if fmt_func:
                    cursor.connect("add", fmt_func)
                else:
                    # Es equivalente a cursor.connect("add", on_add)
                    # Configuración por defecto del tooltip
                    @cursor.connect("add")
                    def _default(sel):
                        sel.annotation.set_text(
                            f"Época: {sel.target[0]:.0f}\nValor: {sel.target[1]:.2f}%"
                        )
                        sel.annotation.get_bbox_patch().set(
                            facecolor="#ffffcc", alpha=0.95, edgecolor="#888888"
                        )

                # Al guardar la referencia a cursor, evita que este sea recolectado por el garbage collector
                # Permite eliminarlos
                self._active_cursors.append(cursor)
            except Exception:
                pass  # mplcursors puede fallar con ciertos artistas

        def _clear_cursors(self):
            """Elimina todos los cursores interactivos activos."""
            # Recorre todos los cursores creados
            for cursor in self._active_cursors:
                try:
                    cursor.remove()
                except Exception:
                    pass

            # Vacía completamente el registro, dejando el sistema límpio
            self._active_cursors.clear()

        # ==============================================================================
        # VENTANA DE PROGRESO
        #
        # Ventana modal que muestra el progreso del entrenamiento
        # Polling de la cola para actualizar barras y mensajes
        # Thread-safe: el hilo de entrenamiento nunca toca directamente los widgets
        # ==============================================================================

        def _create_progress_window(
            self, num_experiments: int, num_epochs: int
        ) -> dict:
            """
            Crea una ventana modal que muestra el progreso del entrenamiento.

            :param num_experiments: Total de experimentos a ejecutar.
            :type num_experiments: int

            :param num_epochs: Total de épocas por experimento.
            :type num_epochs: int

            :return: Diccionario con referencias a los widgets dinámicos.
            :rtype: dict
            """
            # Crea ventana secundaria sobre la principal
            win = tk.Toplevel(self.root)
            win.title("Ejecutando experimentos...")
            win.geometry("520x360")
            win.resizable(False, False)

            # Hace que la ventana sea modal
            # Bloquea interacciones con la ventana principal
            win.grab_set()

            # Desactiva el botón de cerrar
            win.protocol("WM_DELETE_WINDOW", lambda: None)

            # Título
            ttk.Label(
                win, text="Entrenamiento en progreso", font=("Helvetica", 13, "bold")
            ).pack(pady=(18, 4))

            # Etiqueta de fase. Se crea aquí en el orden correcto del árbol
            # de widgets para que pack() la coloque debajo del título.
            # Empieza oculta (no se llama a pack todavía); _poll_queue la
            # hace visible cuando recibe el primer mensaje ("phase", ...).
            phase_var = tk.StringVar(value="")
            phase_label = tk.Label(
                win,
                textvariable=phase_var,
                font=("Helvetica", 10, "bold"),
                fg="white",
                bg="#607D8B",
                width=46,
                pady=4,
            )

            # Experimento actual
            # Se actualizará en tiempo real con el experimento que se esté ejecutando
            exp_label_var = tk.StringVar(value="Inicializando...")
            ttk.Label(win, textvariable=exp_label_var, font=("Helvetica", 10)).pack(
                pady=2
            )

            # Barra de progreso general (experimentos)
            ttk.Label(win, text="Progreso general:").pack(
                anchor=tk.W, padx=24, pady=(10, 0)
            )
            exp_bar = ttk.Progressbar(
                win,
                orient=tk.HORIZONTAL,
                length=470,
                mode="determinate",
                maximum=num_experiments,
            )
            exp_bar.pack(padx=24, pady=4)

            # Barra de épocas del experimento actual
            ttk.Label(win, text="Época actual:").pack(anchor=tk.W, padx=24, pady=(8, 0))
            epoch_bar = ttk.Progressbar(
                win,
                orient=tk.HORIZONTAL,
                length=470,
                mode="determinate",
                maximum=num_epochs,
            )
            epoch_bar.pack(padx=24, pady=4)

            # Último mensaje recibido
            ttk.Label(win, text="Último estado:").pack(
                anchor=tk.W, padx=24, pady=(8, 0)
            )

            # Último mensaje recibido del entrenamiento
            # Funciona igual que el mensaje de experimento actual
            msg_var = tk.StringVar(value="—")
            ttk.Label(
                win,
                textvariable=msg_var,
                font=("Helvetica", 9),
                foreground="#555555",
                wraplength=470,
                justify=tk.LEFT,
            ).pack(anchor=tk.W, padx=24)

            # Diccionario con referencias a todos los widgets que
            # necesitan actualización dinámica desde el hilo de entrenamiento
            return {
                "window": win,
                "exp_label_var": exp_label_var,
                "exp_bar": exp_bar,
                "epoch_bar": epoch_bar,
                "msg_var": msg_var,
                "phase_var": phase_var,
                "phase_label": phase_label,
                "phase_visible": False,
            }

        def _poll_queue(
            self, q, widgets: dict, num_experiments: int, num_epochs: int, on_done
        ) -> None:
            """
            Lee la cola de progreso y actualiza los widgets desde el hilo principal.

            Se reprograma cada 100 ms con root.after hasta recibir "done" o "error".

            Tkinter no es thread-safe: el hilo de entrenamiento nunca toca
            widgets directamente, solo hace ``q.put(tipo, payload)``.
            Este método se reprograma cada 100 ms con root.after hasta que
            recibe el mensaje ``'done'`` o ``'error'``.

            Tipos de mensaje:
                ``('exp', n)``      → avanza barra de experimentos
                ``('epoch', n)``      → avanza barra de épocas
                ``('msg', texto)``  → actualiza etiqueta de estado
                ``('done', result)`` → entrenamiento terminado
                ``('error', exc)``    → error en el hilo secundario

            :param q: Cola compartida con el hilo de entrenamiento.
            :param widgets: Widgets de la ventana de progreso.
            :param num_experiments: Total de experimentos.
            :param num_epochs: Total de épocas por experimento.
            :param on_done: Callback que recibe los resultados al terminar.
            """
            try:
                while True:
                    msg_type, payload = q.get_nowait()
                    if msg_type == "exp":
                        # Usa el total de la fase actual si está disponible,
                        # o el total general si no (modo experimento normal).
                        total = widgets.get("phase_total", num_experiments)
                        widgets["exp_label_var"].set(
                            f"Experimento {payload} de {total}"
                        )
                        widgets["exp_bar"]["value"] = payload - 1
                        widgets["epoch_bar"]["value"] = 0
                    elif msg_type == "epoch":
                        widgets["epoch_bar"]["value"] = payload
                    elif msg_type == "msg":
                        widgets["msg_var"].set(payload)
                    elif msg_type == "phase":
                        # Muestra la etiqueta de fase la primera vez y
                        # actualiza su texto y color según el modo activo.
                        lbl = widgets["phase_label"]
                        if not widgets["phase_visible"]:
                            lbl.pack(fill=tk.X, padx=24, pady=(2, 6))
                            widgets["phase_visible"] = True
                        widgets["phase_var"].set(payload)
                        lbl.configure(
                            bg="#1565C0" if "SECUENCIAL" in payload else "#2E7D32"
                        )
                        # Reinicia las barras y ajusta el máximo al total
                        # de experimentos de esta fase (no del benchmark completo).
                        widgets["exp_bar"]["value"] = 0
                        widgets["epoch_bar"]["value"] = 0
                        widgets["exp_bar"]["maximum"] = widgets["phase_total"]
                    elif msg_type == "done":
                        widgets["exp_bar"]["value"] = num_experiments
                        widgets["epoch_bar"]["value"] = num_epochs
                        widgets["window"].destroy()
                        on_done(payload)
                        return
                    elif msg_type == "error":
                        widgets["window"].destroy()
                        raise payload
            except queue.Empty:
                pass  # No hay mensajes nuevos; sigue esperando
            except Exception as e:
                widgets["window"].destroy()
                messagebox.showerror("Error", f"Error ejecutando experimento:\n{e}")
                self.status_var.set("Error en ejecución")
                return

            # Reprograma el siguiente ciclo de polling cada 100ms
            # Vuelve a ejecutar esta misma función cada 100 ms
            # Crea un polling continuo sin bloquear la interfaz
            self.root.after(
                100,
                lambda: self._poll_queue(
                    q, widgets, num_experiments, num_epochs, on_done
                ),
            )

        # ====================
        # ACCIONES DE BOTONES
        # ====================

        def _collect_params(self) -> dict:
            """
            Lee y valida los parámetros del panel de control.

            :return: Diccionario listo para run_multiple_experiments.
            :rtype: dict
            :raises ValueError: Si algún campo tiene un valor inválido.
            """
            return {
                "num_partitions": self.partitions_var.get(),
                "num_epochs": self.epochs_var.get(),
                "num_experiments": self.experiments_var.get(),
                "hidden_neurons": self.hidden_var.get(),
                "learning_rate": float(self.lr_var.get()),
                "n_train": int(self.n_train_var.get()),
                "parallel": self.parallel_var.get(),
                "verbose": False,
            }

        def _run_experiment(self) -> None:
            """
            Lanza los experimentos en un hilo secundario para no bloquear la UI.

            El hilo de entrenamiento solo escribe en la cola; el hilo principal
            lee la cola cada 100 ms mediante _poll_queue y actualiza la ventana.
            """
            try:
                # Recolecta parámetros
                params = self._collect_params()
            except ValueError as e:
                messagebox.showerror("Error de parámetros", str(e))
                return

            q = queue.Queue()
            num_experiments = params["num_experiments"]
            num_epochs = params["num_epochs"]

            # Crea la ventana de progreso antes de lanzar el hilo
            widgets = self._create_progress_window(num_experiments, num_epochs)
            self.status_var.set("Ejecutando experimentos...")

            # ===========================================================================
            # HILO DE ENTRENAMIENTO
            #
            # Hilo secundario: ejecuta los experimentos y escribe mensajes en la cola
            # Hilo principal: lee la cola y actualiza la GUI
            # ===========================================================================

            def _training_thread():
                """
                Hilo secundario de entrenamiento.

                Nunca accede a widgets de tkinter. Solo escribe en la cola
                mensajes de progreso ("msg", "exp", "epoch"), finalización
                ("done") o error ("error").
                """

                # Detecta mensajes del entrenamiento y los convierte en mensajes para la cola
                def on_progress(msg):
                    # Envía el mensaje genérico a la cola
                    q.put(("msg", msg))

                    # Detecta inicio de nuevo experimento: "EXPERIMENTO X/Y"
                    if msg.startswith("EXPERIMENTO"):
                        try:
                            n = int(msg.split()[1].split("/")[0])
                            q.put(("exp", n))
                        except (IndexError, ValueError):
                            pass
                    # Detecta fin de época: "[Época X/Y — Precisión: ...]"
                    elif msg.startswith("[Época"):
                        try:
                            epoch = int(msg.split()[1].split("/")[0])
                            q.put(("epoch", epoch))
                        except (IndexError, ValueError):
                            pass

                try:
                    results = run_multiple_experiments(
                        **params,  # Pasa todos los parametros recoletados de la UI
                        on_progress=on_progress,  # Envía el callback
                    )
                    # Notifica al hilo principal que el experimento terminó correctamente
                    q.put(("done", results))
                except Exception as e:
                    q.put(("error", e))

            # ========================
            # CALLBACK AL COMPLETAR
            # ========================

            # Función interna que se ejecuta cuando el hilo de entrenamiento termina correctamente
            def _on_done(results):
                # Aquí results es el diccionario devuelto por run_multiple_experiments()
                self.current_results = results

                # Guarda los parámetros con los que se ejecutó el experimento
                self.current_params = params

                self._plot_results(results, params)

                # Cambia el texto de la barra de estado inferior
                self.status_var.set(
                    f"Completado — Precisión final: {results['final_mean_accuracy']:.2f}%"
                )

            # Crea el hilo secundario y lo arranca inmediatamente
            # Un hilo daemon es un hilo secundario dependiente del hilo principal
            threading.Thread(target=_training_thread, daemon=True).start()

            # Llamado inicial a _poll_queue
            self.root.after(
                100,
                lambda: self._poll_queue(
                    q, widgets, num_experiments, num_epochs, _on_done
                ),
            )

        def _run_benchmark(self) -> None:
            """
            Lanza el benchmark secuencial vs paralelo en un hilo secundario.

            Ejecuta ``run_benchmark_comparison`` con los parámetros actuales.
            Al terminar dibuja el panel de benchmark en lugar de los cuatro
            paneles habituales para mostrar la comparativa directamente.
            """
            try:
                params = self._collect_params()
            except ValueError as e:
                messagebox.showerror("Error de parámetros", str(e))
                return

            # El benchmark siempre corre ambos modos con los mismos parámetros;
            # el flag parallel del checkbox se ignora aquí.
            bm_params = {
                k: v for k, v in params.items() if k not in ("parallel", "verbose")
            }

            q = queue.Queue()
            num_experiments = params["num_experiments"]
            num_epochs = params["num_epochs"]

            # El benchmark ejecuta 2 rondas (seq + par), cada una con
            # num_experiments experimentos. La barra se inicializa con el
            # total de una sola fase; phase_total permite mostrar el número
            # correcto en el texto "Experimento X de Y" por fase.
            widgets = self._create_progress_window(num_experiments, num_epochs)
            widgets["phase_total"] = num_experiments
            self.status_var.set("Ejecutando benchmark secuencial vs paralelo...")

            def _training_thread():
                def on_progress(msg):
                    q.put(("msg", msg))
                    if "Fase 1/2" in msg:
                        q.put(("phase", "▶  Fase 1 / 2  —  Modo SECUENCIAL"))
                    elif "Fase 2/2" in msg:
                        q.put(("phase", "▶  Fase 2 / 2  —  Modo PARALELO"))
                    elif msg.startswith("EXPERIMENTO"):
                        try:
                            n = int(msg.split()[1].split("/")[0])
                            q.put(("exp", n))
                        except (IndexError, ValueError):
                            pass
                    elif msg.startswith("[Época"):
                        try:
                            epoch = int(msg.split()[1].split("/")[0])
                            q.put(("epoch", epoch))
                        except (IndexError, ValueError):
                            pass

                try:
                    result = run_benchmark_comparison(
                        **bm_params, verbose=False, on_progress=on_progress
                    )
                    q.put(("done", result))
                except Exception as e:
                    q.put(("error", e))

            def _on_done(bm_result):
                self.current_results = bm_result.get("sequential", {})
                self.current_params = params
                self._plot_benchmark(bm_result)
                comp = bm_result["comparison"]
                self.status_var.set(
                    f"Benchmark completo — Speedup: {comp['speedup']:.2f}×  |  "
                    f"Eficiencia: {comp['efficiency_pct']:.1f}%  |  "
                    f"Δ Precisión: {comp['accuracy_delta']:+.2f}%"
                )

            threading.Thread(target=_training_thread, daemon=True).start()
            self.root.after(
                100,
                lambda: self._poll_queue(
                    q, widgets, num_experiments * 2, num_epochs, _on_done
                ),
            )

        def _plot_benchmark(self, bm_result: dict) -> None:
            """
            Renderiza los cuatro paneles en modo comparación de benchmark.

            Panel 1 — Tiempos por experimento (barras lado a lado)
            Panel 2 — Curvas de aprendizaje secuencial vs paralelo
            Panel 3 — Barras de precisión final por modo
            Panel 4 — Tabla de métricas resumen (speedup, eficiencia, etc.)
            """
            ax1, ax2, ax3, ax4 = self.axes.flatten()
            self._clear_cursors()
            for ax in self.axes.flatten():
                ax.clear()

            bm_data = prepare_benchmark_data(bm_result)
            t_data = bm_data["times"]
            summary = bm_data["summary"]
            comp = bm_result["comparison"]
            seq_h = bm_result["sequential"]["all_histories"]
            par_h = bm_result["parallel"]["all_histories"]

            # Panel 1 - tiempos por experimento
            n = len(t_data["x"])
            x = np.arange(n)
            w = 0.35
            b1 = ax1.bar(
                x - w / 2,
                t_data["seq"],
                w,
                label=f"Secuencial (μ={t_data['seq_mean']:.2f}s)",
                color="#2196F3",
                alpha=0.85,
            )
            b2 = ax1.bar(
                x + w / 2,
                t_data["par"],
                w,
                label=f"Paralelo   (μ={t_data['par_mean']:.2f}s)",
                color="#4CAF50",
                alpha=0.85,
            )
            ax1.axhline(
                t_data["seq_mean"], color="#2196F3", linestyle="--", linewidth=1.5
            )
            ax1.axhline(
                t_data["par_mean"], color="#4CAF50", linestyle="--", linewidth=1.5
            )
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"Exp {i}" for i in t_data["x"]])
            ax1.set(
                xlabel=t_data["xlabel"], ylabel=t_data["ylabel"], title=t_data["title"]
            )
            ax1.legend(loc="upper right", fontsize=8)
            ax1.grid(True, alpha=0.3, axis="y")

            def _fmt_time(sel):
                sel.annotation.set_text(f"Tiempo: {sel.target[1]:.3f}s")
                sel.annotation.get_bbox_patch().set(facecolor="#e3f2fd", alpha=0.95)

            self._add_cursor([b1, b2], _fmt_time)

            # Panel 2 - curvas de aprendizaje de ambos modos
            seq_stats = compute_epoch_statistics(seq_h)
            par_stats = compute_epoch_statistics(par_h)
            epochs_x = list(range(1, len(seq_stats["mean"]) + 1))

            seq_mean = np.array(seq_stats["mean"])
            par_mean = np.array(par_stats["mean"])
            seq_std = np.array(seq_stats["std"])
            par_std = np.array(par_stats["std"])

            (ln_seq,) = ax2.plot(
                epochs_x,
                seq_mean,
                "o-",
                color="#2196F3",
                linewidth=2,
                markersize=4,
                label="Secuencial",
            )
            (ln_par,) = ax2.plot(
                epochs_x,
                par_mean,
                "s-",
                color="#4CAF50",
                linewidth=2,
                markersize=4,
                label="Paralelo",
            )
            ax2.fill_between(
                epochs_x,
                seq_mean - seq_std,
                seq_mean + seq_std,
                alpha=0.15,
                color="#2196F3",
            )
            ax2.fill_between(
                epochs_x,
                par_mean - par_std,
                par_mean + par_std,
                alpha=0.15,
                color="#4CAF50",
            )
            ax2.set(
                xlabel="Época",
                ylabel="Precisión (%)",
                title="Curvas de Aprendizaje: Secuencial vs Paralelo",
            )
            ax2.legend(loc="lower right", fontsize=8)
            ax2.grid(True, alpha=0.3)

            def _fmt_lr(sel):
                sel.annotation.set_text(
                    f"Época: {sel.target[0]:.0f}\nPrecisión: {sel.target[1]:.2f}%"
                )
                sel.annotation.get_bbox_patch().set(facecolor="#ffffcc", alpha=0.95)

            self._add_cursor([ln_seq, ln_par], _fmt_lr)

            # Panel 3 - precisión final por modo
            modes = ["Secuencial", "Paralelo"]
            accs = [comp["seq_mean_accuracy"], comp["par_mean_accuracy"]]
            colors = ["#2196F3", "#4CAF50"]
            bars3 = ax3.bar(modes, accs, color=colors, alpha=0.85, edgecolor="black")
            for bar, val in zip(bars3, accs):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.3,
                    f"{val:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            ax3.set(
                ylabel="Precisión (%)",
                title=f"Precisión Final  |  Δ = {comp['accuracy_delta']:+.2f}%",
            )
            ax3.set_ylim(0, max(accs) * 1.12)
            ax3.grid(True, alpha=0.3, axis="y")

            # Panel 4 - tabla de métricas
            ax4.axis("off")
            rows = [
                ["Métrica", "Valor"],
                ["Speedup", f"{summary['speedup']:.2f}×"],
                ["Eficiencia / núcleo", f"{summary['efficiency_pct']:.1f}%"],
                ["Overhead de procesos", f"{summary['overhead_sec']:.3f}s"],
                ["Tiempo medio sec.", f"{comp['seq_mean_time']:.2f}s"],
                ["Tiempo medio par.", f"{comp['par_mean_time']:.2f}s"],
                ["Precisión sec.", f"{summary['seq_mean_accuracy']:.2f}%"],
                ["Precisión par.", f"{summary['par_mean_accuracy']:.2f}%"],
                ["Δ Precisión (par − sec)", f"{summary['accuracy_delta']:+.2f}%"],
            ]
            table = ax4.table(
                cellText=rows[1:],
                colLabels=rows[0],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.6)

            # Cabecera en azul oscuro, filas alternas en gris suave
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#1565C0")
                    cell.set_text_props(color="white", fontweight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#F5F5F5")

            ax4.set_title(
                "Métricas del Benchmark", fontsize=10, fontweight="bold", pad=8
            )

            self.fig.suptitle(
                "Benchmark — Secuencial vs Paralelo",
                fontsize=14,
                fontweight="bold",
            )
            self.fig.tight_layout()
            self.canvas.draw()

        def _compare_configurations(self):
            """Guarda la configuración actual y lanza un nuevo experimento para comparar."""
            # Si self.current_results está vacío, significa que el usuario aún no ha ejecutado nada
            if not self.current_results:
                messagebox.showwarning("Advertencia", "Primero ejecuta un experimento")
                return

            # Guarda configuración actual antes de ejecutar la nueva
            self.previous_results.append(
                {"results": self.current_results, "params": self.current_params}
            )

            # Se usa para dibujar la nueva configuración con un color diferente
            self.color_index += 1

            # Bandera que indica "No dibujes el próximo experimento normal, sino que haz una comparación"
            self._pending_comparison = True

            self._run_experiment()

        def _clear_plots(self):
            """Limpia todos los gráficos y reinicia el historial de comparaciones."""
            self._clear_cursors()
            for ax in self.axes.flatten():
                ax.clear()

            # Elimina toda la memoria de configuraciones anteriores
            self.previous_results = []

            # Asegura que el próximo experimento use el primer color de la lista
            self.color_index = 0

            # Fuerza a Matplotlib a actualizar la interfaz y actualiza la ventana
            self.canvas.draw()

            self.status_var.set("Gráficos limpiados")

        def _save_results(self):
            """Guarda los resultados del experimento actual en un archivo JSON."""
            # Si self.current_results está vacío, entonces no se ha ejecutado ningún experimento
            if not self.current_results:
                messagebox.showwarning("Advertencia", "No hay resultados para guardar")
                return

            # Crea la carpeta Results si no existe
            os.makedirs("Results", exist_ok=True)
            filename = f"Results/experiment_{self.current_results['timestamp']}.json"
            with open(filename, "w") as f:
                json.dump(
                    {
                        "parameters": self.current_params,
                        "results": {
                            k: self.current_results[k]
                            for k in (
                                "timestamp",
                                "final_mean_accuracy",
                                "final_std_accuracy",
                                "all_histories",
                            )
                        },
                    },
                    f,
                    indent=2,
                    default=str,
                )
            messagebox.showinfo("Éxito", f"Resultados guardados en:\n{filename}")
            self.status_var.set(f"Guardado: {filename}")

        def _current_color(self):
            """Devuelve el color activo para el experimento actual."""
            # Se usa % para que cuando se acaben los colores, vuelva al inicio
            return self.COLORS[self.color_index % len(self.COLORS)]

        # ================
        # RENDERIZADO
        # ================

        def _plot_results(self, results: dict, params: dict) -> None:
            """
            Renderiza los cuatro paneles de análisis.

            Delega la preparación de datos a chart_generator y solo llama
            a matplotlib con los datos ya listos.

            :param results: Resultado de run_multiple_experiments.
            :type results: dict

            :param params: Parámetros del experimento.
            :type params: dict
            """
            ax1, ax2, ax3, ax4 = self.axes.flatten()
            color = self._current_color()
            label = f"P={params['num_partitions']}, E={params['num_epochs']}"

            self._clear_cursors()
            for ax in self.axes.flatten():
                ax.clear()

            histories = results["all_histories"]

            # Actualiza el título de la figura con el tiempo del experimento
            bm = results.get("benchmark", {})
            mode_label = "Paralelo" if params.get("parallel") else "Secuencial"
            if bm:
                total_s = bm.get("total_time", 0.0)
                self.fig.suptitle(
                    f"Análisis de Algoritmo de Diego  —  {mode_label}\n"
                    f"Duración: {total_s:.1f} segundos",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                self.fig.suptitle(
                    "Análisis de Algoritmo de Diego", fontsize=14, fontweight="bold"
                )

            # Panel 1 — Curva de aprendizaje con banda ±1σ
            acc = prepare_accuracy_chart_data(histories)
            (line1,) = ax1.plot(
                acc["x"],
                acc["y_mean"],
                "o-",
                color=color,
                linewidth=2,
                markersize=5,
                label=label,
            )
            ax1.fill_between(
                acc["x"], acc["y_lower"], acc["y_upper"], alpha=0.2, color=color
            )
            ax1.set(
                xlabel=acc["xlabel"],
                ylabel=acc["ylabel"],
                title="Evolución del Promedio (±1σ)",
            )
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)

            def _fmt_acc(sel):
                idx = int(round(sel.target[0])) - 1
                if 0 <= idx < len(acc["y_mean"]):
                    sel.annotation.set_text(
                        f"Época: {idx + 1}\n"
                        f"Media: {acc['y_mean'][idx]:.2f}%\n"
                        f"± Std: {acc['y_std'][idx]:.2f}%"
                    )
                sel.annotation.get_bbox_patch().set(
                    facecolor="#ffffcc", alpha=0.95, edgecolor="#888"
                )

            self._add_cursor([line1], _fmt_acc)

            # Panel 2 — Comparación por partición (último experimento)
            part = prepare_partition_comparison_data([histories[-1]])
            part_lines = []
            if part:
                for p in part["partitions"]:
                    (ln,) = ax2.plot(
                        p["x"],
                        p["y"],
                        "o-",
                        label=f"Partición {p['id']}",
                        alpha=0.7,
                        markersize=4,
                    )
                    part_lines.append(ln)
            ax2.set(
                xlabel=part.get("xlabel", "Época"),
                ylabel=part.get("ylabel", "Precisión (%)"),
                title="Comparación por Partición (último)",
            )
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)

            def _fmt_part(sel):
                sel.annotation.set_text(
                    f"Época: {sel.target[0]:.0f}\nPrecisión: {sel.target[1]:.2f}%"
                )
                sel.annotation.get_bbox_patch().set(
                    facecolor="#e8f5e9", alpha=0.95, edgecolor="#888"
                )

            self._add_cursor(part_lines, _fmt_part)

            # Panel 3 — Distribución de precisión por experimento con RSD
            rsd_data = prepare_experiment_rsd_data(histories)

            # Barras coloreadas según calidad de cada experimento
            bar_colors = [
                "#4CAF50" if y >= 90 else "#FF9800" if y >= 80 else "#F44336"
                for y in rsd_data["y"]
            ]
            bars = ax3.bar(
                rsd_data["x"],
                rsd_data["y"],
                color=bar_colors,
                alpha=0.8,
                edgecolor="black",
                zorder=3,
            )

            # Línea del promedio. Muestra μ ± σ para leer accuracy y dispersión juntos
            ax3.axhline(
                rsd_data["mean"],
                color="red",
                linestyle="--",
                linewidth=2,
                zorder=4,
                label=f"Promedio: {rsd_data['mean']:.2f}% ± {rsd_data['std']:.2f}% (σ)",
            )

            # Bandas ±1σ. Muestra RSD e interpretación cualitativa
            ax3.axhline(
                rsd_data["upper"],
                color="orange",
                linestyle=":",
                linewidth=1.5,
                zorder=4,
                label=f"RSD: {rsd_data['rsd']:.2f}%  ({rsd_data['interpretation']})",
            )
            ax3.axhline(
                rsd_data["lower"],
                color="orange",
                linestyle=":",
                linewidth=1.5,
                zorder=4,
            )
            ax3.fill_between(
                [rsd_data["x"][0] - 0.5, rsd_data["x"][-1] + 0.5],
                rsd_data["lower"],
                rsd_data["upper"],
                alpha=0.1,
                color="orange",
                zorder=1,
            )
            ax3.set_xticks(rsd_data["x"])
            ax3.set(
                xlabel=rsd_data["xlabel"],
                ylabel=rsd_data["ylabel"],
                title=(
                    f"Precisión por Experimento: "
                    f"{rsd_data['mean']:.2f}% ± {rsd_data['std']:.2f}%"
                    f"  (RSD: {rsd_data['rsd']:.2f}%)"
                ),
            )
            ax3.legend(loc="lower right", fontsize=8)
            ax3.grid(True, alpha=0.3, axis="y")

            def _fmt_rsd(sel):
                idx = int(round(sel.target[0])) - 1
                if 0 <= idx < len(rsd_data["y"]):
                    val = rsd_data["y"][idx]
                    diff = val - rsd_data["mean"]
                    sel.annotation.set_text(
                        f"Experimento: {idx + 1}\n"
                        f"Precisión: {val:.2f}%\n"
                        f"vs Promedio: {diff:+.2f}%"
                    )
                sel.annotation.get_bbox_patch().set(
                    facecolor="#fff3e0", alpha=0.95, edgecolor="#888"
                )

            self._add_cursor(bars, _fmt_rsd)

            # Panel 4 — Mejora por época
            conv = prepare_convergence_data(histories)
            bars4 = ax4.bar(conv["x"], conv["y"], color=color, alpha=0.7)
            ax4.axhline(0, color="black", linewidth=0.5)
            ax4.set(xlabel=conv["xlabel"], ylabel=conv["ylabel"], title=conv["title"])
            ax4.grid(True, alpha=0.3)

            def _fmt_conv(sel):
                sel.annotation.set_text(
                    f"Época: {sel.target[0]:.0f}\nMejora: {sel.target[1]:+.2f}%"
                )
                sel.annotation.get_bbox_patch().set(
                    facecolor="#e3f2fd", alpha=0.95, edgecolor="#888"
                )

            self._add_cursor(bars4, _fmt_conv)

            self.fig.tight_layout()
            self.canvas.draw()

            if self._pending_comparison:
                self._pending_comparison = False
                self._plot_comparison()

        def _plot_comparison(self):
            """
            Superpone las curvas de precisión de todas las configuraciones en ax1.

            La configuración actual se dibuja con línea discontinua y mayor
            grosor para distinguirla visualmente de las anteriores.
            """
            if not self.previous_results:
                return

            # .flatten() convierte la matriz plt.subplots(2,2) en lista
            ax1 = self.axes.flatten()[0]
            ax1.clear()
            self._clear_cursors()

            # Construye lista unificada de resultados y etiquetas para chart_generator
            all_results = [p["results"] for p in self.previous_results]
            labels = [
                f"P={p['params']['num_partitions']}, E={p['params']['num_epochs']}"
                for p in self.previous_results
            ]

            # Agrega configuración actual
            if self.current_results:
                all_results.append(self.current_results)
                labels.append(
                    f"P={self.current_params['num_partitions']}, "
                    f"E={self.current_params['num_epochs']} (actual)"
                )

            comp = prepare_comparison_chart_data(all_results, labels)

            # Se usa para detectar cuál es la última configuración (actual)
            n_configs = len(comp["configurations"])
            comp_lines = []

            # Itera sobre cada configuración
            for i, cfg in enumerate(comp["configurations"]):
                # La última configuración es la actual
                is_current = i == n_configs - 1
                (ln,) = ax1.plot(
                    cfg["x"],
                    cfg["y"],
                    "o-",
                    color=self.COLORS[i % len(self.COLORS)],
                    # Configuramos que la línea actual es más gruesa
                    linewidth=3 if is_current else 2,
                    # La línea actual es discontinua y las anteriores sólidas
                    linestyle="--" if is_current else "-",
                    markersize=5,
                    label=f"{cfg['label']} ({cfg['final_accuracy']:.1f}%)",
                )

                # Necesario para añadir tooltips interactivos después
                comp_lines.append(ln)

            ax1.set(xlabel=comp["xlabel"], ylabel=comp["ylabel"], title=comp["title"])
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)

            # Cuando el cursor pasa sobre un punto
            def _fmt_comp(sel):
                sel.annotation.set_text(
                    f"Época: {sel.target[0]:.0f}\nPrecisión: {sel.target[1]:.2f}%"
                )
                sel.annotation.get_bbox_patch().set(
                    facecolor="#ffffcc", alpha=0.95, edgecolor="#888"
                )

            # Asocia el tooltip a todas las líneas
            self._add_cursor(comp_lines, _fmt_comp)

            self.canvas.draw()

    # Crea y ejecuta aplicación
    root = tk.Tk()
    DiegoLearningApp(root)
    root.mainloop()


# ==========
# ENTRADA
# ==========


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NN_practica — Análisis de Algoritmo de Diego para MNIST"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interfaz gráfica"
    )
    parser.add_argument("--partitions", "-p", type=int, default=2)
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--experiments", "-x", type=int, default=5)
    parser.add_argument("--hidden-neurons", "-n", type=int, default=30)
    parser.add_argument("--learning-rate", "-l", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Usa multiprocessing (1 proceso por partición)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Ejecuta ambos modos (secuencial y paralelo) y compara tiempos",
    )

    args = parser.parse_args()
    os.makedirs("Data", exist_ok=True)
    os.makedirs("Results", exist_ok=True)

    if args.interactive:
        run_interactive_mode()
    else:
        run_terminal_mode(args)


if __name__ == "__main__":
    # Requerido en Windows para que multiprocessing funcione correctamente.
    # El método 'spawn' re-importa el módulo __main__ en cada proceso hijo;
    main()
