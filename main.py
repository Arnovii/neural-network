"""
Punto de entrada principal.

Ejecuta experimentos de Algoritmo de Diego con redes neuronales
y genera visualizaciones estadísticas.

MODOS DE USO:
    * Modo terminal (sin GUI)
    * Modo interactivo con GUI
"""

# ================
# IMPORTACIONES
# ================

import argparse
import json
import os
import sys

# Asegura que los módulos locales se puedan importar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Analytics.experiment_runner import run_multiple_experiments
from Analytics.statistics_engine import compute_epoch_statistics
from Analytics.chart_generator import (
    prepare_accuracy_chart_data,
    prepare_comparison_chart_data,
    prepare_convergence_data,
    prepare_distribution_data,
    prepare_partition_comparison_data,
)


# ================
# MODO TERMINAL
# ================


def run_terminal_mode(args: argparse.Namespace) -> None:
    """
    Ejecuta experimentos y muestra resultados en consola.

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
    print("RESUMEN")
    print("=" * 70)
    print(f"Precisión final promedio : {stats['mean'][-1]:.2f}%")
    print(f"Desviación estándar      : {stats['std'][-1]:.2f}%")
    print(f"Mejor precisión          : {max(stats['max']):.2f}%")
    print(f"Resultados en            : Results/experiment_{results['timestamp']}.json")

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
    """Lanza la interfaz gráfica interactiva."""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError as e:
        print(f"Error: no se pueden cargar las librerías gráficas: {e}")
        print("Instala las dependencias con: pip install matplotlib")
        sys.exit(1)

    # Dibuja los mensajes emergentes al pasar el mouse
    class ToolTip:
        """
        Proporciona una pequeña ventana emergente (tooltip) para widgets de Tkinter.

        Permite mostrar información adicional al pasar el cursor sobre un widget.
        Se puede usar para mostrar valores de gráficas, botones o cualquier widget.

        :param widget: Widget de Tkinter al que se asocia el tooltip.
        :type widget: tk.Widget

        :param text: Texto a mostrar en el tooltip.
        :type text: str
        """

        def __init__(self, widget, text):
            self.widget = widget
            self.text = text
            self.tip_window = None

            widget.bind("<Enter>", self.show_tip)
            widget.bind("<Leave>", self.hide_tip)

        def show_tip(self, event=None):
            """
            Muestra el tooltip cerca del cursor.

            Este método crea una ventana Toplevel con el texto del tooltip.
            Se llama automáticamente al entrar el cursor sobre el widget.

            :param event: Evento de Tkinter que dispara la acción (normalmente < Enter >).
                        Contiene información como posición del cursor.
            :type event: tk.Event | None
            """
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
            """
            Oculta el tooltip.

            Este método destruye la ventana del tooltip.
            Se llama automáticamente al salir el cursor del widget.

            :param event: Evento de Tkinter que dispara la acción (normalmente < Leave >).
                        Contiene información como posición del cursor.
            :type event: tk.Event | None
            """
            if self.tip_window:
                self.tip_window.destroy()
                self.tip_window = None

    class FederatedLearningApp:
        # Valores de color para los widgets
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

        # =========================
        # Inicialización de la app y creación de la interfaz
        # - Panel izquierdo: controles y sliders
        # - Panel derecho: gráficos matplotlib
        # - Barra de estado
        # =========================

        def __init__(self, root: tk.Tk) -> None:
            self.root = root

            self.root.columnconfigure(0, weight=0)  # Panel izquierdo fijo
            self.root.columnconfigure(1, weight=1)  # Panel derecho expandible
            self.root.rowconfigure(0, weight=1)

            # Establece el título
            self.root.title("NN_practica — Análisis de Algoritmo de Diego")

            # Ajusta tamaño de ventana al espacio utilizable respetando la barra de tareas
            self.root.state("zoomed")

            # Datos de ejecuciones previas para comparación
            self.previous_results: list = []
            self.color_index: int = 0
            self._pending_comparison: bool = False
            self.current_results: dict = {}
            self.current_params: dict = {}

            self._create_ui()

        # ==========================
        # SISTEMA DE HOVER (TOOLTIP)
        # ==========================

        class _HoverManager:
            """
            Muestra tooltips flotantes al pasar el mouse sobre los elementos
            graficados en una figura de matplotlib embebida en tkinter.

            Diseño:
            - Una sola anotación invisible por cada ejes (ax). Se hace
              visible y se reposiciona cuando el cursor está cerca de un
              punto de interés, y vuelve a ocultarse en caso contrario.
            - Un único event handler ``motion_notify_event`` por figura,
              registrado una sola vez. Al redibujar los paneles se llama
              a ``reset`` para limpiar los elementos registrados del ciclo
              anterior y registrar los nuevos.

            Elementos soportados:
            - ``Line2D``  → puntos de la curva (epoch, valor)
            - ``Rectangle`` (bar) → barras del histograma y del gráfico de mejora

            :param fig: Figura de matplotlib a observar
            :type fig: matplotlib.figure.Figure

            :param canvas: Canvas de tkinter que contiene la figura
            :type canvas: FigureCanvasTkAgg
            """

            # Tolerancia en puntos de pantalla para activar el tooltip
            _PICK_RADIUS = 8

            # Estilo del cuadro del tooltip
            _BBOX_STYLE = dict(
                boxstyle="round,pad=0.45",
                facecolor="#ffffcc",
                edgecolor="#888888",
                linewidth=0.8,
                alpha=0.92,
            )
            _ARROW_STYLE = dict(arrowstyle="->", color="#666666", lw=0.8)

            def __init__(self, fig, canvas) -> None:
                self.fig = fig
                self.canvas = canvas

                # Lista de (ax, lista_de_artistas_con_datos_adjuntos)
                # Se llena con register_* y se vacía con reset().
                self._entries: list = []

                # Una anotación por axes; se crea bajo demanda
                self._annotations: dict = {}

                # ID del handler de matplotlib (para poder desconectarlo)
                self._cid = fig.canvas.mpl_connect("motion_notify_event", self._on_move)

            def reset(self) -> None:
                """
                Elimina todos los artistas registrados y oculta las anotaciones.

                Debe llamarse antes de redibujar los paneles para que las
                referencias a artistas del ciclo anterior no queden activas.
                """
                self._entries.clear()
                for annot in self._annotations.values():
                    annot.set_visible(False)

            def register_lines(self, ax, lines: list, fmt: str) -> None:
                """
                Registra una lista de ``Line2D`` para mostrar tooltip.

                :param ax: Ejes que contienen las líneas
                :param lines: Lista de objetos ``Line2D``
                :param fmt: Cadena de formato. Puede usar ``{x}`` y ``{y}``,
                            o ``{epoch}`` y ``{val}`` como alias de los mismos.
                            Ejemplo: ``"Época: {x}\\nAcc: {y:.2f}%"``
                """
                self._entries.append(("lines", ax, lines, fmt))

            def register_bars(self, ax, bars, fmt: str) -> None:
                """
                Registra un ``BarContainer`` o lista de ``Rectangle`` para tooltip.

                :param ax: Ejes que contienen las barras
                :param bars: ``BarContainer`` devuelto por ``ax.bar``
                :param fmt: Cadena de formato con ``{x}`` (centro de la barra)
                            y ``{y}`` (altura). Ejemplo: ``"Mejora: {y:+.3f}%"``
                """
                self._entries.append(("bars", ax, list(bars), fmt))

            def _get_annotation(self, ax):
                """Devuelve (creando si es necesario) la anotación asociada al ax."""
                if ax not in self._annotations:
                    annot = ax.annotate(
                        "",
                        xy=(0, 0),
                        xytext=(14, 14),
                        textcoords="offset points",
                        bbox=self._BBOX_STYLE,
                        arrowprops=self._ARROW_STYLE,
                        fontsize=8.5,
                        zorder=10,
                    )
                    annot.set_visible(False)
                    self._annotations[ax] = annot
                return self._annotations[ax]

            def _on_move(self, event) -> None:
                """
                Callback de matplotlib llamado en cada movimiento del mouse.

                Itera sobre los artistas registrados para el axes bajo el cursor.
                Si encuentra uno cercano, muestra el tooltip; si no, lo oculta.
                El redibujado se hace con ``draw_idle`` para no bloquear la GUI.
                """
                updated = False

                for entry in self._entries:
                    kind, ax, artists, fmt = entry

                    # El tooltip solo se activa cuando el cursor está en ese axes
                    if event.inaxes != ax:
                        annot = self._annotations.get(ax)
                        if annot and annot.get_visible():
                            annot.set_visible(False)
                            updated = True
                        continue

                    annot = self._get_annotation(ax)
                    hit = False

                    if kind == "lines":
                        for line in artists:
                            cont, ind = line.contains(event)
                            if cont:
                                idx = ind["ind"][0]
                                xd, yd = line.get_xdata(), line.get_ydata()
                                xv, yv = float(xd[idx]), float(yd[idx])
                                text = fmt.format(x=xv, epoch=xv, y=yv, val=yv)
                                annot.xy = (xv, yv)
                                annot.set_text(text)
                                annot.set_visible(True)
                                hit = True
                                break

                    elif kind == "bars":
                        for rect in artists:
                            if not rect.get_visible():
                                continue
                            x0 = rect.get_x()
                            w = rect.get_width()
                            h = rect.get_height()
                            # Comprueba si el cursor está dentro de la barra
                            if x0 <= event.xdata <= x0 + w:
                                cx = x0 + w / 2
                                ybot = rect.get_y()
                                ytop = ybot + h
                                # Posiciona el tooltip en el centro superior
                                text = fmt.format(x=cx, y=h)
                                annot.xy = (cx, ytop)
                                annot.set_text(text)
                                annot.set_visible(True)
                                hit = True
                                break

                    if not hit and annot.get_visible():
                        annot.set_visible(False)
                        updated = True
                    elif hit:
                        updated = True

                if updated:
                    self.canvas.draw_idle()

        # ================
        # CONSTRUCCIÓN UI
        # ================

        def _create_ui(self) -> None:
            def _snap_int(var: tk.IntVar):
                """Fuerza valores enteros en los sliders de tkinter."""
                return lambda v: var.set(int(round(float(v))))

            # Panel izquierdo: controles
            ctrl = ttk.Frame(self.root, padding="10")
            ctrl.grid(row=0, column=0, sticky="ns", padx=5, pady=5)

            ttk.Label(ctrl, text="Parámetros", font=("Helvetica", 14, "bold")).pack(
                pady=10
            )

            def _add_slider(parent, label, var, lo, hi):
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
                ttk.Label(parent, textvariable=var).pack()

            self.partitions_var = tk.IntVar(value=2)
            self.epochs_var = tk.IntVar(value=5)
            self.experiments_var = tk.IntVar(value=5)
            self.hidden_var = tk.IntVar(value=30)

            # Número de particiones
            _add_slider(ctrl, "Particiones:", self.partitions_var, 1, 10)

            # Número de épocas
            _add_slider(ctrl, "Épocas:", self.epochs_var, 1, 50)

            # Número de experimentos
            _add_slider(ctrl, "Experimentos:", self.experiments_var, 1, 20)

            # Neuronas de la capa oculta
            _add_slider(ctrl, "Neuronas de la capa ocultas:", self.hidden_var, 10, 100)

            # Tasa de aprendizaje
            ttk.Label(ctrl, text="Tasa de aprendizaje:").pack(anchor=tk.W, pady=(10, 0))
            self.lr_var = tk.StringVar(value="1.0")
            ttk.Entry(ctrl, textvariable=self.lr_var, width=20).pack(pady=5)

            # Ejemplos de entrenamiento
            ttk.Label(ctrl, text="Ejemplos de entrenamiento:").pack(
                anchor=tk.W, pady=(10, 0)
            )
            self.n_train_var = tk.StringVar(value="5000")
            ttk.Entry(ctrl, textvariable=self.n_train_var, width=20).pack(pady=5)

            # Botones
            ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

            btn_run = ttk.Button(
                ctrl,
                text="Ejecutar Experimento",
                command=self._run_experiment,
            )
            btn_run.pack(fill=tk.X, pady=5)
            ToolTip(btn_run, "Ejecuta el algoritmo con los parámetros actuales")

            btn_compare = ttk.Button(
                ctrl,
                text="Comparar Configuraciones",
                command=self._compare_configurations,
            )
            btn_compare.pack(fill=tk.X, pady=5)
            ToolTip(
                btn_compare,
                "Ejecuta un nuevo experimento y lo superpone\n"
                "con los resultados actualmente mostrados.",
            )

            btn_clear = ttk.Button(
                ctrl,
                text="Limpiar Gráficos",
                command=self._clear_plots,
            )
            btn_clear.pack(fill=tk.X, pady=5)
            ToolTip(btn_clear, "Borra todas las gráficas actuales")

            btn_save = ttk.Button(
                ctrl,
                text="Guardar Resultados",
                command=self._save_results,
            )
            btn_save.pack(fill=tk.X, pady=5)
            ToolTip(btn_save, "Guarda los resultados actuales en un archivo JSON")

            # Panel derecho: gráficos
            self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
            self.fig.suptitle(
                "Análisis de Algoritmo de Diego", fontsize=14, fontweight="bold"
            )

            plot_frame = ttk.Frame(self.root)
            plot_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Hover manager: se instancia una vez y se reutiliza en cada redibujado
            self._hover = FederatedLearningApp._HoverManager(self.fig, self.canvas)

            # Barra de estado
            self.status_var = tk.StringVar(value="Listo")

            status_bar = ttk.Label(
                self.root,
                textvariable=self.status_var,
                relief=tk.SUNKEN,
                anchor=tk.W,
            )

            status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

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

            La ventana no se congela porque el entrenamiento corre en un hilo
            separado que solo escribe en una cola; este hilo principal lee la
            cola cada 100 ms con root.after y actualiza los widgets.

            :param num_experiments: Total de experimentos
            :type num_experiments: int

            :param num_epochs: Total de épocas por experimento
            :type num_epochs: int

            :return: Diccionario con los widgets de la ventana
            :rtype: dict
            """
            win = tk.Toplevel(self.root)
            win.title("Ejecutando experimentos...")
            win.geometry("520x320")
            win.resizable(False, False)
            win.grab_set()
            win.protocol("WM_DELETE_WINDOW", lambda: None)

            # Título
            ttk.Label(
                win, text="Entrenamiento en progreso", font=("Helvetica", 13, "bold")
            ).pack(pady=(18, 4))

            # Experimento actual
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
            msg_var = tk.StringVar(value="—")
            ttk.Label(
                win,
                textvariable=msg_var,
                font=("Helvetica", 9),
                foreground="#555555",
                wraplength=470,
                justify=tk.LEFT,
            ).pack(anchor=tk.W, padx=24)

            return {
                "window": win,
                "exp_label_var": exp_label_var,
                "exp_bar": exp_bar,
                "epoch_bar": epoch_bar,
                "msg_var": msg_var,
            }

        def _poll_queue(
            self, q, widgets: dict, num_experiments: int, num_epochs: int, on_done
        ) -> None:
            """
            Lee la cola de progreso y actualiza los widgets del hilo principal.

            Tkinter no es thread-safe: el hilo de entrenamiento nunca toca
            widgets directamente, solo hace ``q.put(tipo, payload)``.
            Este método se reprograma cada 100 ms con root.after hasta que
            recibe el mensaje ``'done'`` o ``'error'``.

            Tipos de mensaje:
                ``('exp',   n)``      → avanza barra de experimentos
                ``('epoch', n)``      → avanza barra de épocas
                ``('msg',   texto)``  → actualiza etiqueta de estado
                ``('done',  result)`` → entrenamiento terminado
                ``('error', exc)``    → error en el hilo secundario

            :param q: Cola compartida entre hilos, usada para enviar mensajes de progreso
            :type q: queue.Queue[tuple[str, Any]]

            :param widgets: Diccionario con los widgets de la ventana de progreso
            :type widgets: dict[str, tk.Widget | tk.StringVar | ttk.Progressbar]

            :param num_experiments: Total de experimentos a ejecutar
            :type num_experiments: int

            :param num_epochs: Total de épocas por experimento
            :type num_epochs: int

            :param on_done: Función callback que recibe los resultados finales cuando termina
            :type on_done: Callable[[dict], None]
            """
            import queue as queue_module

            try:
                while True:
                    msg_type, payload = q.get_nowait()

                    if msg_type == "exp":
                        widgets["exp_label_var"].set(
                            f"Experimento {payload} de {num_experiments}"
                        )
                        widgets["exp_bar"]["value"] = payload - 1
                        widgets["epoch_bar"]["value"] = 0

                    elif msg_type == "epoch":
                        widgets["epoch_bar"]["value"] = payload

                    elif msg_type == "msg":
                        widgets["msg_var"].set(payload)

                    elif msg_type == "done":
                        widgets["exp_bar"]["value"] = num_experiments
                        widgets["epoch_bar"]["value"] = num_epochs
                        widgets["window"].destroy()
                        on_done(payload)
                        return

                    elif msg_type == "error":
                        widgets["window"].destroy()
                        raise payload

            except queue_module.Empty:
                pass  # No hay mensajes nuevos; seguimos esperando
            except Exception as e:
                widgets["window"].destroy()
                messagebox.showerror("Error", f"Error ejecutando experimento:\n{e}")
                self.status_var.set("Error en ejecución")
                return

            # Reprograma el siguiente ciclo de polling cada 100ms
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

            :return: Diccionario de parámetros listo para run_multiple_experiments
            :rtype: dict
            :raises ValueError: Si algún campo tiene un valor inválido
            """
            return {
                "num_partitions": self.partitions_var.get(),
                "num_epochs": self.epochs_var.get(),
                "num_experiments": self.experiments_var.get(),
                "hidden_neurons": self.hidden_var.get(),
                "learning_rate": float(self.lr_var.get()),
                "n_train": int(self.n_train_var.get()),
                "verbose": False,
            }

        # =====================================================================
        # - Ejecutar experimento en hilo separado
        # - Comparar configuraciones guardando resultados anteriores
        # - Limpiar gráficos
        # - Guardar resultados en JSON
        # =====================================================================

        def _run_experiment(self) -> None:
            """
            Lanza un experimento en un hilo secundario.

            El hilo de entrenamiento nunca toca widgets; solo escribe en la
            cola. El hilo principal lee la cola cada 100 ms mediante
            _poll_queue y actualiza la ventana de progreso.
            """
            import queue
            import threading

            try:
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

            def _training_thread() -> None:
                """
                Hilo secundario de entrenamiento.

                Solo escribe en la cola — nunca accede a widgets de tkinter.
                """
                current_exp = [0]

                def on_progress(msg: str) -> None:
                    q.put(("msg", msg))

                    # Detecta inicio de nuevo experimento: "EXPERIMENTO X/Y"
                    if msg.startswith("EXPERIMENTO"):
                        try:
                            n = int(msg.split()[1].split("/")[0])
                            current_exp[0] = n
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
                        **params, on_progress=on_progress
                    )
                    q.put(("done", results))
                except Exception as e:
                    q.put(("error", e))

            # ========================
            # CALLBACK AL COMPLETAR
            # ========================

            def _on_done(results: dict) -> None:
                self.current_results = results
                self.current_params = params
                self._plot_results(results, params)
                self.status_var.set(
                    f"Completado — Precisión final: "
                    f"{results['final_mean_accuracy']:.2f}%"
                )

            threading.Thread(target=_training_thread, daemon=True).start()
            self.root.after(
                100,
                lambda: self._poll_queue(
                    q, widgets, num_experiments, num_epochs, _on_done
                ),
            )

        def _compare_configurations(self) -> None:
            """Guarda la configuración actual y lanza un nuevo experimento para comparar."""
            if not hasattr(self, "current_results"):
                messagebox.showwarning("Advertencia", "Primero ejecuta un experimento")
                return

            # Guarda configuración actual antes de ejecutar la nueva
            self.previous_results.append(
                {
                    "results": self.current_results,
                    "params": self.current_params,
                }
            )
            self.color_index += 1

            # Marca que al terminar se debe ejecutar _plot_comparison
            self._pending_comparison = True
            self._run_experiment()

        def _clear_plots(self) -> None:
            """Limpia todos los gráficos y reinicia el historial de comparaciones."""
            for ax in self.axes.flatten():
                ax.clear()
            self.previous_results = []
            self.color_index = 0
            self.canvas.draw()
            self.status_var.set("Gráficos limpiados")

        def _save_results(self) -> None:
            """Guarda los resultados del experimento actual en un archivo JSON."""
            if not hasattr(self, "current_results"):
                messagebox.showwarning("Advertencia", "No hay resultados para guardar")
                return

            os.makedirs("Results", exist_ok=True)
            filename = f"results/experiment_{self.current_results['timestamp']}.json"

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
                )

            messagebox.showinfo("Éxito", f"Resultados guardados en:\n{filename}")
            self.status_var.set(f"Guardado: {filename}")

        # ================
        # RENDERIZADO
        # ================

        def _current_color(self) -> str:
            return self.COLORS[self.color_index % len(self.COLORS)]

        # =============================================================
        # Dibujado de gráficos:
        # - Panel 1: Curva de aprendizaje con banda ±1σ
        # - Panel 2: Comparación por partición
        # - Panel 3: Distribución de precisión final
        # - Panel 4: Mejora por época
        # - Comparación de configuraciones anteriores
        # =============================================================

        def _plot_results(
            self, results: dict, params: dict, comparison: bool = False
        ) -> None:
            """
            Renderiza los cuatro paneles de análisis.

            La preparación de datos se delega por completo a chart_generator;
            este método solo llama a matplotlib con datos ya listos.

            :param results: Resultado de run_multiple_experiments
            :type results: dict
            :param params: Parámetros del experimento
            :type params: dict
            :param comparison: Si True, no limpia los ejes antes de dibujar
            :type comparison: bool
            """
            ax1, ax2, ax3, ax4 = self.axes.flatten()
            color = self._current_color()
            label = f"P={params['num_partitions']}, E={params['num_epochs']}"

            # Limpia artistas del ciclo anterior para que el hover no reciba
            # referencias obsoletas a líneas/barras que ya no están en pantalla
            self._hover.reset()

            if not comparison:
                for ax in self.axes.flatten():
                    ax.clear()

            histories = results["all_histories"]

            # Panel 1 — Curva de aprendizaje con banda ±1σ
            acc = prepare_accuracy_chart_data(histories)
            (line1,) = ax1.plot(
                acc["x"], acc["y_mean"], "o-", color=color, linewidth=2, label=label
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
            self._hover.register_lines(
                ax1, [line1], "Época: {x:.0f}\nAcc media: {y:.2f}%"
            )

            # Panel 2 — Comparación por partición (último experimento)
            part = prepare_partition_comparison_data([histories[-1]])
            part_lines = []
            if part:
                for p in part["partitions"]:
                    (ln,) = ax2.plot(
                        p["x"], p["y"], "o-", label=f"Partición {p['id']}", alpha=0.7
                    )
                    part_lines.append((ln, p["id"]))
            ax2.set(
                xlabel=part.get("xlabel", "Época"),
                ylabel=part.get("ylabel", "Precisión (%)"),
                title="Comparación por Partición (último experimento)",
            )
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
            for ln, pid in part_lines:
                self._hover.register_lines(
                    ax2, [ln], f"Partición {pid}\nÉpoca: {{x:.0f}}\nAcc: {{y:.2f}}%"
                )

            # Panel 3 — Distribución de precisión final
            dist = prepare_distribution_data(histories)
            bin_w = dist["bins"][1] - dist["bins"][0]
            bin_centers = [
                (dist["bins"][i] + dist["bins"][i + 1]) / 2
                for i in range(len(dist["bins"]) - 1)
            ]
            bars3 = ax3.bar(
                bin_centers,
                dist["counts"],
                width=bin_w * 0.9,
                color=color,
                alpha=0.7,
                edgecolor="black",
            )
            ax3.axvline(
                dist["mean"],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Media: {dist['mean']:.2f}%",
            )
            ax3.set(xlabel=dist["xlabel"], ylabel=dist["ylabel"], title=dist["title"])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            self._hover.register_bars(ax3, bars3, "Acc: {x:.2f}%\nFrecuencia: {y:.0f}")

            # Panel 4 — Mejora por época
            conv = prepare_convergence_data(histories)
            bars4 = ax4.bar(conv["x"], conv["y"], color=color, alpha=0.7)
            ax4.axhline(0, color="black", linewidth=0.5)
            ax4.set(xlabel=conv["xlabel"], ylabel=conv["ylabel"], title=conv["title"])
            ax4.grid(True, alpha=0.3)
            self._hover.register_bars(ax4, bars4, "Época: {x:.0f}\nMejora: {y:+.3f}%")

            self.fig.tight_layout()
            self.canvas.draw()

            # Si estaba pendiente una comparación, la ejecuta ahora que hay datos frescos
            if self._pending_comparison:
                self._pending_comparison = False
                self._plot_comparison()

        def _plot_comparison(self) -> None:
            """
            Superpone las curvas de precisión de todas las configuraciones.

            La configuración actual se dibuja con línea discontinua y mayor
            grosor para distinguirla visualmente de las anteriores.
            Registra todas las líneas en el hover manager.
            """
            if not self.previous_results:
                return

            ax1 = self.axes.flatten()[0]
            ax1.clear()
            self._hover.reset()

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

            comp = prepare_comparison_chart_data(all_results, labels)
            n_configs = len(comp["configurations"])

            for i, cfg in enumerate(comp["configurations"]):
                is_current = i == n_configs - 1
                (ln,) = ax1.plot(
                    cfg["x"],
                    cfg["y"],
                    "o-",
                    color=self.COLORS[i % len(self.COLORS)],
                    linewidth=3 if is_current else 2,
                    linestyle="--" if is_current else "-",
                    label=f"{cfg['label']} ({cfg['final_accuracy']:.1f}%)",
                )
                lbl = cfg["label"]
                self._hover.register_lines(
                    ax1, [ln], f"{lbl}\nÉpoca: {{x:.0f}}\nAcc: {{y:.2f}}%"
                )

            ax1.set(xlabel=comp["xlabel"], ylabel=comp["ylabel"], title=comp["title"])
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            self.canvas.draw()

    # Crea y ejecuta aplicación
    root = tk.Tk()
    FederatedLearningApp(root)
    root.mainloop()


# ==========
# ENTRADA
# ==========


def main() -> None:
    """
    Parseo de argumentos y elección de modo terminal o GUI
    """
    parser = argparse.ArgumentParser(
        description="NN_practica — Análisis de Algoritmo de Diego para MNIST"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Ejecutar con interfaz gráfica"
    )
    parser.add_argument("--partitions", "-p", type=int, default=2)
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--experiments", "-x", type=int, default=5)
    parser.add_argument("--hidden-neurons", "-n", type=int, default=30)
    parser.add_argument("--learning-rate", "-l", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=5000)

    args = parser.parse_args()
    os.makedirs("Data", exist_ok=True)
    os.makedirs("Results", exist_ok=True)

    if args.interactive:
        run_interactive_mode()
    else:
        run_terminal_mode(args)


if __name__ == "__main__":
    main()
