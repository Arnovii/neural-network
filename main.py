"""
main.py
Punto de entrada principal.

Ejecuta experimentos de Algoritmo de Diego con redes neuronales
y genera visualizaciones estadísticas con gráficas interactivas.

MODOS DE USO:
    * Modo terminal (sin GUI)
    * Modo interactivo con GUI (gráficas con hover tooltips)
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
    prepare_experiment_rsd_data,
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

    class DiegoLearningApp:
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

        # =============================================================
        # Inicialización de la app y creación de la interfaz
        # - Panel izquierdo: controles y sliders
        # - Panel derecho: gráficos matplotlib
        # - Barra de estado
        # =============================================================

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

            # Inicializa atributos para resultados de experimentos
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
                """Fuerza valores enteros en los sliders de tkinter."""
                return lambda v: var.set(int(round(float(v))))

            # Panel izquierdo: controles
            # Crea un frame principal que contendrá todo el panel izquierdo
            ctrl_container = ttk.Frame(self.root, width=260)

            # Configura las propiedades del panel, como su margen y que sólo se estire verticalmente
            ctrl_container.grid(row=0, column=0, sticky="ns", padx=5, pady=5)

            # Evita que el grid lo estire horizontalmente
            ctrl_container.grid_propagate(False)

            # Permite que el panel sea desplazable verticalmente
            canvas = tk.Canvas(ctrl_container, width=260, highlightthickness=0)
            scrollbar = ttk.Scrollbar(
                ctrl_container, orient="vertical", command=canvas.yview
            )

            scrollable_frame = ttk.Frame(canvas, padding="10")

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="y", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Ahora ctrl es el frame interno
            ctrl = scrollable_frame

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

                # Al confirmar (Enter o pérdida de foco) el valor se valida,
                # se clipea al rango [lo, hi] y se escribe en var, lo que
                # mueve el slider automáticamente al ser la misma variable.
                entry = ttk.Entry(parent, textvariable=var, width=6, justify="center")
                entry.pack(pady=(0, 4))

                def _commit(event=None):
                    try:
                        val = int(round(float(var.get())))
                    except (ValueError, tk.TclError):
                        val = lo
                    var.set(max(lo, min(hi, val)))

                entry.bind("<Return>",   _commit)
                entry.bind("<FocusOut>", _commit)

            # Crea variables ligadas con cada uno de los sliders
            self.partitions_var = tk.IntVar(value=2)
            self.epochs_var = tk.IntVar(value=5)
            self.experiments_var = tk.IntVar(value=5)
            self.hidden_var = tk.IntVar(value=30)

            # Número de particiones
            _add_slider(ctrl, "Particiones (1 - 10):", self.partitions_var, 1, 10)

            # Número de experimentos
            _add_slider(ctrl, "Épocas (50 - 1.000):", self.epochs_var, 50, 1000)

            # Neuronas de la capa oculta
            _add_slider(ctrl, "Experimentos (1 - 20):", self.experiments_var, 1, 20)

            # Tasa de aprendizaje
            _add_slider(ctrl, "Neuronas ocultas (10 - 100):", self.hidden_var, 10, 100)

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
                ctrl, text="Ejecutar Experimento", command=self._run_experiment
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
                "Ejecuta un nuevo experimento y lo superpone\ncon los resultados actuales.",
            )

            btn_clear = ttk.Button(
                ctrl, text="Limpiar Gráficos", command=self._clear_plots
            )
            btn_clear.pack(fill=tk.X, pady=5)
            ToolTip(btn_clear, "Borra todas las gráficas actuales")

            btn_save = ttk.Button(
                ctrl, text="Guardar Resultados", command=self._save_results
            )
            btn_save.pack(fill=tk.X, pady=5)
            ToolTip(btn_save, "Guarda los resultados actuales en JSON")

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
            status_bar = ttk.Label(
                self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
            )
            status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

        # ========================
        # TOOLTIPS INTERACTIVOS
        # ========================

        def _add_cursor(self, artists, fmt_func=None):
            """
            Agrega tooltips interactivos a los artistas de matplotlib.

            Al pasar el mouse sobre un punto de datos, muestra una anotación
            con el valor exacto.

            :param artists: Lista de artistas (líneas, barras) de matplotlib
            :param fmt_func: Función opcional para formatear el texto del tooltip
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
                    def on_add(sel):
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
            for c in self._active_cursors:
                try:
                    c.remove()
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
            # Crea ventana secundaria sobre la principal
            win = tk.Toplevel(self.root)
            win.title("Ejecutando experimentos...")
            win.geometry("520x320")
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
                mode="determinate",  # Muestra progreso en proporción a un máximo
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

            :param q: Cola compartida entre hilos, usada para enviar mensajes de progreso como tuplas (tipo_mensaje, payload)
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
                    # q.get_nowait() intenta obtener un mensaje sin bloquear el hilo principal
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
                    elif msg_type == "done":  # El entrenamiento ha terminado
                        widgets["exp_bar"]["value"] = num_experiments
                        widgets["epoch_bar"]["value"] = num_epochs
                        widgets["window"].destroy()  # Cierra la ventana
                        on_done(
                            payload
                        )  # Llama al callback _on_done con los resultados
                        return
                    elif msg_type == "error": # Ocurrió una excepción en el hilo secundario
                        widgets["window"].destroy() # Cierra la ventana
                        raise payload # Propaga la excepción al hilo principal
            except queue_module.Empty:
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

        def _run_experiment(self) -> None:
            """
            Lanza un experimento en un hilo secundario para no bloquear la UI.

            El hilo de entrenamiento nunca toca widgets; solo escribe en la
            cola. El hilo principal lee la cola cada 100 ms mediante
            _poll_queue y actualiza la ventana de progreso.
            """
            import queue
            import threading

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

                Nunca accede a widgets de tkinter. Solo escribe en la cola ``q``
                mensajes de progreso ("msg", "exp", "epoch"), de finalización
                ("done") o errores ("error").

                Solo escribe en la cola, nunca accede a widgets de tkinter.
                """
                current_exp = [0]  # Experimento actual

                # Detecta mensajes del entrenamiento y los convierte en mensajes para la cola
                def on_progress(msg):

                    # Envía el mensaje genérico a la cola
                    q.put(("msg", msg))

                    # Detecta inicio de nuevo experimento: "EXPERIMENTO X/Y"
                    if msg.startswith("EXPERIMENTO"):
                        try:
                            n = int(msg.split()[1].split("/")[0])
                            current_exp[0] = n # Actualiza el experimento actual
                            q.put(("exp", n)) # Envía a la cola para que actualice la barra de progreso
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

        def _compare_configurations(self):
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
            """Retorna el color que debe usarse para el experimento actual"""

            # Se usa % para que cuando se acaben los colores, vuelva al inicio
            return self.COLORS[self.color_index % len(self.COLORS)]

        # ================
        # RENDERIZADO
        # ================

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

            if not comparison:
                self._clear_cursors()
                for ax in self.axes.flatten():
                    ax.clear()

            histories = results["all_histories"]

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

            # Tooltip: muestra época, media y ±std
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

            # Panel 2 — Comparación por partición
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

            # Panel 3 — Precisión por experimento con RSD
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
                label=f"Promedio: {rsd_data['mean']:.2f}% ± {rsd_data['std']:.2f}% (σ)",
                zorder=4,
            )

            # Bandas ±1σ. Muestra RSD e interpretación cualitativa
            ax3.axhline(
                rsd_data["upper"],
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label=f"RSD: {rsd_data['rsd']:.2f}%  ({rsd_data['interpretation']})",
                zorder=4,
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

            # Tooltip: muestra experimento, precisión y distancia al promedio
            rsd_snapshot = rsd_data  # captura para el closure

            def _fmt_rsd(sel):
                idx = int(round(sel.target[0])) - 1
                if 0 <= idx < len(rsd_snapshot["y"]):
                    val = rsd_snapshot["y"][idx]
                    diff = val - rsd_snapshot["mean"]
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
            Superpone las curvas de precisión de todas las configuraciones.

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
                    cfg["x"], # Épocas
                    cfg["y"], # Precisión
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

    args = parser.parse_args()
    os.makedirs("Data", exist_ok=True)
    os.makedirs("Results", exist_ok=True)

    if args.interactive:
        run_interactive_mode()
    else:
        run_terminal_mode(args)


if __name__ == "__main__":
    main()


"""
EXPLICACIÓN SOBRE EL USO DE HILOS PARA LA UI

El problema principal es que Tkinter no es thread-safe, es decir, no se
pueden modificar widgets desde otro hilo.


1. El problema es que la GUI se congela

Tkinter funciona con un hilo principal que:
* Dibuja la interfaz
* Escucha eventos (clics, sliders, etc.)
* Redibuja la pantalla

Ese hilo corre algo llamado event loop:

* root.mainloop()

Si haces directamente con el botón:

* results = run_multiple_experiments(...)

Y el entrenamiento tarda 30 segundos, entonces:

* Durante esos 30 segundos el hilo principal está ocupado
* No puede redibujar
* No puede responder
* La ventana parece "congelada"


2. ¿Qué hace tu código para evitar eso?

Cuando presionas "Ejecutar Experimento", se llama:

* self._run_experiment()

Dentro de ese método ocurre esto:

2.1 Paso 1 — Se crea una cola:

* q = queue.Queue()

Esa cola es un canal seguro de comunicación entre hilos.
* El hilo de entrenamiento escribe en la cola
* El hilo principal lee la cola

Esto es clave porque Tkinter NO es thread-safe.

2.2 Paso 2 — Se crea un hilo secundario:

* threading.Thread(target=_training_thread, daemon=True).start()

Esto significa:

* El entrenamiento corre en un hilo aparte
* El hilo principal queda libre
* La GUI no se bloquea


3. ¿Qué hace exactamente el hilo secundario?

Dentro está esta función:

* def _training_thread():

Ese hilo hace:

* results = run_multiple_experiments(...)

Pero NO toca la GUI.
En vez de eso, manda mensajes a la cola:

* q.put(("msg", texto))
* q.put(("epoch", epoch))
* q.put(("exp", n))
* q.put(("done", results))

Es decir, el significado del mensaje es:

* "msg"	Actualiza texto de estado
* "epoch"	Avanza barra de época
* "exp"	Avanza barra de experimento
* "done"	Terminó todo
* "error"	Hubo error


4. ¿Quién lee la cola?

El hilo principal.
Esto lo hace este método:

* self._poll_queue(...)

Y se programa así:

* self.root.after(100, ...)

¿Qué es root.after(100, ...)?
Le dice a Tkinter:

*  "En 100 milisegundos, ejecuta esta función."

Eso crea un ciclo de polling:

* cada 100 ms: revisar si hay mensajes en la cola


5. Flujo completo del sistema

Voy a dibujártelo mentalmente:

┌────────────────────────┐
│   HILO PRINCIPAL       │
│   (Tkinter GUI)        │
│                        │
│  - Dibuja interfaz     │
│  - Lee cola cada 100ms │
│  - Actualiza barras    │
└───────────▲────────────┘
            │
            │ queue.Queue()
            │
┌───────────┴──────────────┐
│   HILO SECUNDARIO        │
│   (Entrenamiento)        │
│                          │
│ run_multiple_experiments |
│                          │
│ q.put("epoch", n)        │
│ q.put("msg", texto)      │
│ q.put("done", result)    │
└──────────────────────────┘


6. ¿Por qué esto evita que se congele?

Porque ahora:

* El entrenamiento corre en otro hilo
* El hilo principal nunca se bloquea
* Solo hace microtareas cada 100ms

Entonces la GUI siempre responde.


7. Parte clave: el callback on_progress

En experiment_runner.py tienes esto:

def _notify(msg: str):
    if on_progress is not None:
        on_progress(msg)

Ese on_progress viene desde la GUI.

En _training_thread() tú defines:

def on_progress(msg):
    q.put(("msg", msg))

Es decir:

* experiment_runner → llama on_progress
* on_progress → mete mensaje en cola
* cola → la lee el hilo principal

Es un sistema completamente desacoplado.


8. Lo más importante que debes entender

Hay tres reglas fundamentales aquí:

* Regla 1: Tkinter solo puede ser modificado por el hilo principal.

* Regla 2: El entrenamiento pesado nunca debe correr en el hilo principal.

* Regla 3: La comunicación entre hilos debe hacerse con una cola.


9. ¿Qué pasaría si quitaras los hilos?

Si cambiaras esto:

* threading.Thread(...).start()

Por:

* _training_thread()

La ventana se congelaría completamente.


10. Conceptualmente lo que construiste es:

Se puede resumir en:

* El hilo secundario PRODUCE eventos

* El hilo principal CONSUME eventos
"""