"""
ps_gui.py

Interfaz gráfica del Parameter Server para el algoritmo de Diego
distribuido con Data-Oriented Parallelism.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
    python ps_gui.py

Los Workers se lanzan a mano en cada máquina:
    python worker.py --id 0 --server-host <IP_DE_ESTA_MÁQUINA>
    python worker.py --id 1 --server-host <IP_DE_ESTA_MÁQUINA>

──────────────────────────────────────────────────────────────────
ARQUITECTURA DE HILOS
──────────────────────────────────────────────────────────────────
Tkinter no es thread-safe: ningún widget puede tocarse desde un
hilo secundario. La solución es el mismo patrón productor/consumidor
de main.py, adaptado al nuevo flujo:

  ┌─────────────────────────┐        ┌──────────────────────────────┐
  │  HILO PRINCIPAL         │        │  HILO SECUNDARIO             │
  │  (Tkinter event loop)   │        │  (ParameterServer.run())     │
  │                         │        │                              │
  │  - Dibuja interfaz      │◄───────│  on_worker_connected → q     │
  │  - Lee cola cada 100 ms │ Queue()│  on_gradients_received → q   │
  │  - Actualiza tabla WK   │        │  on_epoch_end → q            │
  │  - Actualiza gráficas   │        │                              │
  └─────────────────────────┘        └──────────────────────────────┘

Tipos de mensaje en la cola:
    ("worker_connected",  {"id": int, "addr": str})
    ("gradients_received",{"worker_id": int, "epoch": int,
                           "loss": float, "accuracy": float})
    ("epoch_end",         {"epoch": int, "total": int,
                           "accuracy": float, "loss": float})
    ("training_done",     history: dict)
    ("error",             exc: Exception)

──────────────────────────────────────────────────────────────────
LAYOUT
──────────────────────────────────────────────────────────────────
┌──────────────┬──────────────────────────────────────────────────┐
│  Panel       │  Panel derecho                                   │
│  izquierdo   │                                                  │
│              │  ┌───────────────────────────────────────────┐   │
│  Parámetros  │  │  Tabla de Workers                         │   │
│  del PS      │  │  ID │ Dirección IP  │ Estado  │ Gradientes│   │
│              │  └───────────────────────────────────────────┘   │
│              │  ┌─────────────────┐  ┌──────────────────────┐   │
│  Botón       │  │  Accuracy/época │  │  Loss/época          │   │
│  Iniciar     │  │  (actualización │  │  (actualización      │   │
│              │  │   en tiempo     │  │   en tiempo          │   │
│              │  │   real)         │  │   real)              │   │
│              │  └─────────────────┘  └──────────────────────┘   │
│              │  ┌───────────────────────────────────────────┐   │
│              │  │  Log de mensajes recientes                │   │
│              │  └───────────────────────────────────────────┘   │
└──────────────┴──────────────────────────────────────────────────┘
│  Barra de estado                                                  │
└───────────────────────────────────────────────────────────────────┘
"""

import os
import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError as e:
    print(f"Error: no se pueden cargar las librerías gráficas: {e}")
    print("Instala con: pip install matplotlib")
    sys.exit(1)

import numpy as np

from Distributed.parameter_server import ParameterServer
from Utils.math_utils import xavier_initialization, vector_zeros


# ================================================================
# TOOLTIP (reutilizado de main.py)
# ================================================================


class ToolTip:
    """Muestra un tooltip al pasar el cursor sobre un widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget      = widget
        self.text        = text
        self.tip_window  = None
        self._after_id   = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        if self._after_id:
            self.widget.after_cancel(self._after_id)
        self._after_id = self.widget.after(500, self._show)

    def _show(self):
        self._after_id = None
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left",
                 background="#ffffe0", relief="solid",
                 borderwidth=1, font=("Helvetica", 9)).pack(ipadx=5, ipady=3)

    def hide_tip(self, event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# ================================================================
# APLICACIÓN PRINCIPAL
# ================================================================


class DistributedPSApp:
    """
    Interfaz gráfica del Parameter Server distribuido.

    Permite configurar y lanzar el PS, observar en tiempo real qué
    Workers están conectados, qué gradientes han llegado en cada
    época, y ver las curvas de accuracy y loss actualizarse conforme
    avanza el entrenamiento.
    """

    # Colores por Worker ID (se reutilizan si hay más de 8 workers)
    WORKER_COLORS = [
        "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
        "#F44336", "#00BCD4", "#795548", "#E91E63",
    ]

    # Estados posibles de un Worker
    _ST_WAITING   = "Esperando"
    _ST_CONNECTED = "Conectado"
    _ST_COMPUTING = "Calculando"
    _ST_DONE      = "✓ Listo"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Parameter Server — Algoritmo de Diego Distribuido")
        self.root.state("zoomed")
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Cola de comunicación entre el hilo del PS y Tkinter
        self._q: queue.Queue = queue.Queue()

        # Estado interno de Workers
        self._worker_status: dict  = {}   # {id: str}
        self._worker_grads:  dict  = {}   # {id: int}  gradientes recibidos

        # Historial para las gráficas
        self._acc_history:   list  = []
        self._loss_history:  list  = []

        # Número total de épocas (se fija al iniciar)
        self._total_epochs: int = 0

        # Flag para evitar doble arranque
        self._running: bool = False

        self._build_ui()

    # ================================================================
    # CONSTRUCCIÓN DE LA UI
    # ================================================================

    def _build_ui(self) -> None:
        """Construye todos los widgets de la interfaz."""
        self._build_left_panel()
        self._build_right_panel()
        self._build_status_bar()

    # ── Panel izquierdo: parámetros ───────────────────────────────

    def _build_left_panel(self) -> None:
        """Crea el panel scrollable de configuración del PS."""

        def _snap_int(var: tk.IntVar):
            return lambda v: var.set(int(round(float(v))))

        def _make_int_validator(parent, max_digits: int):
            def _validate(new_value):
                return new_value == "" or (
                    len(new_value) <= max_digits and new_value.isdigit()
                )
            return (parent.register(_validate), "%P")

        def _add_slider(parent, label, var, lo, hi):
            max_digits = len(str(hi))
            ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(10, 0))
            ttk.Scale(parent, from_=lo, to=hi, orient=tk.HORIZONTAL,
                      variable=var, length=200,
                      command=_snap_int(var)).pack(fill=tk.X, pady=5)
            vcmd  = _make_int_validator(parent, max_digits)
            entry = ttk.Entry(parent, textvariable=var, width=max_digits + 1,
                              justify="center", validate="key",
                              validatecommand=vcmd)
            entry.pack(pady=(0, 4))

            def _commit(event=None):
                try:
                    val = int(round(float(var.get())))
                except (ValueError, tk.TclError):
                    val = lo
                var.set(max(lo, min(hi, val)))

            entry.bind("<Return>",   _commit)
            entry.bind("<FocusOut>", _commit)

        def _add_float_input(parent, label, var, lo, hi, max_chars=8):
            ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(10, 0))

            def _validate(new_value):
                return (
                    new_value == ""
                    or (len(new_value) <= max_chars
                        and new_value.count(".") <= 1
                        and all(c in "0123456789." for c in new_value))
                )

            vcmd  = (parent.register(_validate), "%P")
            entry = ttk.Entry(parent, textvariable=var, width=max_chars + 1,
                              justify="center", validate="key",
                              validatecommand=vcmd)
            entry.pack(pady=(0, 6))

            def _commit(event=None):
                try:
                    val = float(var.get())
                except (ValueError, tk.TclError):
                    val = lo
                var.set(max(lo, min(hi, val)))

            entry.bind("<Return>",   _commit)
            entry.bind("<FocusOut>", _commit)

        def _add_integer_input(parent, label, var, lo, hi, max_digits=6):
            ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(10, 0))
            vcmd  = _make_int_validator(parent, max_digits)
            entry = ttk.Entry(parent, textvariable=var, width=max_digits + 1,
                              justify="center", validate="key",
                              validatecommand=vcmd)
            entry.pack(pady=(0, 6))

            def _commit(event=None):
                try:
                    val = int(var.get())
                except (ValueError, tk.TclError):
                    val = lo
                var.set(max(lo, min(hi, val)))

            entry.bind("<Return>",   _commit)
            entry.bind("<FocusOut>", _commit)

        def _add_text_input(parent, label, var, max_chars=20):
            ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(10, 0))
            ttk.Entry(parent, textvariable=var,
                      width=max_chars).pack(pady=(0, 6), fill=tk.X)

        # ── Contenedor scrollable ─────────────────────────────────
        container = ttk.Frame(self.root, width=270)
        container.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        container.grid_propagate(False)

        canvas    = tk.Canvas(container, width=270, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical",
                                  command=canvas.yview)
        frame     = ttk.Frame(canvas, padding="10")
        frame.bind("<Configure>",
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ── Título ────────────────────────────────────────────────
        ttk.Label(frame, text="Parameter Server",
                  font=("Helvetica", 14, "bold")).pack(pady=10)

        # ── Variables ─────────────────────────────────────────────
        self._v_host     = tk.StringVar(value="0.0.0.0")
        self._v_port     = tk.IntVar(value=9999)
        self._v_workers  = tk.IntVar(value=2)
        self._v_epochs   = tk.IntVar(value=10)
        self._v_hidden   = tk.IntVar(value=30)
        self._v_lr       = tk.StringVar(value="0.1")
        self._v_n_train  = tk.StringVar(value="10000")
        self._v_seed     = tk.StringVar(value="")

        # ── Sección: Red ──────────────────────────────────────────
        ttk.Label(frame, text="Red y datos",
                  font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(14, 0))
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        _add_slider(frame,       "Épocas (1 – 200):",               self._v_epochs,  1, 200)
        _add_slider(frame,       "Workers esperados (1 – 16):",      self._v_workers, 1,  16)
        _add_slider(frame,       "Neuronas ocultas (10 – 100):",     self._v_hidden, 10, 100)
        _add_float_input(frame,  "Tasa de aprendizaje:",             self._v_lr,   0.0001, 10.0)
        _add_integer_input(frame,"Ejemplos de entrenamiento:",       self._v_n_train, 100, 60000)
        _add_text_input(frame,   "Semilla aleatoria (vacío = aleat.):", self._v_seed)

        # ── Sección: Conexión ─────────────────────────────────────
        ttk.Label(frame, text="Conexión TCP",
                  font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(18, 0))
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        _add_text_input(frame, "Host (IP de escucha):", self._v_host)
        _add_integer_input(frame, "Puerto:", self._v_port, 1024, 65535, max_digits=5)

        # ── Botones ───────────────────────────────────────────────
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(18, 8))

        self._btn_start = ttk.Button(
            frame, text="▶  Iniciar Parameter Server",
            command=self._start_server,
        )
        self._btn_start.pack(fill=tk.X, pady=5)
        ToolTip(self._btn_start,
                "Abre el socket TCP y espera a que se conecten todos\n"
                "los Workers antes de iniciar el entrenamiento.")

        btn_clear = ttk.Button(frame, text="Limpiar gráficas",
                               command=self._clear_plots)
        btn_clear.pack(fill=tk.X, pady=5)
        ToolTip(btn_clear, "Borra las curvas de accuracy y loss.")

    # ── Panel derecho ─────────────────────────────────────────────

    def _build_right_panel(self) -> None:
        """
        Construye la zona principal: tabla de Workers, gráficas
        en tiempo real y log de mensajes.
        """
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ── Tabla de Workers ──────────────────────────────────────
        workers_frame = ttk.LabelFrame(right, text="Workers", padding=6)
        workers_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        cols = ("ID", "Dirección IP", "Estado", "Gradientes recibidos")
        self._wk_tree = ttk.Treeview(
            workers_frame, columns=cols, show="headings",
            height=5, selectmode="none",
        )
        for col in cols:
            self._wk_tree.heading(col, text=col)
        self._wk_tree.column("ID",                    width=50,  anchor="center")
        self._wk_tree.column("Dirección IP",           width=160, anchor="center")
        self._wk_tree.column("Estado",                 width=120, anchor="center")
        self._wk_tree.column("Gradientes recibidos",   width=160, anchor="center")
        self._wk_tree.pack(fill=tk.X)

        # Barra de progreso de Workers conectados
        wk_prog_frame = ttk.Frame(workers_frame)
        wk_prog_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(wk_prog_frame, text="Conectados:").pack(side=tk.LEFT)
        self._wk_progress = ttk.Progressbar(
            wk_prog_frame, orient=tk.HORIZONTAL, length=300,
            mode="determinate", maximum=1,
        )
        self._wk_progress.pack(side=tk.LEFT, padx=8)
        self._wk_count_var = tk.StringVar(value="0 / ?")
        ttk.Label(wk_prog_frame, textvariable=self._wk_count_var).pack(side=tk.LEFT)

        # ── Gráficas + log ────────────────────────────────────────
        plots_and_log = ttk.Frame(right)
        plots_and_log.grid(row=1, column=0, sticky="nsew")
        plots_and_log.rowconfigure(0, weight=3)
        plots_and_log.rowconfigure(1, weight=1)
        plots_and_log.columnconfigure(0, weight=1)

        # Gráficas matplotlib
        plots_frame = ttk.Frame(plots_and_log)
        plots_frame.grid(row=0, column=0, sticky="nsew")

        self._fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        self._ax_acc, self._ax_loss = axes

        self._fig.suptitle(
            "Entrenamiento Distribuido — Algoritmo de Diego",
            fontsize=13, fontweight="bold",
        )
        self._setup_axes()

        self._canvas = FigureCanvasTkAgg(self._fig, master=plots_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Barra de progreso de épocas
        epoch_frame = ttk.Frame(plots_and_log)
        epoch_frame.grid(row=0, column=0, sticky="s", pady=(0, 4))
        ttk.Label(epoch_frame, text="Época:").pack(side=tk.LEFT)
        self._epoch_bar = ttk.Progressbar(
            epoch_frame, orient=tk.HORIZONTAL, length=500,
            mode="determinate", maximum=1,
        )
        self._epoch_bar.pack(side=tk.LEFT, padx=8)
        self._epoch_var = tk.StringVar(value="—")
        ttk.Label(epoch_frame, textvariable=self._epoch_var,
                  width=12).pack(side=tk.LEFT)

        # Log de mensajes
        log_frame = ttk.LabelFrame(plots_and_log, text="Log", padding=4)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log_text = tk.Text(
            log_frame, height=6, state=tk.DISABLED,
            font=("Courier", 9), bg="#1e1e1e", fg="#d4d4d4",
            wrap=tk.WORD, relief=tk.FLAT,
        )
        log_scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

    def _build_status_bar(self) -> None:
        """Crea la barra de estado en la parte inferior."""
        self._status_var = tk.StringVar(value="Listo — configura los parámetros e inicia el servidor.")
        ttk.Label(
            self.root, textvariable=self._status_var,
            relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2),
        ).grid(row=1, column=0, columnspan=2, sticky="ew")

    # ================================================================
    # HELPERS DE UI
    # ================================================================

    def _setup_axes(self) -> None:
        """Configura los ejes de las dos gráficas."""
        self._ax_acc.set_title("Precisión por época")
        self._ax_acc.set_xlabel("Época")
        self._ax_acc.set_ylabel("Accuracy (%)")
        self._ax_acc.grid(True, alpha=0.3)
        self._ax_acc.set_ylim(0, 100)

        self._ax_loss.set_title("Pérdida por época")
        self._ax_loss.set_xlabel("Época")
        self._ax_loss.set_ylabel("Loss")
        self._ax_loss.grid(True, alpha=0.3)

        self._fig.tight_layout(rect=(0, 0, 1, 0.93))

    def _log(self, msg: str) -> None:
        """
        Añade una línea al log de mensajes.

        Mantiene un máximo de 200 líneas para no crecer sin límite.
        Debe llamarse siempre desde el hilo principal.
        """
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")

        # Limitar a 200 líneas
        lines = int(self._log_text.index(tk.END).split(".")[0])
        if lines > 200:
            self._log_text.delete("1.0", f"{lines - 200}.0")

        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _add_worker_row(self, worker_id: int, addr: str) -> None:
        """Inserta o actualiza una fila en la tabla de Workers."""
        tag = f"w{worker_id}"
        color = self.WORKER_COLORS[worker_id % len(self.WORKER_COLORS)]

        # Si ya existe la fila, actualiza en lugar de insertar
        if self._wk_tree.exists(tag):
            self._wk_tree.item(tag, values=(
                worker_id, addr,
                self._worker_status.get(worker_id, self._ST_CONNECTED),
                self._worker_grads.get(worker_id, 0),
            ))
        else:
            self._wk_tree.insert(
                "", tk.END, iid=tag,
                values=(worker_id, addr,
                        self._ST_CONNECTED,
                        0),
                tags=(tag,),
            )
            self._wk_tree.tag_configure(tag, foreground=color)

    def _update_worker_row(self, worker_id: int) -> None:
        """Refresca estado y contador de gradientes de un Worker."""
        tag = f"w{worker_id}"
        if self._wk_tree.exists(tag):
            current = self._wk_tree.item(tag, "values")
            self._wk_tree.item(tag, values=(
                current[0],
                current[1],
                self._worker_status.get(worker_id, self._ST_CONNECTED),
                self._worker_grads.get(worker_id, 0),
            ))

    def _update_plots(self) -> None:
        """Redibuja las gráficas con los datos del historial actual."""
        epochs = list(range(1, len(self._acc_history) + 1))

        self._ax_acc.clear()
        self._ax_loss.clear()
        self._setup_axes()

        if epochs:
            self._ax_acc.plot(epochs, self._acc_history,
                              "o-", color="#2196F3", linewidth=2,
                              markersize=4, label="Accuracy promedio")
            self._ax_acc.legend(loc="lower right", fontsize=8)

            self._ax_loss.plot(epochs, self._loss_history,
                               "o-", color="#F44336", linewidth=2,
                               markersize=4, label="Loss promedio")
            self._ax_loss.legend(loc="upper right", fontsize=8)

        self._canvas.draw()

    def _clear_plots(self) -> None:
        """Borra el historial y reinicia las gráficas."""
        self._acc_history.clear()
        self._loss_history.clear()
        self._update_plots()
        self._status_var.set("Gráficas limpiadas.")

    # ================================================================
    # ARRANQUE DEL PARAMETER SERVER
    # ================================================================

    def _start_server(self) -> None:
        """
        Valida parámetros, inicializa el PS y lo lanza en un hilo
        secundario. Deshabilita el botón de inicio para evitar
        un doble arranque.
        """
        if self._running:
            messagebox.showwarning("En ejecución",
                                   "El servidor ya está en marcha.")
            return

        # ── Recoger y validar parámetros ──────────────────────────
        try:
            host     = self._v_host.get().strip()
            port     = int(self._v_port.get())
            workers  = int(self._v_workers.get())
            epochs   = int(self._v_epochs.get())
            hidden   = int(self._v_hidden.get())
            lr       = float(self._v_lr.get())
            n_train  = int(self._v_n_train.get())
            seed_str = self._v_seed.get().strip()
            seed     = int(seed_str) if seed_str else None
        except ValueError as exc:
            messagebox.showerror("Parámetro inválido", str(exc))
            return

        # ── Preparar estado inicial ───────────────────────────────
        self._total_epochs = epochs
        self._acc_history.clear()
        self._loss_history.clear()
        self._worker_status.clear()
        self._worker_grads.clear()

        # Limpia la tabla de Workers
        for row in self._wk_tree.get_children():
            self._wk_tree.delete(row)

        # Reinicia barras de progreso con los valores correctos
        self._wk_progress.configure(maximum=workers)
        self._wk_progress["value"] = 0
        self._wk_count_var.set(f"0 / {workers}")
        self._epoch_bar.configure(maximum=epochs)
        self._epoch_bar["value"] = 0
        self._epoch_var.set(f"0 / {epochs}")

        self._update_plots()

        # ── Inicializar parámetros de la red ──────────────────────
        if seed is not None:
            np.random.seed(seed)

        initial_params = {
            "W1": xavier_initialization(784,    hidden),
            "b1": vector_zeros(hidden),
            "W2": xavier_initialization(hidden, 10),
            "b2": vector_zeros(10),
        }

        # ── Callbacks que envían mensajes a la cola ───────────────
        q = self._q

        def _on_worker_connected(worker_id: int, addr: str) -> None:
            q.put(("worker_connected", {"id": worker_id, "addr": addr}))

        def _on_gradients_received(
            worker_id: int, epoch: int, loss: float, accuracy: float
        ) -> None:
            q.put(("gradients_received", {
                "worker_id": worker_id,
                "epoch":     epoch,
                "loss":      loss,
                "accuracy":  accuracy,
            }))

        def _on_epoch_end(
            epoch: int, total: int, accuracy: float, loss: float
        ) -> None:
            q.put(("epoch_end", {
                "epoch":    epoch,
                "total":    total,
                "accuracy": accuracy,
                "loss":     loss,
            }))

        # ── Construir y lanzar el PS ──────────────────────────────
        server = ParameterServer(
            host                 = host,
            port                 = port,
            num_workers          = workers,
            initial_params       = initial_params,
            learning_rate        = lr,
            n_train              = n_train,
            on_worker_connected  = _on_worker_connected,
            on_gradients_received= _on_gradients_received,
            on_epoch_end         = _on_epoch_end,
        )

        def _server_thread() -> None:
            try:
                history = server.run(epochs=epochs)
                q.put(("training_done", history))
            except Exception as exc:
                q.put(("error", exc))

        self._running = True
        self._btn_start.configure(state=tk.DISABLED)
        self._status_var.set(
            f"Servidor escuchando en {host}:{port} — "
            f"esperando {workers} worker(s)..."
        )
        self._log(f"[PS] Iniciado en {host}:{port}  |  workers={workers}  "
                  f"epochs={epochs}  lr={lr}  n_train={n_train}")

        threading.Thread(target=_server_thread, daemon=True).start()

        # Arranca el poller de la cola
        self.root.after(100, self._poll_queue)

    # ================================================================
    # POLLER DE LA COLA
    # ================================================================

    def _poll_queue(self) -> None:
        """
        Lee todos los mensajes disponibles en la cola y actualiza la UI.

        Se reprograma cada 100 ms hasta que llega "training_done"
        o "error".
        """
        try:
            while True:
                msg_type, payload = self._q.get_nowait()

                if msg_type == "worker_connected":
                    self._handle_worker_connected(payload)

                elif msg_type == "gradients_received":
                    self._handle_gradients_received(payload)

                elif msg_type == "epoch_end":
                    self._handle_epoch_end(payload)

                elif msg_type == "training_done":
                    self._handle_training_done(payload)
                    return   # no reprogramar

                elif msg_type == "error":
                    self._handle_error(payload)
                    return

        except queue.Empty:
            pass
        except Exception as exc:
            self._log(f"[ERROR] {exc}")
            self._status_var.set(f"Error: {exc}")
            self._running = False
            self._btn_start.configure(state=tk.NORMAL)
            return

        # Reprograma mientras el entrenamiento siga activo
        self.root.after(100, self._poll_queue)

    # ================================================================
    # MANEJADORES DE MENSAJES
    # ================================================================

    def _handle_worker_connected(self, payload: dict) -> None:
        wid  = payload["id"]
        addr = payload["addr"]

        self._worker_status[wid] = self._ST_CONNECTED
        self._worker_grads[wid]  = 0

        self._add_worker_row(wid, addr)

        connected = len(self._worker_status)
        total     = int(self._v_workers.get())
        self._wk_progress["value"] = connected
        self._wk_count_var.set(f"{connected} / {total}")

        self._log(f"[W{wid}] Conectado desde {addr}")
        self._status_var.set(
            f"Workers conectados: {connected} / {total}"
            + ("  — Iniciando entrenamiento..." if connected == total else "")
        )

    def _handle_gradients_received(self, payload: dict) -> None:
        wid      = payload["worker_id"]
        epoch    = payload["epoch"]
        loss     = payload["loss"]
        accuracy = payload["accuracy"]

        # Incrementa el contador de gradientes del Worker
        self._worker_grads[wid]  = self._worker_grads.get(wid, 0) + 1
        self._worker_status[wid] = self._ST_DONE

        self._update_worker_row(wid)
        self._log(
            f"[W{wid}] Época {epoch} — "
            f"loss={loss:.4f}  acc={accuracy:.2f}%"
        )

    def _handle_epoch_end(self, payload: dict) -> None:
        epoch    = payload["epoch"]
        total    = payload["total"]
        accuracy = payload["accuracy"]
        loss     = payload["loss"]

        self._acc_history.append(accuracy)
        self._loss_history.append(loss)

        # Actualiza barra de épocas
        self._epoch_bar["value"] = epoch
        self._epoch_var.set(f"{epoch} / {total}")

        # Marca a todos los Workers como "Calculando" para la próxima época
        for wid in self._worker_status:
            self._worker_status[wid] = self._ST_COMPUTING
            self._update_worker_row(wid)

        # Actualiza el título de la figura con las métricas actuales
        self._fig.suptitle(
            f"Entrenamiento Distribuido — Época {epoch}/{total}  |  "
            f"Acc: {accuracy:.2f}%  |  Loss: {loss:.4f}",
            fontsize=12, fontweight="bold",
        )

        self._update_plots()
        self._status_var.set(
            f"Época {epoch}/{total} — "
            f"Accuracy: {accuracy:.2f}%  |  Loss: {loss:.4f}"
        )

    def _handle_training_done(self, history: dict) -> None:
        final_acc  = history["accuracies"][-1] if history["accuracies"] else 0.0
        best_acc   = max(history["accuracies"]) if history["accuracies"] else 0.0
        final_loss = history["losses"][-1]      if history["losses"]     else 0.0

        self._fig.suptitle(
            f"Entrenamiento completado  |  "
            f"Mejor acc: {best_acc:.2f}%  |  "
            f"Acc final: {final_acc:.2f}%",
            fontsize=12, fontweight="bold",
        )
        self._update_plots()

        self._log("─" * 50)
        self._log(f"[PS] Entrenamiento completado")
        self._log(f"[PS] Precisión final  : {final_acc:.2f}%")
        self._log(f"[PS] Mejor precisión  : {best_acc:.2f}%")
        self._log(f"[PS] Loss final       : {final_loss:.4f}")
        self._log("─" * 50)

        self._status_var.set(
            f"Completado — Precisión final: {final_acc:.2f}%  |  "
            f"Mejor: {best_acc:.2f}%"
        )
        self._running = False
        self._btn_start.configure(state=tk.NORMAL)

    def _handle_error(self, exc: Exception) -> None:
        self._log(f"[ERROR] {exc}")
        messagebox.showerror("Error del servidor", str(exc))
        self._status_var.set(f"Error: {exc}")
        self._running = False
        self._btn_start.configure(state=tk.NORMAL)


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    root = tk.Tk()
    DistributedPSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
