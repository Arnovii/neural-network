"""
ps_gui.py

Interfaz gráfica del Parameter Server para el algoritmo de Diego
distribuido con Data-Oriented Parallelism.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
    python ps_gui.py

Los Workers se lanzan a mano en cada máquina (sin --id):
    python worker.py --server-host <IP_DE_ESTA_MÁQUINA>

──────────────────────────────────────────────────────────────────
ESTADOS DEL SERVIDOR
──────────────────────────────────────────────────────────────────
    OFFLINE   → El servidor no está iniciado.
    LISTENING → El socket está abierto y acepta Workers. Se puede
                conectar cualquier número de Workers en cualquier
                momento. El botón "Iniciar entrenamiento" aparece
                habilitado cuando hay al menos un Worker conectado.
    TRAINING  → Hay una sesión de entrenamiento en curso.
                Los botones de servidor se deshabilitan temporalmente.

Transiciones:
    OFFLINE  ──[Encender servidor]──►  LISTENING
    LISTENING ──[Iniciar entrenamiento]──►  TRAINING
    TRAINING  ──[fin de época * N]──►  LISTENING  (listo para otra sesión)
    LISTENING ──[Apagar servidor]──►  OFFLINE

──────────────────────────────────────────────────────────────────
ARQUITECTURA DE HILOS
──────────────────────────────────────────────────────────────────
Tkinter no es thread-safe. Toda la UI vive en el hilo principal.
Los callbacks del PS se ejecutan en hilos del PS y comunican con
la UI exclusivamente a través de una Queue.

Tipos de mensaje en la cola:
    ("worker_connected",     {"id": int, "addr": str})
    ("worker_disconnected",  {"id": int})
    ("gradients_received",   {"worker_id": int, "epoch": int,
                              "loss": float, "accuracy": float})
    ("epoch_end",            {"epoch": int, "total": int,
                              "accuracy": float, "loss": float})
    ("training_done",        history: dict)
    ("error",                exc: Exception)
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
from Utils.mnist_loader import load_mnist_labels


# ================================================================
# TOOLTIP
# ================================================================


class ToolTip:
    """Muestra un tooltip al pasar el cursor sobre un widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window = None
        self._after_id = None
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

    Gestiona tres estados: OFFLINE, LISTENING y TRAINING.
    Permite encender/apagar el servidor independientemente de
    lanzar sesiones de entrenamiento.
    """

    WORKER_COLORS = [
        "#2196F3",
        "#4CAF50",
        "#FF9800",
        "#9C27B0",
        "#F44336",
        "#00BCD4",
        "#795548",
        "#E91E63",
    ]

    _ST_CONNECTED = "Conectado"
    _ST_COMPUTING = "Calculando"
    _ST_DONE = "✓ Listo"

    # Estados del servidor
    _S_OFFLINE = "OFFLINE"
    _S_LISTENING = "LISTENING"
    _S_TRAINING = "TRAINING"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Parameter Server — Algoritmo de Diego Distribuido")
        self.root.state("zoomed")
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._q: queue.Queue = queue.Queue()
        self._server: ParameterServer | None = None
        self._state: str = self._S_OFFLINE

        # Datos de Workers y entrenamiento
        self._worker_status: dict = {}
        self._worker_addrs: dict = {}
        self._worker_grads: dict = {}
        self._acc_history: list = []
        self._loss_history: list = []
        self._total_epochs: int = 0

        self._build_ui()
        self._refresh_buttons()

    # ================================================================
    # CONSTRUCCIÓN DE LA UI
    # ================================================================

    def _build_ui(self) -> None:
        self._build_left_panel()
        self._build_right_panel()
        self._build_status_bar()

    # ── Panel izquierdo ──────────────────────────────────────────

    def _build_left_panel(self) -> None:
        def _snap_int(var):
            return lambda v: var.set(int(round(float(v))))

        def _make_int_validator(parent, max_digits):
            def _validate(new_value):
                return new_value == "" or (
                    len(new_value) <= max_digits and new_value.isdigit()
                )

            return (parent.register(_validate), "%P")

        def _add_slider(parent, label, var, lo, hi):
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

        def _add_float_input(parent, label, var, lo, hi, max_chars=8):
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

        def _add_integer_input(parent, label, var, lo, hi, max_digits=6):
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

        def _add_text_input(parent, label, var, max_chars=20):
            ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(10, 0))
            ttk.Entry(parent, textvariable=var, width=max_chars).pack(
                pady=(0, 6), fill=tk.X
            )

        # ── Contenedor scrollable ─────────────────────────────────
        container = ttk.Frame(self.root, width=270)
        container.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        container.grid_propagate(False)

        canvas = tk.Canvas(container, width=270, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas, padding="10")
        frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(frame, text="Parameter Server", font=("Helvetica", 14, "bold")).pack(
            pady=10
        )

        # ── Variables ─────────────────────────────────────────────
        self._v_host = tk.StringVar(value="0.0.0.0")
        self._v_port = tk.IntVar(value=9999)
        self._v_epochs = tk.IntVar(value=50)
        self._v_hidden = tk.IntVar(value=30)
        self._v_lr = tk.StringVar(value="0.1")
        self._v_n_train = tk.StringVar(value="60000")
        self._v_seed = tk.StringVar(value="")

        # ── Sección: Conexión TCP ─────────────────────────────────
        ttk.Label(frame, text="Conexión TCP", font=("Helvetica", 11, "bold")).pack(
            anchor=tk.W, pady=(14, 0)
        )
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        _add_text_input(frame, "Host (IP de escucha):", self._v_host)
        _add_integer_input(frame, "Puerto:", self._v_port, 1024, 65535, max_digits=5)

        # ── Sección: Entrenamiento ────────────────────────────────
        ttk.Label(frame, text="Entrenamiento", font=("Helvetica", 11, "bold")).pack(
            anchor=tk.W, pady=(18, 0)
        )
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        _add_slider(frame, "Épocas (50 – 1000):", self._v_epochs, 50, 1000)
        _add_slider(frame, "Neuronas ocultas (10 – 100):", self._v_hidden, 10, 100)
        _add_float_input(
            frame, "Tasa de aprendizaje\n(0.0001 - 10):", self._v_lr, 0.0001, 10.0
        )
        _add_integer_input(
            frame,
            "Ejemplos de entrenamiento\n(10 - 60000):",
            self._v_n_train,
            100,
            60000,
        )
        _add_text_input(frame, "Semilla (vacío = aleatoria):", self._v_seed)

        # ── Botones ───────────────────────────────────────────────
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(18, 8))

        self._btn_listen = ttk.Button(
            frame,
            text="⚡ Encender servidor",
            command=self._cmd_listen,
        )
        self._btn_listen.pack(fill=tk.X, pady=4)
        ToolTip(
            self._btn_listen,
            "Abre el socket TCP y empieza a aceptar Workers.\n"
            "Los Workers pueden conectarse en cualquier momento.",
        )

        self._btn_train = ttk.Button(
            frame,
            text="▶  Iniciar entrenamiento",
            command=self._cmd_train,
        )
        self._btn_train.pack(fill=tk.X, pady=4)
        ToolTip(
            self._btn_train,
            "Lanza una sesión de entrenamiento con los Workers\n"
            "actualmente conectados. Requiere al menos 1 Worker.",
        )

        self._btn_shutdown = ttk.Button(
            frame,
            text="■  Apagar servidor",
            command=self._cmd_shutdown,
        )
        self._btn_shutdown.pack(fill=tk.X, pady=4)
        ToolTip(
            self._btn_shutdown,
            "Envía STOP a todos los Workers y cierra el socket.\n"
            "Los Workers terminarán limpiamente al recibir la señal.",
        )

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 4))

        btn_clear = ttk.Button(
            frame, text="Limpiar gráficas", command=self._clear_plots
        )
        btn_clear.pack(fill=tk.X, pady=4)
        ToolTip(
            btn_clear,
            "Borra todas las gráficas actuales",
        )

    # ── Panel derecho ─────────────────────────────────────────────

    def _build_right_panel(self) -> None:
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ── Tabla de Workers ──────────────────────────────────────
        wf = ttk.LabelFrame(right, text="Workers conectados", padding=6)
        wf.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        cols = ("ID", "Dirección IP", "Estado", "Épocas completadas")
        self._wk_tree = ttk.Treeview(
            wf,
            columns=cols,
            show="headings",
            height=5,
            selectmode="none",
        )
        for col in cols:
            self._wk_tree.heading(col, text=col)
        self._wk_tree.column("ID", width=50, anchor="center")
        self._wk_tree.column("Dirección IP", width=160, anchor="center")
        self._wk_tree.column("Estado", width=120, anchor="center")
        self._wk_tree.column("Épocas completadas", width=140, anchor="center")
        self._wk_tree.pack(fill=tk.X)

        # Indicador de estado del servidor
        srv_frame = ttk.Frame(wf)
        srv_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(srv_frame, text="Estado del servidor:").pack(side=tk.LEFT)
        self._srv_state_var = tk.StringVar(value="OFFLINE")
        self._srv_state_lbl = tk.Label(
            srv_frame,
            textvariable=self._srv_state_var,
            font=("Helvetica", 10, "bold"),
            fg="white",
            bg="#607D8B",
            padx=8,
            pady=2,
        )
        self._srv_state_lbl.pack(side=tk.LEFT, padx=8)

        # ── Gráficas + log ────────────────────────────────────────
        pal = ttk.Frame(right)
        pal.grid(row=1, column=0, sticky="nsew")
        pal.rowconfigure(0, weight=3)
        pal.rowconfigure(1, weight=1)
        pal.columnconfigure(0, weight=1)

        plots_frame = ttk.Frame(pal)
        plots_frame.grid(row=0, column=0, sticky="nsew")

        self._fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        self._ax_acc, self._ax_loss = axes
        self._fig.suptitle(
            "Entrenamiento Distribuido — Algoritmo de Diego",
            fontsize=13,
            fontweight="bold",
        )
        self._setup_axes()
        self._canvas = FigureCanvasTkAgg(self._fig, master=plots_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Barra de progreso de épocas (debajo de las gráficas)
        epf = ttk.Frame(pal)
        epf.grid(row=0, column=0, sticky="s", pady=(0, 4))
        ttk.Label(epf, text="Época:").pack(side=tk.LEFT)
        self._epoch_bar = ttk.Progressbar(
            epf,
            orient=tk.HORIZONTAL,
            length=500,
            mode="determinate",
            maximum=1,
        )
        self._epoch_bar.pack(side=tk.LEFT, padx=8)
        self._epoch_var = tk.StringVar(value="—")
        ttk.Label(epf, textvariable=self._epoch_var, width=12).pack(side=tk.LEFT)

        # Log
        log_frame = ttk.LabelFrame(pal, text="Log", padding=4)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log_text = tk.Text(
            log_frame,
            height=6,
            state=tk.DISABLED,
            font=("Courier", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            wrap=tk.WORD,
            relief=tk.FLAT,
        )
        log_scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(
            value="Listo. Configura los parámetros y enciende el servidor."
        )
        ttk.Label(
            self.root,
            textvariable=self._status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(6, 2),
        ).grid(row=1, column=0, columnspan=2, sticky="ew")

    # ================================================================
    # GESTIÓN DE ESTADO DE BOTONES
    # ================================================================

    def _refresh_buttons(self) -> None:
        """
        Habilita o deshabilita los botones según el estado actual.

        OFFLINE   → solo "Encender" habilitado
        LISTENING → "Encender" deshabilitado, "Entrenar" habilitado
                    solo si hay Workers conectados, "Apagar" habilitado
        TRAINING  → todos deshabilitados excepto ninguno
        """
        s = self._state
        has_workers = bool(self._worker_status)

        self._btn_listen.configure(
            state=tk.NORMAL if s == self._S_OFFLINE else tk.DISABLED
        )
        self._btn_train.configure(
            state=tk.NORMAL if s == self._S_LISTENING and has_workers else tk.DISABLED
        )
        self._btn_shutdown.configure(
            state=tk.NORMAL if s == self._S_LISTENING else tk.DISABLED
        )

        # Actualiza la pastilla de estado
        labels = {
            self._S_OFFLINE: ("OFFLINE", "#607D8B"),
            self._S_LISTENING: ("LISTENING", "#2E7D32"),
            self._S_TRAINING: ("TRAINING", "#1565C0"),
        }
        text, color = labels.get(s, ("?", "#607D8B"))
        self._srv_state_var.set(text)
        self._srv_state_lbl.configure(bg=color)

    # ================================================================
    # HELPERS DE UI
    # ================================================================

    def _setup_axes(self) -> None:
        self._ax_acc.set_title("Precisión por época")
        self._ax_acc.set_xlabel("Época")
        self._ax_acc.set_ylabel("Precisión (%)")
        self._ax_acc.set_ylim(0, 100)
        self._ax_acc.grid(True, alpha=0.3)

        self._ax_loss.set_title("Pérdida por época")
        self._ax_loss.set_xlabel("Época")
        self._ax_loss.set_ylabel("Pérdida")
        self._ax_loss.grid(True, alpha=0.3)

        self._fig.tight_layout(rect=(0, 0, 1, 0.93))

    def _log(self, msg: str) -> None:
        """Añade una línea al log (máx. 200 líneas). Solo hilo principal."""
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        lines = int(self._log_text.index(tk.END).split(".")[0])
        if lines > 200:
            self._log_text.delete("1.0", f"{lines - 200}.0")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _add_worker_row(self, worker_id: int, addr: str) -> None:
        tag = f"w{worker_id}"
        color = self.WORKER_COLORS[worker_id % len(self.WORKER_COLORS)]
        if self._wk_tree.exists(tag):
            self._wk_tree.item(
                tag,
                values=(
                    worker_id,
                    addr,
                    self._worker_status.get(worker_id, self._ST_CONNECTED),
                    self._worker_grads.get(worker_id, 0),
                ),
            )
        else:
            self._wk_tree.insert(
                "",
                tk.END,
                iid=tag,
                values=(worker_id, addr, self._ST_CONNECTED, 0),
                tags=(tag,),
            )
            self._wk_tree.tag_configure(tag, foreground=color)

    def _remove_worker_row(self, worker_id: int) -> None:
        tag = f"w{worker_id}"
        if self._wk_tree.exists(tag):
            self._wk_tree.delete(tag)

    def _update_worker_row(self, worker_id: int) -> None:
        tag = f"w{worker_id}"
        if self._wk_tree.exists(tag):
            current = self._wk_tree.item(tag, "values")
            self._wk_tree.item(
                tag,
                values=(
                    current[0],
                    current[1],
                    self._worker_status.get(worker_id, self._ST_CONNECTED),
                    self._worker_grads.get(worker_id, 0),
                ),
            )

    def _update_plots(self) -> None:
        epochs = list(range(1, len(self._acc_history) + 1))
        self._ax_acc.clear()
        self._ax_loss.clear()
        self._setup_axes()
        if epochs:
            self._ax_acc.plot(
                epochs,
                self._acc_history,
                "o-",
                color="#2196F3",
                linewidth=2,
                markersize=4,
                label="Precisión promedio",
            )
            self._ax_acc.legend(loc="lower right", fontsize=8)
            self._ax_loss.plot(
                epochs,
                self._loss_history,
                "o-",
                color="#F44336",
                linewidth=2,
                markersize=4,
                label="Pérdida promedio",
            )
            self._ax_loss.legend(loc="upper right", fontsize=8)
        self._canvas.draw()

    def _clear_plots(self) -> None:
        self._acc_history.clear()
        self._loss_history.clear()
        self._update_plots()
        self._status_var.set("Gráficas limpiadas.")

    # ================================================================
    # COMANDOS DE BOTONES
    # ================================================================

    def _cmd_listen(self) -> None:
        """Enciende el servidor: abre el socket y empieza a escuchar."""
        try:
            host = self._v_host.get().strip()
            port = int(self._v_port.get())
        except ValueError as exc:
            messagebox.showerror("Parámetro inválido", str(exc))
            return

        q = self._q

        self._server = ParameterServer(
            host=host,
            port=port,
            on_worker_connected=lambda wid, addr: q.put(
                ("worker_connected", {"id": wid, "addr": addr})
            ),
            on_worker_disconnected=lambda wid: q.put(
                ("worker_disconnected", {"id": wid})
            ),
            on_gradients_received=lambda wid, ep, loss, acc: q.put(
                (
                    "gradients_received",
                    {"worker_id": wid, "epoch": ep, "loss": loss, "accuracy": acc},
                )
            ),
            on_epoch_end=lambda ep, tot, acc, loss: q.put(
                (
                    "epoch_end",
                    {"epoch": ep, "total": tot, "accuracy": acc, "loss": loss},
                )
            ),
        )

        try:
            self._server.listen()
        except Exception as exc:
            messagebox.showerror("Error al encender servidor", str(exc))
            self._server = None
            return

        self._state = self._S_LISTENING
        self._refresh_buttons()
        self._log(f"[PS] Servidor encendido en {host}:{port}")
        self._status_var.set(
            f"Servidor escuchando en {host}:{port}. Esperando Workers..."
        )

        # Arranca el poller
        self.root.after(100, self._poll_queue)

    def _cmd_train(self) -> None:
        """Lanza una sesión de entrenamiento con los Workers conectados."""
        if self._state != self._S_LISTENING:
            return
        if not self._worker_status:
            messagebox.showwarning(
                "Sin Workers", "Conecta al menos un Worker antes de entrenar."
            )
            return

        # Recoger parámetros de entrenamiento
        try:
            epochs = int(self._v_epochs.get())
            hidden = int(self._v_hidden.get())
            lr = float(self._v_lr.get())
            n_train = int(self._v_n_train.get())
            seed_str = self._v_seed.get().strip()
            seed = int(seed_str) if seed_str else None
        except ValueError as exc:
            messagebox.showerror("Parámetro inválido", str(exc))
            return

        # Inicializar pesos
        if seed is not None:
            np.random.seed(seed)

        initial_params = {
            "W1": xavier_initialization(784, hidden),
            "b1": vector_zeros(hidden),
            "W2": xavier_initialization(hidden, 10),
            "b2": vector_zeros(10),
        }

        # Reinicia historial y barras para esta sesión
        self._acc_history.clear()
        self._loss_history.clear()
        self._total_epochs = epochs
        self._epoch_bar.configure(maximum=epochs)
        self._epoch_bar["value"] = 0
        self._epoch_var.set(f"0 / {epochs}")
        self._update_plots()

        # Resetea contadores de Workers
        for wid in self._worker_status:
            self._worker_status[wid] = self._ST_COMPUTING
            self._worker_grads[wid] = 0
            self._update_worker_row(wid)

        self._state = self._S_TRAINING
        self._refresh_buttons()

        n_workers = len(self._worker_status)
        self._log(
            f"[PS] Iniciando entrenamiento — "
            f"{n_workers} worker(s)  epochs={epochs}  "
            f"lr={lr}  n_train={n_train}"
        )
        self._status_var.set(
            f"Entrenando — {n_workers} worker(s)  |  {epochs} épocas  |  lr={lr}"
        )

        q = self._q
        server = self._server

        if server is None:
            messagebox.showerror("Error", "Servidor no disponible")
            self._state = self._S_LISTENING
            self._refresh_buttons()
            return

        def _train_thread() -> None:
            try:
                # Carga solo las etiquetas (~60 KB) para la partición
                # estratificada. No se leen las imágenes (~47 MB).
                Y_train = load_mnist_labels(n_train=n_train)

                history = server.train(
                    epochs=epochs,
                    initial_params=initial_params,
                    learning_rate=lr,
                    n_train=n_train,
                    Y_train=Y_train,
                )
                q.put(("training_done", history))
            except Exception as exc:
                q.put(("error", exc))

        threading.Thread(target=_train_thread, daemon=True).start()

    def _cmd_shutdown(self) -> None:
        """Apaga el servidor enviando STOP a todos los Workers."""
        if self._server is None or self._state == self._S_OFFLINE:
            return
        if self._state == self._S_TRAINING:
            messagebox.showwarning(
                "Entrenamiento en curso",
                "Espera a que termine el entrenamiento antes de apagar.",
            )
            return

        confirm = messagebox.askyesno(
            "Apagar servidor",
            "Se enviará STOP a todos los Workers conectados.\n¿Continuar?",
        )
        if not confirm:
            return

        threading.Thread(target=self._server.shutdown, daemon=True).start()

        self._state = self._S_OFFLINE
        self._server = None
        self._worker_status.clear()
        self._worker_addrs.clear()
        self._worker_grads.clear()

        for row in self._wk_tree.get_children():
            self._wk_tree.delete(row)

        self._refresh_buttons()
        self._log("[PS] Servidor apagado.")
        self._status_var.set("Servidor apagado.")

    # ================================================================
    # POLLER DE LA COLA
    # ================================================================

    def _poll_queue(self) -> None:
        """Lee mensajes de la cola y actualiza la UI. Se reprograma cada 100 ms."""
        try:
            while True:
                msg_type, payload = self._q.get_nowait()

                if msg_type == "worker_connected":
                    self._on_worker_connected(payload)
                elif msg_type == "worker_disconnected":
                    self._on_worker_disconnected(payload)
                elif msg_type == "gradients_received":
                    self._on_gradients_received(payload)
                elif msg_type == "epoch_end":
                    self._on_epoch_end(payload)
                elif msg_type == "training_done":
                    self._on_training_done(payload)
                elif msg_type == "error":
                    self._on_error(payload)

        except queue.Empty:
            pass
        except Exception as exc:
            self._log(f"[ERROR] {exc}")
            self._status_var.set(f"Error: {exc}")

        # Sigue poliando mientras el servidor esté activo
        if self._state != self._S_OFFLINE:
            self.root.after(100, self._poll_queue)

    # ================================================================
    # MANEJADORES DE MENSAJES
    # ================================================================

    def _on_worker_connected(self, payload: dict) -> None:
        wid = payload["id"]
        addr = payload["addr"]

        self._worker_status[wid] = self._ST_CONNECTED
        self._worker_addrs[wid] = addr
        self._worker_grads[wid] = 0

        self._add_worker_row(wid, addr)
        self._refresh_buttons()  # puede habilitar "Iniciar entrenamiento"

        self._log(f"[W{wid}] Conectado desde {addr}")
        self._status_var.set(
            f"Worker {wid} conectado desde {addr}  |  "
            f"Total activos: {len(self._worker_status)}"
        )

    def _on_worker_disconnected(self, payload: dict) -> None:
        wid = payload["id"]

        self._worker_status.pop(wid, None)
        self._worker_addrs.pop(wid, None)
        self._worker_grads.pop(wid, None)

        self._remove_worker_row(wid)
        self._refresh_buttons()

        self._log(f"[W{wid}] Desconectado inesperadamente.")
        self._status_var.set(
            f"Worker {wid} desconectado  |  Activos: {len(self._worker_status)}"
        )

    def _on_gradients_received(self, payload: dict) -> None:
        wid = payload["worker_id"]
        epoch = payload["epoch"]
        loss = payload["loss"]
        accuracy = payload["accuracy"]

        self._worker_grads[wid] = self._worker_grads.get(wid, 0) + 1
        self._worker_status[wid] = self._ST_DONE
        self._update_worker_row(wid)

        self._log(f"[W{wid}] Época {epoch} — loss={loss:.4f}  acc={accuracy:.2f}%")

    def _on_epoch_end(self, payload: dict) -> None:
        epoch = payload["epoch"]
        total = payload["total"]
        accuracy = payload["accuracy"]
        loss = payload["loss"]

        self._acc_history.append(accuracy)
        self._loss_history.append(loss)

        self._epoch_bar["value"] = epoch
        self._epoch_var.set(f"{epoch} / {total}")

        # Marca Workers como "calculando" para la próxima época
        for wid in self._worker_status:
            self._worker_status[wid] = self._ST_COMPUTING
            self._update_worker_row(wid)

        self._fig.suptitle(
            f"Entrenamiento Distribuido — Época {epoch}/{total}  |  "
            f"Precisión: {accuracy:.2f}%  |  Pérdida: {loss:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        self._update_plots()
        self._status_var.set(
            f"Época {epoch}/{total} — Precisión: {accuracy:.2f}%  |  Pérdida: {loss:.4f}"
        )

    def _on_training_done(self, history: dict) -> None:
        final_acc = history["accuracies"][-1] if history["accuracies"] else 0.0
        best_acc = max(history["accuracies"]) if history["accuracies"] else 0.0
        final_loss = history["losses"][-1] if history["losses"] else 0.0

        self._fig.suptitle(
            f"Completado  |  Mejor precisión: {best_acc:.2f}%  |  "
            f"Precisión final: {final_acc:.2f}%",
            fontsize=12,
            fontweight="bold",
        )
        self._update_plots()

        # Marca todos los Workers como "Conectado" (listos para otro entrenamiento)
        for wid in self._worker_status:
            self._worker_status[wid] = self._ST_CONNECTED
            self._update_worker_row(wid)

        self._log("─" * 50)
        self._log("[PS] Entrenamiento completado")
        self._log(f"     Precisión final : {final_acc:.2f}%")
        self._log(f"     Mejor precisión : {best_acc:.2f}%")
        self._log(f"     Pérdida final      : {final_loss:.4f}")
        self._log("─" * 50)

        # Vuelve a LISTENING — los Workers siguen conectados
        self._state = self._S_LISTENING
        self._refresh_buttons()

        self._status_var.set(
            f"Entrenamiento completado — "
            f"Precisión final: {final_acc:.2f}%  |  "
            f"Mejor: {best_acc:.2f}%  |  "
            f"Workers conectados: {len(self._worker_status)}"
        )

    def _on_error(self, exc: Exception) -> None:
        self._log(f"[ERROR] {exc}")
        messagebox.showerror("Error del servidor", str(exc))
        self._status_var.set(f"Error: {exc}")
        # Vuelve a LISTENING para que el usuario pueda reintentar
        if self._state == self._S_TRAINING:
            self._state = self._S_LISTENING
            self._refresh_buttons()


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    root = tk.Tk()
    DistributedPSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
