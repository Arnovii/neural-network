"""
ps_gui_imagenet.py

GUI del Parameter Server asíncrono para entrenamiento distribuido en ImageNet.

FLUJO CORRECTO:
  1. "Encender servidor" → carga CNN+MLP en hilo background (sin congelar GUI)
                         → ps.set_cnn() + ps.set_mlp() + ps.listen()
                         → estado LISTENING
  2. Worker conecta     → PS envía CNN_WEIGHTS + START automáticamente
                         → Worker comienza entrenamiento indefinidamente
  3. GUI recibe primer evento on_step() → transiciona automáticamente a TRAINING
                         → muestra gráficas en tiempo real

MEJORAS VS VERSIÓN ANTERIOR:
  - CNN+MLP se cargan en hilo background → la GUI no se congela (~50MB ResNet-18)
  - Indicador visual "Cargando..." mientras descarga pesos
  - Transición automática a TRAINING → interfaz más intuitiva (sin botón confuso)
  - Logging limpio: sin logs por iteración (solo mensajes críticos)
  - num_batches_tracked excluido del averaging en el PS
  - Label bug corregido en imagenet_streaming.py
"""

import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError as e:
    print(f"Error cargando librerías gráficas: {e}")
    sys.exit(1)

from Distributed.parameter_server import ParameterServer
from Model.cnn_extractor import CNNExtractor
from Model.mlp_pytorch import MLPPyTorch


# ================================================================
# TOOLTIP
# ================================================================


class ToolTip:
    """
    Tooltip flotante que aparece al pasar el mouse sobre un widget tkinter.

    Muestra un label flotante con texto informativo después de 500ms de hover.
    Se destruye automáticamente al salir del widget. Utiliza tkinter.Toplevel
    para renderizar el tooltip fuera de la jerarquía de widgets principal.
    """

    def __init__(self, widget: tk.Widget, text: str) -> None:
        """
        Inicializa tooltip para un widget tkinter.

        Registra eventos de Enter/Leave en el widget para mostrar/ocultar
        el tooltip. El tooltip se mostará después de 500ms persistiendo al
        entrar al widget y desaparecerá al salir del mismo.

        :param widget: Widget tkinter al cual asociar el tooltip.
        :type widget: tk.Widget

        :param text: Texto a mostrar en el tooltip (puede contener saltos de línea).
        :type text: str

        :returns: None
        :rtype: None
        """
        self.widget = widget
        self.text = text
        self._id = None
        self._tip = None
        widget.bind("<Enter>", lambda e: self._schedule())
        widget.bind("<Leave>", lambda e: self._cancel())

    def _schedule(self) -> None:
        """
        Programa la aparición del tooltip después de 500ms.

        Utiliza widget.after() para registrar un callback que ejecutará
        _show() en 500 milisegundos. Si el usuario mueve el mouse fuera
        antes de ese tiempo, _cancel() anulará este callback.

        :returns: None
        :rtype: None
        """
        self._id = self.widget.after(500, self._show)

    def _cancel(self) -> None:
        """
        Cancela la aparición del tooltip y lo destruye si está visible.

        Anula el callback programado (si aún no se ha ejecutado) y destruye
        la ventana flotante del tooltip si ya está visible. Se llama cuando
        el usuario mueve el mouse fuera del widget.

        :returns: None
        :rtype: None
        """
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None

    def _show(self) -> None:
        """
        Muestra el tooltip flotante junto al widget.

        Crea una ventana Toplevel sin decoraciones (overrideredirect=True)
        posicionada 10 píxeles a la derecha y debajo del widget.
        Renderiza un Label con fondo amarillo claro (#ffffe0) con el
        texto del tooltip.

        :returns: None
        :rtype: None
        """
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self.widget)
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


# ================================================================
# APLICACIÓN
# ================================================================


class PSApp:
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
    _S_OFFLINE = "OFFLINE"
    _S_LOADING = "LOADING"  # cargando CNN+MLP (hilo background)
    _S_LISTENING = "LISTENING"
    _S_TRAINING = "TRAINING"

    def __init__(self, root: tk.Tk) -> None:
        """
        Inicializa la interfaz grá fica del Parameter Server.

        Configura widgets, estado interno, plots, y callbacks. Esta interfaz
        permite:
        - Configurar parámetros del PS (CNN, MLP, learning rate, λ, batch-size, image-size)
        - Bloquear parámetros una vez iniciado el servidor
        - Monitorear entrenamiento en tiempo real (loss, accuracy, workers activos)
        - Gestionar lifecycle del servidor (inicio, parada, evaluación)

        :param root: Ventana tkinter raíz (normalmente tk.Tk())
        :type root: tk.Tk
        """
        self.root = root
        self.root.title("Parameter Server — ImageNet-1k Distribuido")
        self.root.state("zoomed")
        self.root.columnconfigure(0, weight=0, minsize=320)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._q: queue.Queue = queue.Queue()
        self._ps: ParameterServer | None = None
        self._state: str = self._S_OFFLINE
        self._freeze_cnn: bool = True

        self._workers: dict = {}
        self._steps_hist: list = []
        self._loss_hist: list = []
        self._acc_hist: list = []
        self._val_steps: list = []
        self._val_loss: list = []
        self._val_acc: list = []
        self._workers_hist: list = []
        self._t_start: float = 0.0
        self._clock_start: float = 0.0  # Clock: tiempo de inicio
        self._clock_running: bool = False  # Clock: corriendo flag
        self._status = tk.StringVar(value="Listo.")

        # Referencia al campo LR CNN para habilitarlo/deshabilitarlo según arch
        self._ent_lr_cnn: ttk.Entry | None

        # Lista de widgets de configuración (para desactivar durante entrenamiento)
        self._config_widgets: list = []

        self._build_ui()
        self._refresh_buttons()
        self._update_lr_cnn_state()  # deshabilitar lr_cnn si resnet18 es default

    # ================================================================
    # UI
    # ================================================================

    def _build_ui(self) -> None:
        """
        Construye la interfaz gráfica completa: panel izquierdo + derecho + status bar.

        Llama a _build_left() para crear panel de configuración (parámetros)
        y _build_right() para crear panel de monitoreo (tablas, gráficas, logs).
        Finalmente agrega status bar en la fila inferior que muestra estado actual.

        :returns: None
        :rtype: None
        """
        self._build_left()
        self._build_right()
        ttk.Label(
            self.root,
            textvariable=self._status,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(6, 2),
        ).grid(row=1, column=0, columnspan=2, sticky="ew")

    def _build_left(self) -> None:
        """
        Construye panel izquierdo con controles de configuración del Parameter Server.

        Crea estructura con Canvas+Scrollbar para permitir scroll vertical en los
        múltiples controles de configuración:
        - Conexión TCP (host, puerto)
        - Dataset (nombre HF, token)
        - Streaming (batch_size, image_size)
        - CNN Extractor (arquitectura: resnet18 / simple)
        - MLP (hidden layer sizes)
        - Learning rates (LR MLP, LR CNN)
        - Async SGD (lambda, windows)
        - Evaluación
        - Botones de acción (listen, train, shutdown, clear)

        :returns: None
        :rtype: None
        """
        cont = ttk.Frame(self.root, width=310)
        cont.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        cont.grid_propagate(False)

        cv = tk.Canvas(cont, highlightthickness=0)
        sb = ttk.Scrollbar(cont, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        frm = ttk.Frame(cv, padding="10")
        frm.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cw = cv.create_window((0, 0), window=frm, anchor="nw")
        cv.bind("<Configure>", lambda e: cv.itemconfig(cw, width=e.width))
        cv.bind(
            "<Enter>",
            lambda e: cv.bind_all(
                "<MouseWheel>",
                lambda ev: cv.yview_scroll(int(-1 * (ev.delta / 120)), "units"),
            ),
        )
        cv.bind("<Leave>", lambda e: cv.unbind_all("<MouseWheel>"))
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        ttk.Label(frm, text="PS — ImageNet-1k", font=("Helvetica", 13, "bold")).pack(
            pady=6
        )

        # ── Conexión ──
        self._section(frm, "Conexión TCP")
        self._v_host = tk.StringVar(value="0.0.0.0")
        self._v_port = tk.IntVar(value=9999)
        ent_host = self._entry(frm, "Host:", self._v_host)
        ent_port = self._entry(frm, "Puerto:", self._v_port, width=10)
        ToolTip(
            ent_host, "IP donde escuchará el servidor (0.0.0.0 = todas las interfaces)"
        )
        ToolTip(ent_port, "Puerto TCP para comunicación con Workers")

        # ── Dataset ──
        self._section(frm, "Dataset")
        self._v_dataset = tk.StringVar(value="ILSVRC/imagenet-1k")
        self._v_hf_token = tk.StringVar(value=os.environ.get("HF_TOKEN", ""))
        ent_dataset = self._entry(frm, "Dataset HF Hub:", self._v_dataset, width=30)
        ttk.Label(frm, text="HF Token:").pack(anchor=tk.W)
        ent_token = ttk.Entry(frm, textvariable=self._v_hf_token, width=30, show="*")
        ent_token.pack(fill=tk.X, pady=2)
        self._config_widgets.append(ent_token)
        ToolTip(
            ent_dataset, "Dataset HF Hub (ej: ILSVRC/imagenet-1k, timm/imagenet-1k-wds)"
        )
        ToolTip(ent_token, "Token de acceso HF para datasets privados.")
        ttk.Label(
            frm,
            text="ℹ ILSVRC/imagenet-1k requiere token con\n  licencia aceptada en HF.",
            font=("Helvetica", 8),
            foreground="#1565C0",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 4))

        # ── Semilla RNG ──
        self._section(frm, "Semilla RNG")
        self._v_seed = tk.StringVar(value="")
        ent_seed = self._entry(frm, "Seed (vacío = aleatorio):", self._v_seed, width=12)
        ToolTip(
            ent_seed,
            "None para reproducibilidad aleatoria, o número entero para reproducir",
        )

        # ── Streaming ──
        self._section(frm, "Streaming")
        self._v_batch_size = tk.IntVar(value=64)
        self._v_image_size = tk.IntVar(value=224)
        ent_bs = self._entry(frm, "Batch size:", self._v_batch_size, width=12)
        ent_is = self._entry(frm, "Image size:", self._v_image_size, width=12)
        ToolTip(ent_bs, "Tamaño de batch para SGD local en cada Worker")
        ToolTip(ent_is, "Resolución de imágenes (ancho y alto, cuadradas)")

        # ── CNN ──
        self._section(frm, "CNN Extractor")
        self._v_arch = tk.StringVar(value="resnet18")
        rb_resnet = ttk.Radiobutton(
            frm,
            text="ResNet-18 + pesos ImageNet (recomendado)",
            variable=self._v_arch,
            value="resnet18",
            command=self._update_lr_cnn_state,
        )
        rb_resnet.pack(anchor=tk.W)
        self._config_widgets.append(rb_resnet)
        rb_simple = ttk.Radiobutton(
            frm,
            text="Simple CNN (sin pretrain)",
            variable=self._v_arch,
            value="simple",
            command=self._update_lr_cnn_state,
        )
        rb_simple.pack(anchor=tk.W)
        self._config_widgets.append(rb_simple)
        ToolTip(rb_resnet, "Extractor preentrenado (más rápido, mejor convergencia)")
        ToolTip(rb_simple, "CNN simple sin preentrenamiento (E2E, más lento)")
        ttk.Label(
            frm,
            text="ℹ resnet18: CNN congelada, solo MLP aprende\n"
            "  simple: CNN + MLP aprenden conjuntamente (E2E)",
            font=("Helvetica", 8),
            foreground="#6A1B9A",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 4))

        # ── MLP ──
        self._section(frm, "Clasificador MLP")
        self._v_h1 = tk.IntVar(value=1024)
        self._v_h2 = tk.IntVar(value=512)
        ent_h1 = self._entry(frm, "Neuronas capa 1:", self._v_h1, width=8)
        ent_h2 = self._entry(frm, "Neuronas capa 2:", self._v_h2, width=8)
        ToolTip(ent_h1, "1ª capa oculta del MLP (features → h1)")
        ToolTip(ent_h2, "2ª capa oculta del MLP (h1 → h2 → 1000)")

        # ── Learning Rates ──
        self._section(frm, "Learning Rates (SGD)")
        self._v_lr = tk.StringVar(value="0.01")
        self._v_lr_cnn = tk.StringVar(value="0.001")

        # LR MLP
        ent_lr_mlp = self._entry(frm, "LR MLP:", self._v_lr, width=12)
        ToolTip(
            ent_lr_mlp,
            "Learning rate del clasificador MLP.\n"
            "Recomendado: 0.01 (resnet18) | 0.01 (simple)\n"
            "El MLP aprende a partir de features buenas (resnet18) o aleatorias (simple).",
        )

        # LR CNN (solo activo en modo simple)
        ttk.Label(frm, text="LR CNN:").pack(anchor=tk.W)
        self._ent_lr_cnn = ttk.Entry(frm, textvariable=self._v_lr_cnn, width=12)
        self._ent_lr_cnn.pack(pady=2)
        self._config_widgets.append(self._ent_lr_cnn)
        ToolTip(
            self._ent_lr_cnn,
            "Learning rate de la CNN (solo en modo simple / E2E).\n"
            "Se deshabilita automáticamente con resnet18 (CNN congelada).\n"
            "Recomendado: 0.001 (10x menor que LR MLP para estabilidad E2E).\n\n"
            "Por qué más pequeño: la CNN parte de pesos aleatorios y sus capas\n"
            "profundas reciben gradientes que ya han sido escalados por las capas\n"
            "superiores. Un LR igual al MLP haría que la CNN oscile en lugar\n"
            "de aprender representaciones estables.",
        )
        self._lbl_lr_cnn_info = ttk.Label(
            frm,
            text="ℹ LR CNN deshabilitado (resnet18 está congelada)",
            font=("Helvetica", 8),
            foreground="#607D8B",
            justify=tk.LEFT,
        )
        self._lbl_lr_cnn_info.pack(anchor=tk.W, pady=(2, 4))

        # ── Async SGD ──
        self._section(frm, "Async SGD")
        self._v_lambda = tk.StringVar(value="0.1")
        self._v_report = tk.IntVar(value=10)
        self._v_window = tk.IntVar(value=50)
        ent_lambda = self._entry(frm, "Staleness λ (0–1):", self._v_lambda, width=12)
        ent_report = self._entry(frm, "Steps por reporte:", self._v_report, width=12)
        ent_window = self._entry(frm, "Ventana métricas:", self._v_window, width=12)
        ToolTip(
            ent_lambda,
            "Factor corrección staleness α(s)=1/(1+λ·s). λ=0: sin corrección, λ=1: fuerte.",
        )
        ToolTip(
            ent_report,
            "Cada cuántos steps actualizar la gráfica.\n"
            "Valor pequeño (1-10): actualización frecuente.\n"
            "Valor grande (50-200): curvas más suaves.",
        )
        ToolTip(
            ent_window,
            "Tamaño de la ventana deslizante de métricas.\n"
            "Ventana=50: refleja los últimos 50 batches.",
        )
        ttk.Label(
            frm,
            text="ℹ λ=0: sin corrección  λ=0.1: moderada  λ=1: fuerte",
            font=("Helvetica", 8),
            foreground="#2E7D32",
        ).pack(anchor=tk.W, pady=(2, 8))

        # ── Evaluación ──
        self._section(frm, "Evaluación")
        self._v_val_batches = tk.IntVar(value=50)
        ent_vb = self._entry(
            frm, "Batches de validación:", self._v_val_batches, width=8
        )
        self._btn_eval = ttk.Button(
            frm, text="Evaluar en validación ahora", command=self._cmd_evaluate
        )
        self._btn_eval.pack(fill=tk.X, pady=6)
        ToolTip(self._btn_eval, "Evalúa modelo global en validación (no bloqueante)")

        # ── Botones ──
        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(18, 8))
        self._btn_listen = ttk.Button(
            frm, text="Encender servidor", command=self._cmd_listen
        )
        self._btn_shutdown = ttk.Button(
            frm, text="■  Detener todo", command=self._cmd_shutdown
        )
        self._btn_clear = ttk.Button(
            frm, text="Limpiar gráficas", command=self._clear_plots
        )
        for btn in (
            self._btn_listen,
            self._btn_shutdown,
            self._btn_clear,
        ):
            btn.pack(fill=tk.X, pady=3)
        self._config_widgets.extend([self._btn_listen, self._btn_clear])

        ToolTip(self._btn_listen, "Carga CNN+MLP y abre socket TCP")
        ToolTip(self._btn_shutdown, "Envía STOP a todos los Workers y cierra el PS")
        ToolTip(self._btn_clear, "Limpia gráficas sin detener entrenamiento")

    def _build_right(self) -> None:
        """
        Construye panel derecho con monitoreo: tabla de Workers, gráficas y log.

        Organiza tres secciones:
        1. Tabla Treeview de Workers conectados (ID, dirección, estado)
        2. Gráficas matplotlib (3 ejes): Pérdida, Precisión, Workers activos
        3. Widget Text para logging con scroll automático

        :returns: None
        :rtype: None
        """
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        # ── Workers ──
        wf = ttk.LabelFrame(right, text="Workers conectados", padding=6)
        wf.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        cols = ("ID", "Dirección", "Estado")
        self._tree = ttk.Treeview(
            wf, columns=cols, show="headings", height=4, selectmode="none"
        )
        for col, w in zip(cols, (60, 200, 120)):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="center")
        self._tree.pack(fill=tk.X)

        srv_row = ttk.Frame(wf)
        srv_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(srv_row, text="Servidor:").pack(side=tk.LEFT)
        self._srv_var = tk.StringVar(value="OFFLINE")
        self._srv_lbl = tk.Label(
            srv_row,
            textvariable=self._srv_var,
            font=("Helvetica", 10, "bold"),
            fg="white",
            bg="#607D8B",
            padx=8,
            pady=2,
        )
        self._srv_lbl.pack(side=tk.LEFT, padx=8)

        m_row = ttk.Frame(wf)
        m_row.pack(fill=tk.X, pady=(4, 0))
        self._m_step = tk.StringVar(value="Step: —")
        self._m_loss = tk.StringVar(value="Loss: —")
        self._m_acc = tk.StringVar(value="Acc: —")
        self._m_stale = tk.StringVar(value="Staleness: —")
        self._m_clock = tk.StringVar(value="Clock: —")
        for v in (
            self._m_step,
            self._m_loss,
            self._m_acc,
            self._m_stale,
            self._m_clock,
        ):
            ttk.Label(m_row, textvariable=v, font=("Courier", 9)).pack(
                side=tk.LEFT, padx=10
            )

        # ── Gráficas ──
        pf = ttk.Frame(right)
        pf.grid(row=1, column=0, sticky="nsew")
        self._fig, (self._ax_loss, self._ax_acc, self._ax_wk) = plt.subplots(
            1, 3, figsize=(13, 4), dpi=95
        )
        self._fig.suptitle(
            "Entrenamiento Distribuido Asíncrono — ImageNet-1k",
            fontsize=12,
            fontweight="bold",
        )
        self._setup_axes()
        self._canvas = FigureCanvasTkAgg(self._fig, master=pf)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Log ──
        lf = ttk.LabelFrame(right, text="Log", padding=4)
        lf.grid(row=2, column=0, sticky="nsew", pady=(3, 0))
        lf.columnconfigure(0, weight=1)
        self._log_txt = tk.Text(
            lf,
            height=5,
            state=tk.DISABLED,
            font=("Courier", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            wrap=tk.WORD,
            relief=tk.FLAT,
        )
        ls = ttk.Scrollbar(lf, command=self._log_txt.yview)
        self._log_txt.configure(yscrollcommand=ls.set)
        self._log_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ls.pack(side=tk.RIGHT, fill=tk.Y)

    @staticmethod
    def _section(parent: ttk.Frame, text: str) -> None:
        """
        Crea sección visual con título separador en el panel izquierdo.

        Agrupa controles relacionados bajo un encabezado en negrita con
        separador horizontal debajo. Mejora legibilidad visual del formulario.

        :param parent: Frame padre donde insertar la sección.
        :type parent: ttk.Frame

        :param text: Título de la sección.
        :type text: str

        :returns: None
        :rtype: None
        """
        ttk.Label(parent, text=text, font=("Helvetica", 10, "bold")).pack(
            anchor=tk.W, pady=(14, 0)
        )
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

    def _entry(
        self, parent: ttk.Frame, label: str, var: tk.Variable, width: int = 22
    ) -> ttk.Entry:
        """
        Crea un par etiqueta+entrada de texto para ingreso de parámetros.

        Genera un Label con descriptivo y un Entry vinculado a una variable tkinter.
        La entrada se desactiva cuando el servidor está activo. Se agrega a
        self._config_widgets para control centralizado de estado.

        :param parent: Frame padre donde insertar el control.
        :type parent: ttk.Frame

        :param label: Texto descriptivo de la etiqueta.
        :type label: str

        :param var: Variable tkinter (StringVar, IntVar, etc) asociada al Entry.
        :type var: tk.Variable

        :param width: Ancho del Entry en caracteres (default: 22).
        :type width: int

        :returns: Widget Entry creado.
        :rtype: ttk.Entry
        """
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        pack_kwargs: dict = {"pady": 2}  # type: ignore[annotation-unchecked]
        if width == 22:
            pack_kwargs["fill"] = "x"
        entry.pack(**pack_kwargs)
        self._config_widgets.append(entry)
        return entry

    def _setup_axes(self) -> None:
        """
        Configura los tres ejes matplotlib con etiquetas, límites y estilos.

        Inicializa:
        - Eje 1 (loss): Pérdida en escala libre
        - Eje 2 (acc): Precisión limitada a 0-100%
        - Eje 3 (workers): Número de workers activos en escala libre

        Todos incluyen grid sutil (alpha=0.3) y etiquetas X/Y.

        :returns: None
        :rtype: None
        """
        for ax, title, ylabel in [
            (self._ax_loss, "Pérdida (ventana deslizante)", "Loss"),
            (self._ax_acc, "Precisión (ventana deslizante)", "Precisión (%)"),
            (self._ax_wk, "Workers activos", "N Workers"),
        ]:
            ax.set_title(title)
            ax.set_xlabel("Steps")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        self._ax_acc.set_ylim(0, 100)
        self._fig.tight_layout(rect=(0, 0, 1, 0.93))

    def _update_lr_cnn_state(self) -> None:
        """
        Habilita o deshabilita el campo LR CNN según la arquitectura seleccionada.

        Lógica:
        - resnet18: CNN siempre congelada → LR CNN no tiene efecto → deshabilitar
        - simple:   CNN entrenable E2E   → LR CNN controla su velocidad → habilitar

        También actualiza el texto informativo debajo del campo para clarificar
        el estado actual de la CNN (congelada vs entrenable).

        :returns: None
        :rtype: None
        """
        if self._ent_lr_cnn is None:
            return
        arch = self._v_arch.get()
        is_simple = arch == "simple"
        self._ent_lr_cnn.configure(state=tk.NORMAL if is_simple else tk.DISABLED)
        if hasattr(self, "_lbl_lr_cnn_info"):
            if is_simple:
                self._lbl_lr_cnn_info.configure(
                    text="ℹ LR CNN activo (modo E2E — CNN + MLP aprenden juntos)",
                    foreground="#2E7D32",
                )
            else:
                self._lbl_lr_cnn_info.configure(
                    text="ℹ LR CNN deshabilitado (resnet18 está congelada)",
                    foreground="#607D8B",
                )

    def _set_config_enabled(self, enabled: bool) -> None:
        """
        Desactiva o activa todos los widgets de configuración.

        Itera sobre self._config_widgets y establece su estado (NORMAL/DISABLED).
        Se utiliza para bloquear parámetros cuando el servidor está activo.
        Ignora errores en widgets que no tengan propiedad state (e.g., Labels).

        :param enabled: True para activar (state=NORMAL), False para desactivar (state=DISABLED).
        :type enabled: bool

        :returns: None
        :rtype: None
        """
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in self._config_widgets:
            try:
                widget.configure(state=state)
            except Exception:
                # Algunos widgets no tienen state (e.g., Labels)
                pass

    # ================================================================
    # BOTONES
    # ================================================================

    def _refresh_buttons(self) -> None:
        """
        Actualiza estado de botones de acción según el estado del servidor.

        Estados posibles:
        - OFFLINE: Solo "Encender servidor" habilitado
        - LOADING: Solo "Detener" habilitado (mientras carga CNN+MLP)
        - LISTENING: "Detener" habilitado, esperando primer step de entrenamiento
        - TRAINING: "Evaluar" (en validación) + "Detener" habilitados

        La transición LISTENING → TRAINING es automática cuando llega el primer
        evento on_step() (sin necesidad de botón manual).

        También actualiza indicador visual del servidor (label con color y estado).

        :returns: None
        :rtype: None
        """
        s = self._state
        self._btn_listen.configure(
            state=tk.NORMAL if s == self._S_OFFLINE else tk.DISABLED
        )
        self._btn_shutdown.configure(
            state=tk.NORMAL
            if s not in (self._S_OFFLINE, self._S_LOADING)
            else tk.DISABLED
        )
        self._btn_eval.configure(
            state=tk.NORMAL if s == self._S_TRAINING else tk.DISABLED
        )

        cfg = {
            self._S_OFFLINE: ("OFFLINE", "#607D8B"),
            self._S_LOADING: ("CARGANDO…", "#F57F17"),
            self._S_LISTENING: ("LISTENING", "#2E7D32"),
            self._S_TRAINING: ("TRAINING", "#1565C0"),
        }
        text, color = cfg[s]
        self._srv_var.set(text)
        self._srv_lbl.configure(bg=color)

    def _cmd_listen(self) -> None:
        """
        Carga CNN+MLP en hilo background e inicia servidor TCP.

        Proceso:
        1. Lee y valida todos los parámetros de GUI (tipos: int, float, str)
        2. Cambia estado a LOADING ('Cargando...')
        3. Lanza hilo daemon con _init():
           - Instancia CNN desde arquitectura seleccionada (resnet18/simple)
           - Instancia MLP con hidden1/hidden2 seleccionados y feature_dim del CNN
           - Instancia ParameterServer con parámetros Async-SGD (lr, lr_cnn, λ, windows)
           - Registra callbacks para eventos (step, report, worker_connected, worker_disconnected)
           - Llama ps.set_cnn(), ps.set_mlp(), ps.listen() para iniciar escucha TCP
        4. Envía evento ps_ready a la cola si éxito, o init_error si falla
        5. Loop principal (_poll) recibe eventos y actualiza GUI en tiempo real

        GUI no se congela durante descarga de ResNet-18 (~50MB) gracias a threading.

        Parámetros parseados:
        - host: dirección IP (default 0.0.0.0)
        - port: puerto TCP (default 9999)
        - lr, lr_cnn: learning rates (float)
        - lam: staleness lambda 0-1 (float)
        - rep, win: steps per report, metrics window (int)
        - h1, h2: hidden layer sizes (int)
        - arch: CNN architecture "resnet18" o "simple" (str)
        - bs, img_sz: batch size, image size (int)
        - seed: reproducibility seed o None (int/None)
        - hf_token: HuggingFace token o None (str/None)

        :returns: None
        :rtype: None

        :raises messagebox.showerror: Si parámetros inválidos (incompatible con tipos esperados)
        """
        try:
            host = self._v_host.get().strip()
            port = int(self._v_port.get())
            lr = float(self._v_lr.get())
            lr_cnn = float(self._v_lr_cnn.get())
            lam = float(self._v_lambda.get())
            rep = int(self._v_report.get())
            win = int(self._v_window.get())
            h1 = int(self._v_h1.get())
            h2 = int(self._v_h2.get())
            arch = self._v_arch.get()
            bs = int(self._v_batch_size.get())
            img_sz = int(self._v_image_size.get())

            # Seed: vacío o "None" → None (Python = aleatorio), o int string → int
            seed_str = self._v_seed.get().strip()
            if not seed_str or seed_str.lower() == "none":
                seed = None
            else:
                seed = int(seed_str)
        except ValueError as e:
            messagebox.showerror("Parámetro inválido", str(e))
            return

        hf_token = self._v_hf_token.get().strip()
        if not hf_token:
            messagebox.showerror("Token requerido", "Ingrese HF Token para streaming")
            return
        if not hf_token.startswith("hf_"):
            messagebox.showerror("Token inválido", "Token debe empezar con 'hf_'")
            return
        if len(hf_token) < 20:
            messagebox.showerror(
                "Token inválido", "Token muy corto (mínimo 20 caracteres)"
            )
            return

        q = self._q

        self._state = self._S_LOADING
        self._refresh_buttons()
        self._set_config_enabled(False)
        mode_str = "freeze" if arch == "resnet18" else "E2E"
        self._status.set(
            f"Cargando {arch} ({mode_str})... (puede tardar en la primera vez)"
        )
        self._log(
            f"[PS] Cargando CNN {arch} ({mode_str}) + MLP {h1}→{h2}→1000 | "
            f"lr_mlp={lr} lr_cnn={lr_cnn} | batch={bs} img={img_sz} seed={seed}"
        )

        def _init():
            try:
                cnn = CNNExtractor(
                    arch=arch,
                    device="cpu",
                    seed=seed,
                )
                mlp = MLPPyTorch(
                    feature_dim=cnn.feature_dim, hidden1=h1, hidden2=h2, n_classes=1000
                )

                ps = ParameterServer(
                    host=host,
                    port=port,
                    learning_rate=lr,
                    learning_rate_cnn=lr_cnn,
                    staleness_lambda=lam,
                    steps_per_report=rep,
                    metrics_window=win,
                    batch_size=bs,
                    image_size=img_sz,
                    seed=seed,
                    hf_token=hf_token,
                    on_step=lambda step, loss, acc, stale, elapsed: q.put(
                        ("step", (step, loss, acc, stale, elapsed))
                    ),
                    on_report=lambda step, loss, acc, elapsed: q.put(
                        ("report", (step, loss, acc, elapsed))
                    ),
                    on_worker_connected=lambda wid, addr: q.put(
                        ("connected", (wid, addr))
                    ),
                    on_worker_disconnected=lambda wid: q.put(("disconnected", (wid,))),
                )

                # set_cnn + set_mlp ANTES de listen → handshake siempre seguro
                ps.set_cnn(cnn)
                ps.set_mlp(mlp.state_dict_numpy())
                ps.listen()

                q.put(
                    (
                        "ps_ready",
                        (
                            ps,
                            host,
                            port,
                            arch,
                            cnn.feature_dim,
                            h1,
                            h2,
                            lr,
                            lr_cnn,
                            bs,
                            img_sz,
                            seed,
                        ),
                    )
                )

            except Exception as e:
                q.put(("init_error", e))

        threading.Thread(target=_init, daemon=True).start()
        self.root.after(100, self._poll)

    def _cmd_shutdown(self) -> None:
        """
        Detiene el servidor PS y desconecta todos los Workers.

        Proceso:
        1. Verifica que ps existe (validación de estado)
        2. Si en TRAINING: solicita confirmación antes de detener
        3. Lanza ps.stop() en hilo daemon:
           - Envía STOP a cada Worker conectado
           - Cierra socket listener TCP
           - Detiene loop de handshake
        4. Cambia estado a OFFLINE
        5. Limpia referencias: self._ps = None, self._workers.clear()
        6. Vacía tabla Treeview de workers
        7. Rehabilita todos los controles de configuración
        8. Actualiza interfaz visual

        Cambio de estado: LOADING/LISTENING/TRAINING → OFFLINE.
        Interfaz: congelada → descongelada.

        :returns: None
        :rtype: None
        """
        if not self._ps:
            return
        if self._state == self._S_TRAINING:
            if not messagebox.askyesno("Detener", "¿Detener el entrenamiento?"):
                return
        threading.Thread(target=self._ps.stop, daemon=True).start()
        self._state = self._S_OFFLINE
        self._ps = None
        self._workers.clear()
        for row in self._tree.get_children():
            self._tree.delete(row)
        self._refresh_buttons()
        self._set_config_enabled(True)
        # Detener Clock
        self._clock_running = False
        self._clock_start = 0.0
        self._m_clock.set("Clock: 00:00:00")
        self._log("[PS] Servidor detenido.")
        self._status.set("Servidor detenido.")

    def _cmd_evaluate(self) -> None:
        """
        Ejecuta evaluación no-bloqueante del modelo en el split de validación.

        Requisitos:
        - Servidor inicializado (self._ps existe)
        - Estado TRAINING (solo evaluar modelo mientras se entrena)

        Proceso:
        1. Lee parámetros de validación de GUI (dataset, token HF, número batches)
        2. Lanza ps.evaluate() en hilo daemon (no bloquea GUI)
        3. Envía evento 'log' para mostrar "Evaluando..."
        4. Al terminar, envía evento 'val_result' con (step, loss, accuracy)
        5. Handler _on_val() agrega resultados a histórico y actualiza gráficas

        Errores se capturan y loguean como eventos 'log' sin interrumpir ejecución.

        :returns: None
        :rtype: None
        """
        if not self._ps or self._state != self._S_TRAINING:
            return
        dataset = self._v_dataset.get().strip()
        hf_token = self._v_hf_token.get().strip() or None
        n_bat = int(self._v_val_batches.get())
        q = self._q
        ps = self._ps  # captura local para el hilo

        def _eval():
            q.put(("log", f"[PS] Evaluando ({n_bat} batches de validación)..."))
            try:
                acc, loss = ps.evaluate(
                    dataset_name=dataset, max_batches=n_bat, hf_token=hf_token
                )
                step = ps.current_version
                q.put(("val_result", (step, loss, acc)))
            except Exception as e:
                q.put(("log", f"[PS] Error en evaluación: {e}"))

        threading.Thread(target=_eval, daemon=True).start()

    # ================================================================
    # POLL
    # ================================================================

    def _poll(self) -> None:
        """
        Loop principal que procesa eventos asíncronos de la cola de eventos.

        Evento queue permite comunicación no-bloqueante entre:
        - Hilo background (_init, _eval, ps.listen callbacks)
        - Main GUI thread (this method)

        Tipos de eventos procesados:
        - ps_ready: PS inicializado → actualiza estado, log
        - init_error: Fallo en _init() → muestra error, revierte a OFFLINE
        - connected/disconnected: Worker cambió estado → tabla + refresh buttons
        - step: Step de entrenamiento → actualiza métricas instantáneas
        - report: Reporte periódico → gráficas + historial
        - val_result: Resultado evaluación → gráficas + historial validación
        - log: Mensaje genérico → agrega a widget Text
        - error: Excepción capturada → muestra error, revierte a LISTENING si TRAINING

        Loop persiste cuando estado != OFFLINE. Timeout 100ms entre calls para
        permitir responsividad de GUI.

        :returns: None
        :rtype: None
        """
        try:
            while True:
                kind, data = self._q.get_nowait()

                if kind == "ps_ready":
                    ps, host, port, arch, fdim, h1, h2, lr, lr_cnn, bs, img_sz, seed = (
                        data
                    )
                    self._ps = ps
                    self._t_start = time.perf_counter()
                    self._clock_start = time.perf_counter()  # Clock independiente
                    self._state = self._S_LISTENING
                    self._refresh_buttons()
                    mode = (
                        "freeze (CNN congelada)"
                        if arch == "resnet18"
                        else "E2E (CNN + MLP)"
                    )
                    self._log(
                        f"[PS] ✓ Servidor en {host}:{port} | arch={arch} ({mode}) | "
                        f"feature_dim={fdim} | MLP {fdim}→{h1}→{h2}→1000 | "
                        f"lr_mlp={lr} lr_cnn={lr_cnn} | batch={bs} img={img_sz} seed={seed}"
                    )
                    self._status.set(
                        f"Escuchando en {host}:{port} — esperando Workers..."
                    )

                elif kind == "init_error":
                    self._state = self._S_OFFLINE
                    self._refresh_buttons()
                    messagebox.showerror("Error inicializando PS", str(data))
                    self._status.set("Error. Revisa los parámetros.")

                elif kind == "connected":
                    self._on_connected(*data)
                elif kind == "disconnected":
                    self._on_disconnected(*data)
                elif kind == "step":
                    self._on_step(*data)
                elif kind == "report":
                    self._on_report(*data)
                elif kind == "val_result":
                    self._on_val(*data)
                elif kind == "log":
                    self._log(data)
                elif kind == "error":
                    self._on_error(data)

        except queue.Empty:
            pass
        except Exception as e:
            self._log(f"[ERROR] {e}")

        if self._state != self._S_OFFLINE:
            self.root.after(100, self._poll)

    # ================================================================
    # HANDLERS
    # ================================================================

    def _on_connected(self, wid: int, addr: str) -> None:
        """
        Maneja evento de conexión de un nuevo Worker.

        Acciones:
        1. Registra el Worker en diccionario self._workers[wid] = addr
        2. Agrega fila a tabla Treeview con (ID, dirección, "Activo")
        3. Asigna color único al Worker usando lista WORKER_COLORS (cíclica)
        4. Loguea mensaje de conexión
        5. Actualiza status bar con recuento de workers conectados

        Nota: El Worker comienza entrenamiento automáticamente (PS envía START
        después del handshake). La GUI transicionará a TRAINING automáticamente
        cuando reciba el primer evento on_step()

        :param wid: ID único del Worker (asignado por PS).
        :type wid: int

        :param addr: Dirección de red del Worker (formato "IP:puerto").
        :type addr: str

        :returns: None
        :rtype: None
        """
        self._workers[wid] = addr
        tag = f"w{wid}"
        color = self.WORKER_COLORS[wid % len(self.WORKER_COLORS)]
        if not self._tree.exists(tag):
            self._tree.insert(
                "", tk.END, iid=tag, values=(wid, addr, "Activo"), tags=(tag,)
            )
            self._tree.tag_configure(tag, foreground=color)
        self._refresh_buttons()
        self._log(f"[W{wid}] Conectado desde {addr}")
        self._status.set(f"Worker {wid} conectado | Total: {len(self._workers)}")

    def _on_disconnected(self, wid: int) -> None:
        """
        Maneja evento de desconexión de un Worker.

        Acciones:
        1. Remueve Worker de diccionario self._workers
        2. Elimina fila correspondiente de tabla Treeview
        3. Actualiza botones (refresca estado de GUI)
        4. Loguea evento de desconexión

        :param wid: ID del Worker desconectado.
        :type wid: int

        :returns: None
        :rtype: None
        """
        self._workers.pop(wid, None)
        if self._tree.exists(f"w{wid}"):
            self._tree.delete(f"w{wid}")
        self._refresh_buttons()
        self._log(f"[W{wid}] Desconectado.")

    def _on_step(
        self, step: int, loss: float, acc: float, stale: int, elapsed: float
    ) -> None:
        """
        Actualiza métricas instantáneas de entrenamiento.

        Se llama después de cada step de entrenamiento en algún Worker.
        Actualiza los labels de status bar con valores actuales (sin agregar a historial).

        En el primer step (cuando aún estamos en LISTENING), transiciona automáticamente
        a TRAINING: esto indica que el entrenamiento asíncrono ha comenzado y los
        Workers están enviando gradientes.

        Métricas mostradas:
        - Step: número de step actual (con separadores de miles)
        - Loss: pérdida actual con 4 decimales
        - Acc: precisión actual con 2 decimales (%)
        - Staleness: versión máxima desincronización observada

        :param step: Número del step de entrenamiento global.
        :type step: int

        :param loss: Valor de pérdida (loss) del batch actual.
        :type loss: float

        :param acc: Precisión en porcentaje (0-100).
        :type acc: float

        :param stale: Máximo staleness observado (versión del modelo más antigua en uso).
        :type stale: int

        :param elapsed: Tiempo elapsed en segundos desde inicio.
        :type elapsed: float

        :returns: None
        :rtype: None
        """
        # Auto-transición LISTENING → TRAINING en el primer step
        if self._state == self._S_LISTENING:
            self._state = self._S_TRAINING
            self._t_start = time.perf_counter()
            self._clock_start = time.perf_counter()  # Iniciar Clock
            self._clock_running = True
            self._refresh_buttons()
            self._log("[PS] ✓ Primer step recibido — Entrenamiento asíncrono activo.")
            self._status.set("Entrenamiento asíncrono en progreso...")
            self._update_clock()  # Iniciar Clock cada segundo

        self._m_step.set(f"Step: {step:,}")
        self._m_loss.set(f"Loss: {loss:.4f}")
        self._m_acc.set(f"Acc: {acc:.2f}%")
        self._m_stale.set(f"Staleness: {stale}")

    def _on_report(self, step: int, loss: float, acc: float, elapsed: float) -> None:
        """
        Agrega métricas a historial y actualiza gráficas (cada steps_per_report steps).

        Se dispara periódicamente (cada N steps según steps_per_report).
        Agrega puntos al historial de entrenamiento para reconstrucción de gráficas.

        Acciones:
        1. Agrega (step, loss, acc) a historial
        2. Registra número de workers activos en este reporte
        3. Redibuja gráficas (ejes loss, acc, workers)
        4. Actualiza status bar con resumen
        5. Loguea el reporte

        :param step: Número de step global del reporte.
        :type step: int

        :param loss: Pérdida promedio en ventana de training.
        :type loss: float

        :param acc: Precisión promedio en ventana de training (%).
        :type acc: float

        :param elapsed: Tiempo elapsed en segundos desde inicio (del PS).
        :type elapsed: float

        :returns: None
        :rtype: None
        """
        self._steps_hist.append(step)
        self._loss_hist.append(loss)
        self._acc_hist.append(acc)
        self._workers_hist.append(len(self._workers))
        self._update_plots()
        self._status.set(
            f"Step {step:,} | loss={loss:.4f} | acc={acc:.2f}% | "
            f"workers={len(self._workers)} | t={elapsed:.0f}s"
        )
        self._log(
            f"[Step {step:,}] loss={loss:.4f} | acc={acc:.2f}% | t={elapsed:.0f}s"
        )

    def _update_clock(self) -> None:
        """
        Actualiza el Clock cada segundo cuando está en TRAINING.

        Muestra el tiempo transcurrido desde el primer step en formato HH:MM:SS.
        Se ejecuta cada 1 segundo via root.after().
        """
        if not self._clock_running:
            return

        if self._state == self._S_TRAINING:
            elapsed = time.perf_counter() - self._clock_start
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self._m_clock.set(f"Clock: {hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self._m_clock.set("Clock: 00:00:00")

        self.root.after(1000, self._update_clock)

    def _on_val(self, step: int, loss: float, acc: float) -> None:
        """
        Agrega métricas de validación a historial y actualiza gráficas.

        Se dispara cuando termina una evaluación en validación (llamada a _cmd_evaluate).
        Agrega puntos al historial de validación que se grafican como scatter plot
        superpuesto sobre las curvas de entrenamiento.

        Acciones:
        1. Agrega (step, loss, acc) a histórico de validación
        2. Redibuja gráficas (superpone scatter points sobre curvas de training)
        3. Loguea resultado de validación
        4. Actualiza status bar con métricas de validación

        :param step: Step del modelo global al momento de la evaluación.
        :type step: int

        :param loss: Pérdida en validación.
        :type loss: float

        :param acc: Precisión en validación (%).
        :type acc: float

        :returns: None
        :rtype: None
        """
        self._val_steps.append(step)
        self._val_loss.append(loss)
        self._val_acc.append(acc)
        self._update_plots()
        self._log(f"[Val] step={step:,} | acc={acc:.2f}% | loss={loss:.4f}")
        self._status.set(f"Validación | acc={acc:.2f}% | loss={loss:.4f}")

    def _on_error(self, exc: Exception) -> None:
        """
        Maneja excepciones capturadas en hilos background.

        Acciones:
        1. Loguea excepción en widget Text
        2. Muestra diálogo de error al usuario
        3. Si estaba en TRAINING: revierte a LISTENING (mantiene setup)
        4. Actualiza interfaz

        Permite al usuario recuperarse de errores sin reiniciar todo
        (e.g., error de red, timeout en evaluación).

        :param exc: Excepción capturada (será convertida a string).
        :type exc: Exception

        :returns: None
        :rtype: None
        """
        self._log(f"[ERROR] {exc}")
        messagebox.showerror("Error", str(exc))
        if self._state == self._S_TRAINING:
            self._state = self._S_LISTENING
            self._refresh_buttons()

    # ================================================================
    # GRÁFICAS
    # ================================================================

    def _update_plots(self) -> None:
        """
        Redibuja las tres gráficas de monitoreo en tiempo real.

        Procesa histor- tales:
        1. Eje izquierda (Pérdida): Línea de training con puntos scatter de validación
        2. Eje centro (Precisión): Línea de training con puntos scatter de validación
        3. Eje derecha (Workers): Gráfico de pasos mostrando número de workers activos

        Limpia ejes previos, redibuja configuración (grid, límites), y renderiza
        puntos nuevos. Se llama después de eventos report y val_result.

        Colores:
        - Training (línea): Rojo (#F44336) para loss, Azul (#2196F3) para accuracy
        - Validación (scatter): Naranja (#FF9800)
        - Workers (step): Verde (#4CAF50)

        :returns: None
        :rtype: None
        """
        for ax in (self._ax_loss, self._ax_acc, self._ax_wk):
            ax.clear()
        self._setup_axes()

        if self._steps_hist:
            self._ax_loss.plot(
                self._steps_hist,
                self._loss_hist,
                "-o",
                color="#F44336",
                lw=2,
                ms=3,
                label="Train",
            )
            if self._val_steps:
                self._ax_loss.scatter(
                    self._val_steps,
                    self._val_loss,
                    color="#FF9800",
                    s=60,
                    zorder=5,
                    label="Val",
                )
            self._ax_loss.legend(fontsize=8)

            self._ax_acc.plot(
                self._steps_hist,
                self._acc_hist,
                "-o",
                color="#2196F3",
                lw=2,
                ms=3,
                label="Train",
            )
            if self._val_steps:
                self._ax_acc.scatter(
                    self._val_steps,
                    self._val_acc,
                    color="#FF9800",
                    s=60,
                    zorder=5,
                    label="Val",
                )
            self._ax_acc.legend(fontsize=8)

            self._ax_wk.step(
                self._steps_hist, self._workers_hist, color="#4CAF50", lw=2
            )
            self._ax_wk.set_ylim(0, max(self._workers_hist, default=1) + 1)

        self._canvas.draw()

    def _clear_plots(self) -> None:
        """
        Limpia todos los historéticos de métricas y borra gráficas.

        Vacía listas de historial:
        - _steps_hist, _loss_hist, _acc_hist: training
        - _val_steps, _val_loss, _val_acc: validación
        - _workers_hist: conteo de workers

        Luego redibuja ejes vacios. No afecta el entrenamiento en progreso
        (los eventos step/report seguirán llegándole a la cola).

        Ústil para limpiar gráficas sin reiniciar el PS ni detener el entrenamiento.

        :returns: None
        :rtype: None
        """
        for lst in (
            self._steps_hist,
            self._loss_hist,
            self._acc_hist,
            self._val_steps,
            self._val_loss,
            self._val_acc,
            self._workers_hist,
        ):
            lst.clear()
        self._update_plots()

    # ================================================================
    # LOG
    # ================================================================

    def _log(self, msg: str) -> None:
        """
        Añade un mensaje al widget de log con manejo de overflow automático.

        Inserta mensaje al final del widget Text, añade salto de línea,
        y mantiene buffer limitado a últimos 300 renglones para evitar
        agotamiento de memoria.

        Automáticamente desplaza al final del log (scroll) y mantiene
        el widget deshabilitado cuando no se está escribiendo.

        Formato esperado: "
        - "[PS] Mensaje del servidor"
        - "[W0] Mensaje del Worker 0"
        - "[Val] Mensaje de validación"
        - "[ERROR] Mensaje de error"

        :param msg: Cadena a agregar al log (sin salto de línea).
        :type msg: str

        :returns: None
        :rtype: None
        """
        self._log_txt.configure(state=tk.NORMAL)
        self._log_txt.insert(tk.END, msg + "\n")
        lines = int(self._log_txt.index(tk.END).split(".")[0])
        if lines > 300:
            self._log_txt.delete("1.0", f"{lines - 300}.0")
        self._log_txt.see(tk.END)
        self._log_txt.configure(state=tk.DISABLED)


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    """
    Punto de entrada principal: inicializa GUI del Parameter Server.

    Crea ventana tkinter raíz, instancia aplicación PSApp, y configura
    handler de cierre. Detecta intento de cierre (X button) y solicita
    confirmación si hay entrenamiento en progreso.

    Cleanup en caso de cierre:
    1. Si TRAINING: confirma "¿Detener entrenamiento y salir?"
    2. Si PS existe: intenta ps.stop() para liberar recursos limpiamente
    3. Destruye ventana tkinter
    4. Exit proceso con os._exit(0)

    El exitway forzado (os._exit) asegura terminación incluso si hay
    threads daemon aún en ejecución.

    :returns: None
    :rtype: None
    """
    root = tk.Tk()
    app = PSApp(root)

    def on_close():
        if app._state == app._S_TRAINING:
            if not messagebox.askyesno("Salir", "¿Detener el entrenamiento y salir?"):
                return
        if app._ps:
            try:
                app._ps.stop()
            except Exception:
                pass
        root.destroy()
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
