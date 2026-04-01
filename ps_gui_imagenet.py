"""
ps_gui_imagenet.py

GUI del Parameter Server asíncrono para entrenamiento distribuido en ImageNet.

DIFERENCIAS CON EL SISTEMA ANTERIOR:
  - Los parámetros del modelo se inicializan con MLPPyTorch.state_dict_numpy()
    y se pasan al PS con ps.set_mlp() — formato PyTorch state_dict nativo.
  - Las gráficas muestran steps (no épocas).
  - Métrica "Workers activos en el tiempo" como tercera gráfica.
  - Evaluación no bloqueante con botón dedicado.
  - El entrenamiento no termina automáticamente: el usuario lo detiene.
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
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._id = None
        self._tip = None
        widget.bind("<Enter>", lambda e: self._schedule())
        widget.bind("<Leave>", lambda e: self._cancel())

    def _schedule(self):
        self._id = self.widget.after(500, self._show)

    def _cancel(self):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None

    def _show(self):
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
    _S_LISTENING = "LISTENING"
    _S_TRAINING = "TRAINING"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Parameter Server — ImageNet-1k Distribuido")
        self.root.state("zoomed")
        self.root.columnconfigure(0, weight=0, minsize=310)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._q: queue.Queue = queue.Queue()
        self._ps: ParameterServer | None = None
        self._state: str = self._S_OFFLINE

        self._workers: dict = {}  # wid → addr
        self._steps_hist: list = []
        self._loss_hist: list = []
        self._acc_hist: list = []
        self._val_steps: list = []
        self._val_loss: list = []
        self._val_acc: list = []
        self._workers_hist: list = []
        self._t_start: float = 0.0
        self._status = tk.StringVar(value="Listo.")

        self._build_ui()
        self._refresh_buttons()

    # ================================================================
    # CONSTRUCCIÓN DE LA UI
    # ================================================================

    def _build_ui(self) -> None:
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
        cont = ttk.Frame(self.root, width=300)
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

        # ── Conexión ──────────────────────────────────────
        self._section(frm, "Conexión TCP")
        self._v_host = tk.StringVar(value="0.0.0.0")
        self._v_port = tk.IntVar(value=9999)
        self._entry(frm, "Host:", self._v_host)
        self._entry(frm, "Puerto:", self._v_port, width=10)

        # ── Dataset ───────────────────────────────────────
        self._section(frm, "Dataset")
        self._v_dataset = tk.StringVar(value="ILSVRC/imagenet-1k")
        self._v_hf_token = tk.StringVar(value=os.environ.get("HF_TOKEN", ""))
        self._entry(frm, "Dataset HF Hub:", self._v_dataset, width=30)
        ttk.Label(frm, text="HF Token:").pack(anchor=tk.W)
        ttk.Entry(frm, textvariable=self._v_hf_token, width=30, show="*").pack(
            fill=tk.X, pady=2
        )
        ttk.Label(
            frm,
            text="ℹ ILSVRC/imagenet-1k requiere token con\n  licencia aceptada en HF.",
            font=("Helvetica", 8),
            foreground="#1565C0",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 4))

        # ── CNN ───────────────────────────────────────────
        self._section(frm, "CNN Extractor")
        self._v_arch = tk.StringVar(value="resnet18")
        ttk.Radiobutton(
            frm,
            text="ResNet-18 + pesos ImageNet (recomendado)",
            variable=self._v_arch,
            value="resnet18",
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            frm, text="Simple CNN (sin pretrain)", variable=self._v_arch, value="simple"
        ).pack(anchor=tk.W)

        # ── MLP ───────────────────────────────────────────
        self._section(frm, "Clasificador MLP")
        self._v_h1 = tk.IntVar(value=1024)
        self._v_h2 = tk.IntVar(value=512)
        self._entry(frm, "Neuronas capa 1:", self._v_h1, width=8)
        self._entry(frm, "Neuronas capa 2:", self._v_h2, width=8)

        # ── Async SGD ─────────────────────────────────────
        self._section(frm, "Async SGD")
        self._v_lr = tk.StringVar(value="0.001")
        self._v_lambda = tk.StringVar(value="0.1")
        self._v_report = tk.IntVar(value=500)
        self._v_window = tk.IntVar(value=200)
        self._entry(frm, "Learning rate:", self._v_lr, width=12)
        self._entry(frm, "Staleness λ (0–1):", self._v_lambda, width=12)
        self._entry(frm, "Steps por reporte:", self._v_report, width=12)
        self._entry(frm, "Ventana métricas:", self._v_window, width=12)
        ttk.Label(
            frm,
            text="ℹ λ=0: sin corrección  λ=0.1: moderada  λ=1: fuerte",
            font=("Helvetica", 8),
            foreground="#2E7D32",
        ).pack(anchor=tk.W, pady=(2, 8))

        # ── Evaluación ────────────────────────────────────
        self._section(frm, "Evaluación")
        self._v_val_batches = tk.IntVar(value=50)
        self._entry(frm, "Batches de validación:", self._v_val_batches, width=8)
        self._btn_eval = ttk.Button(
            frm, text="Evaluar en validación ahora", command=self._cmd_evaluate
        )
        self._btn_eval.pack(fill=tk.X, pady=6)

        # ── Botones ───────────────────────────────────────
        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(18, 8))
        self._btn_listen = ttk.Button(
            frm, text="Encender servidor", command=self._cmd_listen
        )
        self._btn_train = ttk.Button(
            frm, text="▶  Iniciar entrenamiento", command=self._cmd_train
        )
        self._btn_shutdown = ttk.Button(
            frm, text="■  Detener todo", command=self._cmd_shutdown
        )
        self._btn_clear = ttk.Button(
            frm, text="Limpiar gráficas", command=self._clear_plots
        )
        for btn in (
            self._btn_listen,
            self._btn_train,
            self._btn_shutdown,
            self._btn_clear,
        ):
            btn.pack(fill=tk.X, pady=3)

        ToolTip(self._btn_listen, "Abre el socket TCP y espera Workers.")
        ToolTip(self._btn_train, "Configura el modelo y activa el entrenamiento.")
        ToolTip(self._btn_shutdown, "Envía STOP a todos los Workers y cierra el PS.")

    def _build_right(self) -> None:
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ── Workers ───────────────────────────────────────
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

        # Métricas en tiempo real
        m_row = ttk.Frame(wf)
        m_row.pack(fill=tk.X, pady=(4, 0))
        self._m_step = tk.StringVar(value="Step: —")
        self._m_loss = tk.StringVar(value="Loss: —")
        self._m_acc = tk.StringVar(value="Acc: —")
        self._m_stale = tk.StringVar(value="Staleness: —")
        for v in (self._m_step, self._m_loss, self._m_acc, self._m_stale):
            ttk.Label(m_row, textvariable=v, font=("Courier", 9)).pack(
                side=tk.LEFT, padx=10
            )

        # ── Gráficas ──────────────────────────────────────
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

        # ── Log ───────────────────────────────────────────
        lf = ttk.LabelFrame(right, text="Log", padding=4)
        lf.grid(row=2, column=0, sticky="ew", pady=(6, 0))
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

    # ── Helpers de construcción ───────────────────────────

    @staticmethod
    def _section(parent, text):
        ttk.Label(parent, text=text, font=("Helvetica", 10, "bold")).pack(
            anchor=tk.W, pady=(14, 0)
        )
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

    @staticmethod
    def _entry(parent, label, var, width=22):
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        pack_kwargs: dict[str, str | int] = {"pady": 2}  # type: ignore
        if width == 22:
            pack_kwargs["fill"] = "x"  # type: ignore
        entry.pack(**pack_kwargs)  # type: ignore

    def _setup_axes(self):
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

    # ================================================================
    # BOTONES
    # ================================================================

    def _refresh_buttons(self) -> None:
        has_w = bool(self._workers)
        s = self._state
        self._btn_listen.configure(
            state=tk.NORMAL if s == self._S_OFFLINE else tk.DISABLED
        )
        self._btn_train.configure(
            state=tk.NORMAL if s == self._S_LISTENING and has_w else tk.DISABLED
        )
        self._btn_shutdown.configure(
            state=tk.NORMAL if s != self._S_OFFLINE else tk.DISABLED
        )
        self._btn_eval.configure(
            state=tk.NORMAL if s == self._S_TRAINING else tk.DISABLED
        )

        cfg = {
            self._S_OFFLINE: ("OFFLINE", "#607D8B"),
            self._S_LISTENING: ("LISTENING", "#2E7D32"),
            self._S_TRAINING: ("TRAINING", "#1565C0"),
        }
        text, color = cfg[s]
        self._srv_var.set(text)
        self._srv_lbl.configure(bg=color)

    def _cmd_listen(self) -> None:
        try:
            host = self._v_host.get().strip()
            port = int(self._v_port.get())
            lr = float(self._v_lr.get())
            lam = float(self._v_lambda.get())
            rep = int(self._v_report.get())
            win = int(self._v_window.get())
        except ValueError as e:
            messagebox.showerror("Parámetro inválido", str(e))
            return

        q = self._q
        self._ps = ParameterServer(
            host=host,
            port=port,
            learning_rate=lr,
            staleness_lambda=lam,
            steps_per_report=rep,
            metrics_window=win,
            on_step=lambda step, loss, acc, stale: q.put(
                ("step", (step, loss, acc, stale))
            ),
            on_report=lambda step, loss, acc: q.put(("report", (step, loss, acc))),
            on_worker_connected=lambda wid, addr: q.put(("connected", (wid, addr))),
            on_worker_disconnected=lambda wid: q.put(("disconnected", (wid,))),
        )
        try:
            self._ps.listen()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._ps = None
            return

        self._state = self._S_LISTENING
        self._refresh_buttons()
        self._log(f"[PS] Servidor en {host}:{port}")
        self._status.set(f"Escuchando en {host}:{port}...")
        self.root.after(100, self._poll)

    def _cmd_train(self) -> None:
        if self._state != self._S_LISTENING or not self._workers:
            return
        try:
            h1 = int(self._v_h1.get())
            h2 = int(self._v_h2.get())
        except ValueError as e:
            messagebox.showerror("Parámetro inválido", str(e))
            return

        arch = self._v_arch.get()
        hf_token = self._v_hf_token.get().strip() or None
        q = self._q

        def _setup():
            try:
                assert self._ps is not None
                q.put(("log", f"[PS] Cargando CNN {arch}..."))
                cnn = CNNExtractor(
                    arch=arch, pretrained=(arch == "resnet18"), device="cpu", seed=42
                )
                self._ps.set_cnn(cnn)

                mlp = MLPPyTorch(
                    feature_dim=cnn.feature_dim, hidden1=h1, hidden2=h2, n_classes=1000
                )
                self._ps.set_mlp(mlp.state_dict_numpy())

                q.put(
                    (
                        "log",
                        f"[PS] Modelo listo: {arch} | "
                        f"feature_dim={cnn.feature_dim} | "
                        f"MLP {cnn.feature_dim}→{h1}→{h2}→1000",
                    )
                )
                q.put(("ready", None))
            except Exception as e:
                q.put(("error", e))

        threading.Thread(target=_setup, daemon=True).start()

    def _cmd_shutdown(self) -> None:
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
        self._log("[PS] Servidor detenido.")
        self._status.set("Servidor detenido.")

    def _cmd_evaluate(self) -> None:
        if not self._ps or self._state != self._S_TRAINING:
            return
        dataset = self._v_dataset.get().strip()
        hf_token = self._v_hf_token.get().strip() or None
        n_bat = int(self._v_val_batches.get())
        q = self._q

        def _eval():
            assert self._ps is not None
            q.put(("log", f"[PS] Evaluando ({n_bat} batches de validación)..."))
            try:
                acc, loss = self._ps.evaluate(
                    dataset_name=dataset, max_batches=n_bat, hf_token=hf_token
                )
                step = self._ps.current_version
                q.put(("val_result", (step, loss, acc)))
            except Exception as e:
                q.put(("log", f"[PS] Error en evaluación: {e}"))

        threading.Thread(target=_eval, daemon=True).start()

    # ================================================================
    # POLL
    # ================================================================

    def _poll(self) -> None:
        try:
            while True:
                kind, data = self._q.get_nowait()
                if kind == "connected":
                    self._on_connected(*data)
                elif kind == "disconnected":
                    self._on_disconnected(*data)
                elif kind == "step":
                    self._on_step(*data)
                elif kind == "report":
                    self._on_report(*data)
                elif kind == "val_result":
                    self._on_val(*data)
                elif kind == "ready":
                    self._on_ready()
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

    def _on_connected(self, wid, addr) -> None:
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

    def _on_disconnected(self, wid) -> None:
        self._workers.pop(wid, None)
        if self._tree.exists(f"w{wid}"):
            self._tree.delete(f"w{wid}")
        self._refresh_buttons()
        self._log(f"[W{wid}] Desconectado.")

    def _on_step(self, step, loss, acc, stale) -> None:
        self._m_step.set(f"Step: {step:,}")
        self._m_loss.set(f"Loss: {loss:.4f}")
        self._m_acc.set(f"Acc: {acc:.2f}%")
        self._m_stale.set(f"Staleness: {stale}")

    def _on_report(self, step, loss, acc) -> None:
        self._steps_hist.append(step)
        self._loss_hist.append(loss)
        self._acc_hist.append(acc)
        self._workers_hist.append(len(self._workers))
        self._update_plots()
        elapsed = time.perf_counter() - self._t_start
        self._status.set(
            f"Step {step:,} | loss={loss:.4f} | acc={acc:.2f}% | "
            f"workers={len(self._workers)} | t={elapsed:.0f}s"
        )
        self._log(f"[Step {step:,}] loss={loss:.4f} | acc={acc:.2f}%")

    def _on_val(self, step, loss, acc) -> None:
        self._val_steps.append(step)
        self._val_loss.append(loss)
        self._val_acc.append(acc)
        self._update_plots()
        self._log(f"[Val] step={step:,} | acc={acc:.2f}% | loss={loss:.4f}")
        self._status.set(f"Validación | acc={acc:.2f}% | loss={loss:.4f}")

    def _on_ready(self) -> None:
        self._state = self._S_TRAINING
        self._t_start = time.perf_counter()
        self._refresh_buttons()
        self._log("[PS] Entrenamiento asíncrono activo.")
        self._status.set("Entrenamiento asíncrono activo — Workers pueden conectarse.")

    def _on_error(self, exc) -> None:
        self._log(f"[ERROR] {exc}")
        messagebox.showerror("Error", str(exc))
        if self._state == self._S_TRAINING:
            self._state = self._S_LISTENING
            self._refresh_buttons()

    # ================================================================
    # GRÁFICAS
    # ================================================================

    def _update_plots(self) -> None:
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
