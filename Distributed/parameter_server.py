"""
Distributed/parameter_server.py

Parameter Server asíncrono para entrenamiento distribuido en ImageNet.

DISEÑO:
  - Un hilo TCP dedicado por Worker: sin barrera global, sin esperas inter-worker.
  - Los parámetros se actualizan inmediatamente al recibir UPDATES de cualquier Worker.
  - Corrección de staleness: gradientes calculados sobre parámetros viejos reciben
    un factor de atenuación α(s) = 1 / (1 + λ·s), donde s = versión_actual - versión_leída.
  - Todo el modelo (CNN + MLP) se almacena en formato PyTorch state_dict nativo.
    No hay conversión NumPy ↔ PyTorch en el PS: los tensores se serializan como
    numpy arrays para el transporte (pickle eficiente) y se reconstruyen en PyTorch
    solo cuando se necesita evaluación.

FORMATO DE PARÁMETROS INTERNOS:
  _mlp_state:  Dict[str, np.ndarray]  — claves PyTorch: fc1.weight, fc1.bias, …
  _cnn_state:  Dict[str, np.ndarray]  — state_dict completo incluyendo BN buffers

  Esto elimina el mapping W1↔fc1.weight que existía en el sistema anterior.
  El Worker recibe y devuelve exactamente el mismo formato de claves.

MÉTRICAS:
  RunningMetrics mantiene una ventana deslizante de los últimos N batches.
  En entrenamiento asíncrono no existe "época" — se reporta cada K steps
  (configurable como steps_per_report).
"""

import socket
import threading
import time
import collections
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from Distributed.protocol import MsgType, receive_message, send_message
from Model.cnn_extractor import CNNExtractor
from Utils.logging_util import get_logger

_log = get_logger(use_colors=True)


# ================================================================
# MÉTRICAS CON VENTANA DESLIZANTE
# ================================================================


class RunningMetrics:
    """
    Acumulador thread-safe de métricas con ventana deslizante.

    En entrenamiento asíncrono los Workers actualizan el PS de forma
    continua e independiente. Esta clase mantiene los últimos `window`
    valores para calcular promedios representativos del estado reciente,
    sin necesitar sincronización global.
    """

    def __init__(self, window: int = 200) -> None:
        self._lock = threading.Lock()
        self._losses: collections.deque = collections.deque(maxlen=window)
        self._accs: collections.deque = collections.deque(maxlen=window)
        self._total_batches: int = 0

    def update(self, loss: float, acc: float) -> None:
        with self._lock:
            self._losses.append(loss)
            self._accs.append(acc)
            self._total_batches += 1

    @property
    def total_batches(self) -> int:
        with self._lock:
            return self._total_batches

    def snapshot(self) -> Tuple[float, float]:
        """Devuelve (avg_loss, avg_acc) de la ventana actual."""
        with self._lock:
            if not self._losses:
                return 0.0, 0.0
            return float(np.mean(self._losses)), float(np.mean(self._accs))


# ================================================================
# PARAMETER SERVER
# ================================================================


class ParameterServer:
    """
    Parameter Server asíncrono para entrenamiento distribuido en ImageNet.

    Ciclo de vida:
        ps = ParameterServer(...)
        ps.set_cnn(cnn)
        ps.set_mlp(mlp_state_dict)
        ps.listen()          # abre socket, acepta Workers en background
        # ... entrenamiento continuo ...
        ps.stop()

    :param host:              IP de escucha.
    :param port:              Puerto TCP.
    :param learning_rate:     LR base para el SGD local de cada Worker.
    :param staleness_lambda:  Coeficiente de corrección de staleness (0.0–1.0).
    :param steps_per_report:  Steps entre reportes de métricas (on_report callback).
    :param metrics_window:    Tamaño de la ventana deslizante de métricas.
    :param on_step:           Callback por cada step. Firma: (step, loss, acc, staleness).
    :param on_report:         Callback cada steps_per_report. Firma: (step, loss, acc).
    :param on_worker_connected:    Callback al conectar Worker. Firma: (worker_id, addr).
    :param on_worker_disconnected: Callback al desconectar Worker. Firma: (worker_id,).
    """

    def __init__(
        self,
        host: str,
        port: int,
        learning_rate: float = 0.001,
        staleness_lambda: float = 0.1,
        steps_per_report: int = 500,
        metrics_window: int = 200,
        on_step: Optional[Callable] = None,
        on_report: Optional[Callable] = None,
        on_worker_connected: Optional[Callable] = None,
        on_worker_disconnected: Optional[Callable] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.learning_rate = learning_rate
        self.staleness_lambda = staleness_lambda
        self.steps_per_report = steps_per_report

        self.on_step = on_step
        self.on_report = on_report
        self.on_worker_connected = on_worker_connected
        self.on_worker_disconnected = on_worker_disconnected

        # ── Estado del modelo (formato PyTorch state_dict nativo) ──
        # Claves MLP:  fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias
        # Claves CNN:  state_dict completo de la CNN base (incluye BN buffers)
        self._mlp_state: Dict[str, np.ndarray] = {}
        self._cnn_state: Dict[str, np.ndarray] = {}
        self._params_lock = threading.Lock()
        self._version: int = 0

        # ── CNN (solo para distribución inicial y evaluación) ──
        self._cnn: Optional[CNNExtractor] = None

        # ── Workers activos ──
        self._sockets: Dict[int, socket.socket] = {}
        self._addrs: Dict[int, str] = {}
        self._next_id: int = 0
        self._workers_lock = threading.Lock()

        # ── Métricas ──
        self._metrics = RunningMetrics(window=metrics_window)

        # ── Historial (para GUI) ──
        self._history: Dict[str, List] = {
            "steps": [],
            "losses": [],
            "accuracies": [],
            "n_workers": [],
            "timestamps": [],
        }
        self._history_lock = threading.Lock()

        # ── Control del servidor ──
        self._server_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    # ================================================================
    # CONFIGURACIÓN INICIAL
    # ================================================================

    def set_cnn(self, cnn: CNNExtractor) -> None:
        """
        Registra la CNN y serializa su estado inicial en _cnn_state.

        Debe llamarse ANTES de listen() para que los Workers que se
        conecten reciban los pesos correctos.
        """
        self._cnn = cnn
        base = getattr(cnn._model, "model", cnn._model)
        with self._params_lock:
            # Convertir estado a numpy, protegiendo contra tipos que causen problemas
            # (ej: buffers int64 o bool de BatchNorm)
            self._cnn_state = {}
            for name, tensor in base.state_dict().items():
                arr = tensor.cpu().numpy().copy()
                # Convertir tipos no-float a float64 (seguro para averaging y transporte)
                # float64 es más seguro que float32 para evitar problemas de precisión
                if arr.dtype not in [np.float32, np.float64]:
                    arr = arr.astype(np.float64)
                self._cnn_state[name] = arr
        _log.ps(f"CNN lista: arch={cnn.arch}, feature_dim={cnn.feature_dim}")

    def set_mlp(self, mlp_state: Dict[str, np.ndarray]) -> None:
        """
        Registra el estado inicial del MLP en formato PyTorch state_dict.

        Claves esperadas: fc1.weight, fc1.bias, fc2.weight, fc2.bias,
                          fc3.weight, fc3.bias

        Uso desde ps_imagenet.py / ps_gui_imagenet.py:
            from Model.mlp_pytorch import MLPPyTorch
            mlp = MLPPyTorch(feature_dim, hidden1, hidden2, 1000)
            ps.set_mlp({k: v.cpu().numpy() for k, v in mlp.state_dict().items()})
        """
        with self._params_lock:
            self._mlp_state = {k: v.copy() for k, v in mlp_state.items()}
        _log.ps(f"MLP listo: {list(mlp_state.keys())}")

    # ================================================================
    # CICLO DE VIDA
    # ================================================================

    def listen(self) -> None:
        """Abre el socket TCP y comienza a aceptar Workers en background."""
        if self._server_sock is not None:
            raise RuntimeError("El servidor ya está escuchando.")
        self._shutdown.clear()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(32)
        sock.settimeout(1.0)
        self._server_sock = sock
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="ps-accept"
        )
        self._accept_thread.start()
        _log.ps(f"Escuchando en {self.host}:{self.port}")

    def stop(self) -> None:
        """Envía STOP a todos los Workers y cierra el servidor."""
        _log.ps("Deteniendo servidor...")
        self._shutdown.set()
        with self._workers_lock:
            wids = list(self._sockets.keys())
        for wid in wids:
            self._disconnect_worker(wid)
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        if self._accept_thread:
            self._accept_thread.join(timeout=3)
        _log.ps("Servidor detenido.")

    @property
    def connected_workers(self) -> List[int]:
        with self._workers_lock:
            return sorted(self._sockets.keys())

    @property
    def current_version(self) -> int:
        with self._params_lock:
            return self._version

    @property
    def history(self) -> Dict[str, List]:
        with self._history_lock:
            return {k: list(v) for k, v in self._history.items()}

    # ================================================================
    # ACEPTACIÓN DE CONEXIONES
    # ================================================================

    def _accept_loop(self) -> None:
        while not self._shutdown.is_set():
            if not self._server_sock:
                break
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            threading.Thread(
                target=self._handle_new_connection,
                args=(conn, addr),
                daemon=True,
            ).start()

    def _handle_new_connection(self, conn: socket.socket, addr: tuple) -> None:
        """
        Handshake + distribución CNN + loop de servicio para un Worker.

        Secuencia:
          1. Recibir READY
          2. Enviar WORKER_ID
          3. Enviar CNN_WEIGHTS (pesos serializados)
          4. Esperar CNN_ACK
          5. Enviar START
          6. Entrar en _serve_worker (loop REQUEST_PARAMS / UPDATES)
        """
        # ── 1. Handshake ──
        try:
            msg = receive_message(conn)
        except Exception:
            conn.close()
            return
        if msg["type"] != MsgType.READY:
            conn.close()
            return

        with self._workers_lock:
            wid = self._next_id
            self._next_id += 1
            self._sockets[wid] = conn
            self._addrs[wid] = f"{addr[0]}:{addr[1]}"

        try:
            send_message(conn, MsgType.WORKER_ID, {"worker_id": wid})
        except Exception:
            with self._workers_lock:
                self._sockets.pop(wid, None)
                self._addrs.pop(wid, None)
            conn.close()
            return

        _log.ps(f"Worker {wid} conectado desde {addr[0]}:{addr[1]}")
        if self.on_worker_connected:
            self.on_worker_connected(wid, f"{addr[0]}:{addr[1]}")

        # ── 2. Distribuir CNN ──
        if self._cnn is not None:
            try:
                send_message(
                    conn,
                    MsgType.CNN_WEIGHTS,
                    {
                        "arch": self._cnn.arch,
                        "weights_bytes": self._cnn._get_weights_bytes(),
                    },
                )
                ack = receive_message(conn)
                if ack["type"] == MsgType.CNN_ACK:
                    _log.ps(f"Worker {wid}: CNN cargada ✓")
            except Exception as e:
                _log.error(f"Error distribuyendo CNN a Worker {wid}: {e}")
                self._remove_worker(wid)
                return

        # ── 3. Señal de inicio ──
        try:
            send_message(conn, MsgType.START, {})
        except Exception as e:
            _log.error(f"Error enviando START a Worker {wid}: {e}")
            self._remove_worker(wid)
            return

        # ── 4. Loop de servicio ──
        self._serve_worker(wid, conn)

    def _serve_worker(self, wid: int, conn: socket.socket) -> None:
        """
        Loop de servicio asíncrono para un Worker.

        No hay barrera: este hilo atiende solo a `wid` de forma continua.
        Otros Workers tienen sus propios hilos y no se bloquean entre sí.
        """
        # VALIDACIÓN CRÍTICA: MLP DEBE ESTAR INICIALIZADO
        with self._params_lock:
            if not self._mlp_state:
                _log.error(
                    f"[CRÍTICO] MLP NO INICIALIZADO en PS. "
                    f"Llama ps.set_mlp() ANTES de ps.listen(). "
                    f"Worker {wid} recibirá MLP con random init → logs ≈ 0 → accuracy = 0%"
                )
        
        try:
            while not self._shutdown.is_set():
                try:
                    msg = receive_message(conn)
                except Exception:
                    break

                mtype = msg["type"]

                if mtype == MsgType.STOP:
                    break

                elif mtype == MsgType.REQUEST_PARAMS:
                    # Responder inmediatamente con el estado actual
                    with self._params_lock:
                        mlp_copy = {k: v.copy() for k, v in self._mlp_state.items()}
                        cnn_copy = {k: v.copy() for k, v in self._cnn_state.items()}
                        ver = self._version
                    try:
                        send_message(
                            conn,
                            MsgType.PARAMS,
                            {
                                "mlp_state": mlp_copy,
                                "cnn_state": cnn_copy,
                                "version": ver,
                                "lr": self.learning_rate,
                            },
                        )
                    except Exception as e:
                        _log.error(f"Error enviando PARAMS a Worker {wid}: {e}")
                        break

                elif mtype == MsgType.UPDATES:
                    self._apply_update(wid, msg["payload"])

        finally:
            _log.ps(f"Worker {wid} desconectado.")
            self._remove_worker(wid)

    # ================================================================
    # ACTUALIZACIÓN ASÍNCRONA
    # ================================================================

    def _apply_update(self, wid: int, payload: dict) -> None:
        """
        Aplica los pesos actualizados de un Worker con corrección de staleness.

        Algoritmo: promedio ponderado entre estado global y propuesta del Worker.
            θ_new = θ + α(s) · (θ_worker − θ)
            α(s)  = 1 / (1 + λ · s)
            s     = versión_actual − versión_leída_por_worker

        El estado CNN se actualiza de la misma forma, incluyendo los buffers
        de BatchNorm (running_mean, running_var) que forman parte del state_dict.
        Esto es correcto: promediar running stats de Workers que vieron shards
        distintos del mismo dataset produce una estimación global válida.
        """
        loss = payload.get("loss", 0.0)
        acc = payload.get("accuracy", 0.0)
        version_read = payload.get("version_read", 0)
        mlp_weights = payload.get("mlp_weights")  # Dict[str, np.ndarray] PyTorch keys
        cnn_weights = payload.get("cnn_weights")  # Dict[str, np.ndarray] state_dict

        with self._params_lock:
            staleness = max(0, self._version - version_read)
            alpha = 1.0 / (1.0 + self.staleness_lambda * staleness)

            if mlp_weights:
                for key in self._mlp_state:
                    if key in mlp_weights:
                        self._mlp_state[key] += alpha * (
                            mlp_weights[key] - self._mlp_state[key]
                        )

            if cnn_weights:
                for key in self._cnn_state:
                    if key in cnn_weights:
                        # Proteger contra mismatch de tipos (ej: buffer int64 de BN)
                        # Asegurar ambos operandos son float
                        current = self._cnn_state[key]
                        incoming = cnn_weights[key]
                        
                        # Convertir a float64 si no lo son
                        if current.dtype not in [np.float32, np.float64]:
                            current = current.astype(np.float64)
                        if incoming.dtype not in [np.float32, np.float64]:
                            incoming = incoming.astype(np.float64)
                        
                        # Averaging seguro
                        self._cnn_state[key] = current + alpha * (incoming - current)

            self._version += 1
            step = self._version

        # Métricas y callbacks (fuera del lock para no bloquearlo)
        self._metrics.update(loss, acc)
        if self.on_step:
            self.on_step(step, loss, acc, staleness)

        n = self._metrics.total_batches
        if n > 0 and n % self.steps_per_report == 0:
            avg_loss, avg_acc = self._metrics.snapshot()
            with self._workers_lock:
                n_workers = len(self._sockets)
            self._record_history(step, avg_loss, avg_acc, n_workers)
            if self.on_report:
                self.on_report(step, avg_loss, avg_acc)
            _log.train(
                f"Step {step:,}",
                metric=f"loss={avg_loss:.4f} | acc={avg_acc:.2f}% | workers={n_workers}",
            )

    def _record_history(self, step, loss, acc, n_workers) -> None:
        with self._history_lock:
            self._history["steps"].append(step)
            self._history["losses"].append(loss)
            self._history["accuracies"].append(acc)
            self._history["n_workers"].append(n_workers)
            self._history["timestamps"].append(time.time())

    # ================================================================
    # EVALUACIÓN EN VALIDACIÓN
    # ================================================================

    def evaluate(
        self,
        dataset_name: str = "ILSVRC/imagenet-1k",
        max_batches: int = 50,
        batch_size: int = 256,
        hf_token: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Evaluación rápida en el split de validación de ImageNet.

        Usa el estado CNN y MLP más reciente del PS. La CNN se pone en
        eval() antes de extraer features para usar running stats de BN.

        :param max_batches: Número de batches a evaluar (50 × 256 = 12,800 imgs).
                            Para evaluación completa usar max_batches=None.
        :return: (accuracy_pct, mean_loss)
        """
        from Utils.imagenet_streaming import ValidationStream
        from Model.mlp_pytorch import MLPPyTorch

        if self._cnn is None:
            _log.warn("[eval] CNN no configurada.")
            return 0.0, 0.0

        # Tomar snapshot del estado actual
        with self._params_lock:
            mlp_copy = {k: v.copy() for k, v in self._mlp_state.items()}
            cnn_copy = {k: v.copy() for k, v in self._cnn_state.items()}

        # Cargar CNN con estado global actualizado
        base = getattr(self._cnn._model, "model", self._cnn._model)
        sd = base.state_dict()
        with torch.no_grad():
            for name, arr in cnn_copy.items():
                if name in sd:
                    sd[name] = (
                        torch.from_numpy(arr).to(sd[name].device).to(sd[name].dtype)
                    )
            base.load_state_dict(sd)
        self._cnn._model.eval()

        # Construir MLP PyTorch con estado global
        # Las shapes se infieren directamente del state_dict
        fc1w = mlp_copy.get("fc1.weight")
        if fc1w is None:
            _log.warn("[eval] MLP no configurado.")
            return 0.0, 0.0
        hidden1, feature_dim = fc1w.shape
        hidden2 = mlp_copy["fc2.weight"].shape[0]
        n_classes = mlp_copy["fc3.weight"].shape[0]

        mlp = MLPPyTorch(feature_dim, hidden1, hidden2, n_classes).to(self._cnn.device)
        with torch.no_grad():
            for name, param in mlp.named_parameters():
                if name in mlp_copy:
                    param.data.copy_(torch.from_numpy(mlp_copy[name]).to(param.device))
        mlp.eval()

        # Iterar sobre validación
        criterion = torch.nn.CrossEntropyLoss()
        total_correct, total_loss, total_n = 0, 0.0, 0

        for X_np, Y_np in ValidationStream(
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_batches=max_batches,
            hf_token=hf_token,
        ).iterate():
            X = torch.from_numpy(X_np).to(self._cnn.device)
            Y = torch.from_numpy(Y_np).to(self._cnn.device)
            with torch.no_grad():
                features = self._cnn._model(X)
                logits = mlp(features)
                loss = criterion(logits, Y)
                correct = (logits.argmax(1) == Y).sum().item()
            n = len(Y_np)
            total_correct += correct
            total_loss += loss.item() * n
            total_n += n

        if total_n == 0:
            return 0.0, 0.0
        return 100.0 * total_correct / total_n, total_loss / total_n

    # ================================================================
    # UTILIDADES
    # ================================================================

    def get_state(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
        """Devuelve (mlp_state, cnn_state, version) — copia thread-safe."""
        with self._params_lock:
            return (
                {k: v.copy() for k, v in self._mlp_state.items()},
                {k: v.copy() for k, v in self._cnn_state.items()},
                self._version,
            )

    def _disconnect_worker(self, wid: int) -> None:
        with self._workers_lock:
            sock = self._sockets.pop(wid, None)
            self._addrs.pop(wid, None)
        if sock:
            try:
                send_message(sock, MsgType.STOP, None)
                sock.close()
            except Exception:
                pass

    def _remove_worker(self, wid: int) -> None:
        with self._workers_lock:
            sock = self._sockets.pop(wid, None)
            self._addrs.pop(wid, None)
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        if self.on_worker_disconnected:
            self.on_worker_disconnected(wid)
