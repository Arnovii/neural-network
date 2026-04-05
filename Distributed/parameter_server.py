"""
Distributed/parameter_server.py

Parameter Server asíncrono para entrenamiento distribuido en ImageNet.

DISEÑO:
  - Un hilo TCP dedicado por Worker: sin barrera global, sin esperas inter-worker.
  - Los parámetros se actualizan inmediatamente al recibir UPDATES de cualquier Worker.
  - Corrección de staleness: α(s) = 1 / (1 + λ·s), s = versión_actual − versión_leída.
  - Estado interno en formato PyTorch state_dict nativo (numpy arrays para transporte).

HANDSHAKE SEGURO:
  Cada Worker que conecta queda bloqueado hasta que CNN y MLP estén configurados
  (máx 120 s). Esto elimina la race condition entre ps.listen() y ps.set_cnn().

AVERAGING DE CNN:
  Los pesos flotantes (convoluciones, BN running stats) se promedian con FedAvg.
  num_batches_tracked (int64, contador interno de BN) se excluye del averaging
  ya que su promedio no tiene sentido semántico — se mantiene el valor del PS.
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
    Acumulador thread-safe con ventana deslizante.

    Mantiene los últimos `window` valores de loss y accuracy para
    calcular promedios representativos del estado reciente sin
    sincronización global entre Workers.
    """

    def __init__(self, window: int = 200) -> None:
        self._lock = threading.Lock()

        # Deque es una lista eficiente de tamaño finito
        # Si se llena, elimina el más antiguo automáticamente
        self._losses: collections.deque = collections.deque(maxlen=window)
        self._accs: collections.deque = collections.deque(maxlen=window)
        self._total: int = 0  # Cuenta total de batches procesados

    def update(self, loss: float, acc: float) -> None:
        with self._lock:  # Sólo un hilo entra a la vez
            self._losses.append(loss)
            self._accs.append(acc)
            self._total += 1

    @property
    def total_batches(self) -> int:
        with self._lock:
            return self._total

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

    Ciclo de vida recomendado:
        ps = ParameterServer(...)
        ps.set_cnn(cnn)          # ANTES de listen()
        ps.set_mlp(mlp_state)    # ANTES de listen()
        ps.listen()
        # ... entrenamiento continuo ...
        ps.stop()

    Si listen() se llama antes de set_cnn/set_mlp, el handshake de cada
    Worker quedará bloqueado hasta que ambos estén disponibles (máx 120 s).

    :param host:              IP de escucha.
    :param port:              Puerto TCP.
    :param learning_rate:     LR base para SGD local de cada Worker.
    :param staleness_lambda:  Corrección de staleness (0.0 = sin corrección).
    :param steps_per_report:  Steps entre llamadas a on_report.
    :param metrics_window:    Tamaño de la ventana deslizante.
    :param on_step:           Callback por cada step: (step, loss, acc, staleness).
    :param on_report:         Callback cada steps_per_report: (step, loss, acc).
    :param on_worker_connected:    Callback: (worker_id, addr_str).
    :param on_worker_disconnected: Callback: (worker_id,).
    """

    def __init__(
        self,
        host: str,
        port: int,
        learning_rate: float = 0.001,
        staleness_lambda: float = 0.1,
        steps_per_report: int = 500,
        metrics_window: int = 200,
        batch_size: int = 64,
        image_size: int = 224,
        on_step: Optional[Callable] = None,
        on_report: Optional[Callable] = None,
        on_worker_connected: Optional[Callable] = None,
        on_worker_disconnected: Optional[Callable] = None,
    ) -> None:
        """
        Inicializa el Parameter Server para entrenamiento Async-SGD distribuido.

        Crea el servidor de parámetros que coordina múltiples Workers sin barrera
        global. Los Workers entrenan de forma asíncrona, y el PS actualiza el modelo
        global inmediatamente al recibir gradientes. Aplica corrección de staleness
        usando factor α(s) = 1/(1+λ·s) para atenuar gradientes antiguos.

        :param host: IP donde escucha el servidor (ej: '0.0.0.0' o '127.0.0.1')
        :type host: str

        :param port: Puerto TCP para conexión de Workers (ej: 9999)
        :type port: int

        :param learning_rate: Tasa de aprendizaje para SGD (defecto: 0.001)
        :type learning_rate: float

        :param staleness_lambda: Factor de corrección staleness λ en [0,1] (defecto: 0.1).
                                  λ=0 sin corrección, λ=1 fuerte corrección
        :type staleness_lambda: float

        :param steps_per_report: Pasos para agregar y reportar métricas (defecto: 500)
        :type steps_per_report: int

        :param metrics_window: Tamaño ventana deslizante para promedios (defecto: 200)
        :type metrics_window: int

        :param batch_size: Imágenes por batch en entrenamiento (defecto: 64).
                          Se distribuye a todos los Workers via CONFIG
        :type batch_size: int

        :param image_size: Tamaño de crop final post-descarga (defecto: 224).
                          Se distribuye a todos los Workers via CONFIG
        :type image_size: int

        :param on_step: Callback tras cada step de gradiente.
                       Firma: Callable[[int, float, float, float], None]
                       Args: (step, loss, acc, staleness_factor)
        :type on_step: Optional[Callable]

        :param on_report: Callback tras agregación de métricas.
                         Firma: Callable[[int, float, float], None]
                         Args: (step, avg_loss, avg_acc)
        :type on_report: Optional[Callable]

        :param on_worker_connected: Callback cuando Worker conecta.
                                   Firma: Callable[[int, str], None]
                                   Args: (worker_id, address)
        :type on_worker_connected: Optional[Callable]

        :param on_worker_disconnected: Callback cuando Worker desconecta.
                                      Firma: Callable[[int], None]
                                      Args: (worker_id,)
        :type on_worker_disconnected: Optional[Callable]
        """
        self.host = host
        self.port = port
        self.learning_rate = learning_rate
        self.staleness_lambda = staleness_lambda
        self.steps_per_report = steps_per_report
        self.batch_size = batch_size
        self.image_size = image_size

        self.on_step = on_step
        self.on_report = on_report
        self.on_worker_connected = on_worker_connected
        self.on_worker_disconnected = on_worker_disconnected

        # Estado del modelo
        self._mlp_state: Dict[str, np.ndarray] = {}
        self._cnn_state: Dict[str, np.ndarray] = {}

        # Keys de BN que NO se promedian (contador interno, no parámetro)
        self._no_avg_keys: set = set()

        # Evita corrupción cuando múltiples workers actualizan
        self._params_lock = threading.Lock()

        # Contador global de versión
        self._version: int = 0

        # CNN (para distribución inicial y evaluación)
        self._cnn: Optional[CNNExtractor] = None

        # Workers
        self._sockets: Dict[int, socket.socket] = {}
        self._addrs: Dict[int, str] = {}
        self._next_id: int = 0
        self._workers_lock = threading.Lock()

        # Métricas e historial
        self._metrics = RunningMetrics(window=metrics_window)
        self._history: Dict[str, List] = {
            "steps": [],
            "losses": [],
            "accuracies": [],
            "n_workers": [],
            "timestamps": [],
        }
        self._history_lock = threading.Lock()

        # Control
        self._server_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    # ================================================================
    # CONFIGURACIÓN
    # ================================================================

    def set_cnn(self, cnn: CNNExtractor) -> None:
        """
        Registra la CNN y serializa su estado en _cnn_state.

        Debe llamarse idealmente ANTES de listen() para que los Workers
        que conecten reciban los pesos sin espera.

        Los tensores int64 (num_batches_tracked de BatchNorm) se guardan
        en _no_avg_keys y se excluyen del averaging en _apply_update.
        """
        self._cnn = cnn
        base = getattr(cnn._model, "model", cnn._model)  # Accede al modelo interno
        no_avg: set = set()
        cnn_state: Dict[str, np.ndarray] = {}

        # Recorremos los pesos del modelo
        for name, tensor in base.state_dict().items():
            arr = tensor.cpu().numpy().copy()
            if arr.dtype == np.int64 or tensor.dtype == torch.int64:
                # num_batches_tracked: almacenar como int64, excluir del averaging
                # No lo promedia ya que es un contador
                no_avg.add(name)
            cnn_state[name] = arr  # Guardamos pesos

        with self._params_lock:
            self._cnn_state = cnn_state
            self._no_avg_keys = no_avg

        _log.ps(
            f"CNN lista: arch={cnn.arch} | feature_dim={cnn.feature_dim} | "
            f"params={len(cnn_state)} ({len(no_avg)} excluidos del avg)"
        )

    def set_mlp(self, mlp_state: Dict[str, np.ndarray]) -> None:
        """
        Registra el estado inicial del MLP en formato PyTorch state_dict.

        Claves esperadas: fc1.weight, fc1.bias, fc2.weight, fc2.bias,
                          fc3.weight, fc3.bias.
        """
        with self._params_lock:
            self._mlp_state = {key: value.copy() for key, value in mlp_state.items()}
        _log.ps(f"MLP listo: {list(mlp_state.keys())}")

    # ================================================================
    # CICLO DE VIDA
    # ================================================================

    def listen(self) -> None:
        """Abre el socket TCP y comienza a aceptar Workers en background."""
        if self._server_sock is not None:
            raise RuntimeError("El servidor ya está escuchando.")
        self._shutdown.clear()

        # AF_INET -> IPv4
        # SOCK_STREAM -> TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Permite reutilizar el puerto
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Asocia IP y puerto
        sock.bind((self.host, self.port))

        # Número de conexiones en cola
        sock.listen(32)

        # Timeout del socket
        sock.settimeout(1.0)

        # Guarda socket
        self._server_sock = sock

        # Crea hilo de aceptación
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="ps-accept"
        )
        self._accept_thread.start()
        _log.ps(f"Escuchando en {self.host}:{self.port}")

    def stop(self) -> None:
        """Envía STOP a todos los Workers y cierra el servidor."""
        _log.ps("Deteniendo servidor...")
        self._shutdown.set()  # Bandera global de apagado
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
    # ACEPTACIÓN
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

            # Crea hilo por worker
            threading.Thread(
                target=self._handle_new_connection,
                args=(conn, addr),
                daemon=True,
            ).start()

    def _handle_new_connection(self, conn: socket.socket, addr: tuple) -> None:
        """
        Handshake completo para un Worker nuevo.

        Secuencia:
          1. Recibir READY
          2. Enviar WORKER_ID
          3. Esperar a que CNN + MLP estén listos (máx 120 s)
          4. Enviar CNN_WEIGHTS (siempre, nunca opcional)
          5. Esperar CNN_ACK con verificación de arquitectura
          6. Enviar START
          7. Loop _serve_worker
        """
        # ----------------- 1. READY -----------------
        try:
            msg = receive_message(conn)
        except Exception:
            conn.close()
            return
        if msg["type"] != MsgType.READY:
            conn.close()
            return

        with self._workers_lock:
            # Creamos id único
            # Guardamos socket y dirección
            wid = self._next_id
            self._next_id += 1
            self._sockets[wid] = conn
            self._addrs[wid] = f"{addr[0]}:{addr[1]}"

        try:
            send_message(conn, MsgType.WORKER_ID, {"worker_id": wid})

            # Enviar configuración global inmediatamente
            send_message(
                conn,
                MsgType.CONFIG,
                {
                    "batch_size": self.batch_size,
                    "image_size": self.image_size,
                },
            )
        except Exception:
            with self._workers_lock:
                self._sockets.pop(wid, None)
                self._addrs.pop(wid, None)
            conn.close()
            return

        _log.ps(f"Worker {wid} conectado desde {addr[0]}:{addr[1]}")
        if self.on_worker_connected:
            self.on_worker_connected(wid, f"{addr[0]}:{addr[1]}")

        # ----------------- 2. Esperar CNN + MLP (bloqueo seguro) -----------------

        # Esto evita un race condition, donde el worker se conecte antes de que el modelo esté listo.
        # En otras palabras, el worker dice "dame la CNN", pero el PS no la tiene aún

        deadline = 120.0
        waited = 0.0
        poll = 0.25
        _log.ps(
            f"Worker {wid}: esperando configuración del modelo (máx {int(deadline)} s)..."
        )

        while True:
            with self._params_lock:
                ready = (  # No basta con crear la CNN, sus pesos deben estar listos
                    self._cnn is not None
                    and bool(self._mlp_state)
                    and bool(self._cnn_state)
                )
            if ready:
                break
            if waited >= deadline:
                _log.error(
                    f"Worker {wid}: timeout ({int(deadline)} s) esperando CNN + MLP. "
                    "Llama ps.set_cnn() y ps.set_mlp() antes de ps.listen()."
                )
                self._remove_worker(wid)
                conn.close()
                return
            if self._shutdown.is_set():
                self._remove_worker(wid)
                conn.close()
                return
            time.sleep(poll)  # Espera poll segundos
            waited += poll

        assert self._cnn is not None

        # ----------------- 3. Enviar CNN_WEIGHTS (siempre) -----------------
        try:
            arch = self._cnn.arch
            weights_bytes = self._cnn._get_weights_bytes()
            with self._params_lock:
                mlp_keys = list(self._mlp_state.keys())
                cnn_key_count = len(self._cnn_state)  # Número de parámetros de la CNN

            send_message(
                conn,
                MsgType.CNN_WEIGHTS,
                {
                    "arch": arch,
                    "weights_bytes": weights_bytes,
                    "mlp_keys": mlp_keys,
                    "cnn_key_count": cnn_key_count,
                },
            )
            _log.ps(
                f"Worker {wid}: CNN enviada — arch={arch} | "
                f"cnn_params={cnn_key_count} | mlp_keys={mlp_keys}"
            )

            ack = receive_message(conn)
            if ack["type"] != MsgType.CNN_ACK:
                _log.error(f"Worker {wid}: esperaba CNN_ACK, recibió {ack['type']}")
                self._remove_worker(wid)
                return

            # Verifica que el Worker confirmó la misma arquitectura
            ack_payload = ack.get("payload") or {}
            ack_arch = (
                ack_payload.get("arch") if isinstance(ack_payload, dict) else None
            )
            if ack_arch and ack_arch != arch:
                _log.error(
                    f"Worker {wid}: mismatch de arquitectura — "
                    f"PS={arch}, Worker={ack_arch}. Desconectando."
                )
                self._remove_worker(wid)
                return

            _log.ps(f"Worker {wid}: CNN cargada ✓ arch={arch}")

        except Exception as e:
            _log.error(f"Error distribuyendo CNN a Worker {wid}: {e}")
            self._remove_worker(wid)
            return

        # ----------------- 4. START -----------------
        try:
            send_message(conn, MsgType.START, {})
        except Exception as e:
            _log.error(f"Error enviando START a Worker {wid}: {e}")
            self._remove_worker(wid)
            return

        # ----------------- 5. Loop de servicio -----------------
        self._serve_worker(wid, conn)

    def _serve_worker(self, wid: int, conn: socket.socket) -> None:
        """
        Loop de servicio asíncrono para un Worker.

        Atiende solo a `wid`. Otros Workers tienen sus propios hilos.

        :param wid: Worker id
        :type wid: int

        :param conn: Socket de conexión
        :type conn: socket.socket
        """
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
                    with self._params_lock:
                        mlp_copy = {
                            key: value.copy() for key, value in self._mlp_state.items()
                        }
                        cnn_copy = {
                            key: value.copy() for key, value in self._cnn_state.items()
                        }
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
        Aplica los pesos actualizados con corrección de staleness.

            θ_new = θ + α(s) · (θ_worker − θ)
            α(s) = 1 / (1 + λ · s)
            s = versión_actual − versión_leída

        Los tensores marcados en _no_avg_keys (num_batches_tracked de BN)
        no se promedian: mantienen el valor del PS. Promediar un contador
        de batches no tiene significado semántico y puede distorsionar
        el comportamiento de BatchNorm en eval().
        """
        loss = payload.get("loss", 0.0)
        acc = payload.get("accuracy", 0.0)
        version_read = payload.get("version_read", 0)
        mlp_weights = payload.get("mlp_weights")
        cnn_weights = payload.get("cnn_weights")

        with self._params_lock:
            # Calcula factor de corrección Alpha
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
                    if key in self._no_avg_keys:
                        continue  # num_batches_tracked: no promediar
                    if key in cnn_weights:
                        curr = self._cnn_state[key].astype(
                            np.float64
                        )  # Previene errores numéricos
                        incoming = cnn_weights[key]
                        if incoming.dtype not in (
                            np.float32,
                            np.float64,
                        ):  # Convierte si hace falta
                            incoming = incoming.astype(np.float64)
                        self._cnn_state[key] = (
                            curr + alpha * (incoming - curr)
                        ).astype(
                            self._cnn_state[key].dtype
                        )  # Mantiene compatibilidad con PyTorch

            self._version += 1
            step = self._version

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
    # EVALUACIÓN
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

        Lo que hace es:
            1. Copia pesos actuales
            2. Carga la CNN con esos pesos
            3. Reconstruye el MLP
            4. Pasa datos de validación
            5. Calcula accuracy y loss

        :param max_batches: 50 × 256 = 12,800 imágenes por defecto.
        :return: (accuracy_pct, mean_loss)
        """
        from Utils.imagenet_streaming import ValidationStream
        from Model.mlp_pytorch import MLPPyTorch

        if self._cnn is None:
            _log.warn("[eval] CNN no configurada.")
            return 0.0, 0.0

        with self._params_lock:
            mlp_copy = {key: value.copy() for key, value in self._mlp_state.items()}
            cnn_copy = {key: value.copy() for key, value in self._cnn_state.items()}

        # Cargar CNN con estado global
        base = getattr(self._cnn._model, "model", self._cnn._model)
        sd = base.state_dict()
        with torch.no_grad():
            for name, arr in cnn_copy.items():
                if name in sd:
                    # Convierte Numpy a Tensor PyTorch
                    sd[name] = (
                        torch.from_numpy(arr).to(sd[name].device).to(sd[name].dtype)
                    )
            base.load_state_dict(sd)
        self._cnn._model.eval()

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

        criterion = torch.nn.CrossEntropyLoss()  # Función de pérdida de clasificación
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
                # Forward Pass
                features = self._cnn._model(X)
                logits = mlp(features)

                # Pérdida
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
                {key: value.copy() for key, value in self._mlp_state.items()},
                {key: value.copy() for key, value in self._cnn_state.items()},
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
