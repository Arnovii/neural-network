"""
Distributed/parameter_server.py

Parameter Server asíncrono para entrenamiento distribuido en ImageNet.

DISEÑO:
  - Un hilo TCP dedicado por Worker: sin barrera global, sin esperas inter-worker.
  - Los parámetros se actualizan inmediatamente al recibir UPDATES de cualquier Worker.
  - Corrección de staleness: α(s) = 1 / (1 + λ·s), s = versión_actual − versión_leída.
  - Estado interno en formato PyTorch state_dict nativo (numpy arrays para transporte).

OPTIMIZADOR:
  El Worker usa SGD puro con gradient clipping. No se usa Adam porque sus momentos
  (m, v) son locales al Worker y se dessincronizan cuando el PS hace FedAvg con
  múltiples workers: los momentos apuntan a una trayectoria que ya no corresponde
  al punto de partida devuelto por el PS.

  El PS envía dos LRs en PARAMS:
    lr      → Learning rate del MLP (siempre activo)
    lr_cnn  → Learning rate de la CNN (solo activo en modo E2E / simple)

  En modo resnet18 (freeze), lr_cnn viaja en el mensaje pero el Worker lo ignora
  porque la CNN no recibe gradientes.

LRs SEPARADOS:
  Motivación: en E2E desde cero, la CNN necesita un LR más bajo que el MLP.
  La CNN parte de pesos aleatorios y sus capas convolucionales profundas reciben
  gradientes más pequeños que el MLP. Un LR igual causaría que la CNN avance
  demasiado despacio mientras el MLP sobreajusta las features actuales.

AVERAGING DE CNN:
  Los pesos flotantes (convoluciones, BN running stats) se promedian con FedAvg.
  num_batches_tracked (int64, contador interno de BN) se excluye del averaging
  ya que su promedio no tiene sentido semántico — se mantiene el valor del PS.

FUNCIÓN AUXILIAR: suggest_lr(lr_base, n_workers)
  Calcula el learning rate sugerido para entrenamiento con múltiples Workers
  usando la Linear Scaling Rule adaptada para FedAvg asíncrono:
    lr_sugerido = lr_base × √n_workers
  No modifica el sistema — es puramente informativa.
  Con 1 worker: sin cambio. Con 4 workers: lr × 2.
"""

import socket
import threading
import time
import collections
import math as _math
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from Distributed.protocol import MsgType, receive_message, send_message
from Model.cnn_extractor import CNNExtractor
from Utils.logging_util import get_logger
from Utils.results_exporter import ResultsExporter
from Utils.constants import (
    HF_DATASET_DEFAULT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_LR_CNN,
    DEFAULT_SEED,
    DEFAULT_STALENESS_LAMBDA,
    EXPORT_DIR_DEFAULT,
    HIDDEN1_DEFAULT,
    HIDDEN2_DEFAULT,
    IMAGE_SIZE,
    METRICS_WINDOW_DEFAULT,
    STEPS_PER_REPORT_DEFAULT,
)

_log = get_logger(use_colors=True)


# ================================================================
# UTILIDAD: LINEAR SCALING RULE PARA LR
# ================================================================


def suggest_lr(lr_base: float, n_workers: int) -> float:
    """
    Calcula el learning rate sugerido según la Linear Scaling Rule.

    En entrenamiento distribuido síncrono, la regla lineal de Goyal et al.
    (2017) recomienda escalar lr ∝ n_workers para mantener la misma
    dinámica de descenso de gradiente que con un solo worker.

    En entrenamiento ASÍNCRONO con FedAvg y corrección de staleness,
    la escala lineal es demasiado agresiva: el staleness ya atenúa
    algunos updates, por lo que se usa escala por raíz cuadrada:

        lr_sugerido = lr_base × √n_workers

    Esta fórmula es más conservadora y apropiada cuando los workers
    tienen asincronía variable (staleness heterogéneo).

    Con 1 worker: lr_sugerido = lr_base (sin cambio).
    Con 4 workers: lr_sugerido ≈ 2 × lr_base.

    La función es informativa — no modifica el sistema. Úsela como
    referencia al configurar learning_rate en ParameterServer.

    :param lr_base:    LR de referencia para 1 worker.
    :type lr_base:     float

    :param n_workers:  Número de Workers que se planea conectar.
    :type n_workers:   int

    :returns: LR ajustado según √n_workers.
    :rtype:   float

    :example:
        >>> suggest_lr(0.01, 1)   # → 0.01
        >>> suggest_lr(0.01, 4)   # → 0.02
        >>> suggest_lr(0.001, 2)  # → 0.00141...
    """
    if n_workers <= 1:
        return lr_base
    return lr_base * _math.sqrt(n_workers)


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
        """Inicializa el acumulador de metricas con ventana deslizante.

        :param window: Tamaño de la ventana deslizante (ultimos N valores).
        :type window: int
        """
        self._lock = threading.Lock()

        # Deque es una lista eficiente de tamaño finito
        # Si se llena, elimina el más antiguo automáticamente
        self._losses: collections.deque = collections.deque(maxlen=window)
        self._accs: collections.deque = collections.deque(maxlen=window)
        self._total: int = 0  # Cuenta total de batches procesados

    def update(self, loss: float, acc: float) -> None:
        """
        Registra un batch de métricas en la ventana deslizante.

        Añade loss y accuracy a sus deques correspondientes. Si la ventana
        está llena, elimina automáticamente el valor más antiguo.

        :param loss: Valor de pérdida del batch actual.
        :type loss: float

        :param acc: Precisión del batch actual (%).
        :type acc: float

        :returns: None
        :rtype: None
        """
        with self._lock:  # Solo un hilo entra a la vez
            self._losses.append(loss)
            self._accs.append(acc)
            self._total += 1

    @property
    def total_batches(self) -> int:
        """
        Retorna el número total de batches procesados desde el inicio.

        Contador acumulado que no se reinicia con la ventana deslizante.
        Útil para calcular progreso global del entrenamiento.

        :returns: Número total de batches procesados.
        :rtype: int
        """
        with self._lock:
            return self._total

    def snapshot(self) -> Tuple[float, float, float, float]:
        """Calcula estadisticas de la ventana deslizante actual.

        :returns: Tupla (avg_loss, avg_acc, std_loss, std_acc) de la ventana actual.
            Si la ventana esta vacia, retorna (0.0, 0.0, 0.0, 0.0).
        :rtype: Tuple[float, float, float, float]
        """
        with self._lock:
            if not self._losses:
                return 0.0, 0.0, 0.0, 0.0

            mean_loss = float(np.mean(self._losses))
            mean_acc = float(np.mean(self._accs))
            if len(self._losses) < 2:
                return mean_loss, mean_acc, 0.0, 0.0

            return (
                mean_loss,
                mean_acc,
                float(np.std(self._losses)),
                float(np.std(self._accs)),
            )


# ================================================================
# PARAMETER SERVER
# ================================================================


class ParameterServer:
    """
    Parameter Server asíncrono para entrenamiento distribuido en ImageNet.
    """

    def __init__(
        self,
        host: str,
        port: int,
        learning_rate: float = DEFAULT_LR,
        learning_rate_cnn: float = DEFAULT_LR_CNN,
        staleness_lambda: float = DEFAULT_STALENESS_LAMBDA,
        steps_per_report: int = STEPS_PER_REPORT_DEFAULT,
        metrics_window: int = METRICS_WINDOW_DEFAULT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        image_size: int = IMAGE_SIZE,
        dataset_name: str = HF_DATASET_DEFAULT,
        seed: int | None = DEFAULT_SEED,
        hf_token: str | None = None,
        export_dir: str = EXPORT_DIR_DEFAULT,
        hidden1: int = HIDDEN1_DEFAULT,
        hidden2: int = HIDDEN2_DEFAULT,
        on_step: Callable | None = None,
        on_report: Callable | None = None,
        on_worker_connected: Callable | None = None,
        on_worker_disconnected: Callable | None = None,
        on_start_sent: Callable | None = None,
    ) -> None:
        """
        Inicializa el Parameter Server para entrenamiento Async-SGD distribuido.

        Crea el servidor de parametros que coordina multiple Workers sin barrera
        global. Los Workers entrenan de forma asincrona, y el PS actualiza el modelo
        global inmediatamente al recibir gradientes. Aplica correccion de staleness
        usando factor alpha(s) = 1/(1+lambda*s) para atenuar gradientes antiguos.

        :param host: IP donde escucha el servidor (ej: '0.0.0.0' o '127.0.0.1').
        :type host: str

        :param port: Puerto TCP para conexion de Workers (ej: 9999).
        :type port: int

        :param learning_rate: Tasa de aprendizaje para SGD (defecto: 0.001).
        :type learning_rate: float

        :param learning_rate_cnn: Tasa de aprendizaje para la CNN en modo E2E (defecto: 0.001).
        :type learning_rate_cnn: float

        :param staleness_lambda: Factor de correccion staleness lambda en [0,1] (defecto: 0.1).
            lambda=0 sin correccion, lambda=1 fuerte correccion.
        :type staleness_lambda: float

        :param steps_per_report: Pasos para agregar y reportar metricas (defecto: 500).
        :type steps_per_report: int

        :param metrics_window: Tamano ventana deslizante para promedios (defecto: 200).
        :type metrics_window: int

        :param batch_size: Imagenes por batch en entrenamiento (defecto: 64).
            Se distribuye a todos los Workers via CONFIG.
        :type batch_size: int

        :param image_size: Tamano de crop final post-descarga (defecto: 224).
            Se distribuye a todos los Workers via CONFIG.
        :type image_size: int

        :param dataset_name: Nombre del dataset HuggingFace (defecto: 'ILSVRC/imagenet-1k').
        :type dataset_name: str

        :param seed: Semilla RNG para reproducibilidad (None = aleatorio, enviado en CONFIG).
        :type seed: int | None

        :param hf_token: Token de HuggingFace para acceso al dataset.
        :type hf_token: str | None

        :param export_dir: Directorio base para exportar resultados (defecto: './Exports').
        :type export_dir: str

        :param hidden1: Tamano de la primera capa oculta del MLP (defecto: 1024).
        :type hidden1: int

        :param hidden2: Tamano de la segunda capa oculta del MLP (defecto: 512).
        :type hidden2: int

        :param on_step: Callback tras cada step de gradiente.
            Firma: Callable[[int, float, float, float, float], None].
            Args: (step, loss, acc, staleness_factor, elapsed).
        :type on_step: Callable | None

        :param on_report: Callback tras agregacion de metricas.
            Firma: Callable[[int, float, float, float], None].
            Args: (step, avg_loss, avg_acc, elapsed).
        :type on_report: Callable | None

        :param on_worker_connected: Callback cuando Worker conecta.
            Firma: Callable[[int, str], None].
            Args: (worker_id, address).
        :type on_worker_connected: Callable | None

        :param on_worker_disconnected: Callback cuando Worker desconecta.
            Firma: Callable[[int], None].
            Args: (worker_id,).
        :type on_worker_disconnected: Callable | None

        :param on_start_sent: Callback cuando se envia START a un Worker.
            Firma: Callable[[int], None].
            Args: (worker_id,).
        :type on_start_sent: Callable | None

        :returns: None
        :rtype: None
        """
        self.host = host
        self.port = port
        self.learning_rate = learning_rate  # LR del MLP
        self.learning_rate_cnn = learning_rate_cnn  # LR de la CNN (E2E)
        self.staleness_lambda = staleness_lambda
        self.steps_per_report = steps_per_report
        self.metrics_window = metrics_window
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.seed = seed
        self.hf_token = hf_token
        self.export_dir = export_dir
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.on_step = on_step
        self.on_report = on_report
        self.on_worker_connected = on_worker_connected
        self.on_worker_disconnected = on_worker_disconnected
        self.on_start_sent = on_start_sent

        # Estado del modelo
        self._mlp_state: Dict[str, np.ndarray] = {}
        self._cnn_state: Dict[str, np.ndarray] = {}

        # Keys de BN que NO se promedian (contador interno, no parámetro)
        self._no_avg_keys: set = set()  # BN buffers de CNN (no promediar)
        self._no_avg_mlp_keys: set = set()  # BN buffers de MLP (no promediar)

        # Evita corrupción cuando múltiples workers actualizan
        self._params_lock = threading.Lock()

        # Contador global de versión
        self._version: int = 0
        self._t_start: float = 0.0  # Tiempo de inicio (para elapsed)
        self._t_start_initialized: bool = (
            False  # Flag para inicializar solo una vez en START
        )

        # CNN (para distribución inicial y evaluación)
        self._cnn: CNNExtractor | None = None

        # Workers
        self._sockets: Dict[int, socket.socket] = {}
        self._addrs: Dict[int, str] = {}
        self._worker_freeze: Dict[int, bool] = {}  # wid -> freeze_cnn
        self._worker_rank: Dict[
            int, int
        ] = {}  # wid -> rank (0-based, asignado dinámicamente)
        self._next_id: int = 0
        self._next_rank: int = 0  # Contador para asignar ranks secuenciales
        self._workers_lock = threading.Lock()

        # Métricas e historial
        self._metrics = RunningMetrics(window=metrics_window)
        self._history: Dict[str, List] = {
            "steps": [],
            "losses": [],
            "accuracies": [],
            "n_workers": [],
            "elapsed": [],
            "timestamps": [],
        }
        self._history_lock = threading.Lock()

        # Control
        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

        # Exportador de resultados (desacoplado, se inicializa posteriormente)
        self._results_exporter: ResultsExporter | None = None

        # Contadores para estadísticas
        self._nan_rejected: int = 0
        self._total_requests: int = 0

    @property
    def nan_rejected_count(self) -> int:
        """Cantidad de actualizaciones rechazadas por contener valores NaN.

        :returns: Numero de actualizaciones rechazadas.
        :rtype: int
        """
        return self._nan_rejected

    @property
    def tcp_request_count(self) -> int:
        """Cantidad total de requests TCP recibidos por el servidor.

        :returns: Numero total de requests TCP.
        :rtype: int
        """
        return self._total_requests

    # ================================================================
    # CONFIGURACIÓN
    # ================================================================

    def set_cnn(self, cnn: CNNExtractor) -> None:
        """
        Registra la CNN global y serializa su estado inicial en _cnn_state.

        Debe llamarse idealmente ANTES de listen() para que los Workers
        que conecten reciban los pesos sin espera.

        Los tensores int64 (num_batches_tracked de BatchNorm) se guardan
        en _no_avg_keys y se excluyen del averaging en _apply_update.

        :param cnn: Extractor CNN a registrar como modelo global.
        :type cnn: CNNExtractor

        :returns: None
        :rtype: None
        """
        self._cnn = cnn
        base = getattr(cnn._model, "model", cnn._model)  # Accede al modelo interno
        no_avg: set = set()
        cnn_state: Dict[str, np.ndarray] = {}

        # Recorremos los pesos del modelo
        for name, tensor in base.state_dict().items():
            arr = tensor.cpu().numpy().copy()
            # Excluir num_batches_tracked (int64): es un contador, no un parámetro
            if arr.dtype == np.int64 or tensor.dtype == torch.int64:
                no_avg.add(name)
            # Excluir running_mean y running_var de BatchNorm:
            # son estadísticas descriptivas locales de cada Worker.
            # Promediarlas con FedAvg introduce sesgo cuando cada Worker
            # tiene un shard diferente de datos (distribución local ≠ global).
            elif "running_mean" in name or "running_var" in name:
                no_avg.add(name)
            cnn_state[name] = arr  # Guardamos pesos

        with self._params_lock:
            self._cnn_state = cnn_state
            self._no_avg_keys = no_avg

        n_running = sum(1 for k in no_avg if "running" in k)
        n_tracked = sum(1 for k in no_avg if "tracked" in k)
        _log.ps(
            f"CNN lista: arch={cnn.arch} | feature_dim={cnn.feature_dim} | "
            f"params={len(cnn_state)} | no_avg={len(no_avg)} "
            f"(running={n_running}, tracked={n_tracked})"
        )

    def set_mlp(self, mlp_state: Dict[str, np.ndarray]) -> None:
        """
        Registra el estado inicial del MLP en formato PyTorch state_dict.

        Detecta keys de BatchNorm que NO deben promediarse (running_mean, running_var, num_batches_tracked).
        Claves esperadas: fc*.weight, fc*.bias, bn*.weight, bn*.bias, bn*.running_mean, bn*.running_var, bn*.num_batches_tracked.

        :param mlp_state: Estado del MLP (diccionario de pesos/sesgos/buffers en numpy).
        :type mlp_state: Dict[str, np.ndarray]
        :returns: None
        :rtype: None
        """
        # Detecta keys de BN que NO deben promediarse
        no_avg: set = set()
        for key, arr in mlp_state.items():
            if (
                "running_mean" in key
                or "running_var" in key
                or "num_batches_tracked" in key
            ):
                no_avg.add(key)
        with self._params_lock:
            self._mlp_state = {key: value.copy() for key, value in mlp_state.items()}
            self._no_avg_mlp_keys = no_avg
        _log.ps(f"MLP listo: {list(mlp_state.keys())}")
        if no_avg:
            _log.ps(f"MLP no_avg_keys: {sorted(no_avg)}")

    # ================================================================
    # CICLO DE VIDA
    # ================================================================

    def listen(self) -> None:
        """Abre el socket TCP y comienza a aceptar Workers en background.

        Crea un socket TCP, lo vincula al host y puerto configurados,
        e inicia un hilo daemon (_accept_loop) para aceptar conexiones entrantes.
        Tambien inicializa el ResultsExporter para registrar metricas.

        :returns: None
        :rtype: None

        :raises RuntimeError: Si el servidor ya esta escuchando.
        """
        if self._server_sock is not None:
            raise RuntimeError("El servidor ya está escuchando.")
        self._shutdown.clear()

        # Inicializa exportador de resultados
        config = {
            "lr": self.learning_rate,
            "lr_cnn": self.learning_rate_cnn,
            "staleness_lambda": self.staleness_lambda,
            "metrics_window": self.metrics_window,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "host": self.host,
            "port": self.port,
            "cnn_arch": getattr(
                self._cnn, "arch", "unknown"
            ),  # Obtiene arquitectura CNN
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "steps_per_report": self.steps_per_report,
            "description": "Distributed Async-SGD on ImageNet-1k",
        }
        self._results_exporter = ResultsExporter(
            config=config, export_dir=self.export_dir
        )

        # Registra handler de logs en el logger global
        _log.add_log_handler(self._results_exporter.record_log)

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
        """Envia STOP a todos los Workers y cierra el servidor.

        Senala el shutdown, desconecta todos los workers activos,
        cierra el socket listener y espera a que el hilo de aceptacion termine.

        :returns: None
        :rtype: None
        """
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
                pass  # noqa: S110 (cleanup code, socket may already be closed)
            self._server_sock = None
        if self._accept_thread:
            self._accept_thread.join(timeout=3)

        # Finaliza exportador de resultados
        if self._results_exporter is not None:
            # Desregistra handler de logs
            _log.remove_log_handler(self._results_exporter.record_log)
            self._results_exporter.tcp_request_count = self._total_requests
            self._results_exporter.nan_rejected_count = self._nan_rejected
            # Finaliza y exporta
            export_path = self._results_exporter.finalize()
            _log.ps(f"Resultados exportados a: {export_path}")

        _log.ps("Servidor detenido.")

    @property
    def connected_workers(self) -> List[int]:
        """
        Retorna la lista ordenada de IDs de Workers conectados.

        :returns: Lista de IDs de workers activos.
        :rtype: List[int]
        """
        with self._workers_lock:
            return sorted(self._sockets.keys())

    @property
    def current_version(self) -> int:
        """
        Retorna el número de versión actual del modelo global.

        El versionado se incrementa con cada actualización aplicada.
        Útil para monitorear el progreso de sincronización.

        :returns: Número de versión actual.
        :rtype: int
        """
        with self._params_lock:
            return self._version

    @property
    def history(self) -> Dict[str, List]:
        """
        Retorna copia del historial de métricas del entrenamiento.

        Incluye: steps, losses, accuracies, n_workers, elapsed, timestamps.

        :returns: Diccionario con listas de histórico.
        :rtype: Dict[str, List]
        """
        with self._history_lock:
            return {k: list(v) for k, v in self._history.items()}

    # ================================================================
    # ACEPTACIÓN
    # ================================================================

    def _accept_loop(self) -> None:
        """Bucle de aceptacion de conexiones entrantes (hilo dedicado).

        Acepta conexiones TCP entrantes en un bucle hasta que se
        senale el shutdown. Cada conexion se delega a un hilo dedicado
        que ejecuta _handle_new_connection.

        :returns: None
        :rtype: None
        """
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
        """Handshake completo para un Worker recien conectado.

        Secuencia:

        1. Recibir READY
        2. Enviar WORKER_ID
        3. Enviar CONFIG (batch_size, image_size, dataset, seed, rank, etc.)
        4. Enviar CNN_WEIGHTS (siempre, nunca opcional)
        5. Esperar CNN_ACK con verificacion de arquitectura
        6. Enviar START
        7. Iniciar loop _serve_worker para atender mensajes

        :param conn: Socket de conexion entrante desde el Worker.
        :type conn: socket.socket

        :param addr: Tupla (IP, puerto) del cliente.
        :type addr: tuple

        :returns: None. El metodo gestiona el handshake y lanza el thread de servicio.
        :rtype: None

        .. note::
            Timeout implicito: si CNN+MLP no estan listo en 120s, el Worker
            puede timeout waiting for weights.
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
            rank = self._next_rank
            self._next_rank += 1
            self._sockets[wid] = conn
            self._addrs[wid] = f"{addr[0]}:{addr[1]}"
            self._worker_freeze[wid] = False
            self._worker_rank[wid] = rank

        try:
            send_message(conn, MsgType.WORKER_ID, {"worker_id": wid})
        except Exception:
            with self._workers_lock:
                self._sockets.pop(wid, None)
                self._addrs.pop(wid, None)
                self._worker_freeze.pop(wid, None)
            conn.close()
            return

        # Enviar CONFIG con parámetros de streaming
        try:
            config = {
                "batch_size": self.batch_size,
                "image_size": self.image_size,
                "dataset_name": self.dataset_name,
                "seed": self.seed,
                "rank": rank,
                "num_workers": self._next_rank,  # Total de workers conectados (incluyendo el actual)
                "hf_token": self.hf_token,  # Token HF para streaming
            }
            send_message(conn, MsgType.CONFIG, config)
        except Exception:
            with self._workers_lock:
                self._sockets.pop(wid, None)
                self._addrs.pop(wid, None)
                self._worker_freeze.pop(wid, None)
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
        _log.ps(f"Worker {wid}: esperando modelo (máx {int(deadline)} s)...")

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
                _log.error(f"Worker {wid}: timeout esperando CNN+MLP.")
                self._remove_worker(wid)
                conn.close()
                return
            if self._shutdown.is_set():
                self._remove_worker(wid)
                conn.close()
                return
            time.sleep(poll)  # Espera poll segundos
            waited += poll

        if self._cnn is None:
            raise RuntimeError(
                "El modelo de CNN no se ha cargado antes de enviar los pesos"
            )

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

            ack_pl = ack.get("payload") or {}
            if isinstance(ack_pl, dict):
                ack_arch = ack_pl.get("arch")
                freeze_flag = bool(ack_pl.get("freeze_cnn", False))
                if ack_arch and ack_arch != arch:
                    _log.error(
                        f"Worker {wid}: mismatch arch PS={arch}, Worker={ack_arch}"
                    )
                    self._remove_worker(wid)
                    return
                with self._workers_lock:
                    self._worker_freeze[wid] = freeze_flag
                mode = "freeze" if freeze_flag else "E2E"
                _log.ps(f"Worker {wid}: CNN cargada ✓ arch={arch} mode={mode}")
            else:
                _log.ps(f"Worker {wid}: CNN cargada ✓ arch={arch}")

        except Exception as e:
            _log.error(f"Error distribuyendo CNN a Worker {wid}: {e}")
            self._remove_worker(wid)
            return

        # ----------------- 4. START -----------------
        try:
            send_message(conn, MsgType.START, {})

            # Inicia temporizador (punto de inicio real del entrenamiento del worker)
            with self._params_lock:
                if not self._t_start_initialized:
                    self._t_start = time.perf_counter()
                    self._t_start_initialized = True

            # Notifica a la GUI que el primer worker ha recibido START
            if self.on_start_sent:
                self.on_start_sent(wid)

            if self._results_exporter is not None:
                with self._workers_lock:
                    worker_addr = self._addrs.get(wid, f"{addr[0]}:{addr[1]}")
                self._results_exporter.record_worker_event(
                    step=self.current_version,
                    event_type="connected",
                    worker_id=wid,
                    worker_addr=worker_addr,
                )
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

        Envía PARAMS con dos LRs:
          - lr:     LR del MLP (siempre)
          - lr_cnn: LR de la CNN (solo relevante en modo E2E; ignorado en freeze)

        La CNN state solo se envía en modo E2E (optimización de bandwidth).

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
                        self._total_requests += 1
                        mlp_copy = {
                            key: value.copy() for key, value in self._mlp_state.items()
                        }
                        ver = self._version
                        with self._workers_lock:
                            is_freeze = self._worker_freeze.get(wid, False)
                        cnn_copy = (
                            {}
                            if is_freeze
                            else {k: v.copy() for k, v in self._cnn_state.items()}
                        )

                    try:
                        send_message(
                            conn,
                            MsgType.PARAMS,
                            {
                                "mlp_state": mlp_copy,
                                "cnn_state": cnn_copy,
                                "version": ver,
                                "lr": self.learning_rate,  # LR del MLP
                                "lr_cnn": self.learning_rate_cnn,  # LR de la CNN
                            },
                        )
                    except Exception as e:
                        _log.error(f"Error enviando PARAMS a Worker {wid}: {e}")
                        break

                elif mtype == MsgType.UPDATES:
                    with self._params_lock:
                        self._total_requests += 1
                    self._apply_update(msg["payload"])

        finally:
            _log.ps(f"Worker {wid} desconectado.")
            self._remove_worker(wid)

    # ================================================================
    # ACTUALIZACIÓN ASÍNCRONA
    # ================================================================

    def _apply_update(self, payload: dict) -> None:
        """
        Aplica los pesos actualizados con corrección de staleness.

            θ_new = θ + α(s) · (θ_worker − θ)
            α(s) = 1 / (1 + λ · s)
            s = versión_actual − versión_leída

        Los tensores marcados en _no_avg_keys (num_batches_tracked de BN)
        no se promedian: mantienen el valor del PS. Promediar un contador
        de batches no tiene significado semántico y puede distorsionar
        el comportamiento de BatchNorm en eval().

        VALIDACIÓN DE NaN: Si loss o accuracy contienen NaN, se rechaza la
        actualización y se registra un error. Esto previene que pesos corruptos
        se promedien en el modelo global.

        :param payload: Diccionario con actualización del Worker: loss, accuracy, version_read, mlp_weights, cnn_weights.
        :type payload: dict

        :returns: None
        :rtype: None
        """
        loss = payload.get("loss", 0.0)
        acc = payload.get("accuracy", 0.0)
        version_read = payload.get("version_read", 0)
        mlp_weights = payload.get("mlp_weights")
        cnn_weights = payload.get("cnn_weights")

        # VALIDACIÓN: Rechaza actualizaciones con NaN
        if np.isnan(loss) or np.isnan(acc):
            self._nan_rejected += 1
            _log.error(
                f"RECHAZADA actualización con NaN: loss={loss}, acc={acc}. "
                f"Posible inestabilidad numérica en Worker. "
                f"Considera aumentar staleness_lambda o reducir learning_rate."
            )
            return

        with self._params_lock:
            # Calcula factor de corrección Alpha
            staleness = max(0, self._version - version_read)
            alpha = 1.0 / (1.0 + self.staleness_lambda * staleness)

            if mlp_weights:
                for key in mlp_weights:
                    if key in self._no_avg_mlp_keys or key not in self._mlp_state:
                        continue  # running_mean/var y num_batches_tracked: no promediar
                    self._mlp_state[key] += alpha * (
                        mlp_weights[key] - self._mlp_state[key]
                    )

            if cnn_weights:
                for key in cnn_weights:
                    if key in self._no_avg_keys or key not in self._cnn_state:
                        continue
                    arr = self._cnn_state[key]  # float32 ya en el tipo correcto
                    inc = cnn_weights[key]
                    # OPTIMIZACIÓN: FedAvg en float32 directamente.
                    # La diferencia numérica vs float64 es ~1e-7 relativa,
                    # negligible para señales de gradiente (~1e-3 a 1e-5).
                    # arr += alpha*(inc-arr) es in-place: evita asignación de nuevo array.
                    # Versión anterior: astype(float64) → aritmética → astype(float32)
                    # → 2 copias extra + aritmética 2× más lenta en numpy.
                    if inc.dtype != arr.dtype:
                        inc = inc.astype(arr.dtype)
                    arr += alpha * (inc - arr)

            self._version += 1
            step = self._version

        # Tiempo de entrenamiento desde que se envió START al primer worker
        elapsed = time.perf_counter() - self._t_start if self._t_start != 0.0 else 0.0

        staleness = max(0, self._version - version_read)
        alpha = 1.0 / (1.0 + self.staleness_lambda * staleness)

        self._metrics.update(loss, acc)
        n = self._metrics.total_batches
        if n > 0:
            avg_loss, avg_acc, std_loss, std_acc = self._metrics.snapshot()
        else:
            avg_loss, avg_acc, std_loss, std_acc = loss, acc, 0.0, 0.0

        if self.on_step:
            self.on_step(step, loss, acc, staleness, elapsed)

        if self._results_exporter is not None:
            with self._workers_lock:
                n_workers = len(self._sockets)
            self._results_exporter.record_metric(
                step,
                avg_loss,
                avg_acc,
                n_workers,
                elapsed,
                loss_std=std_loss,
                acc_std=std_acc,
                staleness=staleness,
                alpha=alpha,
            )

        if n > 0 and n % self.steps_per_report == 0:
            with self._workers_lock:
                n_workers = len(self._sockets)
            self._record_history(step, avg_loss, avg_acc, n_workers, elapsed)
            if self.on_report:
                self.on_report(step, avg_loss, avg_acc, elapsed)
            _log.train(
                f"Step {step:,}",
                metric=f"loss={avg_loss:.4f} | acc={avg_acc:.2f}% | workers={n_workers} | {elapsed:.0f}s",
            )

    def _record_history(self, step, loss, acc, n_workers, elapsed) -> None:
        """Registra un punto de metrica en el historial completo de entrenamiento.

        Almacena step, loss, accuracy, numero de workers y timestamp
        en listas thread-safe para posterior exportacion.

        :param step: Numero de paso actual.
        :type step: int

        :param loss: Valor de loss del paso.
        :type loss: float

        :param acc: Valor de accuracy del paso.
        :type acc: float

        :param n_workers: Numero de workers activos.
        :type n_workers: int

        :param elapsed: Tiempo transcurrido desde inicio.
        :type elapsed: float

        :returns: None
        :rtype: None
        """
        with self._history_lock:
            self._history["steps"].append(step)
            self._history["losses"].append(loss)
            self._history["accuracies"].append(acc)
            self._history["n_workers"].append(n_workers)
            self._history["elapsed"].append(elapsed)
            self._history["timestamps"].append(time.time())

    # ================================================================
    # EVALUACIÓN
    # ================================================================

    def evaluate(
        self,
        dataset_name: str = "ILSVRC/imagenet-1k",
        max_batches: int = 50,
        batch_size: int = 256,
        hf_token: str | None = None,
    ) -> Tuple[float, float]:
        """
        Evaluación rápida del modelo global en el split de validación de ImageNet.

        Proceso:
            1. Copia pesos actuales del estado global
            2. Carga la CNN con esos pesos
            3. Reconstruye el MLP
            4. Procesa batches de validación
            5. Calcula accuracy y loss promedio

        :param dataset_name: Nombre del dataset en HuggingFace Hub.
        :type dataset_name: str

        :param max_batches: Numero de batches a evaluar (50 x 256 = 12,800 imagenes por defecto).
        :type max_batches: int

        :param batch_size: Imagenes por batch de evaluacion.
        :type batch_size: int

        :param hf_token: Token de autenticacion HuggingFace.
        :type hf_token: str | None

        :returns: Tupla (accuracy_percentaje, mean_loss)
        :rtype: Tuple[float, float]
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
        """
        Obtiene copia thread-safe del estado actual del modelo global.

        Devuelve una tupla con copias de los pesos del MLP, CNN y la versión
        actual. Las copias evitan carreras de datos cuando múltiples threads
        acceden simultáneamente.

        :returns: Tupla (mlp_state, cnn_state, version) donde mlp_state y cnn_state son Dict[str, np.ndarray] y version es int.
        :rtype: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]
        """
        with self._params_lock:
            return (
                {key: value.copy() for key, value in self._mlp_state.items()},
                {key: value.copy() for key, value in self._cnn_state.items()},
                self._version,
            )

    def _disconnect_worker(self, wid: int) -> None:
        """
        Desconecta un Worker enviándole mensaje STOP y cerrando el socket.

        Remueve el Worker de las estructuras internas (_sockets, _addrs, _worker_freeze, _worker_rank)
        de forma thread-safe. Intenta enviar STOP antes de cerrar; ignora errores
        de red (Worker puede haber desconectado ya).

        :param wid: Id único del Worker a desconectar.
        :type wid: int

        :returns: None
        :rtype: None
        """
        with self._workers_lock:
            sock = self._sockets.pop(wid, None)
            self._addrs.pop(wid, None)
            self._worker_freeze.pop(wid, None)
            self._worker_rank.pop(wid, None)
        if sock:
            try:
                send_message(sock, MsgType.STOP, None)
                sock.close()
            except Exception:
                pass  # noqa: S110 (cleanup code, worker may be disconnected)

    def _remove_worker(self, wid: int) -> None:
        """
        Remueve un Worker sin comunicación activa y dispara callback on_worker_disconnected.

        Se diferencia de _disconnect_worker en que NO envía STOP (asume que el
        Worker ya desconectó) y SÍ dispara el callback on_worker_disconnected para
        notificar a la aplicación que un Worker se fue. Se usa en el finally de
        _serve_worker cuando la conexión se cae.

        :param wid: Id único del Worker a remover.
        :type wid: int

        :returns: None
        :rtype: None
        """
        with self._workers_lock:
            worker_addr = self._addrs.get(wid, "")
            sock = self._sockets.pop(wid, None)
            self._addrs.pop(wid, None)
            self._worker_freeze.pop(wid, None)
        if sock:
            try:
                sock.close()
            except Exception:
                pass  # noqa: S110 (cleanup code, socket may already be closed)
        if self._results_exporter is not None:
            self._results_exporter.record_worker_event(
                step=self.current_version,
                event_type="disconnected",
                worker_id=wid,
                worker_addr=worker_addr,
            )
        if self.on_worker_disconnected:
            self.on_worker_disconnected(wid)
