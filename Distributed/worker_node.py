"""
Distributed/worker_node.py

Worker asíncrono para entrenamiento distribuido en ImageNet.

MODOS DE ENTRENAMIENTO (determinados automáticamente por arquitectura CNN):

  "resnet18" (freeze_cnn=True):
    - CNN congelada desde CNNExtractor.__init__ (requires_grad=False permanente)
    - forward con torch.no_grad() → features sin grad_fn → 0 overhead de memoria
    - Solo el MLP recibe gradientes y se actualiza con lr (del PS)
    - cnn_weights=None en UPDATES → PS no toca la CNN global
    - lr_cnn viaja en PARAMS pero se ignora (CNN no recibe gradientes)

  "simple" (freeze_cnn=False):
    - CNN entrenable desde CNNExtractor.__init__ (requires_grad=True permanente)
    - forward en train() → features con grad_fn → backward fluye por toda la red
    - CNN se actualiza con lr_cnn (del PS), MLP con lr (del PS)
    - gradient clipping conjunto (CNN + MLP) antes del SGD step
    - cnn_weights enviado al PS → FedAvg sobre CNN global

OPTIMIZADOR: SGD puro con gradient clipping.
  No se usa Adam porque sus momentos (m, v) son locales al Worker y se
  dessincronizan cuando el PS hace FedAvg con 2+ workers:
    - El Worker calcula gradientes sobre los pesos del PS
    - Adam acumula momentos sobre la trayectoria LOCAL del Worker
    - El PS devuelve pesos promediados (FedAvg) que pueden diferir mucho
    - Los momentos de Adam ya no corresponden al nuevo punto de partida
  SGD sin estado extra garantiza que cada paso sea correcto respecto
  a los pesos actuales del PS.

LRs SEPARADOS:
  El PS envía dos valores en cada PARAMS:
    lr     → LR del MLP
    lr_cnn → LR de la CNN (solo usado en modo simple/E2E)

GRADIENT CLIPPING:
  En modo E2E, el grad norm conjunto (CNN+MLP) puede ser alto porque la
  CNN parte de pesos aleatorios. clip_grad_norm_(all_params, max_norm=1.0)
  estabiliza el entrenamiento sin necesidad de un LR muy pequeño.
"""

import socket
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from Distributed.protocol import MsgType, receive_message, send_message
from Model.cnn_extractor import CNNExtractor
from Model.mlp_pytorch import MLPPyTorch
from Utils.imagenet_streaming import PrefetchBuffer, build_worker_stream
from Utils.logging_util import get_logger

_log = get_logger(use_colors=True)

_IMAGENET_CLASSES = 1000
_GRAD_CLIP_MAX_NORM = 1.0  # Umbral de gradient clipping (E2E)


class WorkerNode:
    """
    Nodo Worker asíncrono para entrenamiento distribuido en ImageNet-1k.

    Implementa el cliente de un sistema de entrenamiento distribuido que se conecta
    a un Parameter Server centralizado. Soporta dos modos de entrenamiento determinados
    automáticamente por la arquitectura CNN recibida del PS:

    - **ResNet-18 (MLP-only)**: CNN congelada permanentemente (requires_grad=False).
      Solo MLP se entrena.

    - **SimpleCNN (E2E)**: CNN entrenable (requires_grad=True). Ambas redes se entrenan.

    El Worker ejecuta un loop infinito: solicita parámetros globales del PS,
    entrena localmente durante accum_steps batches, y envía las actualizaciones
    locales de vuelta al PS que aplica Async-FedAvg.

    :param server_host: Dirección IP o nombre de host del Parameter Server.
    :type server_host: str

    :param server_port: Puerto TCP en el que escucha el Parameter Server.
    :type server_port: int

    :param dataset_name: Nombre del dataset en HuggingFace Hub para streaming.
    :type dataset_name: str

    :param worker_rank: Índice único de este Worker (0-based), usado para sharding del dataset.
    :type worker_rank: int

    :param num_workers: Cantidad total de Workers en el entrenamiento (necesario para sharding).
    :type num_workers: int

    :param device: Dispositivo PyTorch donde ejecutar ('cpu', 'cuda', 'cuda:0', 'mps').
    :type device: str

    :param shuffle_buffer: Número de imágenes en el buffer de shuffle local.
    :type shuffle_buffer: int

    :param prefetch_batches: Número de batches a pre-cargar en background.
    :type prefetch_batches: int

    :param seed: Semilla para RNG (None = determinismo deshabilitado, aleatorio).
    :type seed: Optional[int]

    :param hf_token: Token de autenticación de HuggingFace (requerido para ILSVRC/imagenet-1k).
    :type hf_token: Optional[str]

    :param accum_steps: Número de batches a acumular antes de enviar UPDATES al PS.
    :type accum_steps: int

    :param verbose: Imprimir logs de progreso cada 10 batches.
    :type verbose: bool

    :note: batch_size e image_size se reciben del PS mediante mensaje CONFIG durante conexión. lr y lr_cnn se reciben del PS mediante PARAMS en cada iteración.

    :raises ConnectionError: Si falla la conexión inicial con el Parameter Server.
    :raises RuntimeError: Si hay mismatch de parámetros con la CNN recibida del PS.
    """

    def __init__(
        self,
        server_host: str,
        server_port: int,
        dataset_name: str = "ILSVRC/imagenet-1k",
        worker_rank: int = 0,
        num_workers: int = 1,
        device: str = "cpu",
        shuffle_buffer: int = 1000,
        prefetch_batches: int = 4,
        seed: Optional[int] = None,
        hf_token: Optional[str] = None,
        accum_steps: int = 1,
        verbose: bool = True,
    ) -> None:
        # Configuración de red
        self.server_host = server_host
        self.server_port = server_port

        # Dataset a utilizar
        self.dataset_name = dataset_name
        self.hf_token = hf_token

        # Identificadores para paralelismo
        self.worker_rank = worker_rank
        self.num_workers = num_workers

        # Buffer y prefetch
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_batches = prefetch_batches

        # Otros ajustes necesarios
        self.seed = seed
        self.device = torch.device(device)
        self.accum_steps = accum_steps
        self.verbose = verbose

        # Recibidos via CONFIG del PS
        self.batch_size: Optional[int] = None
        self.image_size: Optional[int] = None

        # Estado interno del worker
        self._worker_id: Optional[int] = None
        self._sock: Optional[socket.socket] = None
        self._cnn: Optional[CNNExtractor] = None
        self._mlp: Optional[MLPPyTorch] = None
        self._stream: Optional[PrefetchBuffer] = None
        self._batches_done = 0

        # Determinado en _load_cnn() a partir del arch recibido del PS.
        # Refleja directamente el estado de requires_grad en CNNExtractor:
        #   resnet18 → True  (CNN siempre congelada)
        #   simple   → False (CNN siempre entrenable)
        self._freeze_cnn: bool = True

    # ================================================================
    # PUNTO DE ENTRADA
    # ================================================================

    def run(self) -> None:
        """
        Punto de entrada principal del Worker. Ejecuta la secuencia completa de inicio.

        Realiza las siguientes operaciones en orden:

        1. Conecta al Parameter Server y recibe configuración (batch_size, image_size).
        2. Inicializa el pipeline de streaming de ImageNet-1k.
        3. Realiza handshake con el PS para recibir estados iniciales de CNN y MLP.
        4. Ejecuta el loop infinito de entrenamiento asincrónico.

        Garantiza liberación de recursos (socket, stream) en caso de excepción
        mediante bloque finally.

        :returns: None
        :rtype: None

        :raises ConnectionError: Si falla conexión TCP con Parameter Server.
        :raises RuntimeError: Si hay inconsistencias en mensajes o estados recibidos.
        """
        self._connect()
        _log.worker_msg(
            self._worker_id,
            f"Conectado | rank={self.worker_rank}/{self.num_workers} | "
            f"device={self.device} | batch={self.batch_size} | "
            f"accum={self.accum_steps}",
        )
        self._init_stream()
        try:
            self._handshake_loop()
        finally:
            self._cleanup()

    # ================================================================
    # CONEXIÓN
    # ================================================================

    def _connect(self) -> None:
        """
        Establece conexión TCP con el Parameter Server e intercambia configuración inicial.

        Ejecuta la secuencia de handshake de conexión:

        1. Crea socket TCP y conecta a ``server_host:server_port``.
        2. Envía mensaje READY indicando que el Worker está listo.
        3. Recibe mensaje WORKER_ID asignado por el PS.
        4. Recibe mensaje CONFIG con batch_size e image_size configurados globalmente.

        Modifica atributos internos: ``self._sock``, ``self._worker_id``,
        ``self.batch_size``, ``self.image_size``.

        :returns: None
        :rtype: None

        :raises ConnectionError: Si falla conexión TCP, timeout, o se reciben mensajes
                                  fuera de orden que no coinciden con READY/WORKER_ID/CONFIG.
        :raises socket.error: Si error de socket subyacente.
        """
        # AF_INET -> IPv4
        # SOCK_STREAM -> TCP
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.server_host, self.server_port))
        send_message(self._sock, MsgType.READY, {})

        # Recibe WORKER_ID
        msg = receive_message(self._sock)
        if msg["type"] != MsgType.WORKER_ID:
            raise ConnectionError(f"Esperaba WORKER_ID, recibí {msg['type']}")
        self._worker_id = msg["payload"]["worker_id"]

        # Recibe CONFIG (batch_size, image_size desde PS)
        msg = receive_message(self._sock)
        if msg["type"] != MsgType.CONFIG:
            raise ConnectionError(f"Esperaba CONFIG, recibí {msg['type']}")
        config = msg["payload"]
        self.batch_size = config["batch_size"]
        self.image_size = config["image_size"]
        _log.worker_msg(
            self._worker_id,
            f"CONFIG recibida: batch_size={self.batch_size}, image_size={self.image_size}",
        )

    def _cleanup(self) -> None:
        """
        Libera recursos del Worker (stream de datos y socket TCP).

        Cierra el stream de datos del dataset si está activo, luego cierra
        la conexión TCP. Atrapa excepciones para asegurar que ambas operaciones
        se ejecuten aunque una falle.

        :returns: None
        :rtype: None

        :note: Se ejecuta automáticamente en bloque finally de run().
        """
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        _log.worker_msg(self._worker_id, "Recursos liberados.")

    # ================================================================
    # STREAM DE DATOS
    # ================================================================

    def _init_stream(self) -> None:
        """
        Construye e inicia el pipeline de streaming de ImageNet-1k con prefetching asíncrono.

        Crea un buffer de prefetching que descarga batches de imágenes en hilo
        background mientras el Worker entrena. El streaming respeta el sharding automático:
        Worker k obtiene imágenes con índices k, k+num_workers, k+2*num_workers, ...
        evitando solapamiento entre Workers.

        Precondición: ``self.batch_size`` y ``self.image_size`` deben haber sido
        configurados anteriormente mediante mensaje CONFIG en _connect().

        :returns: None
        :rtype: None

        :raises AssertionError: Si batch_size o image_size no han sido configurados.
        :raises Exception: Si falla descarga inicial o construcción del stream.
        """

        # Garantiza que el batch_size e image_size fueron recibidos en CONFIG
        assert self.batch_size is not None, "batch_size debe ser configurado por CONFIG"
        assert self.image_size is not None, "image_size debe ser configurado por CONFIG"

        self._stream = build_worker_stream(
            worker_rank=self.worker_rank,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            dataset_name=self.dataset_name,
            image_size=self.image_size,
            shuffle_buffer=self.shuffle_buffer,
            prefetch_batches=self.prefetch_batches,
            seed=(self.seed + self.worker_rank) if self.seed is not None else None,
            hf_token=self.hf_token,
        )
        self._stream.start()
        _log.worker_msg(
            self._worker_id,
            f"Stream iniciado: {self.dataset_name} | "
            f"shard {self.worker_rank}/{self.num_workers}",
        )

    # ================================================================
    # HANDSHAKE
    # ================================================================

    def _handshake_loop(self) -> None:
        """
        Realiza handshake de entrenamiento con el Parameter Server.

        Espera secuencialmente:

        1. Mensaje CNN_WEIGHTS: Recibe la arquitectura CNN (ResNet-18 o SimpleCNN)
           y sus pesos iniciales. Llama a _load_cnn() para instanciar y sincronizar.
        2. Mensaje START: Señal del PS indicando que todos los Workers están listos.
           Dispara transición a loop de entrenamiento infinito.
        3. Mensaje STOP: Detiene el Worker de forma ordenada (abandona handshake).

        El PS solo envía CNN_WEIGHTS después de que se ha configurado la CNN
        globalmente, por lo que no hay race conditions desde el lado Worker.

        :returns: None (retorna cuando recibe START o STOP)
        :rtype: None

        :raises RuntimeError: Si CNN_WEIGHTS tiene arquitectura desconocida.
        """
        assert self._sock is not None

        while True:
            msg = receive_message(self._sock)
            type_msg = msg["type"]

            if type_msg == MsgType.STOP:
                _log.worker_msg(self._worker_id, "STOP recibido durante handshake.")
                return
            elif type_msg == MsgType.CNN_WEIGHTS:
                self._load_cnn(msg["payload"])
            elif type_msg == MsgType.START:
                _log.worker_msg(
                    self._worker_id, "START recibido — iniciando loop de entrenamiento."
                )
                self._training_loop()
                return

    def _load_cnn(self, payload: dict) -> None:
        """
        Recibe, instancia y sincroniza la CNN desde el Parameter Server.

        Procesa el payload CNN_WEIGHTS que contiene:

        - ``payload['arch']``: Arquitectura CNN ('resnet18' o 'simple')
        - ``payload['weights_bytes']``: Estado serializado de la CNN
        - ``payload['mlp_keys']``: Claves esperadas del MLP para validación
        - ``payload['cnn_key_count']``: Número de parámetros de CNN para validación

        Instancia CNNExtractor con la arquitectura correcta. Esta instanciación
        establece requires_grad de forma permanente:

        - `arch='resnet18'`: requires_grad=False (CNN congelada, MLP-only)
        - `arch='simple'`: requires_grad=True (CNN entrenable, E2E)

        El estado requires_grad se preserva a través de load_state_dict(),
        así que sincronizaciones posteriores no lo modifican.

        Valida que números de parámetros y claves MLP coincidan con lo esperado.
        Envía CNN_ACK al PS confirmando carga exitosa.

        :param payload: Diccionario con 'arch', 'weights_bytes', 'mlp_keys', 'cnn_key_count'
        :type payload: dict

        :returns: None
        :rtype: None

        :raises RuntimeError: Si arquitectura es inválida, o hay mismatch de parámetros
                              entre CNN recibida y localmente instantiada.
        """
        arch = payload["arch"]
        weights_bytes = payload["weights_bytes"]
        mlp_keys = payload.get("mlp_keys", [])
        cnn_key_count = payload.get("cnn_key_count", 0)

        if arch not in ("simple", "resnet18"):
            raise RuntimeError(
                f"[W{self._worker_id}] Arquitectura desconocida del PS: '{arch}'. "
                "Valores válidos: 'simple', 'resnet18'."
            )

        # freeze_cnn refleja el estado permanente de requires_grad en CNNExtractor
        self._freeze_cnn = arch == "resnet18"

        # Solo crea la CNN si no existe aún, o hubo un cambio de arquitectura
        if self._cnn is None or self._cnn.arch != arch:
            _log.worker_msg(
                self._worker_id, f"Instanciando CNN arch={arch} en {self.device}"
            )
            # CNNExtractor.__init__ establece requires_grad según arch:
            #   resnet18 → False (congelada), simple → True (entrenable)
            self._cnn = CNNExtractor(arch=arch, device=str(self.device), seed=self.seed)

        # load_weights_from_bytes -> load_state_dict -> NO modifica requires_grad
        self._cnn.load_weights_from_bytes(weights_bytes)

        # Verifica que el número de parámetros coincide
        actual_keys = len(self._cnn._model.state_dict())

        # Verifica que el modelo del servidor coincida con el del worker
        if cnn_key_count > 0 and actual_keys != cnn_key_count:
            raise RuntimeError(
                f"[W{self._worker_id}] Mismatch CNN params: "
                f"PS={cnn_key_count}, Worker={actual_keys}"
            )

        # Verifica que las claves MLP son las esperadas
        expected_mlp = {
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
            "fc3.weight",
            "fc3.bias",
        }
        if mlp_keys and set(mlp_keys) != expected_mlp:
            raise RuntimeError(
                f"[W{self._worker_id}] MLP keys inesperadas del PS: {set(mlp_keys)}."
            )

        mode = "freeze (solo MLP)" if self._freeze_cnn else "E2E (CNN + MLP)"
        _log.worker_msg(
            self._worker_id,
            f"CNN cargada ✓ arch={arch} | feature_dim={self._cnn.feature_dim} | "
            f"params={actual_keys} | modo={mode}",
        )

        assert self._sock is not None
        send_message(
            self._sock,
            MsgType.CNN_ACK,
            {
                "worker_id": self._worker_id,
                "arch": arch,
                "feature_dim": self._cnn.feature_dim,
                "freeze_cnn": self._freeze_cnn,
            },
        )

    # ================================================================
    # LOOP DE ENTRENAMIENTO ASÍNCRONO
    # ================================================================

    def _training_loop(self) -> None:
        """
        Loop infinito de entrenamiento asincrónico sin sincronización con otros Workers.

        Ejecuta iterativamente la secuencia:

        1. **REQUEST_PARAMS**: Solicita parámetros globales (CNN + MLP) al PS.
        2. **Sincronización**: Descarga y carga los parámetros recibidos en modelos locales.
        3. **Entrenamiento**: Entrena ``accum_steps`` batches usando SGD local:
           - En modo ResNet-18 (congelado): Solo MLP recibe gradientes.
           - En modo SimpleCNN (E2E): CNN y MLP reciben gradientes.
        4. **UPDATES**: Envía parámetros entrenados localmente de vuelta al PS.
           - CNN=None si congelada, CNN=state_dict si entrenable.
           - MLP siempre se envía.

        El PS aplica Async-FedAvg a las actualizaciones recibidas con corrección
        de staleness según la versión de parámetros que el Worker leyó.

        Precondiciones (garantizadas por _handshake_loop()):
        - CNN fue cargada, verificada e instanciada.
        - Stream de datos fue inicializado.
        - Socket TCP está activo.

        :returns: None (retorna si recibe STOP o error de comunicación)
        :rtype: None

        :raises RuntimeError: Si CNN no fue inicializada antes de entrar al loop.
        """
        if self._cnn is None:
            raise RuntimeError(
                f"[W{self._worker_id}] _training_loop sin CNN inicializada."
            )

        assert self._sock is not None
        assert self._cnn is not None
        assert self._stream is not None

        _log.worker_msg(
            self._worker_id,
            f"Entrenamiento | arch={self._cnn.arch} | "
            f"feature_dim={self._cnn.feature_dim} | device={self.device} | "
            f"freeze_cnn={self._freeze_cnn}",
        )

        stream_iter = self._stream.__iter__()
        version_read = 0
        first_params_logged = False

        while True:
            # ----------------- 1. Pedir parámetros globales -----------------
            try:
                send_message(self._sock, MsgType.REQUEST_PARAMS, {})
                msg = receive_message(self._sock)
            except Exception as e:
                _log.worker_msg(self._worker_id, f"Error de comunicación: {e}")
                return

            if msg["type"] == MsgType.STOP:
                return
            if msg["type"] != MsgType.PARAMS:
                _log.worker_msg(self._worker_id, f"Mensaje inesperado: {msg['type']}")
                continue

            payload = msg["payload"]
            mlp_state = payload["mlp_state"]
            cnn_state = payload["cnn_state"]
            version_read = payload["version"]
            lr = payload["lr"]  # LR del MLP
            lr_cnn = payload.get("lr_cnn", lr * 0.1)  # LR de la CNN

            if not mlp_state:
                raise RuntimeError(f"[W{self._worker_id}] PS devolvió mlp_state vacío.")
            if not cnn_state and not self._freeze_cnn:
                raise RuntimeError(
                    f"[W{self._worker_id}] PS devolvió cnn_state vacío en modo E2E."
                )

            if not first_params_logged:
                fc1_shape = mlp_state.get("fc1.weight", np.array([])).shape
                _log.worker_msg(
                    self._worker_id,
                    f"Primer PARAMS — v={version_read} | lr={lr} | "
                    f"mlp fc1.weight={fc1_shape} | cnn_params={len(cnn_state)}",
                )
                first_params_logged = True

            # ----------------- 2. Sincronizar estado global del PS -----------------

            # _sync_cnn usa load_state_dict -> no modifica requires_grad
            # El estado correcto (False para resnet18, True para simple)
            # se mantiene intacto tras cada sincronización.
            self._sync_cnn(cnn_state)
            self._mlp = self._sync_mlp(mlp_state, self._mlp)

            # ----------------- 3. Entrenar accum_steps batches -----------------
            total_loss, total_acc, total_n = 0.0, 0.0, 0

            for _ in range(self.accum_steps):
                try:
                    X_np, Y_np = next(stream_iter)  # Consigue batch
                except StopIteration:
                    stream_iter = self._stream.__iter__()  # Reinicia el stream
                    X_np, Y_np = next(stream_iter)

                loss, acc, n = self._train_batch(X_np, Y_np, lr, lr_cnn)
                total_loss += loss * n
                total_acc += acc * n
                total_n += n

            if total_n == 0:
                continue

            avg_loss = total_loss / total_n
            avg_acc = total_acc / total_n
            self._batches_done += self.accum_steps

            if self._batches_done % 10 == 0:
                _log.worker_msg(
                    self._worker_id,
                    f"batch={self._batches_done} | "
                    f"loss={avg_loss:.4f} | acc={avg_acc:.2f}% | "
                    f"v={version_read} | q={self._stream.queue_size}",
                )

            # ----------------- 4. Enviar actualizaciones al PS -----------------

            # cnn_weights=None si freeze → PS no actualiza la CNN global
            # cnn_weights=state_dict si E2E → PS aplica FedAvg sobre CNN
            try:
                send_message(
                    self._sock,
                    MsgType.UPDATES,
                    {
                        "loss": avg_loss,
                        "accuracy": avg_acc,
                        "batch_size": total_n,
                        "version_read": version_read,
                        "mlp_weights": self._serialize_mlp(),
                        "cnn_weights": None
                        if self._freeze_cnn
                        else self._serialize_cnn(),
                    },
                )
            except Exception as e:
                _log.worker_msg(self._worker_id, f"Error enviando UPDATES: {e}")
                return

    # ================================================================
    # FORWARD + BACKWARD
    # ================================================================

    def _train_batch(
        self,
        X_np: np.ndarray,
        Y_np: np.ndarray,
        lr: float,
        lr_cnn: float,
    ) -> Tuple[float, float, int]:
        """
        Ejecuta un paso forward-backward-SGD de entrenamiento local.

        El modo de entrenamiento se determina por ``self._freeze_cnn``, que
        refleja el estado permanente de requires_grad establecido en CNNExtractor.__init__:

        **Modo ResNet-18 (requires_grad=False permanente)**:
        - Forward CNN en eval() con torch.no_grad(): sin grafo de computación.
        - Features sin grad_fn: backward de MLP no puede fluir hacia CNN.
        - Solo MLP recibe gradientes y se actualiza con SGD local.
        - Reducción de memoria gracias a disposición de activaciones.

        **Modo SimpleCNN (requires_grad=True permanente)**:
        - Forward CNN en train(): BN actualiza running_mean/running_var.
        - Features con grad_fn: backward fluye a través de MLP → CNN.
        - CNN y MLP reciben gradientes y se actualizan con SGD local.
        - Entrenamiento End-to-End (E2E).

        No hay llamadas a requires_grad_(True/False): el estado fue
        establecido en CNNExtractor.__init__() y se preserva en load_state_dict().

        Calcula loss (CrossEntropyLoss) y accuracy (top-1).

        :param X_np: Batch de imágenes de entrada.
        :type X_np: np.ndarray

        :param Y_np: Batch de etiquetas (índices de clase 0-999).
        :type Y_np: np.ndarray

        :param lr: Learning rate para SGD local.
        :type lr: float

        :returns: Tupla (loss_escalar, accuracy_porcentaje, batch_size)
        :rtype: Tuple[float, float, int]

        :raises AssertionError: Si CNN o MLP no están inicializados.
        """
        assert self._cnn is not None
        assert self._mlp is not None

        # Convierte numpy a torch.Tensor
        X = torch.from_numpy(X_np).to(self.device)
        Y = torch.from_numpy(Y_np.astype(np.int64)).to(self.device)

        if self._freeze_cnn:
            # ---------- Modo resnet18: CNN fija ----------

            # requires_grad=False ya establecido en CNNExtractor.__init__
            # torch.no_grad() añade una garantía explícita + ahorra memoria
            self._cnn._model.eval()
            with torch.no_grad():
                features = self._cnn._model(X)

            self._mlp.train()  # Pone el MLP en modo entrenamiento
            self._mlp.zero_grad()  # Limpia gradientes anteriores
            logits = self._mlp(features)
            loss_t = nn.functional.cross_entropy(
                logits, Y
            )  # Calcula CrossEntropyLoss entre predicciones y etiquetas reales
            loss_t.backward()  # Gradientes se calculan solo para parámetros del MLP

            with torch.no_grad():
                for param in self._mlp.parameters():
                    if param.grad is not None:
                        param.data -= lr * param.grad

        else:
            # ---------- Modo simple: E2E ----------

            # requires_grad=True ya establecido en CNNExtractor.__init__
            # train() necesario para que BN actualice running_mean/running_var
            self._cnn._model.train()
            self._mlp.train()

            # Limpia gradientes anteriores de toda la red
            self._cnn._model.zero_grad()
            self._mlp.zero_grad()

            features = self._cnn._model(X)  # grad_fn presente
            logits = self._mlp(features)
            loss_t = nn.functional.cross_entropy(logits, Y)
            loss_t.backward()  # gradientes en CNN + MLP

            # Gradient clipping sobre CNN + MLP conjuntamente.
            # Necesario en E2E desde cero: sin pretrain, los gradientes
            # de la CNN pueden ser desproporcionados respecto al MLP,
            # causando oscilaciones que enlentecen la convergencia.
            # max_norm=1.0 es el umbral estándar para redes sin pretrain.
            # En modo resnet18 (freeze) este bloque no se ejecuta.
            all_params = list(self._cnn._model.parameters()) + list(
                self._mlp.parameters()
            )
            nn.utils.clip_grad_norm_(all_params, max_norm=_GRAD_CLIP_MAX_NORM)

            # LRs separados: lr_cnn < lr para que la CNN aprenda más
            # despacio que el MLP (la CNN parte de representaciones aleatorias).
            with torch.no_grad():
                for param in self._cnn._model.parameters():
                    if param.grad is not None:
                        param.data -= lr * param.grad
                for param in self._mlp.parameters():
                    if param.grad is not None:
                        param.data -= lr * param.grad

            # eval() tras el step: BN usa running stats en inferencia/sync
            self._cnn._model.eval()

        with torch.no_grad():
            correct = (logits.argmax(1) == Y).sum().item()

        n = len(Y_np)
        loss_val = loss_t.item()
        acc_val = 100.0 * correct / n

        del X, Y, features, logits, loss_t
        return loss_val, acc_val, n

    # ================================================================
    # SINCRONIZACIÓN
    # ================================================================

    def _sync_cnn(self, cnn_state: Dict[str, np.ndarray]) -> None:
        """
        Sincroniza los parámetros de la CNN local con el estado global del PS.

        Carga el state_dict de la CNN recibido del PS en la CNN local.
        PyTorch's load_state_dict() copia valores de tensores pero NO modifica
        requires_grad, así que el estado correcto establecido en CNNExtractor.__init__
        se preserva:

        - ResNet-18: Permanece en requires_grad=False después de cada sync.
        - SimpleCNN: Permanece en requires_grad=True después de cada sync.

        Tras cargar, fuerza la CNN a modo eval() para sincronización correcta
        de BatchNorm (usa running stats en lugar de estadísticas de batch).

        :param cnn_state: State_dict de la CNN serializado como Dict[str, np.ndarray].
        :type cnn_state: Dict[str, np.ndarray]

        :returns: None
        :rtype: None

        :raises AssertionError: Si CNN no fue inicializada.
        """
        assert self._cnn is not None
        if not cnn_state:
            return
        base = getattr(self._cnn._model, "model", self._cnn._model)
        sd = base.state_dict()
        with torch.no_grad():
            # Itera sobre todos los parámetros enviados por el PS
            for name, arr in cnn_state.items():
                if name not in sd:
                    continue
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr)
                sd[name] = torch.from_numpy(arr).to(sd[name].device).to(sd[name].dtype)
            base.load_state_dict(sd)
        self._cnn._model.eval()

    def _sync_mlp(
        self,
        mlp_state: Dict[str, np.ndarray],
        existing: Optional[MLPPyTorch],
    ) -> MLPPyTorch:
        """
        Sincroniza los parámetros del MLP local con el estado global del PS.

        Si es la primera sincronización (``existing`` es None), crea una nueva
        instancia de MLPPyTorch deduciendo dimensiones del state_dict recibido:

        - feature_dim (capa entrada): shape[1] de fc1.weight
        - hidden1 (capa oculta 1): shape[0] de fc1.weight
        - hidden2 (capa oculta 2): shape[0] de fc2.weight
        - n_classes: _IMAGENET_CLASSES (1000)

        Si ya existe, reutiliza la instancia para evitar reconstruir el módulo.
        Copia parámetros del estado recibido del PS en lugar de usar load_state_dict()
        para mayor control sobre la transferencia.

        :param mlp_state: State_dict del MLP serializado como Dict[str, np.ndarray].
        :type mlp_state: Dict[str, np.ndarray]

        :param existing: Instancia existente de MLPPyTorch o None.
        :type existing: Optional[MLPPyTorch]

        :returns: Instancia de MLPPyTorch sincronizada.
        :rtype: MLPPyTorch

        :raises RuntimeError: Si mlp_state no contiene 'fc1.weight' (inválido).
        """
        if existing is None:
            if "fc1.weight" not in mlp_state:
                raise RuntimeError(
                    f"[W{self._worker_id}] mlp_state no contiene fc1.weight."
                )
            feature_dim = mlp_state["fc1.weight"].shape[1]
            hidden1 = mlp_state["fc1.weight"].shape[0]
            hidden2 = mlp_state["fc2.weight"].shape[0]
            existing = MLPPyTorch(feature_dim, hidden1, hidden2, _IMAGENET_CLASSES).to(
                self.device
            )
            _log.worker_msg(
                self._worker_id,
                f"MLP creado desde PS: {feature_dim}→{hidden1}→{hidden2}→{_IMAGENET_CLASSES}",
            )

        with torch.no_grad():
            for name, param in existing.named_parameters():
                if name in mlp_state:
                    param.data.copy_(torch.from_numpy(mlp_state[name]).to(param.device))
        return existing

    # ================================================================
    # SERIALIZACIÓN
    # ================================================================

    def _serialize_cnn(self) -> Dict[str, np.ndarray]:
        """
        Serializa el state_dict completo de la CNN a formato numpy.

        Extrae todos los parámetros de la CNN (pesos y biases) y los convierte
        a arrays numpy para transmisión TCP al Parameter Server.

        Esta función solo se llama cuando la CNN es entrenable (modo SimpleCNN).
        En modo ResNet-18 (congelado), el Worker envía cnn_weights=None al PS.

        :returns: Dict mapeando nombres de parámetros a arrays numpy.
        :rtype: Dict[str, np.ndarray]

        :raises AssertionError: Si CNN no fue inicializada.
        """
        assert self._cnn is not None
        base = getattr(self._cnn._model, "model", self._cnn._model)
        return {
            name: tensor.cpu().numpy().copy()
            for name, tensor in base.state_dict().items()
        }

    def _serialize_mlp(self) -> Dict[str, np.ndarray]:
        """
        Serializa los parámetros del MLP a formato numpy para transmisión al PS.

        Extrae todos los parámetros del MLP (pesos fc1, fc2, fc3 y sesgos)
        y los convierte a arrays numpy. MLP siempre se serializa y se envía
        al PS en ambos modos (ResNet-18 y SimpleCNN).

        :returns: Dict mapeando nombres de parámetros a arrays numpy.
        :rtype: Dict[str, np.ndarray]

        :raises AssertionError: Si MLP no fue inicializado.
        """
        assert self._mlp is not None
        return {
            name: param.data.cpu().numpy().copy()
            for name, param in self._mlp.named_parameters()
        }
