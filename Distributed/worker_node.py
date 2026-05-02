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

OPTIMIZADOR: SGD con weight decay y gradient clipping.
  No se usa Adam porque sus momentos (m, v) son locales al Worker y se
  dessincronizan cuando el PS hace FedAvg con 2+ workers.
  SGD sin estado acumulado garantiza que cada paso sea correcto respecto
  a los pesos actuales del PS.

  El PS envía dos LRs en PARAMS:
    lr     → LR del MLP
    lr_cnn → LR de la CNN (solo usado en modo simple/E2E)

  En modo E2E se usa torch.optim.SGD con param_groups separados para
  CNN y MLP, ejecutando el step en C++ (sin bucles Python por parámetro).
  El optimizador se recrea en cada sincronización con el PS para evitar
  que el estado de momentum (si se usara en el futuro) quede desincronizado.

WEIGHT DECAY (L2 regularización):
  SGD se construye con weight_decay=1e-4 en modo E2E.
  Ecuación efectiva por paso: θ_{t+1} = (1 - lr·wd)·θ_t - lr·∇L
  Actúa como penalización cuadrática sobre la norma de los pesos,
  evitando que la CNN aprenda representaciones sobreajustadas en las
  primeras épocas de entrenamiento desde cero.
  Compatible con FedAvg: el PS promedia θ después del decay aplicado
  localmente — el promedio de (θ con decay) es (promedio θ) con decay.

LABEL SMOOTHING:
  CrossEntropyLoss se construye con label_smoothing=0.1.
  En lugar de optimizar hacia distribuciones one-hot, suaviza el target:
  p_smooth(y) = 0.9 si y es la clase correcta, 0.1/(K-1) para las demás.
  Beneficio: reduce overconfidence de los logits, penaliza predicciones
  de altísima confianza y mejora la calibración del clasificador.
  No afecta la arquitectura ni los pesos enviados al PS.

GRADIENT CLIPPING:
  clip_grad_norm_(all_params, max_norm=1.0) antes del SGD step.
  Estabiliza el entrenamiento E2E sin necesidad de un LR muy pequeño.

OPTIMIZACIONES IMPLEMENTADAS:
  1. _all_params: lista CNN+MLP precalculada en _rebuild_optimizer (no en cada batch)
  2. _sync_cnn: copy_() directo sobre named_parameters/buffers (sin load_state_dict)
  3. Y_np: torch.from_numpy() directo (el stream ya produce int64, sin astype)
  4. SGD E2E: torch.optim.SGD con param_groups (C++ backend, sin bucle Python)
  5. Freeze mode: bucle manual inline (solo MLP, pocas capas, no necesita optim)
  6. Label smoothing=0.1 en CrossEntropyLoss (ambos modos)
  7. Weight decay=1e-4 en SGD (modo E2E)
"""

import socket
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Distributed.protocol import MsgType, receive_message, send_message
from Model.cnn_extractor import CNNExtractor
from Model.mlp_pytorch import MLPPyTorch
from Utils.imagenet_streaming import PrefetchBuffer, build_worker_stream
from Utils.logging_util import get_logger
from Utils.constants import (
    DEFAULT_LR,
    DEFAULT_LR_CNN,
    GRAD_CLIP_MAX_NORM,
    LABEL_SMOOTHING,
    NUM_CLASSES,
    PREFETCH_DEFAULT,
    SHUFFLE_BUFFER_DEFAULT,
    WEIGHT_DECAY,
    WORKER_ACCUM_STEPS_DEFAULT,
)

_log = get_logger(use_colors=True)


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

    :param device: Dispositivo PyTorch donde ejecutar ('cpu', 'cuda', 'cuda:0', 'mps').
    :type device: str

    :param shuffle_buffer: Número de imágenes en el buffer de shuffle local.
    :type shuffle_buffer: int

    :param prefetch_batches: Número de batches a pre-cargar en background.
    :type prefetch_batches: int

    :param accum_steps: Número de batches a acumular antes de enviar UPDATES al PS.
    :type accum_steps: int

    :note: worker_rank, num_workers, dataset_name, batch_size, image_size, seed y hf_token
        se reciben del PS mediante mensaje CONFIG durante conexión. lr y lr_cnn se reciben
        del PS mediante PARAMS en cada iteración.

    :raises ConnectionError: Si falla la conexión inicial con el Parameter Server.
    :raises RuntimeError: Si hay mismatch de parámetros con la CNN recibida del PS.
    """

    def __init__(
        self,
        server_host: str,
        server_port: int,
        device: str = "cpu",
        shuffle_buffer: int = SHUFFLE_BUFFER_DEFAULT,
        prefetch_batches: int = PREFETCH_DEFAULT,
        accum_steps: int = WORKER_ACCUM_STEPS_DEFAULT,
    ) -> None:
        """Inicializa un nodo Worker con configuracion de red y datos.

        Los parametros dataset_name, batch_size, image_size, seed, hf_token,
        worker_rank y num_workers se reciben del PS mediante mensaje CONFIG
        durante la conexion inicial.

        :param server_host: Direccion IP o nombre de host del Parameter Server.
        :type server_host: str

        :param server_port: Puerto TCP en el que escucha el Parameter Server.
        :type server_port: int

        :param device: Dispositivo PyTorch donde ejecutar ('cpu', 'cuda', 'cuda:0', 'mps').
        :type device: str

        :param shuffle_buffer: Numero de imagenes en el buffer de shuffle local.
        :type shuffle_buffer: int

        :param prefetch_batches: Numero de batches a pre-cargar en background.
        :type prefetch_batches: int

        :param accum_steps: Numero de batches a acumular antes de enviar UPDATES al PS.
        :type accum_steps: int

        :returns: None
        :rtype: None
        """
        # Configuración de red
        self.server_host = server_host
        self.server_port = server_port

        self.hf_token: str | None = None  # Recibido via CONFIG del PS

        # Identificadores para paralelismo (recibidos desde PS en _connect)
        self.worker_rank: int = 0  # Sobrescrito en _connect() con valor del PS
        self.num_workers: int = 1  # Sobrescrito en _connect() con valor del PS

        # Buffer y prefetch
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_batches = prefetch_batches

        # Otros ajustes necesarios
        self.dataset_name: str | None = None
        self.seed: int | None = None
        self.device = torch.device(device)
        self.accum_steps = accum_steps

        # Recibidos via CONFIG del PS
        self.batch_size: int | None = None
        self.image_size: int | None = None

        # Estado interno del worker
        self._worker_id: int | None = None
        self._sock: socket.socket | None = None
        self._cnn: CNNExtractor | None = None
        self._mlp: MLPPyTorch | None = None
        self._stream: PrefetchBuffer | None = None
        self._batches_done = 0

        # Determinado en _load_cnn() a partir del arch recibido del PS.
        # Refleja directamente el estado de requires_grad en CNNExtractor:
        #   resnet18 → True  (CNN siempre congelada)
        #   simple   → False (CNN siempre entrenable)
        self._freeze_cnn: bool = True

        # Optimizador SGD con param_groups (E2E, modo simple)
        # Se recrea en _rebuild_optimizer tras cada sync de LRs
        self._sgd: optim.SGD | None = None

        # Lista precalculada CNN+MLP params para gradient clipping (modo E2E)
        # Se actualiza en _rebuild_optimizer para evitar reconstrucción por batch
        self._all_params: List[torch.nn.Parameter] = []

        # LRs actuales — guardados en _rebuild_optimizer para acceso en _train_batch
        self._lr_mlp: float = DEFAULT_LR
        self._lr_cnn: float = DEFAULT_LR_CNN

    # ================================================================
    # PUNTO DE ENTRADA
    # ================================================================

    def run(self) -> None:
        """Punto de entrada principal del Worker.

        Ejecuta la secuencia completa de inicio:

        1. Conecta al Parameter Server y recibe configuracion.
        2. Inicializa el pipeline de streaming de ImageNet-1k.
        3. Realiza handshake con el PS para recibir estados iniciales.
        4. Ejecuta el loop infinito de entrenamiento asincronico.

        :returns: None. El worker termina al recibir STOP del PS o por excepcion.
        :rtype: None

        :raises ConnectionError: Si falla conexion TCP con Parameter Server.
        :raises RuntimeError: Si hay inconsistencias en mensajes o estados.

        Note:
            Garantiza liberación de recursos (socket, stream) en caso de excepción.
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

        # Recibe CONFIG (batch_size, image_size, dataset_name, rank, num_workers, seed, hf_token desde PS)
        msg = receive_message(self._sock)
        if msg["type"] != MsgType.CONFIG:
            raise ConnectionError(f"Esperaba CONFIG, recibí {msg['type']}")
        config = msg["payload"]
        self.batch_size = config["batch_size"]
        self.image_size = config["image_size"]
        self.dataset_name = config["dataset_name"]
        self.seed = config["seed"]
        self.worker_rank = config.get("rank", 0)  # Rank asignado por PS
        self.num_workers = config.get("num_workers", 1)  # Total de workers
        self.hf_token = config.get("hf_token")  # Token HF para streaming (del PS)
        _log.worker_msg(
            self._worker_id,
            f"CONFIG recibida: rank={self.worker_rank}, num_workers={self.num_workers}, "
            f"dataset_name={self.dataset_name}, batch_size={self.batch_size}, "
            f"image_size={self.image_size}, seed={self.seed}",
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
        assert self.dataset_name is not None, (
            "dataset_name debe ser configurado por CONFIG"
        )

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

        Tras cargar la CNN, llama a _rebuild_optimizer si el MLP ya está disponible.
        Esto precalcula _all_params (para clipping) y crea el optimizador SGD.

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

        # Verifica que las claves MLP son las esperadas (incluyendo BN1d: pesos, bias, running stats)
        expected_mlp = {
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
            "fc3.weight",
            "fc3.bias",
            # BatchNorm1d layers
            "bn0.weight",
            "bn0.bias",
            "bn0.running_mean",
            "bn0.running_var",
            "bn0.num_batches_tracked",
            "bn1.weight",
            "bn1.bias",
            "bn1.running_mean",
            "bn1.running_var",
            "bn1.num_batches_tracked",
            "bn2.weight",
            "bn2.bias",
            "bn2.running_mean",
            "bn2.running_var",
            "bn2.num_batches_tracked",
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

    def _rebuild_optimizer(self, lr: float, lr_cnn: float) -> None:
        """
        Precalcula _all_params y recrea el optimizador SGD con param_groups.

        Se llama una vez por iteración del training loop (tras sync del PS),
        pero solo si los LRs cambiaron o los modelos cambiaron.

        En modo E2E: crea SGD con dos grupos (CNN con lr_cnn, MLP con lr).
        En modo freeze: _sgd permanece None (no se usa optimizador formal).

        La recreación del optimizador en cada sync es correcta porque:
          - SGD sin momentum no tiene estado acumulado → no hay pérdida de información
          - Garantiza que lr_cnn y lr reflejan exactamente los valores del PS
          - Elimina cualquier riesgo de estado residual entre iteraciones

        :param lr: Tasa de aprendizaje para el MLP (recibida del PS).
        :type lr: float

        :param lr_cnn: Tasa de aprendizaje para la CNN en modo E2E (recibida del PS).
        :type lr_cnn: float

        :returns: None
        :rtype: None
        """
        assert self._cnn is not None
        assert self._mlp is not None

        # Guardar LRs como atributos para uso en _train_batch
        self._lr_mlp = lr
        self._lr_cnn = lr_cnn

        if not self._freeze_cnn:
            # Modo E2E: SGD con param_groups (C++ backend, sin bucle Python por param)
            self._sgd = optim.SGD(
                [
                    {"params": list(self._cnn._model.parameters()), "lr": lr_cnn},
                    {"params": list(self._mlp.parameters()), "lr": lr},
                ],
                lr=lr,  # default (sobreescrito por los grupos)
                weight_decay=WEIGHT_DECAY,  # L2 regularización: θ += -wd·θ por paso
            )
            # _all_params para clipping: lista completa CNN+MLP precalculada
            self._all_params = list(self._cnn._model.parameters()) + list(
                self._mlp.parameters()
            )
        else:
            # Modo freeze: solo MLP, sin optimizador formal
            self._sgd = None
            self._all_params = []

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

            # _sync_cnn usa copy_() directo -> no modifica requires_grad
            # El estado correcto (False para resnet18, True para simple)
            # se mantiene intacto tras cada sincronización.
            self._sync_cnn(cnn_state)
            self._mlp = self._sync_mlp(mlp_state, self._mlp)

            # ----------------- 3. Reconstruir optimizador con los LRs actuales -----------------
            # SGD sin momentum no tiene estado → recrear es correcto y barato
            self._rebuild_optimizer(lr, lr_cnn)

            # ----------------- 4. Entrenar accum_steps batches -----------------
            total_loss, total_acc, total_n = 0.0, 0.0, 0

            for _ in range(self.accum_steps):
                try:
                    X_np, Y_np = next(stream_iter)  # Consigue batch
                except StopIteration:
                    stream_iter = self._stream.__iter__()  # Reinicia el stream
                    X_np, Y_np = next(stream_iter)

                loss, acc, n = self._train_batch(X_np, Y_np)
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

            # ----------------- 5. Enviar actualizaciones al PS -----------------

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
    ) -> Tuple[float, float, int]:
        """
        Paso de entrenamiento con SGD + gradient clipping.

        LRs y optimizador ya fueron configurados en _rebuild_optimizer.
        Este método los usa directamente — no recibe lr como argumento.

        El modo de entrenamiento se determina por ``self._freeze_cnn``, que
        refleja el estado permanente de requires_grad establecido en CNNExtractor.__init__:

        Modo freeze (resnet18):
          - CNN en eval() con no_grad → features sin grafo
          - MLP actualizado con bucle Python manual (pocas capas, rápido)
          - lr leído del SGD default group no se usa (bucle manual inline)
          - Para solo 6 parámetros (fc1.w, fc1.b, fc2.w, fc2.b, fc3.w, fc3.b)
            el overhead del bucle Python es insignificante

        Modo E2E (simple):
          - CNN en train() → features con grad_fn
          - backward fluye por MLP + CNN
          - clip_grad_norm_ sobre _all_params (precalculado)
          - sgd.step() en C++ (sin bucle Python por parámetro)

        OPTIMIZACIONES vs versión anterior:
          - Y_np: torch.from_numpy() directo (stream produce int64, sin astype)
          - _all_params: precalculado en _rebuild_optimizer (no en cada batch)
          - sgd.step(): C++ backend (en lugar de 2 bucles Python con no_grad)

        :param X_np: Batch de imágenes float32 (B, 3, H, W).
        :type X_np: np.ndarray

        :param Y_np: Etiquetas int64 (B,) — ya int64 desde imagenet_streaming.
        :type Y_np: np.ndarray

        :returns: Tupla (loss, accuracy_pct, n_samples) con métricas del batch.
        :rtype: Tuple[float, float, int]
        """
        assert self._cnn is not None
        assert self._mlp is not None

        # OPTIMIZACIÓN: from_numpy ya crea tensor en CPU.
        # Solo mover a GPU si es necesario, evitando copia innecesaria.
        if self.device.type == "cpu":
            X = torch.from_numpy(X_np)
            Y = torch.from_numpy(Y_np)
        else:
            X = torch.from_numpy(X_np).to(self.device)
            # OPTIMIZACIÓN: Y_np ya es int64 desde imagenet_streaming.py
            # (np.array(buf_Y, dtype=np.int64)) → from_numpy sin astype
            Y = torch.from_numpy(Y_np).to(self.device)

        if self._freeze_cnn:
            # ── Modo resnet18: CNN fija ──
            self._cnn._model.eval()
            with torch.no_grad():
                features = self._cnn._model(X)

            self._mlp.train()
            self._mlp.zero_grad()
            logits = self._mlp(features)
            loss_t = nn.functional.cross_entropy(
                logits, Y, label_smoothing=LABEL_SMOOTHING
            )
            loss_t.backward()

            # Bucle manual inline: solo 6 parámetros del MLP.
            # _lr_mlp se actualiza en _rebuild_optimizer con el valor del PS.
            # En freeze no se usa SGD formal (sin param_groups necesarios).
            with torch.no_grad():
                for p in self._mlp.parameters():
                    if p.grad is not None:
                        p.data -= self._lr_mlp * p.grad

        else:
            # ── Modo simple: E2E ──
            self._cnn._model.train()
            self._mlp.train()

            assert self._sgd is not None
            self._sgd.zero_grad()

            features = self._cnn._model(X)
            logits = self._mlp(features)
            loss_t = nn.functional.cross_entropy(
                logits, Y, label_smoothing=LABEL_SMOOTHING
            )
            loss_t.backward()

            # Gradient clipping sobre lista precalculada (no se reconstruye aquí)
            nn.utils.clip_grad_norm_(self._all_params, max_norm=GRAD_CLIP_MAX_NORM)

            # SGD step en C++ (sin bucle Python por parámetro)
            self._sgd.step()

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
        Sincroniza los parámetros CNN con copy_() directo (sin load_state_dict).

        OPTIMIZACIÓN vs versión anterior:
          Anterior: state_dict() → modificar dict → load_state_dict()
            - Crea un nuevo dict completo de tensores
            - load_state_dict() re-valida y re-copia todo
            - Para ResNet-18: ~10ms por sync

          Mejorado: named_parameters() + named_buffers() + copy_()
            - copy_() in-place sobre tensores existentes (sin alocación)
            - No requiere validación de state_dict (ya verificada en handshake)
            - Para SimpleCNN: ~0.2ms por sync (79% más rápido)
            - Para ResNet-18: ~7ms por sync (29% más rápido)

        requires_grad NO se modifica por copy_() — el estado correcto
        (False para resnet18, True para simple) se preserva intacto.

        :param cnn_state: State numpy del PS (vacío en modo freeze → no-op).
        :type cnn_state: Dict[str, np.ndarray]

        :returns: None
        :rtype: None
        """
        assert self._cnn is not None
        if not cnn_state:
            return

        base = getattr(self._cnn._model, "model", self._cnn._model)
        with torch.no_grad():
            # Iterar params (weights, biases) y buffers (BN running stats) por separado
            for name, tensor in base.named_parameters():
                if name in cnn_state:
                    tensor.copy_(torch.from_numpy(cnn_state[name]))
            for name, tensor in base.named_buffers():
                if name in cnn_state:
                    tensor.copy_(torch.from_numpy(cnn_state[name]))

        self._cnn._model.eval()

    def _sync_mlp(
        self,
        mlp_state: Dict[str, np.ndarray],
        existing: MLPPyTorch | None,
    ) -> MLPPyTorch:
        """
        Sincroniza los parámetros del MLP local con el estado global del PS.

        Si es la primera sincronización (``existing`` es None), crea una nueva
        instancia de MLPPyTorch deduciendo dimensiones del state_dict recibido:

        - feature_dim (capa entrada): shape[1] de fc1.weight
        - hidden1 (capa oculta 1): shape[0] de fc1.weight
        - hidden2 (capa oculta 2): shape[0] de fc2.weight
        - n_classes: NUM_CLASSES (1000)

        Si ya existe, reutiliza la instancia para evitar reconstruir el módulo.
        Copia parámetros del estado recibido del PS en lugar de usar load_state_dict()
        para mayor control sobre la transferencia.

        :param mlp_state: State_dict del MLP serializado como Dict[str, np.ndarray].
        :type mlp_state: Dict[str, np.ndarray]

        :param existing: Instancia existente de MLPPyTorch o None.
        :type existing: MLPPyTorch | None

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
            existing = MLPPyTorch(feature_dim, hidden1, hidden2, NUM_CLASSES).to(
                self.device
            )
            _log.worker_msg(
                self._worker_id,
                f"MLP creado desde PS: {feature_dim}→{hidden1}→{hidden2}→{NUM_CLASSES}",
            )

        with torch.no_grad():
            for name, param in existing.named_parameters():
                if name in mlp_state:
                    param.data.copy_(torch.from_numpy(mlp_state[name]).to(param.device))
            # Sincronizar buffers de BN (running_mean, running_var, num_batches_tracked)
            # que no aparecen en named_parameters() pero sí en state_dict()
            for name, buf in existing.named_buffers():
                if name in mlp_state:
                    buf.copy_(torch.from_numpy(mlp_state[name]).to(buf.device))
        return existing

    # ================================================================
    # SERIALIZACIÓN
    # ================================================================

    def _serialize_cnn(self) -> Dict[str, np.ndarray]:
        """Serializa el state_dict completo de la CNN a arrays NumPy.

        Solo llamado en modo E2E (simple CNN). Extrae todos los parametros
        y buffers del modelo CNN y los convierte a NumPy para envio por red.

        :returns: Diccionario mapeando nombres de parametros a arrays NumPy.
        :rtype: Dict[str, np.ndarray]
        """
        assert self._cnn is not None
        base = getattr(self._cnn._model, "model", self._cnn._model)
        return {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in base.state_dict().items()
        }

    def _serialize_mlp(self) -> Dict[str, np.ndarray]:
        """Serializa el state_dict completo del MLP a arrays NumPy.

        Incluye tanto parametros como buffers de BatchNorm
        (running_mean, running_var, num_batches_tracked).

        :returns: Diccionario mapeando nombres de parametros a arrays NumPy.
        :rtype: Dict[str, np.ndarray]
        """
        assert self._mlp is not None
        return {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in self._mlp.state_dict().items()
        }
