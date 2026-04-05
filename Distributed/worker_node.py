"""
Distributed/worker_node.py

Worker asíncrono para entrenamiento distribuido en ImageNet.

MODOS DE ENTRENAMIENTO (determinados automáticamente por arquitectura CNN):

  "resnet18" (freeze_cnn=True):
    - CNN congelada desde CNNExtractor.__init__ (requires_grad=False permanente)
    - forward con torch.no_grad() → features sin grad_fn → 0 overhead de memoria
    - Solo el MLP recibe gradientes y se actualiza
    - cnn_weights=None en UPDATES → PS no toca la CNN global

  "simple" (freeze_cnn=False):
    - CNN entrenable desde CNNExtractor.__init__ (requires_grad=True permanente)
    - forward en train() → features con grad_fn → backward fluye por toda la red
    - CNN y MLP se actualizan con SGD local
    - cnn_weights enviado al PS → FedAvg sobre CNN global

El modo se fija en _load_cnn() al recibir el arch del PS. No hay gestión
manual de requires_grad en _train_batch: el estado es correcto desde la
construcción de CNNExtractor y se preserva a través de load_state_dict().
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


class WorkerNode:
    """
    Worker asíncrono para entrenamiento distribuido en ImageNet.

    :param server_host:      IP del Parameter Server.
    :param server_port:      Puerto TCP.
    :param dataset_name:     Dataset HF Hub.
    :param worker_rank:      Índice de este Worker (0-based, para sharding).
    :param num_workers:      Total de Workers.
    :param device:           Dispositivo PyTorch ('cpu', 'cuda', 'cuda:0', 'mps').
    :param shuffle_buffer:   Imágenes en buffer de shuffle.
    :param prefetch_batches: Batches pre-cargados en background.
    :param seed:             Semilla RNG (None = aleatorio).
    :param hf_token:         Token HuggingFace.
    :param accum_steps:      Batches a acumular antes de enviar UPDATES.
    :param verbose:          Imprimir progreso cada 10 batches.

    NOTA: batch_size e image_size se reciben del PS mediante CONFIG.
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
        self.server_host = server_host
        self.server_port = server_port
        self.dataset_name = dataset_name
        self.worker_rank = worker_rank
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_batches = prefetch_batches
        self.seed = seed
        self.hf_token = hf_token
        self.accum_steps = accum_steps
        self.verbose = verbose

        # Recibidos via CONFIG del PS
        self.batch_size: Optional[int] = None
        self.image_size: Optional[int] = None

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
        self._connect()
        self._log(
            f"Conectado | rank={self.worker_rank}/{self.num_workers} | "
            f"device={self.device} | batch={self.batch_size} | "
            f"accum={self.accum_steps}"
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
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.server_host, self.server_port))
        send_message(self._sock, MsgType.READY, {})

        msg = receive_message(self._sock)
        if msg["type"] != MsgType.WORKER_ID:
            raise ConnectionError(f"Esperaba WORKER_ID, recibí {msg['type']}")
        self._worker_id = msg["payload"]["worker_id"]

        msg = receive_message(self._sock)
        if msg["type"] != MsgType.CONFIG:
            raise ConnectionError(f"Esperaba CONFIG, recibí {msg['type']}")
        config = msg["payload"]
        self.batch_size = config["batch_size"]
        self.image_size = config["image_size"]
        self._log(
            f"CONFIG recibida: batch_size={self.batch_size}, image_size={self.image_size}"
        )

    def _cleanup(self) -> None:
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
        self._log("Recursos liberados.")

    # ================================================================
    # STREAM DE DATOS
    # ================================================================

    def _init_stream(self) -> None:
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
        self._log(
            f"Stream iniciado: {self.dataset_name} | "
            f"shard {self.worker_rank}/{self.num_workers}"
        )

    # ================================================================
    # HANDSHAKE
    # ================================================================

    def _handshake_loop(self) -> None:
        assert self._sock is not None

        while True:
            msg = receive_message(self._sock)
            t = msg["type"]

            if t == MsgType.STOP:
                self._log("STOP recibido durante handshake.")
                return
            elif t == MsgType.CNN_WEIGHTS:
                self._load_cnn(msg["payload"])
            elif t == MsgType.START:
                self._log("START recibido — iniciando loop de entrenamiento.")
                self._training_loop()
                return

    def _load_cnn(self, payload: dict) -> None:
        """
        Carga la CNN recibida del PS y fija el modo de entrenamiento.

        CNNExtractor se instancia con el arch correcto, lo que establece
        requires_grad de forma permanente y coherente con la semántica:
          - resnet18 → requires_grad=False en todos los parámetros
          - simple   → requires_grad=True  en todos los parámetros

        load_weights_from_bytes usa load_state_dict, que no modifica
        requires_grad, así que el estado se preserva tras cargar los pesos.
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

        if self._cnn is None or self._cnn.arch != arch:
            self._log(f"Instanciando CNN arch={arch} en {self.device}")
            # CNNExtractor.__init__ establece requires_grad según arch:
            #   resnet18 → False (congelada), simple → True (entrenable)
            self._cnn = CNNExtractor(arch=arch, device=str(self.device), seed=self.seed)

        # load_weights_from_bytes → load_state_dict → NO modifica requires_grad
        self._cnn.load_weights_from_bytes(weights_bytes)

        actual_keys = len(self._cnn._model.state_dict())
        if cnn_key_count > 0 and actual_keys != cnn_key_count:
            raise RuntimeError(
                f"[W{self._worker_id}] Mismatch CNN params: "
                f"PS={cnn_key_count}, Worker={actual_keys}"
            )

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
        self._log(
            f"CNN cargada ✓ arch={arch} | feature_dim={self._cnn.feature_dim} | "
            f"params={actual_keys} | modo={mode}"
        )

        assert self._sock is not None
        send_message(
            self._sock,
            MsgType.CNN_ACK,
            {
                "worker_id": self._worker_id,
                "arch": arch,
                "feature_dim": self._cnn.feature_dim,
            },
        )

    # ================================================================
    # LOOP DE ENTRENAMIENTO ASÍNCRONO
    # ================================================================

    def _training_loop(self) -> None:
        """Loop: REQUEST_PARAMS → sincronizar → entrenar → UPDATES."""
        if self._cnn is None:
            raise RuntimeError(
                f"[W{self._worker_id}] _training_loop sin CNN inicializada."
            )

        assert self._sock is not None
        assert self._cnn is not None
        assert self._stream is not None

        self._log(
            f"Entrenamiento | arch={self._cnn.arch} | "
            f"feature_dim={self._cnn.feature_dim} | device={self.device} | "
            f"freeze_cnn={self._freeze_cnn}"
        )

        stream_iter = self._stream.__iter__()
        version_read = 0
        first_params_logged = False

        while True:
            # ── 1. Pedir parámetros globales ──
            try:
                send_message(self._sock, MsgType.REQUEST_PARAMS, {})
                msg = receive_message(self._sock)
            except Exception as e:
                self._log(f"Error de comunicación: {e}")
                return

            if msg["type"] == MsgType.STOP:
                return
            if msg["type"] != MsgType.PARAMS:
                self._log(f"Mensaje inesperado: {msg['type']}")
                continue

            payload = msg["payload"]
            mlp_state = payload["mlp_state"]
            cnn_state = payload["cnn_state"]
            version_read = payload["version"]
            lr = payload["lr"]

            if not mlp_state:
                raise RuntimeError(f"[W{self._worker_id}] PS devolvió mlp_state vacío.")
            if not cnn_state:
                raise RuntimeError(f"[W{self._worker_id}] PS devolvió cnn_state vacío.")

            if not first_params_logged:
                fc1_shape = mlp_state.get("fc1.weight", np.array([])).shape
                self._log(
                    f"Primer PARAMS — v={version_read} | lr={lr} | "
                    f"mlp fc1.weight={fc1_shape} | cnn_params={len(cnn_state)}"
                )
                first_params_logged = True

            # ── 2. Sincronizar estado global del PS ──
            # _sync_cnn usa load_state_dict → no modifica requires_grad
            # El estado correcto (False para resnet18, True para simple)
            # se mantiene intacto tras cada sincronización.
            self._sync_cnn(cnn_state)
            self._mlp = self._sync_mlp(mlp_state, self._mlp)

            # ── 3. Entrenar accum_steps batches ──
            total_loss, total_acc, total_n = 0.0, 0.0, 0

            for _ in range(self.accum_steps):
                try:
                    X_np, Y_np = next(stream_iter)
                except StopIteration:
                    stream_iter = self._stream.__iter__()
                    X_np, Y_np = next(stream_iter)

                loss, acc, n = self._train_batch(X_np, Y_np, lr)
                total_loss += loss * n
                total_acc += acc * n
                total_n += n

            if total_n == 0:
                continue

            avg_loss = total_loss / total_n
            avg_acc = total_acc / total_n
            self._batches_done += self.accum_steps

            if self.verbose and self._batches_done % 10 == 0:
                self._log(
                    f"batch={self._batches_done} | "
                    f"loss={avg_loss:.4f} | acc={avg_acc:.2f}% | "
                    f"v={version_read} | q={self._stream.queue_size}"
                )

            # ── 4. Enviar actualizaciones al PS ──
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
                self._log(f"Error enviando UPDATES: {e}")
                return

    # ================================================================
    # FORWARD + BACKWARD
    # ================================================================

    def _train_batch(
        self, X_np: np.ndarray, Y_np: np.ndarray, lr: float
    ) -> Tuple[float, float, int]:
        """
        Paso de entrenamiento. El comportamiento depende de self._freeze_cnn,
        que a su vez refleja el estado permanente de requires_grad en la CNN:

        Modo freeze (resnet18, requires_grad=False permanente):
          - torch.no_grad() en el forward CNN: sin grafo, sin memoria de activaciones
          - features sin grad_fn: backward del MLP no puede atravesarlas hacia la CNN
          - Solo MLP recibe gradientes y se actualiza

        Modo E2E (simple, requires_grad=True permanente):
          - CNN en train(): BN actualiza running stats durante entrenamiento
          - forward con grafo: features tienen grad_fn
          - backward fluye por MLP y CNN
          - Ambos se actualizan con SGD local

        No hay llamadas a requires_grad_(True/False) aquí: el estado es
        correcto desde CNNExtractor.__init__ y se preserva en load_state_dict.
        """
        assert self._cnn is not None
        assert self._mlp is not None

        X = torch.from_numpy(X_np).to(self.device)
        Y = torch.from_numpy(Y_np.astype(np.int64)).to(self.device)

        if self._freeze_cnn:
            # ── Modo resnet18: CNN fija ──
            # requires_grad=False ya establecido en CNNExtractor.__init__
            # torch.no_grad() añade una garantía explícita + ahorra memoria
            self._cnn._model.eval()
            with torch.no_grad():
                features = self._cnn._model(X)

            self._mlp.train()
            self._mlp.zero_grad()
            logits = self._mlp(features)
            loss_t = nn.functional.cross_entropy(logits, Y)
            loss_t.backward()

            with torch.no_grad():
                for p in self._mlp.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad

        else:
            # ── Modo simple: E2E ──
            # requires_grad=True ya establecido en CNNExtractor.__init__
            # train() necesario para que BN actualice running_mean/running_var
            self._cnn._model.train()
            self._mlp.train()

            self._cnn._model.zero_grad()
            self._mlp.zero_grad()

            features = self._cnn._model(X)  # grad_fn presente
            logits = self._mlp(features)
            loss_t = nn.functional.cross_entropy(logits, Y)
            loss_t.backward()  # gradientes en CNN + MLP

            with torch.no_grad():
                for p in self._cnn._model.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad
                for p in self._mlp.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad

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
        Carga el estado global de la CNN desde el PS.

        load_state_dict copia valores de tensores pero NO modifica requires_grad.
        El estado correcto establecido en CNNExtractor.__init__ se preserva:
          - resnet18: sigue en requires_grad=False tras cada sync
          - simple:   sigue en requires_grad=True  tras cada sync
        """
        assert self._cnn is not None
        base = getattr(self._cnn._model, "model", self._cnn._model)
        sd = base.state_dict()
        with torch.no_grad():
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
        Carga el estado global del MLP desde el PS.

        Reutiliza el objeto si ya existe para evitar reconstruir nn.Module.
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
            self._log(
                f"MLP creado desde PS: {feature_dim}→{hidden1}→{hidden2}→{_IMAGENET_CLASSES}"
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
        """State_dict completo de la CNN (solo llamado en modo E2E)."""
        assert self._cnn is not None
        base = getattr(self._cnn._model, "model", self._cnn._model)
        return {
            name: tensor.cpu().numpy().copy()
            for name, tensor in base.state_dict().items()
        }

    def _serialize_mlp(self) -> Dict[str, np.ndarray]:
        """State_dict del MLP en formato PyTorch nativo."""
        assert self._mlp is not None
        return {
            name: param.data.cpu().numpy().copy()
            for name, param in self._mlp.named_parameters()
        }

    # ================================================================
    # LOG
    # ================================================================

    def _log(self, msg: str) -> None:
        if self.verbose:
            wid = self._worker_id if self._worker_id is not None else "?"
            print(f"[W{wid}] {msg}", flush=True)
