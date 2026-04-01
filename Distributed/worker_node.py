"""
Distributed/worker_node.py

Worker asíncrono para entrenamiento distribuido E2E en ImageNet.

DISEÑO:
  - Streaming puro desde HuggingFace: nunca más de `prefetch_batches`
    batches en RAM simultáneamente.
  - Loop completamente autónomo: el Worker no espera a ningún otro Worker.
  - Formato PyTorch state_dict nativo en todo el pipeline:
      PS → Worker: mlp_state (fc1.weight / fc1.bias / …) + cnn_state
      Worker → PS: los mismos keys, sin conversión alguna
    Esto elimina el mapping W1↔fc1.weight del sistema anterior.
  - El MLP PyTorch se reutiliza entre batches (sin reconstruir el módulo),
    reduciendo el overhead de inicialización.

FLUJO POR ITERACIÓN:
    1. REQUEST_PARAMS  → PS responde con PARAMS
    2. _sync_model()   → carga CNN + MLP con el estado global
    3. _train_batch()  → forward E2E + backward + SGD local
    4. UPDATES         → enviar pesos actualizados + métricas al PS

GRADIENT ACCUMULATION:
    Si accum_steps > 1, el Worker acumula N batches antes de enviar
    UPDATES. Reduce la frecuencia de comunicación a costa de mayor staleness.
    Recomendado: 1–4 para redes de alta velocidad.
"""

import socket
import time
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
    Worker asíncrono para entrenamiento E2E distribuido en ImageNet.

    :param server_host:    IP del Parameter Server.
    :param server_port:    Puerto TCP.
    :param dataset_name:   Dataset en HF Hub (e.g. 'ILSVRC/imagenet-1k').
    :param worker_rank:    Índice de este Worker (0-based, para sharding).
    :param num_workers:    Total de Workers (para sharding del stream).
    :param batch_size:     Imágenes por batch de entrenamiento.
    :param hidden1:        Neuronas capa oculta 1 del MLP.
    :param hidden2:        Neuronas capa oculta 2 del MLP.
    :param device:         Dispositivo PyTorch ('cpu', 'cuda', 'cuda:0', 'mps').
    :param shuffle_buffer: Imágenes en el buffer de shuffle del stream HF.
    :param prefetch_batches: Batches pre-cargados en hilo background.
    :param image_size:     Tamaño de imagen tras crop (224 estándar).
    :param hf_token:       Token HuggingFace para datasets con licencia.
    :param accum_steps:    Batches a acumular antes de enviar UPDATES al PS.
    :param verbose:        Imprimir progreso cada batch.
    """

    def __init__(
        self,
        server_host: str,
        server_port: int,
        dataset_name: str = "ILSVRC/imagenet-1k",
        worker_rank: int = 0,
        num_workers: int = 1,
        batch_size: int = 64,
        hidden1: int = 1024,
        hidden2: int = 512,
        device: str = "cpu",
        shuffle_buffer: int = 1000,
        prefetch_batches: int = 4,
        image_size: int = 224,
        hf_token: Optional[str] = None,
        accum_steps: int = 1,
        verbose: bool = True,
    ) -> None:
        self.server_host    = server_host
        self.server_port    = server_port
        self.dataset_name   = dataset_name
        self.worker_rank    = worker_rank
        self.num_workers    = num_workers
        self.batch_size     = batch_size
        self.hidden1        = hidden1
        self.hidden2        = hidden2
        self.device         = torch.device(device)
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_batches = prefetch_batches
        self.image_size     = image_size
        self.hf_token       = hf_token
        self.accum_steps    = accum_steps
        self.verbose        = verbose

        self._worker_id: Optional[int] = None
        self._sock:      Optional[socket.socket] = None
        self._cnn:       Optional[CNNExtractor] = None
        self._mlp:       Optional[MLPPyTorch] = None
        self._stream:    Optional[PrefetchBuffer] = None
        self._batches_done = 0

    # ================================================================
    # PUNTO DE ENTRADA
    # ================================================================

    def run(self) -> None:
        """Conecta al PS y ejecuta el loop de entrenamiento asíncrono."""
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
        """Construye el pipeline de streaming con prefetching en background."""
        self._stream = build_worker_stream(
            worker_rank     = self.worker_rank,
            num_workers     = self.num_workers,
            batch_size      = self.batch_size,
            dataset_name    = self.dataset_name,
            image_size      = self.image_size,
            shuffle_buffer  = self.shuffle_buffer,
            prefetch_batches= self.prefetch_batches,
            seed            = 42 + self.worker_rank,
            hf_token        = self.hf_token,
        )
        self._stream.start()
        self._log(f"Stream iniciado: {self.dataset_name} | shard {self.worker_rank}/{self.num_workers}")

    # ================================================================
    # HANDSHAKE + LOOP PRINCIPAL
    # ================================================================

    def _handshake_loop(self) -> None:
        """
        Espera CNN_WEIGHTS, confirma CNN_ACK, espera START, entra en training loop.
        """
        assert self._sock is not None

        while True:
            msg = receive_message(self._sock)
            t = msg["type"]

            if t == MsgType.STOP:
                self._log("STOP recibido.")
                return

            elif t == MsgType.CNN_WEIGHTS:
                self._load_cnn(msg["payload"])

            elif t == MsgType.START:
                self._log("START recibido — iniciando loop de entrenamiento.")
                self._training_loop()
                return

    def _load_cnn(self, payload: dict) -> None:
        """Carga la CNN recibida del PS y confirma con CNN_ACK."""
        arch          = payload["arch"]
        weights_bytes = payload["weights_bytes"]
        assert self._sock is not None

        if self._cnn is None or self._cnn.arch != arch:
            self._cnn = CNNExtractor(arch=arch, device=str(self.device), seed=42)

        self._cnn.load_weights_from_bytes(weights_bytes)
        self._cnn._model.eval()

        self._log(f"CNN cargada: arch={arch}, feature_dim={self._cnn.feature_dim}")
        send_message(self._sock, MsgType.CNN_ACK, {"worker_id": self._worker_id})

    # ================================================================
    # LOOP DE ENTRENAMIENTO ASÍNCRONO
    # ================================================================

    def _training_loop(self) -> None:
        """
        Loop continuo: REQUEST_PARAMS → entrenar → UPDATES.

        No hay barrera con otros Workers. Cada iteración es completamente
        independiente. El PS aplica las actualizaciones inmediatamente.
        """
        assert self._sock is not None
        assert self._cnn is not None
        assert self._stream is not None

        stream_iter = iter(self._stream)
        version_read = 0

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

            payload      = msg["payload"]
            mlp_state    = payload["mlp_state"]     # Dict[str, np.ndarray] PyTorch keys
            cnn_state    = payload["cnn_state"]     # Dict[str, np.ndarray] state_dict
            version_read = payload["version"]
            lr           = payload["lr"]

            # ── 2. Sincronizar modelo local con el estado global ──
            self._sync_cnn(cnn_state)
            self._mlp = self._sync_mlp(mlp_state, self._mlp)

            # ── 3. Entrenar accum_steps batches ──
            total_loss, total_acc, total_n = 0.0, 0.0, 0

            for _ in range(self.accum_steps):
                try:
                    X_np, Y_np = next(stream_iter)
                except StopIteration:
                    assert self._stream is not None
                    stream_iter = iter(self._stream)
                    X_np, Y_np = next(stream_iter)

                loss, acc, n = self._train_batch(X_np, Y_np, lr)
                total_loss += loss * n
                total_acc  += acc  * n
                total_n    += n

            if total_n == 0:
                continue

            avg_loss = total_loss / total_n
            avg_acc  = total_acc  / total_n
            self._batches_done += self.accum_steps

            if self.verbose and self._batches_done % 10 == 0:
                assert self._stream is not None
                self._log(
                    f"batch={self._batches_done} | "
                    f"loss={avg_loss:.4f} | acc={avg_acc:.2f}% | "
                    f"v={version_read} | q={self._stream.queue_size}"
                )

            # ── 4. Enviar actualizaciones al PS ──
            try:
                send_message(self._sock, MsgType.UPDATES, {
                    "loss":         avg_loss,
                    "accuracy":     avg_acc,
                    "batch_size":   total_n,
                    "version_read": version_read,
                    "mlp_weights":  self._serialize_mlp(),
                    "cnn_weights":  self._serialize_cnn(),
                })
            except Exception as e:
                self._log(f"Error enviando UPDATES: {e}")
                return

    # ================================================================
    # FORWARD + BACKWARD (un batch)
    # ================================================================

    def _train_batch(
        self, X_np: np.ndarray, Y_np: np.ndarray, lr: float
    ) -> Tuple[float, float, int]:
        """
        Ejecuta un paso E2E completo: imagen → CNN → features → MLP → loss → backward.

        :return: (loss, accuracy_pct, n_samples)
        """
        assert self._cnn is not None
        assert self._mlp is not None
        
        X = torch.from_numpy(X_np).to(self.device)   # (N, 3, 224, 224)
        Y = torch.from_numpy(Y_np.astype(np.int64)).to(self.device)

        # Activar modo entrenamiento
        self._cnn._model.train()
        for p in self._cnn._model.parameters():
            p.requires_grad_(True)
        self._mlp.train()

        # Zero grad
        self._cnn._model.zero_grad()
        self._mlp.zero_grad()

        # Forward E2E
        features = self._cnn._model(X)           # (N, feature_dim)
        logits   = self._mlp(features)            # (N, 1000)
        loss_t   = nn.functional.cross_entropy(logits, Y)

        # Backward
        loss_t.backward()

        # SGD local (el PS promedia con FedAvg asíncrono)
        with torch.no_grad():
            for p in self._cnn._model.parameters():
                if p.grad is not None:
                    p.data -= lr * p.grad
            for p in self._mlp.parameters():
                if p.grad is not None:
                    p.data -= lr * p.grad

        # Métricas
        with torch.no_grad():
            correct = (logits.argmax(1) == Y).sum().item()

        n        = len(Y_np)
        loss_val = loss_t.item()
        acc_val  = 100.0 * correct / n

        # Restaurar CNN a eval
        self._cnn._model.eval()
        for p in self._cnn._model.parameters():
            p.requires_grad_(False)

        # Liberar tensores
        del X, Y, features, logits, loss_t
        return loss_val, acc_val, n

    # ================================================================
    # SINCRONIZACIÓN CNN + MLP
    # ================================================================

    def _sync_cnn(self, cnn_state: Dict[str, np.ndarray]) -> None:
        """
        Carga el estado global de la CNN (state_dict completo con BN buffers).
        Pone la CNN en eval() después para usar running stats de BatchNorm.
        """
        assert self._cnn is not None
        base = getattr(self._cnn._model, "model", self._cnn._model)
        sd   = base.state_dict()
        with torch.no_grad():
            for name, arr in cnn_state.items():
                if name in sd:
                    sd[name] = (
                        torch.from_numpy(arr)
                        .to(sd[name].device)
                        .to(sd[name].dtype)
                    )
            base.load_state_dict(sd)
        self._cnn._model.eval()

    def _sync_mlp(
        self,
        mlp_state: Dict[str, np.ndarray],
        existing: Optional[MLPPyTorch],
    ) -> MLPPyTorch:
        """
        Carga el estado global del MLP (formato PyTorch state_dict nativo).

        Si `existing` ya existe, reutiliza el objeto en lugar de crear uno nuevo.
        Esto evita la sobrecarga de construir un nn.Module cada iteración.
        """
        if existing is None:
            # Inferir dimensiones del state_dict recibido
            feature_dim = mlp_state["fc1.weight"].shape[1]
            hidden1     = mlp_state["fc1.weight"].shape[0]
            hidden2     = mlp_state["fc2.weight"].shape[0]
            existing    = MLPPyTorch(
                feature_dim, hidden1, hidden2, _IMAGENET_CLASSES
            ).to(self.device)

        with torch.no_grad():
            for name, param in existing.named_parameters():
                if name in mlp_state:
                    param.data.copy_(
                        torch.from_numpy(mlp_state[name]).to(param.device)
                    )
        return existing

    # ================================================================
    # SERIALIZACIÓN (modelo → numpy para transporte)
    # ================================================================

    def _serialize_cnn(self) -> Dict[str, np.ndarray]:
        """State_dict completo de la CNN (parámetros + BN buffers)."""
        assert self._cnn is not None
        base = getattr(self._cnn._model, "model", self._cnn._model)
        return {
            name: tensor.cpu().numpy().copy()
            for name, tensor in base.state_dict().items()
        }

    def _serialize_mlp(self) -> Dict[str, np.ndarray]:
        """State_dict del MLP en formato PyTorch nativo (fc1.weight, …)."""
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
            print(f"[W{wid}] {msg}")