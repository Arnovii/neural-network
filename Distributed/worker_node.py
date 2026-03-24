"""
Distributed/worker_node.py

Worker Node para el Algoritmo de Diego distribuido — ImageNet.

──────────────────────────────────────────────────────────────────
MODOS DE DATOS
──────────────────────────────────────────────────────────────────
Este Worker detecta automáticamente el origen del dataset:

  MODO LOCAL (rápido, offline)
  ─────────────────────────────
  Si data_dir contiene train/ y val/ → usa torchvision.ImageFolder.
  Requiere ~150 GB en disco pero no necesita internet en runtime.

  MODO STREAM (sin descarga previa)
  ───────────────────────────────────
  Si data_dir es None o no existe → usa HuggingFace Hub en streaming.
  Las imágenes llegan bajo demanda. Solo se guardan los features
  extraídos (~2.6 GB de shards .npy). Requiere token HuggingFace
  con acceso a ILSVRC/imagenet-1k.
  Después de la primera sesión, los shards .npy permiten entrenar
  completamente offline.

  DETECCIÓN: detect_data_source(data_dir) en imagenet_loader.py.

──────────────────────────────────────────────────────────────────
PIPELINE
──────────────────────────────────────────────────────────────────
  1. PS envía CNN_WEIGHTS (ResNet-18, ~44 MB)
  2. Worker extrae features por shards de 50k imágenes
     → guarda shard_XXXX_X.npy en cache_dir
  3. PS pide features de val con REQUEST_TEST_FEATURES
  4. Entrenamiento MLP epoch por epoch con carga selectiva de shards
"""

import os
import socket
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from Distributed.protocol import MsgType, receive_message, send_message
from Model.cnn_extractor import CNNExtractor, FEATURE_DIM
from Utils.imagenet_loader import (
    NUM_CLASSES, SHARD_SIZE,
    detect_data_source,
    get_imagenet_dataloader,
    get_imagenet_stream_dataloader,
    get_stream_shard_size,
    load_imagenet_labels,
    load_imagenet_labels_stream,
)
from Utils.feature_scaler import FeatureScaler


class WorkerNode:
    """
    Worker Node persistente para ImageNet con extracción de features por shards.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        server_host: str = "127.0.0.1",
        server_port: int = 9999,
        device: str = "cpu",
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        hf_token: str = "",
    ) -> None:
        self._data_dir    = data_dir
        # Inicializar worker_id antes de cualquier _log
        self.worker_id:   int  = -1
        self._server_host = server_host
        self._server_port = server_port
        self.verbose      = verbose
        self._hf_token    = hf_token or os.environ.get("HF_TOKEN", "")

        # Detectar origen del dataset
        self._data_source = detect_data_source(data_dir)

        # Cargar etiquetas de train en RAM (solo metadatos, sin imágenes).
        # Modo local:  lee desde disco (ms).
        # Modo stream: descarga solo la columna "label" de HF (~400 KB).
        if self._data_source == "local":
            # Modo local: leer etiquetas del disco (instantáneo)
            self._Y_raw   = load_imagenet_labels(
                split="train", data_dir=data_dir
            )
            self._n_train = len(self._Y_raw)
        else:
            # Modo stream: NO descargar 1.28M etiquetas ahora.
            # Las etiquetas quedan en los shards .npy tras _handle_cnn_weights.
            # Usar estimación oficial del tamaño del split.
            self._log("Modo streaming detectado. "
                      "Etiquetas de train se cargan tras extraer features.")
            self._Y_raw   = None
            self._n_train = get_stream_shard_size(
                split="train", num_shards=1, shard_index=0
            )
        self.Y_train  = self._Y_raw  # puede ser None en stream

        self._cnn = CNNExtractor(
            arch="resnet18",
            pretrained=True,
            device=device,
            seed=None,
            cache_dir=cache_dir,
            input_size=224,  # ImageNet nativo — sin upscale
        )

        self._sock:        Optional[socket.socket] = None
        self._scaler:      Optional[FeatureScaler] = None
        # En modo stream _Y_raw es None hasta que se extraen los shards.
        self._class_indices: List[np.ndarray] = (
            [np.where(self._Y_raw == c)[0] for c in range(NUM_CLASSES)]
            if self._Y_raw is not None else []
        )

        self._log(
            f"WorkerNode inicializado. "
            f"Dataset: {self._n_train} imgs, device={device}"
        )

    def run(self) -> None:
        """
        Conecta al PS, recibe el ID asignado y entra en el bucle
        persistente de espera de sesiones de entrenamiento.

        Bloquea hasta recibir STOP o hasta que la conexión se pierda.
        """
        self._connect()
        self._log(
            f"Conectado a {self._server_host}:{self._server_port}  "
            f"| ID={self.worker_id}"
        )
        self._log("Esperando sesión de entrenamiento del Parameter Server...")
        self._main_loop()
        self._disconnect()

    # ================================================================
    # CONEXIÓN
    # ================================================================


    def _connect(self) -> None:
        """
        Establece la conexión TCP y completa el handshake con el PS.

        Envía READY (sin ID) y espera WORKER_ID con el ID asignado.
        """

        # AF_INET = IPv4.
        # SOCK_STREAM = TCP
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self._server_host, self._server_port))

        # Handshake: el Worker no declara ID; el PS lo asigna
        send_message(self._sock, MsgType.READY, {})

        msg = receive_message(self._sock)
        if msg["type"] != MsgType.WORKER_ID:
            raise ConnectionError(f"Esperaba WORKER_ID, llegó: {msg['type']}")
        self.worker_id = msg["payload"]["worker_id"]

    def _disconnect(self) -> None:
        """Cierra la conexión TCP."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._log("Conexión cerrada.")

    # ================================================================
    # BUCLE PRINCIPAL
    # ================================================================


    def _main_loop(self) -> None:
        """
        Bucle principal: espera mensajes del PS y los despacha.
        Soporta múltiples sesiones de entrenamiento sin reconexión.
        """
        assert self._sock is not None
        while True:
            msg = receive_message(self._sock)

            if msg["type"] == MsgType.STOP:
                self._log("STOP recibido. Finalizando.")
                break

            elif msg["type"] == MsgType.REQUEST_TEST_FEATURES:
                # PS pide features de prueba explícitamente (un solo Worker)
                self._handle_request_test_features()

            elif msg["type"] == MsgType.TRAIN_SAMPLE:
                # Fallback: PS pide muestra de train para pretrain externo
                self._handle_train_sample(msg["payload"])

            elif msg["type"] == MsgType.CNN_WEIGHTS:
                self._handle_cnn_weights(msg["payload"])

            elif msg["type"] == MsgType.TRAIN_START:
                p = msg["payload"]
                self._log(
                    f"TRAIN_START — {p['epochs']} épocas  "
                    f"n_train={p['n_train']}  rank={p['worker_rank']}/{p['n_workers']}"
                )
                self._run_training_session(
                    p["epochs"], p["n_train"],
                    p["n_workers"], p["worker_rank"]
                )

    def _handle_request_test_features(self) -> None:
        """
        Extrae y envía features de prueba al PS bajo petición explícita.

        Solo un Worker recibe esta petición (el PS usa failover).
        La CNN ya está cargada desde _handle_cnn_weights.
        """
        self._log("PS solicitó features de prueba. Cargando val split...")
        if self._data_source == "local":
            loader = get_imagenet_dataloader(
                split="val",
                data_dir=self._data_dir,
                batch_size=self._optimal_batch_size(),
                num_workers=4,
            )
        else:
            self._log("Stream: descargando val desde HuggingFace...")
            loader = get_imagenet_stream_dataloader(
                split="val",
                token=self._hf_token,
                batch_size=self._optimal_batch_size(),
                shard_index=0,
                num_shards=1,
            )
        feats_list, labels_list = [], []
        self._cnn._model.eval()
        with torch.inference_mode():
            for imgs, labels in loader:
                imgs = imgs.to(self._cnn.device)
                feats_list.append(self._cnn._model(imgs).cpu().numpy())
                labels_list.append(labels.numpy().astype(np.int32))

        X_feat = np.concatenate(feats_list,  axis=0)
        Y_feat = np.concatenate(labels_list, axis=0)
        self._log(
            f"Features de prueba extraídos: {X_feat.shape} "
            f"({X_feat.nbytes // 1024 // 1024} MB). Enviando al PS..."
        )
        assert self._sock is not None
        send_message(
            self._sock,
            MsgType.TEST_FEATURES,
            {"X_test_features": X_feat, "Y_test": Y_feat},
        )
        self._log("Features de prueba enviados.")

    def _handle_train_sample(self, payload: dict) -> None:
        """
        Responde al PS con una muestra aleatoria de imagenes de train.
        Usado por el PS para pretrain de CNN simple. Envia imagenes RAW.
        Soporta modo local (ImageFolder) y modo stream (HuggingFace).
        """
        n_samples = min(payload.get("n_samples", 5000), self._n_train)
        rng       = np.random.RandomState(42)
        indices   = rng.choice(self._n_train, size=n_samples, replace=False)

        if self._data_source == "local":
            loader = get_imagenet_dataloader(
                split="train",
                data_dir=self._data_dir,
                batch_size=self._optimal_batch_size(),
                num_workers=4,
                indices=indices,
            )
        else:
            # Modo stream: pedir n_samples imagenes del stream
            loader = get_imagenet_stream_dataloader(
                split="train",
                token=self._hf_token,
                batch_size=self._optimal_batch_size(),
                shard_index=0,
                num_shards=1,
            )
        imgs_list, labels_list = [], []
        for imgs, labels in loader:
            imgs_list.append(imgs.numpy())
            labels_list.append(labels.numpy().astype(np.int32))

        X_sample = np.concatenate(imgs_list,   axis=0)
        Y_sample = np.concatenate(labels_list, axis=0)
        self._log(
            f"Enviando muestra de train al PS "
            f"({n_samples} imgs, {X_sample.nbytes // 1024 // 1024} MB)..."
        )
        assert self._sock is not None
        send_message(
            self._sock,
            MsgType.TRAIN_SAMPLE_DATA,
            {"X_sample": X_sample, "Y_sample": Y_sample},
        )
        self._log("Muestra de train enviada.")

    def _handle_cnn_weights(self, payload: dict) -> None:
        """
        Carga pesos CNN del PS y extrae features de ImageNet por shards.

        Flujo:
          1. Cargar pesos recibidos (reconstruir CNN si arch cambió)
          2. Contar shards ya en caché para este hash de pesos
          3. Extraer shards faltantes uno a uno, liberando RAM entre ellos
          4. Calcular/cargar FeatureScaler sobre shard 0
          5. Reconstruir índices por clase
          6. Enviar CNN_READY al PS
        """
        arch          = payload["arch"]
        weights_bytes = payload["weights_bytes"]
        self._log(f"CNN_WEIGHTS recibido (arch={arch}). Cargando pesos...")

        if self._cnn.arch != arch:
            self._log(f"Arquitectura cambió → reconstruyendo CNN...")
            self._cnn = CNNExtractor(
                arch=arch, device=str(self._cnn.device),
                seed=self._cnn.seed, cache_dir=self._cnn._cache_dir,
                input_size=224,
            )

        self._cnn.load_weights_from_bytes(weights_bytes)
        wh = self._cnn._weights_hash()
        self._log(f"Pesos cargados (hash={wh}).")

        # ── Extracción por shards ──────────────────────────────────
        n_shards = (self._n_train + SHARD_SIZE - 1) // SHARD_SIZE
        n_cached = self._cnn.count_cached_shards("train")

        if n_cached >= n_shards:
            self._log(f"Todos los shards ({n_shards}) ya en caché.")
        else:
            self._log(
                f"Extrayendo shards {n_cached}–{n_shards-1} "
                f"({self._n_train} imgs, {SHARD_SIZE}/shard)..."
            )
            for sid in range(n_cached, n_shards):
                start   = sid * SHARD_SIZE
                end     = min(start + SHARD_SIZE, self._n_train)
                indices = np.arange(start, end)
                self._log(f"  Shard {sid}/{n_shards-1}: imgs {start}–{end-1}...")

                loader = get_imagenet_dataloader(
                    split="train", data_dir=self._data_dir,
                    batch_size=self._optimal_batch_size(),
                    num_workers=4, indices=indices,
                )
                feats_list, labels_list = [], []
                self._cnn._model.eval()
                with torch.inference_mode():
                    for imgs, labels in loader:
                        imgs = imgs.to(self._cnn.device)
                        feats_list.append(self._cnn._model(imgs).cpu().numpy())
                        labels_list.append(labels.numpy().astype(np.int32))

                X_s = np.concatenate(feats_list,  axis=0)
                Y_s = np.concatenate(labels_list, axis=0)
                self._cnn.save_shard(sid, "train", X_s, Y_s)
                self._log(f"  Shard {sid} guardado ({X_s.nbytes // 1024 // 1024} MB)")
                del X_s, Y_s, feats_list, labels_list

        # ── FeatureScaler sobre shard 0 ────────────────────────────
        if not FeatureScaler.exists(self._cnn._cache_dir, wh):
            self._log("Calculando FeatureScaler sobre shard 0...")
            s0 = self._cnn.load_shard(0, "train")
            if s0 is not None:
                self._scaler = FeatureScaler().fit(s0[0])
                self._scaler.save(self._cnn._cache_dir, wh)
                del s0
                self._log("FeatureScaler guardado en caché.")
        else:
            self._scaler = FeatureScaler.load(self._cnn._cache_dir, wh)
            self._log("FeatureScaler cargado desde caché.")

        # Reconstruir índices por clase
        # Reconstruir _class_indices desde los shards ya cacheados
        # si estamos en modo stream y _Y_raw aún no está cargado.
        if self._Y_raw is None:
            all_labels: list = []
            n_shards = self._cnn.count_cached_shards("train")
            for sid in range(n_shards):
                shard = self._cnn.load_shard(sid, "train")
                if shard is not None:
                    all_labels.append(shard[1])
            if all_labels:
                self._Y_raw   = np.concatenate(all_labels)
                self.Y_train  = self._Y_raw
                self._n_train = len(self._Y_raw)
        self._class_indices = [
            np.where(self._Y_raw == c)[0] for c in range(NUM_CLASSES)
        ] if self._Y_raw is not None else []

        self._log("Features listos. Enviando CNN_READY al PS.")
        assert self._sock is not None
        send_message(self._sock, MsgType.CNN_READY, {"worker_id": self.worker_id})

    def _run_training_session(
        self,
        epochs: int,
        n_train: int,
        n_workers: int,
        worker_rank: int,
    ) -> None:
        """
        Bucle de épocas. Por cada época:
          1. Recibir PARAMS (pesos MLP + epoch_seed)
          2. Reconstruir índices localmente con epoch_seed
          3. Cargar features de los shards necesarios
          4. Aplicar FeatureScaler
          5. forward + backward MLP → enviar GRADIENTS
        """
        assert self._sock is not None

        for epoch in range(1, epochs + 1):
            msg = receive_message(self._sock)
            if msg["type"] == MsgType.STOP:
                self._log("STOP recibido durante entrenamiento.")
                raise SystemExit(0)
            if msg["type"] != MsgType.PARAMS:
                continue

            params     = msg["payload"]["params"]
            epoch_seed = msg["payload"]["seed"]

            indices = self._reconstruct_indices(
                epoch_seed, n_train, n_workers, worker_rank
            )

            X_epoch, Y_epoch = self._load_features_for_indices(indices)

            if self._scaler is not None:
                X_epoch = self._scaler.transform(X_epoch)

            from Model.mlp import forward_and_gradients
            gradients, loss, accuracy = forward_and_gradients(
                params, X_epoch, Y_epoch
            )

            send_message(
                self._sock,
                MsgType.GRADIENTS,
                {
                    "worker_id": self.worker_id,
                    "epoch":     epoch,
                    "gradients": gradients,
                    "loss":      loss,
                    "accuracy":  accuracy,
                },
            )


    def _get_train_loader(
        self,
        indices: np.ndarray,
        shard_idx: int,
        n_shards: int,
        start_in_shard: int = 0,
    ) -> DataLoader:
        """
        Devuelve un DataLoader para un rango de imágenes de train.

        Modo local:  usa get_imagenet_dataloader con índices concretos.
        Modo stream: usa get_imagenet_stream_dataloader con sharding HF.

        En modo stream el parámetro 'indices' se ignora — HuggingFace
        no admite indexación aleatoria. El sharding garantiza que cada
        Worker procesa una porción distinta del dataset.

        :param indices: Índices globales (usado en modo local).
        :param shard_idx: Índice del shard (0-based).
        :param n_shards: Total de shards del dataset.
        :param start_in_shard: Elemento de inicio (para reanudación en stream).
        :return: DataLoader iterable.
        """
        bs = self._optimal_batch_size()
        if self._data_source == "local":
            return get_imagenet_dataloader(
                split="train",
                data_dir=self._data_dir,
                batch_size=bs,
                num_workers=4,
                indices=indices,
            )
        else:
            # Stream: el shard HF corresponde al shard de features
            return get_imagenet_stream_dataloader(
                split="train",
                token=self._hf_token,
                batch_size=bs,
                shard_index=shard_idx,
                num_shards=n_shards,
                start_index=start_in_shard,
            )

    def _load_features_for_indices(
        self, indices: np.ndarray
    ) -> "tuple[np.ndarray, np.ndarray]":
        """
        Carga los features de los shards que cubren los índices dados.
        Solo carga los shards necesarios, liberando cada uno tras extraer
        las filas pedidas — nunca más de ~200 MB en RAM simultáneamente.
        """
        shard_ids = np.unique(indices // SHARD_SIZE)
        feat_parts: list  = []
        label_parts: list = []

        for sid in shard_ids:
            shard_data = self._cnn.load_shard(int(sid), "train")
            if shard_data is None:
                continue
            X_s, Y_s    = shard_data
            shard_start = int(sid) * SHARD_SIZE
            mask        = (indices >= shard_start) &                           (indices < shard_start + len(X_s))
            local_idx   = indices[mask] - shard_start
            feat_parts.append(X_s[local_idx])
            label_parts.append(Y_s[local_idx])
            del X_s, Y_s

        if not feat_parts:
            s0 = self._cnn.load_shard(0, "train")
            if s0 is not None:
                n = min(len(indices), len(s0[0]))
                return s0[0][:n], s0[1][:n]
            return (
                np.empty((0, FEATURE_DIM), np.float32),
                np.empty((0,), np.int32),
            )

        return (
            np.concatenate(feat_parts,  axis=0),
            np.concatenate(label_parts, axis=0),
        )

    def _optimal_batch_size(self, base: int = 256) -> int:
        """Batch size óptimo según dispositivo."""
        device_type = str(self._cnn.device).split(":")[0]
        if device_type == "cuda":
            return 512
        if device_type == "mps":
            return 256
        return base  # cpu

    def _reconstruct_indices(
        self,
        seed: int,
        n_train: int,
        n_workers: int,
        worker_rank: int,
    ) -> np.ndarray:
        """
        Reconstruye el chunk de índices de este Worker para una época.

        Aplica Round Robin estratificado por clase usando índices
        precalculados en ``__init__`` (``self._class_indices``).
        Esto evita ejecutar ``np.where`` en cada época.

        1. Para cada clase: shufflea una copia completa de sus índices
           antes de recortar, garantizando que todos los ejemplos del
           dataset puedan aparecer en cualquier época (crítico cuando
           ``n_train`` es pequeño).
        2. Recorta proporcionalmente a ``n_train``.
        3. Slicing Round Robin ``[worker_rank::n_workers]``.
        4. Concatena y shufflea el chunk resultante.

        El uso de la misma semilla garantiza que todos los Workers
        reproduzcan exactamente la misma asignación global y cada uno
        extraiga su propio chunk sin recibir ningún índice por red.

        :param seed: Semilla de época enviada por el PS.
        :type seed: int

        :param n_train: Total de ejemplos de entrenamiento.
        :type n_train: int

        :param n_workers: Número de Workers en la sesión.
        :type n_workers: int

        :param worker_rank: Posición de este Worker (0-based).
        :type worker_rank: int

        :return: Array de índices para este Worker en esta época.
        :rtype: np.ndarray
        """
        # Crea un generador de números aleatorios determinístico
        rng = np.random.RandomState(seed)

        # Proporción de n_train respecto al dataset completo.
        # Permite recortar cada clase proporcionalmente sin búsquedas.
        # Fallback para modo stream antes de tener _class_indices completos.
        if not self._class_indices or self.Y_train is None:
            rng2 = np.random.RandomState(seed)
            idx_all = rng2.permutation(self._n_train)[:n_train]
            return idx_all[worker_rank::n_workers]

        ratio = n_train / len(self.Y_train)

        # Aquí se guardarán los índices que le corresponden a este worker.
        my_indices = []
        for class_idx in self._class_indices:
            # Copia y shufflea la clase completa antes de recortar.
            # Garantiza que todos los ejemplos tengan posibilidad de
            # aparecer en cada época, incluso con n_train pequeño.
            shuffled = class_idx.copy()
            rng.shuffle(shuffled)

            # Define cuántos ejemplos de esta clase usar.
            # max(1, ...) evita que desaparezcan clases si n_train es pequeño.
            n_class = max(1, round(len(class_idx) * ratio))

            # Round Robin: este Worker toma 1 de cada n_workers elementos
            my_indices.append(shuffled[:n_class][worker_rank::n_workers])

        my_indices = np.concatenate(my_indices)
        rng.shuffle(my_indices)
        return my_indices

    # ================================================================
    # LOG
    # ================================================================


    def _log(self, msg: str) -> None:
        """Imprime un mensaje con el prefijo del Worker si verbose=True."""
        if self.verbose:
            wid = self.worker_id if self.worker_id is not None else "?"
            print(f"[W{wid}] {msg}")