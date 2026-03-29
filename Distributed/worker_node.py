"""
Distributed/worker_node.py

Implementación del Worker para el Algoritmo de Diego distribuido
con pipeline CNN (extractor) + MLP (clasificador).

──────────────────────────────────────────────────────────────────
ROL DEL WORKER EN LA PIPELINE CNN + MLP
──────────────────────────────────────────────────────────────────
Cada Worker es un proceso persistente que:

    1. Se conecta al Parameter Server enviando READY (sin ID propio).
    2. Recibe su ID asignado por el PS (mensaje WORKER_ID).
    3. Entra en un bucle de espera permanente:
         a. Espera TRAIN_START → comienza una sesión de entrenamiento.
         b. Por cada época: recibe PARAMS, calcula gradientes, envía
            GRADIENTS al PS.
         c. Al terminar todas las épocas, vuelve a esperar TRAIN_START.
    4. Al recibir STOP, cierra la conexión limpiamente.

El Worker nunca se desconecta entre sesiones de entrenamiento.
Permanece activo hasta que el PS envíe STOP o el proceso se
interrumpa manualmente.

El Worker ejecuta dos etapas en cada época:

    1. EXTRACCIÓN (CNN — PyTorch, pesos fijos):
       X_batch (N, 3, 32, 32) ──► CNN ──► features (N, feature_dim)
       La CNN no se entrena: sus pesos son idénticos en todos los
       Workers y no cambian durante el entrenamiento.

    2. FORWARD + BACKWARD (MLP — NumPy):
       features (N, feature_dim) ──► MLP ──► gradientes
       Solo los gradientes del MLP viajan por la red al PS.

Esta separación mantiene el Algoritmo de Diego intacto.

──────────────────────────────────────────────────────────────────
EXTRACCIÓN PREPROCESADA UNA SOLA VEZ
──────────────────────────────────────────────────────────────────
Al arrancar, el Worker extrae los features de las 50 000 imágenes
una sola vez y los almacena en self._X_features (50000, feature_dim).
En cada época solo se indexan las filas correspondientes al chunk.
Esto es correcto porque la CNN es fija: los features no cambian.

Coste único: ~50 000 forward passes CNN al arrancar (~segundos).
Coste por época: indexación + MLP forward/backward (puro NumPy).
"""

import os
import socket
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from Distributed.protocol import MsgType, receive_message, send_message
from Model.cnn_extractor import CNNExtractor
from Model.mlp import forward_and_gradients
from Utils.logging_util import get_logger

_logger = get_logger(use_colors=True)


class WorkerNode:
    """
    Nodo Worker persistente para entrenamiento distribuido con CNN + MLP.

    :param server_host: IP del Parameter Server.
    :param server_port: Puerto TCP del Parameter Server.
    :param X_train: Imágenes (N, 3, 32, 32) float32 NCHW normalizadas.
    :param Y_train: Etiquetas (N,) int32.
    :param cnn_device: Dispositivo PyTorch: "cpu", "cuda", "mps".
    :param cnn_seed: Semilla para inicialización CNN.
    :param hidden1: Neuronas capa oculta 1 del MLP.
    :param hidden2: Neuronas capa oculta 2 del MLP.
    :param cnn_batch_size: Batch size para extracción inicial de features.
    :param verbose: Imprime progreso por época.
    """

    def __init__(
        self,
        server_host: str,
        server_port: int,
        X_train: "np.ndarray",
        Y_train: "np.ndarray",
        X_test: "np.ndarray | None" = None,
        Y_test: "np.ndarray | None" = None,
        cnn_device: str = "cpu",
        cnn_seed: int | None = 42,
        hidden1: int = 256,
        hidden2: int = 128,
        cnn_batch_size: int = 2048,
        verbose: bool = True,
        training_mode: str = "precomputed",
    ) -> None:
        self.server_host = server_host
        self.server_port = server_port
        self.Y_train = Y_train
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.verbose = verbose
        self.worker_id: Optional[int] = None
        self._sock: Optional[socket.socket] = None
        self.training_mode = training_mode

        self._log(
            "Inicializando extractor CNN (pesos temporales, el PS los sobreescribirá)..."
        )
        self._cnn = CNNExtractor(
            arch="simple",
            device=cnn_device,
            seed=cnn_seed,
        )

        # Guardar los datos raw para poder re-extraer features cuando
        # el PS envíe una nueva CNN (mensaje CNN_WEIGHTS).
        self._X_raw: np.ndarray = X_train
        self._Y_raw: np.ndarray = Y_train
        self._X_test: "np.ndarray | None" = X_test
        self._Y_test: "np.ndarray | None" = Y_test

        # No extraemos features aquí — el PS enviará CNN_WEIGHTS con sus
        # pesos antes de TRAIN_START, y _handle_cnn_weights() hará la
        # extracción completa con la CNN correcta (con caché).
        self._X_features: np.ndarray = np.empty((0,), dtype=np.float32)

        # ── Índices por clase precalculados ───────────────────────
        # np.where se ejecuta una sola vez por clase al arrancar.
        # _reconstruct_indices los reutiliza cada época sin recalcularlos.
        self._class_indices: List[np.ndarray] = [
            np.where(Y_train == digit)[0] for digit in range(10)
        ]

    # ================================================================
    # PUNTO DE ENTRADA
    # ================================================================

    def run(self) -> None:
        """
        Conecta al PS, recibe el ID asignado y entra en el bucle
        persistente de espera de sesiones de entrenamiento.

        Bloquea hasta recibir STOP o hasta que la conexión se pierda.
        """
        self._connect()
        self._log(
            f"Conectado a {self.server_host}:{self.server_port}  "
            f"| ID={self.worker_id}  "
            f"| features={self._X_features.shape}  "
            f"| MLP hidden=({self.hidden1},{self.hidden2})"
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
        self._sock.connect((self.server_host, self.server_port))

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
        Bucle persistente que alterna entre:
            - Esperar TRAIN_START (nueva sesión de entrenamiento)
            - Procesar épocas de esa sesión (PARAMS → GRADIENTS)
            - Volver a esperar

        Sale cuando recibe STOP.
        """
        assert self._sock is not None

        while True:
            msg = receive_message(self._sock)

            if msg["type"] == MsgType.STOP:
                self._log("STOP recibido. Finalizando.")
                break

            if msg["type"] == MsgType.REQUEST_TEST_FEATURES:
                # El PS pide los features de prueba explícitamente.
                self._handle_request_test_features()

            elif msg["type"] == MsgType.TRAIN_SAMPLE:
                # El PS pide una muestra de imágenes de train.
                self._handle_train_sample(msg["payload"])

            elif msg["type"] == MsgType.CNN_WEIGHTS:
                # El PS envía sus pesos CNN antes de TRAIN_START.
                # El Worker los carga, extrae sus features de train
                # con esa CNN, y confirma con CNN_READY.
                self._handle_cnn_weights(msg["payload"])

            elif msg["type"] == MsgType.TRAIN_START:
                p = msg["payload"]

                # [INSTRUMENTACIÓN] Log del estado ANTES de sincronización
                training_mode_before = self.training_mode
                _logger.worker(
                    "[INSTRUM] RECIBIENDO TRAIN_START",
                    progress=f"training_mode_ANTES={training_mode_before} | "
                    f"payload_keys={list(p.keys())} | "
                    f"training_mode_EN_PAYLOAD={'PRESENTE' if 'training_mode' in p else 'AUSENTE'}",
                )

                # [SINCRONIZACIÓN] Actualizar training_mode desde PS
                if "training_mode" in p:
                    training_mode = p["training_mode"]
                    if training_mode not in ("precomputed", "end_to_end"):
                        raise RuntimeError(
                            f"[ERROR] training_mode inválido: {training_mode}. "
                            f"Debe ser 'precomputed' o 'end_to_end'."
                        )
                    if training_mode != self.training_mode:
                        self._log(
                            f"Sincronizando training_mode: {self.training_mode} → {training_mode}"
                        )
                    self.training_mode = training_mode
                else:
                    # Fallback para compatibilidad (PS viejo sin training_mode)
                    self._log(
                        "[ADVERTENCIA] TRAIN_START sin training_mode. Usando predeterminado."
                    )

                # [INSTRUMENTACIÓN] Log del estado DESPUÉS de sincronización
                _logger.worker(
                    "[INSTRUM] TRAIN_START PROCESADO",
                    progress=f"training_mode_DESPUÉS={self.training_mode} | "
                    f"epochs={p['epochs']} | n_train={p['n_train']} | "
                    f"worker_rank={p['worker_rank']}/{p['n_workers']}",
                )

                self._log(
                    f"TRAIN_START — {p['epochs']} épocas  "
                    f"n_train={p['n_train']}  rank={p['worker_rank']}/{p['n_workers']}  "
                    f"mode={self.training_mode}"
                )
                self._run_training_session(
                    p["epochs"], p["n_train"], p["n_workers"], p["worker_rank"]
                )

    def _optimal_batch_size(self) -> int:
        """
        Calcula el batch size óptimo mediante heurística adaptativa.

        Considera:
        - Arquitectura CNN (simple es más ligera que resnet18)
        - Dispositivo (CPU tiene restricciones severas)
        - Número de CPUs disponibles

        Dataset: CIFAR-10 (imágenes 32×32), no ImageNet.

        Heurística conservadora para evitar congelamiento:
        - CNN simple: 512-2048 (arquitectura ligera, más batches tolerables)
        - ResNet18: 64-256 (arquitectura pesada, batches muy pequeños)
        - CPU: reducción del 50% vs GPU (mucho más lenta)
        """
        device_type = str(self._cnn.device).split(":")[0]
        arch = self._cnn.arch
        n_cpus = os.cpu_count() or 1

        # Escalar según CPUs disponibles
        if n_cpus <= 2:
            cpu_factor = 1.0
        elif n_cpus <= 8:
            cpu_factor = 1.5
        else:
            cpu_factor = 2.0

        # Base según arquitectura CNN
        if arch == "resnet18":
            # ResNet-18 es muy profunda (18 capas convolucionales + upscale 32→224)
            # Batches pequeños incluso en CPU para CIFAR-10
            if device_type == "cpu":
                return max(32, min(128, int(64 * cpu_factor)))
            elif device_type == "cuda":
                return max(128, min(512, int(256 * cpu_factor)))
            elif device_type == "mps":
                return max(64, min(256, int(128 * cpu_factor)))
            else:
                return 128
        else:
            # CNN simple es más ligera (3 bloques convolucionales)
            # Permite batches más grandes
            if device_type == "cpu":
                return max(256, min(1024, int(512 * cpu_factor)))
            elif device_type == "cuda":
                return max(512, min(4096, int(2048 * cpu_factor)))
            elif device_type == "mps":
                return max(256, min(2048, int(1024 * cpu_factor)))
            else:
                return 512

    def _handle_request_test_features(self) -> None:
        """
        Responde al PS con los features de prueba extraídos con la CNN actual.

        El PS llama a este método (via REQUEST_TEST_FEATURES) después de la
        barrera CNN_READY, cuando ya sabe que este Worker tiene la CNN cargada.
        Solo un Worker recibe esta petición — el resto no hace nada.

        Features de test están cacheados con la misma estrategia que train:
        si el hash de pesos CNN no cambia, se reutiliza el caché (< 0.5s)
        en lugar de re-extraer (~ 5-10s).
        """
        if self._X_test is None or self._Y_test is None:
            self._log("Sin datos de prueba — no puedo enviar TEST_FEATURES.")
            return
        self._log(
            f"PS solicitó features de prueba ({len(self._X_test)} imgs). "
            f"Verificando caché..."
        )
        bs = self._optimal_batch_size()
        # Cargar features de test con caché inteligente
        # Si el hash de CNN es el mismo, evita re-extraer
        X_test_feat, _ = self._load_features_with_cache(
            self._X_test,
            self._Y_test,
            arch=self._cnn.arch,
            batch_size=bs,
            split="test",
        )
        self._log(
            f"Enviando features de prueba al PS "
            f"({X_test_feat.nbytes // 1024 // 1024} MB)..."
        )
        assert self._sock is not None
        send_message(
            self._sock,
            MsgType.TEST_FEATURES,
            {"X_test_features": X_test_feat, "Y_test": self._Y_test},
        )
        self._log("Features de prueba enviados al PS.")

    def _handle_train_sample(self, payload: dict) -> None:
        """
        Responde al PS con una muestra aleatoria de imágenes de train.

        El PS usa esta muestra para preentrenar la CNN sin necesidad
        de usar los datos de prueba, eliminando el sesgo de evaluación.
        Solo se envían imágenes raw (no features) para que el PS
        pueda preentrenar con distintas CNNs sin re-solicitar datos.
        """
        n_samples = payload.get("n_samples", 10000)
        n_samples = min(n_samples, len(self._X_raw))

        rng = np.random.RandomState(42)
        indices = rng.choice(len(self._X_raw), size=n_samples, replace=False)
        X_sample = self._X_raw[indices]
        Y_sample = self._Y_raw[indices]

        self._log(
            f"Enviando muestra de train al PS "
            f"({n_samples} imgs, "
            f"{X_sample.nbytes // 1024 // 1024} MB)..."
        )
        assert self._sock is not None
        send_message(
            self._sock,
            MsgType.TRAIN_SAMPLE_DATA,
            {"X_sample": X_sample, "Y_sample": Y_sample},
        )
        self._log("Muestra de train enviada al PS.")

    def _load_features_with_cache(
        self,
        X_raw: np.ndarray,
        Y_raw: np.ndarray,
        arch: str,
        batch_size: int,
        split: str = "train",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Carga features con caché inteligente basado en hash de CNN.

        Flujo:
        1. Calcula hash MD5 de pesos CNN actuales
        2. Define ruta de caché: Data/feature_cache/{arch}_{hash}_{split}_{X|Y}.npy
        3. Si archivo existe y es válido → [CACHE HIT] carga instantaneamente
        4. Si no existe o está corrupto → [CACHE MISS] extrae y guarda
        5. Valida shape según split:
           - "train": (50000, feature_dim)
           - "test":  (10000, feature_dim)

        Logging diferenciado:
        - [CACHE HIT][TRAIN]    / [CACHE HIT][TEST]    Carga desde disco
        - [CACHE MISS][TRAIN]   / [CACHE MISS][TEST]   Genera nuevo
        - [CACHE CORRUPT][TRAIN] / [CACHE CORRUPT][TEST] Corrupto, regenerando
        - [SHAPE INVALID]       Shape no coincide, regenerando

        :param X_raw: Imágenes de entrada (N, 3, 32, 32)
        :param Y_raw: Etiquetas (N,)
        :param arch: Arquitectura CNN
        :param batch_size: Batch size para extracción
        :param split: Tipo de split: "train" (50k) o "test" (10k), default="train"
        :return: Tupla (X_features, Y) donde X_features shape (N, feature_dim)
        """
        import os

        weights_hash = self._cnn._weights_hash()
        expected_feature_dim = self._cnn.feature_dim
        n_samples = len(X_raw)

        # Validar split válido
        if split not in ("train", "test"):
            raise ValueError(f"split debe ser 'train' o 'test', recibido: {split}")

        # Definir rutas de caché (mismo formato que cnn_extractor)
        cache_dir = os.path.join("Data", "feature_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Cache key incluye el split
        cache_key = f"{arch}_{weights_hash}_{split}"
        cache_X_path = os.path.join(cache_dir, f"{cache_key}_X.npy")
        cache_Y_path = os.path.join(cache_dir, f"{cache_key}_Y.npy")

        # ────────────────────────────────────────────────────────────
        # INTENTO 1: Cargar desde caché existente
        # ────────────────────────────────────────────────────────────
        if os.path.exists(cache_X_path) and os.path.exists(cache_Y_path):
            try:
                X_feat = np.load(cache_X_path, allow_pickle=False)
                Y_cached = np.load(cache_Y_path, allow_pickle=False)

                if X_feat.shape == (n_samples, expected_feature_dim):
                    split_upper = split.upper()
                    self._log(
                        f"[CACHE HIT][{split_upper}] Features cargados desde caché "
                        f"(hash={weights_hash}, shape={X_feat.shape}). {X_feat.nbytes // 1024 // 1024} MB."
                    )
                    return X_feat, Y_cached

                else:
                    # Shape inválido → regenerar
                    self._log(
                        f"[SHAPE INVALID][{split.upper()}] Caché tiene shape {X_feat.shape}, "
                        f"pero esperamos ({n_samples}, {expected_feature_dim}). "
                        f"Regenerando features..."
                    )

            except Exception as e:
                # Caché corrupto → regenerar
                self._log(
                    f"[CACHE CORRUPT][{split.upper()}] Error cargando caché: {e}. "
                    f"Regenerando features..."
                )

        # ────────────────────────────────────────────────────────────
        # INTENTO 2: Extraer nuevas features (CACHE MISS)
        # ────────────────────────────────────────────────────────────
        split_upper = split.upper()
        self._log(
            f"[CACHE MISS][{split_upper}] Extrayendo features con CNN "
            f"(arch={arch}, hash={weights_hash}, split={split}, N={n_samples})..."
        )

        t0 = time.perf_counter()
        X_feat = self._cnn.extract_batched(
            X_raw, batch_size=batch_size, verbose=self.verbose
        )
        elapsed = time.perf_counter() - t0

        # Validar shape después de extraer
        assert X_feat.shape == (n_samples, expected_feature_dim), (
            f"Shape inválido tras extracción: {X_feat.shape} vs esperado ({n_samples}, {expected_feature_dim})"
        )

        # Guardar en caché
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_X_path, X_feat)
            np.save(cache_Y_path, Y_raw)
            self._log(
                f"[CACHE SAVE][{split_upper}] Features guardados en caché: "
                f"{cache_key} "
                f"({X_feat.nbytes // 1024 // 1024} MB, {elapsed:.1f}s)"
            )
        except Exception as e:
            self._log(
                f"[CACHE SAVE ERROR][{split_upper}] No se guardó caché: {e}. Continuando..."
            )

        return X_feat, Y_raw

    def _handle_cnn_weights(self, payload: dict) -> None:
        """
        Procesa CNN_WEIGHTS del PS: reconstruye la CNN si el arch cambió,
        carga los pesos, regenera features (si precomputed) y confirma con CNN_READY.

        ╔════════════════════════════════════════════════════════════════╗
        ║ DOS FLUJOS MUTUAMENTE EXCLUYENTES SEGÚN training_mode         ║
        ╚════════════════════════════════════════════════════════════════╝

        PRECOMPUTED:
        ───────────
        - Congelar CNN (set_trainable=False)
        - Extraer y cachear features de todo el dataset AQUÍ (no en cada época)
        - Esto es correcto porque la CNN es fija: los features no cambian
        - Costo: ~50 000 forward passes CNN UNA SOLA VEZ (~30-60s)
        - Beneficio: Cada época cuesta solo ~1-5s en MLP (NumPy)

        END-TO-END:
        ───────────
        - Habilitar CNN (set_trainable=True)
        - NO extraer features precalculados
        - Guardar imágenes raw en memoria
        - Features se calculan dinámicamente en cada época forward
        - Costo setup: <1s (solo carga pesos)
        - Costo por época: ~5-30s (CNN forward/backward PyTorch + MLP)

        :param payload: Dict con ``arch`` y ``weights_bytes``
        """
        # [INSTRUMENTACIÓN] Log del estado ANTES de procesar CNN_WEIGHTS
        _logger.worker(
            "[INSTRUM] RECIBIENDO CNN_WEIGHTS",
            progress=f"training_mode={self.training_mode} | "
            f"payload_keys={list(payload.keys())} | "
            f"arch={payload.get('arch', 'N/A')} | "
            f"weights_size={len(payload.get('weights_bytes', b''))} bytes",
        )

        arch = payload["arch"]
        weights_bytes = payload["weights_bytes"]

        self._log(f"CNN_WEIGHTS recibido (arch={arch}). Cargando pesos...")

        # Reconstruir CNN si la arquitectura cambió
        if self._cnn.arch != arch:
            self._log(
                f"Arquitectura cambió ({self._cnn.arch} → {arch}). Reconstruyendo..."
            )
            self._cnn = CNNExtractor(
                arch=arch,
                device=str(self._cnn.device),
                seed=self._cnn.seed,
                cache_dir=self._cnn._cache_dir,
            )

        self._cnn.load_weights_from_bytes(weights_bytes)
        wh = self._cnn._weights_hash()

        # [INSTRUMENTACIÓN] Log ANTES de branch selection
        _logger.worker(
            "[INSTRUM] CNN_WEIGHTS CARGADO",
            progress=f"arch={arch} | weights_hash={wh} | "
            f"training_mode_AHORA={self.training_mode} | "
            f"BRANCH_SERÁ={'PRECOMPUTED' if self.training_mode == 'precomputed' else 'END-TO-END'}",
        )

        # ════════════════════════════════════════════════════════════════
        # RAMA 1: PRECOMPUTED — CNN CONGELADA, FEATURES CACHEADOS
        # ════════════════════════════════════════════════════════════════
        if self.training_mode == "precomputed":
            _logger.worker(
                "[INSTRUM] EJECUTANDO RAMA PRECOMPUTED",
                progress="action=freeze_cnn | action=extract_features",
            )

            self._log(
                f"[PRECOMPUTED] Congelando CNN y extrayendo features (hash={wh})..."
            )

            # [R1.2] Congelar CNN: no se entrenan gradientes
            self._cnn.set_trainable(False)

            # Calcular batch size óptimo
            optimal_bs = self._optimal_batch_size()
            n_cpus = os.cpu_count() or 1
            device_str = str(self._cnn.device)

            self._log(
                f"Batch size dinámico: {optimal_bs} "
                f"(arch={arch}, CPUs={n_cpus}, device={device_str})"
            )

            # [R1.2] Extraer features con caché inteligente
            # Si los pesos (hash) son iguales a una sesión anterior,
            # reutiliza las features del caché (< 0.5s) en lugar de re-extraer (30-60s)
            self._X_features, self.Y_train = self._load_features_with_cache(
                self._X_raw,
                self._Y_raw,
                arch,
                batch_size=optimal_bs,
            )

            # Reconstruir índices estratificados
            self._class_indices = [
                np.where(self.Y_train == digit)[0] for digit in range(10)
            ]

            self._log(
                f"[PRECOMPUTED] Features listos (shape={self._X_features.shape}). "
                f"CNN_READY ✓"
            )

        # ════════════════════════════════════════════════════════════════
        # RAMA 2: END-TO-END — CNN ENTRENABLE, SIN CACHEAR FEATURES
        # ════════════════════════════════════════════════════════════════
        else:  # end_to_end
            _logger.worker(
                "[INSTRUM] EJECUTANDO RAMA END-TO-END",
                progress="action=enable_cnn | action=skip_feature_extraction",
            )

            self._log(f"[END-TO-END] Habilitando CNN para entrenamiento (hash={wh})...")

            # [R2.1/R2.6] Habilitar CNN: entrenable con gradientes
            self._cnn.set_trainable(True)

            # [R2.3] NO extraer features: se calcularán cada época
            # Placeholder para mantener consistencia de atributos
            self._X_features = np.empty((0,), dtype=np.float32)
            self._Y_train = self._Y_raw.copy()

            # Índices estratificados para distribución de datos
            self._class_indices = [
                np.where(self.Y_train == digit)[0] for digit in range(10)
            ]

            self._log(
                "[END-TO-END] CNN entrenable. "
                "Features dinámicos (calculados por época). "
                "CNN_READY ✓"
            )

        # ═══════════════════════════════════════════════════════════════
        # CONFIRMACIÓN
        # ═══════════════════════════════════════════════════════════════
        # [INSTRUMENTACIÓN] Log ANTES de enviar CNN_READY
        cnn_has_grad = any(p.requires_grad for p in self._cnn._model.parameters())
        _logger.worker(
            "[INSTRUM] ENVIANDO CNN_READY",
            progress=f"training_mode_FINAL={self.training_mode} | "
            f"cnn_requires_grad={cnn_has_grad} | "
            f"features_shape={self._X_features.shape}",
        )

        assert self._sock is not None
        try:
            send_message(self._sock, MsgType.CNN_READY, {"worker_id": self.worker_id})
        except Exception as exc:
            self._log(f"ERROR enviando CNN_READY: {exc}")
            raise

        # Los features de prueba se envían solo cuando el PS
        # los solicita explícitamente con REQUEST_TEST_FEATURES.

    def _run_training_session(
        self,
        epochs: int,
        n_train: int,
        n_workers: int,
        worker_rank: int,
    ) -> None:
        """
        Procesa todas las épocas de una sesión de entrenamiento.

        Por cada época: recibe PARAMS (con semilla), reconstruye los índices
        localmente, calcula gradientes y envía GRADIENTS.
        Al terminar ``epochs`` épocas vuelve a _main_loop para esperar
        el siguiente TRAIN_START.

        :param epochs: Número de épocas en esta sesión.
        :param n_train: Total de ejemplos de entrenamiento (para estratificación).
        :param n_workers: Número de Workers en esta sesión.
        :param worker_rank: Posición de este Worker en la sesión (0-based).
        """
        assert self._sock is not None

        for _ in range(epochs):
            msg = receive_message(self._sock)

            if msg["type"] == MsgType.STOP:
                # Defensa ante un apagado forzado del PS (p.ej. desde ps_terminal.py
                # o si el PS falla). La GUI lo previene, pero el Worker no puede
                # asumir que siempre hay una GUI de por medio.
                self._log("STOP recibido durante entrenamiento. Finalizando.")
                raise SystemExit(0)

            if msg["type"] == MsgType.PARAMS:
                self._handle_params(msg["payload"], n_train, n_workers, worker_rank)

    # ================================================================
    # PROCESAMIENTO DE UNA ÉPOCA
    # ================================================================

    def _handle_params(
        self,
        payload: Dict[str, Any],
        n_train: int,
        n_workers: int,
        worker_rank: int,
    ) -> None:
        """
        Procesa PARAMS en una época.

        FIXES aplicados:
          [P3] LR efectivo = learning_rate / n_batches para evitar divergencia FedAvg.
          [P4] Métricas ponderadas por tamaño real de mini-batch.
          [P5] Pesos MLP enviados en formato PyTorch nativo (sin .T).
               El PS lee directamente en ese formato.
        """
        epoch = payload["epoch"]
        mlp_params = payload["params"]
        seed = payload["seed"]
        cnn_params = payload.get("cnn_params")

        # [INSTRUMENTACIÓN] Log del estado ANTES de branch selection
        _logger.worker(
            f"[INSTRUM] RECIBIENDO PARAMS | Época {epoch}",
            progress=f"training_mode={self.training_mode} | "
            f"cnn_params={'PRESENTE' if cnn_params is not None else 'AUSENTE'} | "
            f"payload_keys={list(payload.keys())} | "
            f"BRANCH_SERÁ={'PRECOMPUTED' if self.training_mode == 'precomputed' else 'END-TO-END'}",
        )

        indices = self._reconstruct_indices(seed, n_train, n_workers, worker_rank)
        self._log(f"Época {epoch} — {len(indices)} ejemplos")

        t_start = time.perf_counter()

        # Initialize variables for each training mode (ensures Pylance knows they're defined)
        gradients: dict | None = None
        cnn_gradients: dict | None = None
        updated_cnn_weights: dict = {}
        updated_mlp_weights: dict = {}

        # ════════════════════════════════════════════════════════════════
        # RAMA 1: PRECOMPUTED — MLP DISTRIBUIDO, CNN FIJA
        # ════════════════════════════════════════════════════════════════
        if self.training_mode == "precomputed":
            # [INSTRUMENTACIÓN] Log cuando entra a rama PRECOMPUTED
            cnn_has_grad = any(p.requires_grad for p in self._cnn._model.parameters())
            _logger.worker(
                f"[INSTRUM] EJECUTANDO RAMA PRECOMPUTED | Época {epoch}",
                progress=f"cnn_params_recibido={'SÍ (ERROR!)' if cnn_params is not None else 'NO (correcto)'} | "
                f"cnn_requires_grad={cnn_has_grad} | "
                f"features_shape={self._X_features.shape}",
            )

            # Validación [R1.3]: NO debe haber cnn_params en precomputed
            if cnn_params is not None:
                raise RuntimeError(
                    "[VALIDACIÓN PRECOMPUTED] Recibí cnn_params pero "
                    "NO debo recibirlos en precomputed (invariante [R1.3])"
                )

            self._log("[PRECOMPUTED] Forward/backward MLP...")

            # [R1.2] Features ya cacheados (extraídos en _handle_cnn_weights)
            F_batch = self._X_features[indices]
            Y_batch = self.Y_train[indices]

            # Forward MLP + backward MLP
            gradients, loss, accuracy = forward_and_gradients(
                mlp_params, F_batch, Y_batch
            )

            # [R1.3] NO hay gradientes CNN en precomputed
            cnn_gradients = None

        # ════════════════════════════════════════════════════════════════
        # RAMA 2: END-TO-END (FedAvg con LR efectivo ajustado)
        # ════════════════════════════════════════════════════════════════
        else:  # end_to_end
            from Model.mlp_pytorch import MLPPyTorch

            # [INSTRUMENTACIÓN]
            _logger.worker(
                f"[INSTRUM] EJECUTANDO RAMA END-TO-END | Época {epoch}",
                progress=f"cnn_params_recibido={'SÍ' if cnn_params is not None else 'NO'} | "
                f"training_mode={self.training_mode}",
            )

            # Validación: DEBE haber cnn_params en E2E
            if cnn_params is None:
                raise RuntimeError(
                    "[E2E] Recibí None para cnn_params pero son obligatorios en E2E"
                )

            # LR recibido del PS
            learning_rate = payload.get("learning_rate", 1e-3)

            # Inicializar MLP PyTorch con pesos del PS
            mlp = MLPPyTorch(
                feature_dim=self._cnn.feature_dim,
                hidden1=self.hidden1,
                hidden2=self.hidden2,
                n_classes=10,
            ).to(self._cnn.device)

            # ── [P5] Cargar pesos MLP en formato PyTorch NATIVO (sin .T) ──
            # El PS ahora envía los pesos en formato PyTorch (fc1.weight shape:
            # hidden1 × feature_dim). No se aplica ninguna transposición.
            mlp_state = {}

            W1_np = mlp_params["W1"]
            # W1 en NumPy MLP es (hidden1, feature_dim) — mismo que fc1.weight PyTorch
            if W1_np.shape == (self.hidden1, self._cnn.feature_dim):
                mlp_state["fc1.weight"] = torch.from_numpy(W1_np.copy())
            elif W1_np.shape == (self._cnn.feature_dim, self.hidden1):
                # Formato transpuesto heredado: corregir
                mlp_state["fc1.weight"] = torch.from_numpy(W1_np.T.copy())
            else:
                raise ValueError(f"W1 shape {W1_np.shape} invalida")

            mlp_state["fc1.bias"] = torch.from_numpy(mlp_params["b1"].copy())

            W2_np = mlp_params["W2"]
            # W2 en NumPy MLP es (hidden2, hidden1) — mismo que fc2.weight PyTorch
            if W2_np.shape == (self.hidden2, self.hidden1):
                mlp_state["fc2.weight"] = torch.from_numpy(W2_np.copy())
            elif W2_np.shape == (self.hidden1, self.hidden2):
                mlp_state["fc2.weight"] = torch.from_numpy(W2_np.T.copy())
            else:
                raise ValueError(f"W2 shape {W2_np.shape} invalida")

            mlp_state["fc2.bias"] = torch.from_numpy(mlp_params["b2"].copy())

            W3_np = mlp_params["W3"]
            # W3 en NumPy MLP es (n_classes, hidden2) — mismo que fc3.weight PyTorch
            if W3_np.shape == (10, self.hidden2):
                mlp_state["fc3.weight"] = torch.from_numpy(W3_np.copy())
            elif W3_np.shape == (self.hidden2, 10):
                mlp_state["fc3.weight"] = torch.from_numpy(W3_np.T.copy())
            else:
                raise ValueError(f"W3 shape {W3_np.shape} invalida")

            mlp_state["fc3.bias"] = torch.from_numpy(mlp_params["b3"].copy())

            for name, param in mlp.named_parameters():
                if name in mlp_state:
                    param.data = mlp_state[name].to(param.device).to(torch.float32)

            # ── Sincronizar CNN con pesos globales del PS ──
            # [P2] state_dict completo (parámetros + BN buffers)
            base_model = getattr(self._cnn._model, "model", self._cnn._model)
            with torch.no_grad():
                current_sd = base_model.state_dict()
                for name, arr in cnn_params.items():
                    if name in current_sd:
                        current_sd[name] = (
                            torch.from_numpy(arr)
                            .to(current_sd[name].device)
                            .to(current_sd[name].dtype)
                        )
                base_model.load_state_dict(current_sd)

            # Activar gradientes
            self._cnn._model.train()
            for param in self._cnn._model.parameters():
                param.requires_grad_(True)
            mlp.train()

            # ── Mini-batching adaptativo ──
            base_opt_bs = self._optimal_batch_size()
            mini_bs = max(16, int(base_opt_bs / 2.5))
            n_total = len(indices)
            n_batches = (n_total + mini_bs - 1) // mini_bs

            # ── [P3] LR efectivo ajustado por número de steps locales ──
            # En FedAvg cada Worker hace n_batches steps SGD locales.
            # El LR efectivo acumulado sería learning_rate * n_batches sin ajuste.
            # Dividir por n_batches mantiene la magnitud de actualización equivalente
            # a un solo step con todos los datos, evitando divergencia.
            effective_lr = learning_rate / max(1, n_batches)

            self._log(
                f"[E2E] Entrenamiento local: {n_total} ejemplos en {n_batches} "
                f"mini-batches (size={mini_bs}, lr={learning_rate:.2e}, "
                f"lr_efectivo={effective_lr:.2e})"
            )

            # ── [P4] Acumuladores ponderados por tamaño de batch ──
            total_loss_weighted = 0.0
            total_correct = 0
            total_samples = 0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * mini_bs
                end_idx = min(start_idx + mini_bs, n_total)
                mini_indices = indices[start_idx:end_idx]
                batch_size_actual = len(
                    mini_indices
                )  # puede ser < mini_bs en el último

                X_mini = self._X_raw[mini_indices].astype(np.float32)
                Y_mini = self._Y_raw[mini_indices].astype(np.int64)

                if batch_idx % max(1, n_batches // 5) == 0:
                    self._log(f"  Batch {batch_idx + 1}/{n_batches}")

                # Forward: X → CNN → Features → MLP → Logits → Loss
                X_mini_torch = torch.from_numpy(X_mini).to(self._cnn.device)
                Y_mini_torch = torch.from_numpy(Y_mini).to(self._cnn.device)

                # Forward: CNN → features → MLP → logits
                features = self._cnn._model(X_mini_torch)
                logits = mlp(features)

                # Loss
                loss_tensor = torch.nn.functional.cross_entropy(logits, Y_mini_torch)

                # Backward
                self._cnn._model.zero_grad()
                mlp.zero_grad()
                loss_tensor.backward()

                # [P3] Usar effective_lr en lugar de learning_rate
                with torch.no_grad():
                    for param in self._cnn._model.parameters():
                        if param.grad is not None:
                            param.data -= effective_lr * param.grad

                    for param in mlp.parameters():
                        if param.grad is not None:
                            param.data -= effective_lr * param.grad

                # [P4] Acumular ponderado por tamaño de batch
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    correct_count = (preds == Y_mini_torch).sum().item()

                total_loss_weighted += loss_tensor.item() * batch_size_actual
                total_correct += correct_count
                total_samples += batch_size_actual

            # [P4] Métricas correctamente ponderadas
            loss = total_loss_weighted / max(1, total_samples)
            accuracy = 100.0 * total_correct / max(1, total_samples)

            # ── [P2] Serializar CNN state_dict COMPLETO (params + BN buffers) ──
            # FIX P2: state_dict() incluye running_mean, running_var, num_batches_tracked
            # Esto garantiza que el PS pueda reconstruir exactamente el mismo estado BN
            # al promediar, eliminando la desincronización que causaba las oscilaciones.
            updated_cnn_weights = {}
            for name, tensor in base_model.state_dict().items():
                updated_cnn_weights[name] = tensor.cpu().numpy().copy()

            # ── [P5] Serializar pesos MLP en formato PyTorch NATIVO (sin .T) ──
            # El PS leerá fc1.weight como (hidden1, feature_dim) directamente.
            # Cero ambigüedad, cero riesgo de doble transposición.
            updated_mlp_weights = {}
            for name, param in mlp.named_parameters():
                updated_mlp_weights[name] = param.data.cpu().numpy().copy()

            # [P1] Restaurar CNN a eval mode tras entrenamiento
            self._cnn._model.eval()
            for param in self._cnn._model.parameters():
                param.requires_grad_(False)

        elapsed = time.perf_counter() - t_start
        self._log(f"  loss={loss:.4f}  acc={accuracy:.2f}%  ({elapsed:.3f}s)")

        # ════════════════════════════════════════════════════════════════
        # ENVIAR RESULTADOS (diferente según rama)
        # ════════════════════════════════════════════════════════════════
        payload_send = {
            "worker_id": self.worker_id,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "training_mode": self.training_mode,
        }

        if self.training_mode == "precomputed":
            # Precomputed: enviar gradientes MLP solamente
            payload_send["gradients"] = gradients
            payload_send["cnn_gradients"] = None
        else:
            # E2E: enviar pesos actualizados (FedAvg)
            payload_send["cnn_weights"] = updated_cnn_weights
            payload_send["mlp_weights"] = updated_mlp_weights

        assert self._sock is not None
        send_message(
            self._sock,
            MsgType.GRADIENTS,
            payload_send,
        )

    # ================================================================
    # RECONSTRUCCIÓN DE ÍNDICES
    # ================================================================

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
