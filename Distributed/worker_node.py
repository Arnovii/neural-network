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

from Distributed.protocol import MsgType, receive_message, send_message
from Model.cnn_extractor import CNNExtractor
from Model.mlp import forward_and_gradients


class WorkerNode:
    """
    Nodo Worker persistente para entrenamiento distribuido con CNN + MLP.

    :param server_host: IP del Parameter Server.
    :param server_port: Puerto TCP del Parameter Server.
    :param X_train: Imágenes (N, 3, 32, 32) float32 NCHW normalizadas.
    :param Y_train: Etiquetas (N,) int32.
    :param cnn_device: Dispositivo PyTorch: "cpu", "cuda", "mps".
    :param cnn_seed: Semilla para inicialización CNN (no crítica — el PS sobreescribirá los pesos).
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
        self.training_mode = training_mode  # "precomputed" o "end_to_end"
        # ── Extractor CNN — placeholder hasta recibir CNN_WEIGHTS del PS ────
        # El Worker arranca con una CNN mínima (simple) solo para tener
        # la estructura en memoria. Al recibir CNN_WEIGHTS del PS,
        # _handle_cnn_weights() la reconstruirá con la arquitectura
        # y pesos correctos. El usuario no necesita especificar arch.
        self._log(
            "Inicializando extractor CNN (pesos temporales, el PS los sobreescribirá)..."
        )
        self._cnn = CNNExtractor(
            arch="simple",  # placeholder — se reconstruye en _handle_cnn_weights
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
        """
        if self._X_test is None or self._Y_test is None:
            self._log("Sin datos de prueba — no puedo enviar TEST_FEATURES.")
            return
        self._log(
            f"PS solicitó features de prueba. "
            f"Extrayendo {len(self._X_test)} imgs con CNN actual..."
        )
        bs = self._optimal_batch_size()
        X_test_feat = self._cnn.extract_batched(
            self._X_test, batch_size=bs, verbose=self.verbose
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

        # ════════════════════════════════════════════════════════════════
        # RAMA 1: PRECOMPUTED — CNN CONGELADA, FEATURES CACHEADOS
        # ════════════════════════════════════════════════════════════════
        if self.training_mode == "precomputed":
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

            # [R1.2] Extraer y cachear features UNA SOLA VEZ
            # prepare() valida caché automáticamente por hash de pesos
            self._X_features, self.Y_train = self._cnn.prepare(
                self._X_raw,
                self._Y_raw,
                split="train",
                pretrain_epochs=0,  # CNN ya viene del PS, no preentrenar
                batch_size=optimal_bs,
                verbose=self.verbose,
            )

            # Reconstruir índices estratificados
            self._class_indices = [
                np.where(self.Y_train == digit)[0] for digit in range(10)
            ]

            self._log(
                f"[PRECOMPUTED] Features cacheados (shape={self._X_features.shape}). "
                f"CNN_READY ✓"
            )

        # ════════════════════════════════════════════════════════════════
        # RAMA 2: END-TO-END — CNN ENTRENABLE, SIN CACHEAR FEATURES
        # ════════════════════════════════════════════════════════════════
        else:  # end_to_end
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
        # CONFIRMACIÓN (mismo mensaje para ambas ramas, pero con diferentes estados)
        # ═══════════════════════════════════════════════════════════════
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
        Procesa PARAMS en una época: ejecuta forward/backward según training_mode.

        ╔════════════════════════════════════════════════════════════════╗
        ║ DOS FLUJOS MUTUAMENTE EXCLUYENTES SEGÚN training_mode         ║
        ╚════════════════════════════════════════════════════════════════╝

        PRECOMPUTED:
        ───────────
        - Recibe: PARAMS con {"epoch", "params" (MLP), "seed"}
        - Lo que NO recibe: "cnn_params" (invariante [R1.3])
        - Procesa: Features ya cacheados + MLP forward/backward
        - Envía: GRADIENTS con gradientes MLP, cnn_gradients=None
        - CNN nunca se actualiza (invariante [R1.1])

        END-TO-END:
        ───────────
        - Recibe: PARAMS con {"epoch", "params" (MLP), "seed", "cnn_params" (CNN)}
        - Si NO recibe "cnn_params": ERROR (invariante [R2.4])
        - Procesa: Raw images + CNN forward/backward + MLP forward/backward
        - Envía: GRADIENTS con gradientes MLP + CNN (ambos obligatorios)
        - CNN se actualiza cada época (invariante [R2.1])

        :param payload:     Dict con parámetros y configuración de época
        :param n_train:     Total de ejemplos (para reconstruir índices)
        :param n_workers:   Número de Workers en sesión
        :param worker_rank: Índice de este Worker (0-based)
        """
        import torch
        from Model.mlp import mlp_backward_to_input

        epoch = payload["epoch"]
        mlp_params = payload["params"]
        seed = payload["seed"]
        cnn_params = payload.get("cnn_params")  # None en precomputed

        indices = self._reconstruct_indices(seed, n_train, n_workers, worker_rank)
        self._log(f"Época {epoch} — {len(indices)} ejemplos")

        t_start = time.perf_counter()

        # ════════════════════════════════════════════════════════════════
        # RAMA 1: PRECOMPUTED — MLP DISTRIBUIDO, CNN FIJA
        # ════════════════════════════════════════════════════════════════
        if self.training_mode == "precomputed":
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
        # RAMA 2: END-TO-END — CNN + MLP CONJUNTAMENTE CON MINI-BATCHING
        # ════════════════════════════════════════════════════════════════
        else:  # end_to_end
            # Validación [R2.4]: DEBE haber cnn_params en E2E
            if cnn_params is None:
                raise RuntimeError(
                    "[VALIDACIÓN E2E] NO recibí cnn_params pero SON "
                    "obligatorios en end_to_end (invariante [R2.4])"
                )

            # ✓ CRÍTICO: Habilitar gradientes en la CNN para mode entrenamiento
            # [R2.1] La CNN fue inicializada con requires_grad=False (modo precomputed).
            # En end_to_end, necesitamos entrenar la CNN, así que:
            # 1. Cambiar modelo a train() — BatchNorm diferenciable, Dropout activo
            # 2. Habilitar requires_grad en todos los parámetros — computation graph activo
            self._cnn._model.train()
            for param in self._cnn._model.parameters():
                param.requires_grad_(True)

            # ─ MINI-BATCHING ─
            # Forward+backward de CNN es 2-3× más caro que solo forward.
            # Usar _optimal_batch_size() calculado para forward, dividir entre 2.5
            # para acomodar backward + acumulación sin congelamiento.
            base_opt_bs = self._optimal_batch_size()
            mini_bs = max(16, int(base_opt_bs / 2.5))
            n_total = len(indices)
            n_batches = (n_total + mini_bs - 1) // mini_bs

            self._log(
                f"[END-TO-END] Procesando {n_total} ejemplos en {n_batches} "
                f"mini-batches (size={mini_bs})..."
            )

            # Acumuladores para gradientes, loss, accuracy
            # Se promediarán al final de todos los mini-batches
            accumulated_mlp_grads: List[Dict[str, np.ndarray]] = []
            accumulated_cnn_grads: Dict[str, np.ndarray] = {}
            accumulated_losses: List[float] = []
            accumulated_accs: List[float] = []

            # ── LOOP DE MINI-BATCHES ──────────────────────────────────────
            for batch_idx in range(n_batches):
                start_idx = batch_idx * mini_bs
                end_idx = min(start_idx + mini_bs, n_total)
                mini_indices = indices[start_idx:end_idx]

                X_mini = self._X_raw[mini_indices]
                Y_mini = self._Y_raw[mini_indices]

                if batch_idx % max(1, n_batches // 5) == 0:  # Log cada 20%
                    self._log(
                        f"  [END-TO-END] Batch {batch_idx + 1}/{n_batches}  "
                        f"size={len(mini_indices)}"
                    )

                # ─ Forward CNN (PyTorch) en modo gradiente ─
                X_mini_torch = torch.from_numpy(X_mini).to(self._cnn.device)
                with torch.enable_grad():
                    features_torch = self._cnn._model(X_mini_torch)
                    features = features_torch.detach().cpu().numpy()

                # ─ Backward MLP para obtener ∇L/∂features ─
                dX_mini, mlp_grads_mini, loss_mini, acc_mini = mlp_backward_to_input(
                    mlp_params, features, Y_mini
                )

                accumulated_mlp_grads.append(mlp_grads_mini)
                accumulated_losses.append(loss_mini)
                accumulated_accs.append(acc_mini)

                # ─ Backward CNN usando proxy loss ─
                # Recalcular forward en modo gradiente para backward
                # (forward anterior fue .detach(), no propagaba gradientes)
                self._cnn._model.zero_grad()  # Limpiar gradientes previos

                X_mini_torch = torch.from_numpy(X_mini).to(self._cnn.device)

                # ✓ CRÍTICO: torch.enable_grad() para construir el computation graph
                # Sin esto, PyTorch no rastreará operaciones y .backward() fallará
                with torch.enable_grad():
                    features_torch = self._cnn._model(X_mini_torch)

                    # ━━━ PROXY LOSS ━━━
                    # dX_mini es ∂L/∂features, shape (N, D)
                    # features_torch es features, shape (N, D)
                    # Proxy loss = Σ(features_torch * dX_mini) / N
                    # ⚠️ NO transponer dX_mini — ya está en forma correcta
                    loss_proxy = (
                        features_torch
                        * torch.from_numpy(dX_mini / len(Y_mini)).to(self._cnn.device)
                    ).sum()

                    loss_proxy.backward()

                # ━━━ VALIDACIONES DE SHAPES ━━━
                # [DEBUG] Detectar mismatches temprano
                assert features_torch.shape[0] == len(Y_mini), (
                    f"[E2E] features batch size {features_torch.shape[0]} != Y size {len(Y_mini)}"
                )
                assert dX_mini.shape == (len(Y_mini), 512), (
                    f"[E2E] dX_mini shape {dX_mini.shape} != expected ({len(Y_mini)}, 512)"
                )
                assert features_torch.shape == dX_mini.shape, (
                    f"[E2E] features_torch {features_torch.shape} != dX_mini {dX_mini.shape}"
                )

                # Extraer y acumular gradientes CNN (SIN normalizar aquí)
                # [R2.1] CNN se actualiza acumulando gradientes de mini-batches
                # Se normalizarán al final por n_total
                for name, param in self._cnn._model.named_parameters():
                    if param.grad is not None:
                        grad_np = param.grad.detach().cpu().numpy()
                        if name not in accumulated_cnn_grads:
                            accumulated_cnn_grads[name] = grad_np.copy()
                        else:
                            accumulated_cnn_grads[name] += grad_np

            # ── PROMEDIADO FINAL ──────────────────────────────────────────
            # Promediar gradientes MLP
            gradients = {}
            for key in accumulated_mlp_grads[0].keys():
                gradients[key] = np.mean(
                    [g[key] for g in accumulated_mlp_grads], axis=0
                )

            # Normalizar gradientes CNN por número TOTAL de ejemplos (no por n_batches)
            # Esto es matemáticamente correcto: suma(gradientes) / n_total
            cnn_gradients = {
                name: grad / n_total for name, grad in accumulated_cnn_grads.items()
            }

            # Promediar loss y accuracy
            loss = float(np.mean(accumulated_losses))
            accuracy = float(np.mean(accumulated_accs))

            # ✓ RESTAURAR CNN a su estado inicial (eval mode, requires_grad=False)
            # Esto garantiza que:
            # 1. En la próxima época precomputed, we no entrenamos la CNN (invariante [R1.3])
            # 2. Las operaciones forward futuras son eficientes (eval mode)
            # 3. El estado es reproducible entre worker/PS
            self._cnn._model.eval()
            for param in self._cnn._model.parameters():
                param.requires_grad_(False)

        elapsed = time.perf_counter() - t_start
        self._log(f"  loss={loss:.4f}  acc={accuracy:.2f}%  ({elapsed:.3f}s)")

        # ════════════════════════════════════════════════════════════════
        # ENVIAR GRADIENTES (distinto según rama)
        # ════════════════════════════════════════════════════════════════
        payload_send = {
            "worker_id": self.worker_id,
            "epoch": epoch,
            "gradients": gradients,
            "loss": loss,
            "accuracy": accuracy,
        }

        # Incluir cnn_gradients solo si no es None (E2E)
        if cnn_gradients is not None:
            # [R2.4] En E2E, SIEMPRE incluir cnn_gradients
            payload_send["cnn_gradients"] = cnn_gradients
        else:
            # [R1.3] En precomputed, nunca incluir cnn_gradients (None es la marca)
            # El PS espera que NO exista la clave o sea None
            pass

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
