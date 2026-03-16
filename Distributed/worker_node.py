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

import socket
import time
from typing import Any, Dict, List, Optional, Tuple

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
    :param cnn_arch: Arquitectura CNN: "simple" o "resnet18".
    :param cnn_pretrained: Cargar pesos ImageNet (solo resnet18).
    :param cnn_device: Dispositivo PyTorch: "cpu", "cuda", "mps".
    :param cnn_seed: Semilla para inicialización CNN (reproducibilidad entre Workers).
    :param hidden1: Neuronas capa oculta 1 del MLP.
    :param hidden2: Neuronas capa oculta 2 del MLP.
    :param cnn_batch_size: Batch size para extracción inicial de features.
    :param cnn_pretrain_epochs: Épocas de preentrenamiento CNN simple (0 = no pretrain).
    :param cnn_pretrain_lr: Learning rate para el preentrenamiento CNN.
    :param verbose: Imprime progreso por época.
    """

    def __init__(
        self,
        server_host: str,
        server_port: int,
        X_train: "np.ndarray",
        Y_train: "np.ndarray",
        cnn_arch: str = "simple",
        cnn_pretrained: bool = False,
        cnn_device: str = "cpu",
        cnn_seed: int | None = 42,
        hidden1: int = 256,
        hidden2: int = 128,
        cnn_batch_size: int = 2048,
        cnn_pretrain_epochs: int = 10,
        cnn_pretrain_lr: float = 1e-3,
        verbose: bool = True,
    ) -> None:
        self.server_host = server_host
        self.server_port = server_port
        self.Y_train = Y_train
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.verbose = verbose
        self.worker_id: Optional[int] = None
        self._sock: Optional[socket.socket] = None
        self._cnn_pretrain_epochs = cnn_pretrain_epochs
        self._cnn_pretrain_lr = cnn_pretrain_lr

        # ── Extractor CNN (pesos fijos) ───────────────────────────
        self._log("Construyendo extractor CNN...")
        self._cnn = CNNExtractor(
            arch=cnn_arch,
            pretrained=cnn_pretrained,
            device=cnn_device,
            seed=cnn_seed,
        )

        # Guardar los datos raw para poder re-extraer features cuando
        # el PS envíe una nueva CNN (mensaje CNN_WEIGHTS).
        self._X_raw: np.ndarray = X_train
        self._Y_raw: np.ndarray = Y_train

        # Extracción inicial con pesos locales (si los hay en caché).
        # Cuando llegue CNN_WEIGHTS del PS, se re-extraerán con la CNN definitiva.
        self._X_features: np.ndarray
        self._X_features, self.Y_train = self._cnn.prepare(
            X_train,
            Y_train,
            split="train",
            pretrain_epochs=cnn_pretrain_epochs,
            pretrain_lr=cnn_pretrain_lr,
            batch_size=cnn_batch_size,
            verbose=verbose,
        )

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

            if msg["type"] == MsgType.CNN_WEIGHTS:
                # El PS envía sus pesos CNN antes de TRAIN_START.
                # El Worker los carga, extrae sus features de train
                # con esa CNN, y confirma con CNN_READY.
                self._handle_cnn_weights(msg["payload"])

            elif msg["type"] == MsgType.TRAIN_START:
                p = msg["payload"]
                self._log(
                    f"TRAIN_START — {p['epochs']} épocas  "
                    f"n_train={p['n_train']}  rank={p['worker_rank']}/{p['n_workers']}"
                )
                self._run_training_session(
                    p["epochs"], p["n_train"], p["n_workers"], p["worker_rank"]
                )

    def _handle_cnn_weights(self, payload: dict) -> None:
        """
        Procesa CNN_WEIGHTS del PS: reconstruye la CNN si el arch cambió,
        carga los pesos, regenera features y confirma con CNN_READY.

        El Worker puede haber arrancado con arch=simple y recibir pesos
        de resnet18 (o viceversa) si el usuario cambió la arquitectura en
        el PS entre sesiones. En ese caso se reconstruye el modelo antes
        de cargar los pesos para evitar un RuntimeError de state_dict.
        """
        arch = payload["arch"]
        weights_bytes = payload["weights_bytes"]

        self._log(f"CNN_WEIGHTS recibido del PS (arch={arch}). Cargando pesos...")

        # Reconstruir el extractor si la arquitectura es diferente a la actual.
        # Es necesario porque load_state_dict falla si los pesos no coinciden
        # con la arquitectura del modelo — por ejemplo, ResNet-18 sobre _SimpleCNN.
        if self._cnn.arch != arch:
            self._log(
                f"Arquitectura cambió ({self._cnn.arch} → {arch}). "
                f"Reconstruyendo extractor CNN..."
            )
            self._cnn = CNNExtractor(
                arch=arch,
                device=str(self._cnn.device),
                seed=self._cnn.seed,
                cache_dir=self._cnn._cache_dir,
            )

        self._cnn.load_weights_from_bytes(weights_bytes)
        wh = self._cnn._weights_hash()
        self._log(f"Pesos cargados (hash={wh}). Preparando features de train...")

        # Regenerar features con la CNN del PS.
        # prepare() comprueba la caché local primero — si ya existe
        # {arch}_{wh}_train_50000_X.npy, la carga sin re-extraer.
        self._X_features, self.Y_train = self._cnn.prepare(
            self._X_raw,
            self._Y_raw,
            split="train",
            pretrain_epochs=0,  # pesos ya vienen del PS, no reentrenar
            batch_size=2048,
            verbose=self.verbose,
        )

        # Reconstruir índices por clase con el Y_train actualizado
        self._class_indices = [
            np.where(self.Y_train == digit)[0] for digit in range(10)
        ]

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
        Recibe PARAMS (pesos MLP + semilla), reconstruye los índices localmente
        a partir de la semilla, calcula gradientes y envía GRADIENTS.

        No hay forward CNN aquí: self._X_features ya tiene todos los features.
        Solo se indexan las filas correspondientes al chunk de este Worker.

        La partición es un Round Robin estratificado por clase (0-9)
        con la misma semilla, es decir, mismo resultado. Esto significa
        cero índices por red.

        :param payload:     Dict con ``epoch``, ``params``, ``seed``.
        :param n_train:     Total de ejemplos (recibido en TRAIN_START).
        :param n_workers:   Número de Workers en la sesión.
        :param worker_rank: Posición de este Worker (0-based).
        """
        epoch = payload["epoch"]
        params = payload["params"]
        seed = payload["seed"]

        indices = self._reconstruct_indices(seed, n_train, n_workers, worker_rank)
        self._log(f"Época {epoch} — {len(indices)} ejemplos")

        t_start = time.perf_counter()
        F_batch = self._X_features[indices]
        Y_batch = self.Y_train[indices]
        gradients, loss, accuracy = forward_and_gradients(params, F_batch, Y_batch)
        elapsed = time.perf_counter() - t_start

        self._log(f"  loss={loss:.4f}  acc={accuracy:.2f}%  ({elapsed:.3f}s)")

        assert self._sock is not None
        send_message(
            self._sock,
            MsgType.GRADIENTS,
            {
                "worker_id": self.worker_id,
                "epoch": epoch,
                "gradients": gradients,
                "loss": loss,
                "accuracy": accuracy,
            },
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
