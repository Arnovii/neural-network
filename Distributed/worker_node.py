"""
Distributed/worker_node.py

Implementación del Worker para el algoritmo de Diego distribuido.

──────────────────────────────────────────────────────────────────
ROL DEL WORKER
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

──────────────────────────────────────────────────────────────────
POR QUÉ ÍNDICES Y NO DATOS
──────────────────────────────────────────────────────────────────
Enviar los datos de entrenamiento por red en cada época sería muy
costoso: 60 000 imágenes × 784 floats × 8 bytes ≈ 376 MB por época.
En cambio, enviar los índices cuesta < 1 KB. Cada Worker tiene MNIST
localmente y extrae su batch en microsegundos.

──────────────────────────────────────────────────────────────────
CÁLCULO DE GRADIENTES
──────────────────────────────────────────────────────────────────
El Worker NO actualiza sus pesos. Solo calcula gradientes y los
envía. La actualización θ ← θ − lr * ∇θ la hace exclusivamente el PS.

Forward:   Z1=W1@X.T+b1  ->  A1=σ(Z1)  ->  Z2=W2@A1+b2  ->  A2=softmax(Z2)
Backward:  δ2=A2−Y_hot  ->  dW2=(1/n)δ2@A1.T  ->  db2=(1/n)Σδ2
           δ1=(W2.T@δ2)⊙σ'(A1)  ->  dW1=(1/n)δ1@X  ->  db1=(1/n)Σδ1

Las funciones de activación se importan de Utils/math_utils.py,
compartidas con el resto del proyecto.
"""

import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Distributed.protocol import MsgType, receive_message, send_message
from Model.nn import cross_entropy_loss, forward_pass
from Utils.math_utils import sigmoid_derivative_from_activation


class WorkerNode:
    """
    Nodo Worker persistente para el entrenamiento distribuido.

    Se conecta al PS, recibe su ID asignado, y permanece activo
    esperando sesiones de entrenamiento hasta recibir STOP.

    :param server_host: Dirección IP del Parameter Server.
    :type server_host: str

    :param server_port: Puerto TCP del Parameter Server.
    :type server_port: int

    :param X_train: Dataset completo de entrenamiento, forma ``(N, 784)``.
    :type X_train: np.ndarray

    :param Y_train: Etiquetas de entrenamiento, forma ``(N,)``.
    :type Y_train: np.ndarray

    :param input_size: Neuronas de entrada (784 para MNIST).
    :type input_size: int

    :param hidden_size: Neuronas en la capa oculta.
    :type hidden_size: int

    :param output_size: Número de clases (10 para MNIST).
    :type output_size: int

    :param verbose: Si True imprime progreso por época.
    :type verbose: bool
    """

    def __init__(
        self,
        server_host: str,
        server_port: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        input_size: int = 784,
        hidden_size: int = 30,
        output_size: int = 10,
        verbose: bool = True,
    ) -> None:
        self.server_host = server_host
        self.server_port = server_port
        self.X_train = X_train
        self.Y_train = Y_train
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.verbose = verbose

        # Índices por clase precalculados una sola vez al arrancar.
        # _reconstruct_indices los reutiliza cada época sin recalcularlos.
        self._class_indices: List[np.ndarray] = [
            np.where(Y_train == digit)[0] for digit in range(10)
        ]

        # Asignado por el PS durante el handshake
        self.worker_id: Optional[int] = None

        self._sock: Optional[socket.socket] = None

    # ================================================================
    # PUNTO DE ENTRADA PRINCIPAL
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
            f"| ID asignado: {self.worker_id}  "
            f"| Dataset: {len(self.X_train)} ejemplos"
        )
        self._log("Esperando sesión de entrenamiento del Parameter Server...")
        self._main_loop()
        self._disconnect()

    # ================================================================
    # CONEXIÓN Y HANDSHAKE
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
            raise ConnectionError(f"Se esperaba WORKER_ID, llegó: {msg['type']}")
        self.worker_id = msg["payload"]["worker_id"]

    def _disconnect(self) -> None:
        """Cierra la conexión TCP."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._log("Conexión cerrada. Worker finalizado.")

    # ================================================================
    # BUCLE PRINCIPAL PERSISTENTE
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
                self._log("Señal STOP recibida. Finalizando.")
                break

            if msg["type"] == MsgType.TRAIN_START:
                epochs = msg["payload"]["epochs"]
                n_train = msg["payload"]["n_train"]
                n_workers = msg["payload"]["n_workers"]
                worker_rank = msg["payload"]["worker_rank"]
                self._log(
                    f"TRAIN_START recibido — "
                    f"{epochs} épocas  |  n_train={n_train}  "
                    f"|  rank={worker_rank}/{n_workers}"
                )
                self._run_training_session(epochs, n_train, n_workers, worker_rank)

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
        Procesa un mensaje PARAMS: reconstruye los índices localmente
        a partir de la semilla, calcula gradientes y los envía.

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

        self._log(f"Época {epoch} — batch {len(indices)} ejemplos")

        t_start = time.perf_counter()

        X_batch = self.X_train[indices]
        Y_batch = self.Y_train[indices]

        gradients, loss, accuracy = self._compute_gradients(params, X_batch, Y_batch)

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
    # FORWARD + BACKWARD
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

            # Define cuántos ejemplos de esta clase usar
            # max(1, ...) evita que desaparezcan clases si n_train es pequeño
            n_class = max(1, round(len(class_idx) * ratio))

            # Round Robin: este Worker toma 1 de cada n_workers elementos
            my_indices.append(shuffled[:n_class][worker_rank::n_workers])

        my_indices = np.concatenate(my_indices)
        rng.shuffle(my_indices)
        return my_indices

    def _compute_gradients(
        self,
        params: Dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, float]:
        """
        Forward pass y backward pass sobre el batch.

        No modifica los parámetros: devuelve gradientes, loss y accuracy.

        :param params: Parámetros globales (W1, b1, W2, b2).
        :type params: Dict[str, np.ndarray]

        :param X: Batch de imágenes ``(n, input_size)``.
        :type X: np.ndarray

        :param Y: Etiquetas del batch ``(n,)``.
        :type Y: np.ndarray

        :return: ``(gradients, loss, accuracy)``
        """
        W2 = params["W2"]
        num_imagenes = len(X)

        # Forward
        A1, A2 = forward_pass(params, X)

        # Métricas
        predictions = np.argmax(A2, axis=0)  # (N,)
        correct = int(np.sum(predictions == Y))
        mean_loss = cross_entropy_loss(A2, Y)

        # Backward
        Y_onehot = np.zeros((self.output_size, num_imagenes))
        Y_onehot[Y, np.arange(num_imagenes)] = 1.0

        delta2 = A2 - Y_onehot  # (output, N)
        dW2 = (1.0 / num_imagenes) * (delta2 @ A1.T)  # (output, hidden)
        db2 = (1.0 / num_imagenes) * np.sum(delta2, axis=1)  # (output,)

        delta1 = (W2.T @ delta2) * sigmoid_derivative_from_activation(A1)  # (hidden, N)
        dW1 = (1.0 / num_imagenes) * (delta1 @ X)  # (hidden, input)
        db1 = (1.0 / num_imagenes) * np.sum(delta1, axis=1)  # (hidden,)

        return (
            {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2},
            mean_loss,
            100.0 * correct / num_imagenes,
        )

    # ================================================================
    # LOG
    # ================================================================

    def _log(self, msg: str) -> None:
        """Imprime un mensaje con el prefijo del Worker si verbose=True."""
        if self.verbose:
            wid = self.worker_id if self.worker_id is not None else "?"
            print(f"[W{wid}] {msg}")
