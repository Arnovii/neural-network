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
"""

import socket
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from Distributed.protocol import MsgType, receive_message, send_message
from Utils.math_utils import sigmoid, sigmoid_derivative_from_activation, softmax


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
                self._log(
                    f"TRAIN_START recibido — {epochs} épocas  |  n_train={n_train}"
                )
                self._run_training_session(epochs)

    def _run_training_session(self, epochs: int) -> None:
        """
        Procesa todas las épocas de una sesión de entrenamiento.

        Por cada época: recibe PARAMS, calcula gradientes, envía GRADIENTS.
        Al terminar ``epochs`` épocas vuelve a _main_loop para esperar
        el siguiente TRAIN_START.

        :param epochs: Número de épocas en esta sesión.
        :type epochs: int
        """
        assert self._sock is not None

        for _ in range(epochs):
            msg = receive_message(self._sock)

            if msg["type"] == MsgType.STOP:
                # El PS puede apagarse incluso en medio de un entrenamiento
                self._log("STOP recibido durante entrenamiento. Finalizando.")
                raise SystemExit(0)

            if msg["type"] == MsgType.PARAMS:
                self._handle_params(msg["payload"])

    # ================================================================
    # PROCESAMIENTO DE UNA ÉPOCA
    # ================================================================

    def _handle_params(self, payload: Dict[str, Any]) -> None:
        """
        Procesa un mensaje PARAMS: calcula gradientes y los envía.

        :param payload: Dict con ``epoch``, ``params``, ``indices``.
        :type payload: Dict[str, Any]
        """
        epoch = payload["epoch"]
        params = payload["params"]
        indices = payload["indices"]

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
        W1, b1 = params["W1"], params["b1"]
        W2, b2 = params["W2"], params["b2"]
        num_imagenes = len(X)

        # Forward

        # Multiplica los pesos por cada entrada: W1 @ X_T -> (hidden_size, N),
        # b1 tiene forma (hidden_size,), una dimensión por debajo de W1 @ X_T
        # np.newaxis agrega una dimensión extra, convirtiendo (hidden_size,) en (hidden_size, 1)
        Z1 = W1 @ X.T + b1[:, np.newaxis]  # (hidden, N)

        A1 = sigmoid(Z1)  # (hidden, N)
        Z2 = W2 @ A1 + b2[:, np.newaxis]  # (output, N)
        A2 = softmax(Z2)  # (output, N)

        # Métricas

        # Escoge la clase más probable
        predictions = np.argmax(A2, axis=0)  # (N,)

        # Cuenta cuántas predicciones fueron correctas
        correct = int(np.sum(predictions == Y))

        # Calcula el error usando "cross-entropy"
        log_probs = np.log(np.clip(A2, 1e-15, 1.0))

        # log_probs[Y, np.arange(n)] significa:
        # Para cada muestra, toma el logaritmo de la probabilidad de su clase correcta
        total_loss = -float(np.sum(log_probs[Y, np.arange(num_imagenes)]))

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
            total_loss / num_imagenes,
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
