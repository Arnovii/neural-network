"""
Distributed/worker_node.py

Implementación del Worker para el algoritmo de Diego distribuido.

──────────────────────────────────────────────────────────────────
ROL DEL WORKER
──────────────────────────────────────────────────────────────────
Cada Worker es una máquina independiente que:

    1. Tiene una copia local de MNIST (descargada automáticamente).
    2. Se conecta al Parameter Server y envía READY.
    3. Espera el mensaje PARAMS con los pesos globales y los índices
       de los ejemplos que debe procesar en esta época.
    4. Extrae su batch usando los índices recibidos.
    5. Realiza el forward pass y el backward pass sobre ese batch.
    6. Envía los gradientes calculados al Parameter Server (PUSH).
    7. Vuelve al paso 3 para la siguiente época.
    8. Al recibir STOP, cierra la conexión limpiamente.

──────────────────────────────────────────────────────────────────
POR QUÉ ÍNDICES Y NO DATOS
──────────────────────────────────────────────────────────────────
Enviar los datos de entrenamiento por red en cada época sería muy
costoso: 60 000 imágenes × 784 floats × 4 bytes ≈ 188 MB por época.

En cambio, enviar los índices cuesta < 1 KB. Cada Worker tiene MNIST
localmente y puede extraer su batch en microsegundos a partir de los
índices recibidos.

──────────────────────────────────────────────────────────────────
CÁLCULO DE GRADIENTES
──────────────────────────────────────────────────────────────────
El Worker NO actualiza sus pesos. Solo calcula gradientes y los
envía. La actualización de pesos la hace exclusivamente el PS.

Forward pass:
    Z1 = W1 @ X.T + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

Backward pass:
    δ2 = A2 − Y_onehot
    dW2 = (1/n) * δ2 @ A1.T
    db2 = (1/n) * sum(δ2, axis=1)
    δ1 = (W2.T @ δ2) * sigmoid'(A1)
    dW1 = (1/n) * δ1 @ X
    db1 = (1/n) * sum(δ1, axis=1)
"""

import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Distributed.protocol import MsgType, receive_message, send_message


class WorkerNode:
    """
    Nodo Worker para el entrenamiento distribuido.

    Conecta con el Parameter Server, recibe parámetros e índices,
    calcula gradientes sobre su batch y los devuelve al servidor.

    :param worker_id: Identificador único de este Worker (entero ≥ 0).
    :type worker_id: int

    :param server_host: Dirección IP del Parameter Server.
    :type server_host: str

    :param server_port: Puerto TCP del Parameter Server.
    :type server_port: int

    :param X_train: Dataset completo de entrenamiento, forma (N, 784).
    :type X_train: np.ndarray

    :param Y_train: Etiquetas de entrenamiento, forma (N,).
    :type Y_train: np.ndarray

    :param input_size: Tamaño de entrada de la red (784 para MNIST).
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
        worker_id: int,
        server_host: str,
        server_port: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        input_size: int  = 784,
        hidden_size: int = 30,
        output_size: int = 10,
        verbose: bool    = True,
    ) -> None:
        self.worker_id   = worker_id
        self.server_host = server_host
        self.server_port = server_port
        self.X_train     = X_train
        self.Y_train     = Y_train
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.verbose     = verbose

        self._sock: Optional[socket.socket] = None

    # ================================================================
    # PUNTO DE ENTRADA PRINCIPAL
    # ================================================================

    def run(self) -> None:
        """
        Conecta al Parameter Server y ejecuta el loop de entrenamiento.

        Bloquea hasta recibir la señal STOP del servidor.
        """
        self._connect()

        if self.verbose:
            print(f"[W{self.worker_id}] Conectado a {self.server_host}:{self.server_port}")
            print(f"[W{self.worker_id}] Dataset local: {len(self.X_train)} ejemplos")
            print(f"[W{self.worker_id}] Esperando instrucciones del Parameter Server...")

        self._training_loop()
        self._disconnect()

    # ================================================================
    # CONEXIÓN
    # ================================================================

    def _connect(self) -> None:
        """
        Establece la conexión TCP con el Parameter Server y envía READY.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.server_host, self.server_port))
        send_message(self._sock, MsgType.READY, {"worker_id": self.worker_id})

    def _disconnect(self) -> None:
        """
        Cierra la conexión TCP con el Parameter Server.
        """
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

        if self.verbose:
            print(f"[W{self.worker_id}] Conexión cerrada. Worker finalizado.")

    # ================================================================
    # LOOP PRINCIPAL
    # ================================================================

    def _training_loop(self) -> None:
        """
        Espera mensajes del Parameter Server y responde con gradientes.

        El loop termina cuando recibe un mensaje STOP.
        """
        assert self._sock is not None, "Socket no inicializado"

        while True:
            msg = receive_message(self._sock)

            if msg["type"] == MsgType.STOP:
                if self.verbose:
                    print(f"[W{self.worker_id}] Señal STOP recibida. Finalizando.")
                break

            if msg["type"] == MsgType.PARAMS:
                self._handle_params(msg["payload"])

    # ================================================================
    # PROCESAMIENTO DE UNA ÉPOCA
    # ================================================================

    def _handle_params(self, payload: Dict[str, Any]) -> None:
        """
        Procesa un mensaje PARAMS: calcula gradientes y los envía.

        :param payload: Diccionario con ``epoch``, ``params``, ``indices``.
        :type payload: Dict[str, Any]
        """
        epoch   = payload["epoch"]
        params  = payload["params"]
        indices = payload["indices"]

        if self.verbose:
            print(
                f"[W{self.worker_id}] Época {epoch} — "
                f"batch {len(indices)} ejemplos"
            )

        t_start = time.perf_counter()

        # Extrae el batch local usando los índices recibidos
        X_batch = self.X_train[indices]
        Y_batch = self.Y_train[indices]

        # Forward + Backward — NO actualiza pesos, solo calcula gradientes
        gradients, loss, accuracy = self._compute_gradients(
            params, X_batch, Y_batch
        )

        elapsed = time.perf_counter() - t_start

        if self.verbose:
            print(
                f"[W{self.worker_id}]   loss={loss:.4f}  "
                f"acc={accuracy:.2f}%  ({elapsed:.3f}s)"
            )

        # Push de gradientes al Parameter Server
        assert self._sock is not None
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

    # ================================================================
    # FORWARD + BACKWARD (sin actualización de pesos)
    # ================================================================

    def _compute_gradients(
        self,
        params: Dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, float]:
        """
        Realiza el forward pass y el backward pass sobre el batch.

        No modifica los parámetros: solo calcula y devuelve gradientes.

        Forward pass:
            Z1 = W1 @ X.T + b1[:, newaxis]
            A1 = sigmoid(Z1)
            Z2 = W2 @ A1 + b2[:, newaxis]
            A2 = softmax(Z2)

        Backward pass:
            δ2 = A2 − Y_onehot
            dW2 = (1/n) * δ2 @ A1.T
            db2 = (1/n) * Σ δ2
            δ1 = (W2.T @ δ2) ⊙ σ'(A1)
            dW1 = (1/n) * δ1 @ X
            db1 = (1/n) * Σ δ1

        :param params: Parámetros globales actuales (W1, b1, W2, b2).
        :type params: Dict[str, np.ndarray]

        :param X: Batch de imágenes, forma ``(n, input_size)``.
        :type X: np.ndarray

        :param Y: Etiquetas del batch, forma ``(n,)``.
        :type Y: np.ndarray

        :return: Tupla ``(gradients, loss, accuracy)`` donde
                 ``gradients`` tiene claves dW1, db1, dW2, db2.
        :rtype: Tuple[Dict[str, np.ndarray], float, float]
        """
        W1, b1 = params["W1"], params["b1"]
        W2, b2 = params["W2"], params["b2"]
        n = len(X)

        # ── Forward pass ─────────────────────────────────────────────
        Z1 = W1 @ X.T + b1[:, np.newaxis]          # (hidden, n)
        A1 = self._sigmoid(Z1)                      # (hidden, n)

        Z2 = W2 @ A1 + b2[:, np.newaxis]            # (output, n)
        A2 = self._softmax(Z2)                       # (output, n)

        # ── Métricas ──────────────────────────────────────────────────
        predictions = np.argmax(A2, axis=0)          # (n,)
        correct     = int(np.sum(predictions == Y))
        log_probs   = np.log(np.clip(A2, 1e-15, 1.0))
        total_loss  = -float(np.sum(log_probs[Y, np.arange(n)]))

        # ── Backward pass ─────────────────────────────────────────────
        Y_onehot = np.zeros((self.output_size, n))
        Y_onehot[Y, np.arange(n)] = 1.0

        delta2 = A2 - Y_onehot                       # (output, n)

        dW2 = (1.0 / n) * (delta2 @ A1.T)           # (output, hidden)
        db2 = (1.0 / n) * np.sum(delta2, axis=1)    # (output,)

        delta1 = (W2.T @ delta2) * self._sigmoid_deriv(A1)  # (hidden, n)

        dW1 = (1.0 / n) * (delta1 @ X)              # (hidden, input)
        db1 = (1.0 / n) * np.sum(delta1, axis=1)    # (hidden,)

        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        loss      = total_loss / n
        accuracy  = 100.0 * correct / n

        return gradients, loss, accuracy

    # ================================================================
    # FUNCIONES DE ACTIVACIÓN (inline para no depender de Utils)
    # ================================================================

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """σ(z) = 1 / (1 + e^{-z}) con clipping para estabilidad numérica."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _sigmoid_deriv(a: np.ndarray) -> np.ndarray:
        """σ'(z) = a * (1 - a) a partir de la activación."""
        return a * (1.0 - a)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Softmax estabilizado sobre columnas de una matriz (output, n)."""
        z_stable = z - np.max(z, axis=0, keepdims=True)
        exp_z    = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
