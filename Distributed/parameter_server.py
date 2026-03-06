"""
Distributed/parameter_server.py

Implementación del Parameter Server para el algoritmo de Diego
distribuido con Data-Oriented Parallelism sobre sockets TCP.

──────────────────────────────────────────────────────────────────
CICLO DE VIDA
──────────────────────────────────────────────────────────────────
El PS tiene tres fases independientes:

    listen()   → Abre el socket y acepta Workers indefinidamente
                 en un hilo de fondo. Los Workers que se conectan
                 quedan registrados y esperan instrucciones.
                 El PS les asigna un ID secuencial (0, 1, 2, …).

    train()    → Ejecuta una sesión de entrenamiento con los Workers
                 actualmente conectados. Puede llamarse múltiples
                 veces sin reiniciar el servidor ni reconectar Workers.

    shutdown() → Envía STOP a todos los Workers, cierra conexiones
                 y detiene el hilo de aceptación.

──────────────────────────────────────────────────────────────────
ASIGNACIÓN DE IDs
──────────────────────────────────────────────────────────────────
El Worker ya no declara su propio ID. Al conectarse envía READY
con payload vacío y el PS responde con WORKER_ID asignando el
siguiente entero disponible (0, 1, 2, …). Esto evita colisiones
cuando varios Workers se conectan simultáneamente.

──────────────────────────────────────────────────────────────────
WORKERS PERSISTENTES
──────────────────────────────────────────────────────────────────
Los Workers no se desconectan al finalizar un entrenamiento.
Permanecen conectados esperando el siguiente TRAIN_START. El PS
puede lanzar múltiples sesiones de entrenamiento sin que los
Workers se reinicien.

──────────────────────────────────────────────────────────────────
CALLBACKS DISPONIBLES
──────────────────────────────────────────────────────────────────
on_worker_connected(worker_id, addr)
    Llamado cuando un Worker envía READY y queda registrado.

on_worker_disconnected(worker_id)
    Llamado cuando un Worker pierde la conexión inesperadamente.

on_gradients_received(worker_id, epoch, loss, accuracy)
    Llamado cada vez que se reciben los gradientes de un Worker.

on_epoch_end(epoch, total_epochs, train_accuracy, train_loss, test_accuracy, test_loss)
    Llamado tras promediar gradientes y actualizar pesos.
    ``test_accuracy`` y ``test_loss`` son None si no se pasaron datos de prueba.
"""

import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from Distributed.protocol import MsgType, receive_message, send_message
from Utils.math_utils import sigmoid, softmax


class ParameterServer:
    """
    Parameter Server persistente para entrenamiento distribuido.

    El servidor acepta conexiones continuamente hasta llamar a
    ``shutdown()``. Una sesión de entrenamiento se inicia con
    ``train()`` y puede repetirse sin reconectar Workers.

    :param host: Dirección IP en la que escucha el servidor.
    :type host: str

    :param port: Puerto TCP.
    :type port: int

    :param on_worker_connected: Callback cuando un Worker se registra.
                                Firma: ``(worker_id: int, addr: str)``
    :type on_worker_connected: Callable | None

    :param on_worker_disconnected: Callback cuando un Worker se pierde.
                                   Firma: ``(worker_id: int)``
    :type on_worker_disconnected: Callable | None

    :param on_gradients_received: Callback al recibir gradientes.
                                  Firma: ``(worker_id, epoch, loss, accuracy)``
    :type on_gradients_received: Callable | None

    :param on_epoch_end: Callback al final de cada época.
                           Firma: ``(epoch, total_epochs, train_accuracy, train_loss,
                           test_accuracy, test_loss)``.
                           ``test_accuracy`` y ``test_loss`` son None si no se
                           proporcionaron datos de prueba.
    :type on_epoch_end: Callable | None
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_worker_connected: Optional[Callable] = None,
        on_worker_disconnected: Optional[Callable] = None,
        on_gradients_received: Optional[Callable] = None,
        on_epoch_end: Optional[Callable] = None,
    ) -> None:
        self.host = host
        self.port = port

        self.on_worker_connected = on_worker_connected
        self.on_worker_disconnected = on_worker_disconnected
        self.on_gradients_received = on_gradients_received
        self.on_epoch_end = on_epoch_end

        # Sockets y metadatos de Workers activos
        self._worker_sockets: Dict[int, socket.socket] = {}
        self._worker_addrs: Dict[int, str] = {}
        self._next_id: int = 0  # Contador para asignar IDs
        self._lock = threading.Lock()  # Evita que múltiples hilos modifiquen las estructuras anteriores al mismo tiempo

        # Servidor TCP
        self._server_sock: Optional[socket.socket] = None  # Socket principal
        self._accept_thread: Optional[threading.Thread] = (
            None  # Hilo que acepta conexiones
        )
        self._shutdown_flag = threading.Event()  # Bandera para detener el servidor

        # Gradientes y métricas de la época actual (reutilizados por train)
        self._epoch_gradients: Dict[int, Dict[str, np.ndarray]] = {}
        self._epoch_metrics: Dict[int, Tuple[float, float]] = {}

    # ================================================================
    # CICLO DE VIDA DEL SERVIDOR
    # ================================================================

    def listen(self) -> None:
        """
        Abre el socket TCP y comienza a aceptar Workers en un hilo
        de fondo. Retorna inmediatamente; las conexiones se procesan
        de forma asíncrona.

        :raises RuntimeError: Si el servidor ya está escuchando.
        """
        if self._server_sock is not None:
            raise RuntimeError("El servidor ya está escuchando.")

        self._shutdown_flag.clear()

        # AF_INET = IPv4
        # SOCK_STREAM = TCP
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Permite reiniciar el servidor inmediatamente sin tener que esperar
        # a que el sistema operativo libere el puerto.
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))  # Asocia IP y puerto
        self._server_sock.listen(32)  # Permite hasta 32 conexiones en cola.
        # Timeout corto para que el hilo de aceptación pueda comprobar
        # el flag de apagado sin bloquearse indefinidamente en accept().
        self._server_sock.settimeout(1.0)

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        print(f"[PS] Escuchando en {self.host}:{self.port}")

    def shutdown(self) -> None:
        """
        Envía STOP a todos los Workers, cierra conexiones y detiene
        el hilo de aceptación.
        """
        print("[PS] Apagando servidor...")
        self._shutdown_flag.set()

        with self._lock:
            worker_ids = list(self._worker_sockets.keys())

        for wid in worker_ids:
            self._stop_worker(wid)

        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=3)  # Espera a que el hilo termine
            self._accept_thread = None

        print("[PS] Servidor apagado.")

    @property
    def connected_workers(self) -> List[int]:
        """IDs de los Workers actualmente conectados, ordenados."""
        with self._lock:
            return sorted(self._worker_sockets.keys())

    # ================================================================
    # HILO DE ACEPTACIÓN
    # ================================================================

    def _accept_loop(self) -> None:
        """
        Acepta conexiones entrantes indefinidamente hasta que se activa
        el flag de apagado.

        Por cada conexión: lee READY, asigna un ID, responde con
        WORKER_ID y llama al callback on_worker_connected.
        """
        while not self._shutdown_flag.is_set():
            if self._server_sock is None:
                break
            try:
                # conn = nuevo socket exclusivo para el Worker
                # addr = dirección IP y puerto del Worker
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            # Maneja el handshake en un hilo aparte para no bloquear accept()
            threading.Thread(
                target=self._handshake,
                args=(conn, addr),
                daemon=True,
            ).start()

    def _handshake(self, conn: socket.socket, addr: tuple) -> None:
        """
        Realiza el handshake inicial con un Worker que acaba de conectarse.

        Lee READY, asigna ID, responde con WORKER_ID y registra el Worker.
        Si el mensaje no es READY, cierra la conexión sin registrar nada.

        :param conn: Socket para comunicarse con ese Worker
        :type conn: socket.socket

        :param addr: Dirección del Worker (IP y puerto)
        :type addr: tuple
        """
        try:
            msg = receive_message(conn)
        except Exception:
            conn.close()
            return

        if msg["type"] != MsgType.READY:
            conn.close()
            return

        # Asigna ID único a Worker
        with self._lock:
            worker_id = self._next_id
            self._next_id += 1
            self._worker_sockets[worker_id] = conn
            self._worker_addrs[worker_id] = f"{addr[0]}:{addr[1]}"

        # Informa al Worker su ID asignado
        try:
            send_message(conn, MsgType.WORKER_ID, {"worker_id": worker_id})
        except Exception:
            with self._lock:
                self._worker_sockets.pop(worker_id, None)
                self._worker_addrs.pop(worker_id, None)
            conn.close()
            return

        addr_str = f"{addr[0]}:{addr[1]}"
        print(f"[PS] Worker {worker_id} conectado desde {addr_str}")

        if self.on_worker_connected is not None:
            self.on_worker_connected(worker_id, addr_str)

    # ================================================================
    # SESIÓN DE ENTRENAMIENTO
    # ================================================================

    def train(
        self,
        epochs: int,
        initial_params: Dict[str, np.ndarray],
        learning_rate: float,
        n_train: int,
        Y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        Y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Ejecuta una sesión de entrenamiento con los Workers conectados.

        Puede llamarse múltiples veces; cada llamada es independiente
        y comienza desde ``initial_params``.

        :param epochs: Número de épocas a entrenar.
        :type epochs: int

        :param initial_params: Pesos iniciales de la red (W1, b1, W2, b2).
        :type initial_params: Dict[str, np.ndarray]

        :param learning_rate: Tasa de aprendizaje.
        :type learning_rate: float

        :param n_train: Total de ejemplos de entrenamiento.
        :type n_train: int

        :param Y_train: Etiquetas de entrenamiento, forma ``(n_train,)``.
                        Se usa exclusivamente para la partición estratificada;
                        los datos nunca se envían por red.
        :type Y_train: np.ndarray

        :param X_test: Imágenes del conjunto de prueba, forma ``(N_test, 784)``.
                       Si se proporciona junto con ``Y_test``, el PS evaluará
                       el modelo global después de cada época.
        :type X_test: np.ndarray | None

        :param Y_test: Etiquetas del conjunto de prueba, forma ``(N_test,)``.
        :type Y_test: np.ndarray | None

        :return: Historial con ``"accuracies"``, ``"losses"``,
                 ``"test_accuracies"`` y ``"test_losses"`` por época.
                 Las listas de test están vacías si no se pasaron datos de prueba.
        :rtype: Dict[str, List[float]]

        :raises RuntimeError: Si no hay Workers conectados.
        """
        worker_ids = self.connected_workers
        if not worker_ids:
            raise RuntimeError(
                "No hay Workers conectados. "
                "Inicia al menos un Worker antes de entrenar."
            )

        # Crea una copia de los parámetros iniciales
        params = {tipo: datos.copy() for tipo, datos in initial_params.items()}

        history: Dict[str, List[float]] = {
            "accuracies": [],
            "losses": [],
            "test_accuracies": [],
            "test_losses": [],
        }

        print("=" * 70)
        print("PARAMETER SERVER — INICIANDO ENTRENAMIENTO")
        print("=" * 70)
        print(f"  Workers activos : {worker_ids}")
        print(f"  Épocas          : {epochs}")
        print(f"  Learning rate   : {learning_rate}")
        print(f"  Ejemplos train  : {n_train}")
        print("=" * 70)

        # Notifica a los Workers que va a comenzar una sesión
        self._broadcast(
            MsgType.TRAIN_START,
            {"epochs": epochs, "n_train": n_train},
            worker_ids,
        )

        for epoch in range(1, epochs + 1):
            print(f"[PS] ── Época {epoch}/{epochs} ──────────────────────────")
            t_start = time.perf_counter()

            # Limpia los gradientes anteriores
            self._epoch_gradients.clear()
            self._epoch_metrics.clear()

            index_chunks = self._split_indices(worker_ids, n_train, Y_train)

            done_event = threading.Event()

            # Se usa una lista con un solo elemento para poder modificar
            # ese valor desde varios hilos dentro de una función interna.
            received_count = [0]

            def _receive_from_worker(wid: int) -> None:
                try:
                    msg = receive_message(self._worker_sockets[wid])
                    if msg["type"] == MsgType.GRADIENTS:
                        payload = msg["payload"]
                        loss = payload["loss"]
                        accuracy = payload["accuracy"]

                        with self._lock:
                            self._epoch_gradients[wid] = payload["gradients"]
                            self._epoch_metrics[wid] = (loss, accuracy)
                            received_count[0] += 1
                            all_done = received_count[0] == len(worker_ids)

                        if self.on_gradients_received is not None:
                            self.on_gradients_received(wid, epoch, loss, accuracy)

                        if all_done:
                            done_event.set()

                except Exception as exc:
                    print(f"[PS] Error recibiendo de Worker {wid}: {exc}")
                    self._remove_worker(wid)
                    with self._lock:
                        received_count[0] += 1
                        if received_count[0] == len(worker_ids):
                            done_event.set()

            # Broadcast: params + índices
            for wid in worker_ids:
                try:
                    send_message(
                        self._worker_sockets[wid],
                        MsgType.PARAMS,
                        {
                            "epoch": epoch,
                            "params": params,
                            "indices": index_chunks[wid],
                        },
                    )
                except Exception as exc:
                    print(f"[PS] Error enviando a Worker {wid}: {exc}")
                    self._remove_worker(wid)

            # Lanza receptores
            threads = [
                threading.Thread(
                    target=_receive_from_worker,
                    args=(wid,),
                    daemon=True,
                )
                for wid in worker_ids
                if wid in self._worker_sockets
            ]
            for t in threads:
                t.start()

            # El PS se queda esperando hasta que todos
            # los Workers manden gradientes (o fallen).
            done_event.wait()
            for t in threads:
                t.join()

            if not self._epoch_gradients:
                print("[PS] Sin gradientes — todos los Workers fallaron.")
                break

            avg_grads = self._average_gradients(list(self._epoch_gradients.values()))
            self._apply_gradients(params, avg_grads, learning_rate)

            # Promedia métricas
            losses = [m[0] for m in self._epoch_metrics.values()]
            accuracies = [m[1] for m in self._epoch_metrics.values()]
            epoch_loss = float(np.mean(losses))
            epoch_acc = float(np.mean(accuracies))

            history["losses"].append(epoch_loss)
            history["accuracies"].append(epoch_acc)

            # Evalúa sobre datos de prueba (si se proporcionaron)
            test_acc: Optional[float] = None
            test_loss: Optional[float] = None
            if X_test is not None and Y_test is not None:
                test_acc, test_loss = self._evaluate(params, X_test, Y_test)
                history["test_accuracies"].append(test_acc)
                history["test_losses"].append(test_loss)

            elapsed = time.perf_counter() - t_start
            test_str = (
                f"  precisión_prueba={test_acc:.2f}%  pérdida_prueba={test_loss:.4f}"
                if test_acc is not None
                else ""
            )
            print(
                f"[PS]   precisión={epoch_acc:.2f}%  pérdida={epoch_loss:.4f}"
                f"{test_str}  ({elapsed:.2f}s)"
            )

            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, epochs, epoch_acc, epoch_loss, test_acc, test_loss)

        print("[PS] Entrenamiento completado.\n")
        return history

    # ================================================================
    # HELPERS INTERNOS
    # ================================================================

    def _broadcast(
        self,
        msg_type: MsgType,
        payload: Any,
        worker_ids: List[int],
    ) -> None:
        """
        Envía el mismo mensaje a todos los Workers indicados.

        :param msg_type: Tipo de mensaje
        :type msg_type: MsgType

        :param payload: Datos que se envían
        :type payload: Any

        :param worker_ids: Lista de IDs a los que se enviará
        :type worker_ids: List[int]
        """
        for wid in worker_ids:
            with self._lock:
                sock = self._worker_sockets.get(wid)
            if sock is None:
                continue
            try:
                send_message(sock, msg_type, payload)
            except Exception as exc:
                print(f"[PS] Error haciendo broadcast a Worker {wid}: {exc}")
                self._remove_worker(wid)

    def _stop_worker(self, worker_id: int) -> None:
        """Envía STOP y cierra el socket de un Worker."""
        with self._lock:
            sock = self._worker_sockets.pop(worker_id, None)
            self._worker_addrs.pop(worker_id, None)
        if sock is not None:
            try:
                send_message(sock, MsgType.STOP, None)
                sock.close()
            except Exception:
                pass

    def _remove_worker(self, worker_id: int) -> None:
        """Elimina un Worker que perdió la conexión, sin enviar STOP."""
        with self._lock:
            sock = self._worker_sockets.pop(worker_id, None)
            self._worker_addrs.pop(worker_id, None)
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if self.on_worker_disconnected is not None:
            self.on_worker_disconnected(worker_id)

    def _split_indices(
        self,
        worker_ids: List[int],
        n_train: int,
        Y_train: np.ndarray,
    ) -> Dict[int, List[int]]:
        """
        Divide los índices de entrenamiento en chunks disjuntos y
        estratificados, uno por Worker.

        Estratificación significa que cada Worker recibe aproximadamente
        la misma proporción de cada clase (dígito 0–9). Esto se logra
        con un reparto Round Robin por clase, igual que en
        ``data_partitioner.py``, pero devolviendo solo índices (no datos)
        para no romper el protocolo distribuido.

        Sin estratificación, una permutación aleatoria con pocos Workers
        puede producir batches desbalanceados, lo que sesga los gradientes
        de cada Worker aunque el promediado lo amortigüe parcialmente.

        :param worker_ids: IDs de los Workers conectados.
        :type worker_ids: List[int]

        :param n_train: Total de ejemplos de entrenamiento.
        :type n_train: int

        :param Y_train: Etiquetas ``(n_train,)``, solo para estratificar.
        :type Y_train: np.ndarray

        :return: Diccionario ``{worker_id: [índices]}``.
        """
        n = len(worker_ids)
        partitions: Dict[int, List[int]] = {wid: [] for wid in worker_ids}

        for digit in range(10):
            # Índices de todos los ejemplos de esta clase
            class_indices = np.where(Y_train[:n_train] == digit)[0].tolist()
            np.random.shuffle(class_indices)

            # Round Robin: reparte los índices de esta clase entre Workers
            for i, idx in enumerate(class_indices):
                partitions[worker_ids[i % n]].append(idx)

        # Mezcla dentro de cada partición para que no quede ordenada por clase
        for wid in worker_ids:
            np.random.shuffle(partitions[wid])

        return partitions

    def _average_gradients(
        self, gradients_list: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Calcula el promedio elemento a elemento de los gradientes
        enviados por múltiples Workers.

        Implementa la operación:

            ∇θ = (1/N) * Σᵢ ∇θᵢ

        donde N es el número de Workers y ∇θᵢ representa el gradiente
        calculado localmente por el Worker i sobre su subconjunto de datos.

        Se asume que todos los diccionarios de ``gradients_list`` contienen
        exactamente las mismas claves.

        :param gradients_list: Lista de diccionarios de gradientes,
                            uno por Worker. Cada diccionario debe
                            contener las mismas claves (por ejemplo,
                            ``"dW1"``, ``"db1"``, etc.) y valores de tipo
                            ``np.ndarray``.
        :type gradients_list: List[Dict[str, np.ndarray]]

        :return: Diccionario con los gradientes promediados para cada
                parámetro del modelo.
        :rtype: Dict[str, np.ndarray]

        :raises ValueError: Si ``gradients_list`` está vacío.
        """
        averaged: Dict[str, np.ndarray] = {}
        for key in gradients_list[0]:
            stacked = np.array([g[key] for g in gradients_list])
            averaged[key] = np.mean(stacked, axis=0)
        return averaged

    def _apply_gradients(
        self,
        params: Dict[str, np.ndarray],
        gradients: Dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        """
        Actualiza los parámetros del modelo  usando los gradientes proporcionados.

        Implementa la regla de actualización:

            θ ← θ − lr * ∇θ

        donde:
            θ   = parámetros del modelo
            lr  = learning rate
            ∇θ  = gradientes promediados

        La actualización se realiza in-place sobre el diccionario ``params``.

        :param params: Diccionario de parámetros del modelo a actualizar
                    (por ejemplo, ``"W1"``, ``"b1"``, ``"W2"``, ``"b2"``).
        :type params: Dict[str, np.ndarray]

        :param gradients: Diccionario con los gradientes correspondientes
                        a cada parámetro (por ejemplo, ``"dW1"``,
                        ``"db1"``, etc.).
        :type gradients: Dict[str, np.ndarray]

        :param learning_rate: Tasa de aprendizaje utilizada para escalar
                            el gradiente antes de aplicarlo.
        :type learning_rate: float

        :return: None
        :rtype: None
        """
        params["W1"] -= learning_rate * gradients["dW1"]
        params["b1"] -= learning_rate * gradients["db1"]
        params["W2"] -= learning_rate * gradients["dW2"]
        params["b2"] -= learning_rate * gradients["db2"]

    def _evaluate(
        self,
        params: Dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Hace Forward Pass sobre ``X`` con los parámetros
        actuales y devuelve precisión y pérdida sobre el conjunto dado.

        No modifica ``params`` ni calcula gradientes.

        :param params: Parámetros globales actualizados (W1, b1, W2, b2).
        :type params: Dict[str, np.ndarray]

        :param X: Imágenes de evaluación, forma ``(N, 784)``.
        :type X: np.ndarray

        :param Y: Etiquetas, forma ``(N,)``.
        :type Y: np.ndarray

        :return: ``(accuracy_pct, mean_loss)``
        :rtype: Tuple[float, float]
        """
        W1, b1 = params["W1"], params["b1"]
        W2, b2 = params["W2"], params["b2"]
        n = X.shape[0]

        Z1 = W1 @ X.T + b1[:, np.newaxis]
        A1 = sigmoid(Z1)
        Z2 = W2 @ A1 + b2[:, np.newaxis]
        A2 = softmax(Z2)

        predictions = np.argmax(A2, axis=0)
        accuracy = 100.0 * float(np.sum(predictions == Y)) / n

        log_probs = np.log(np.clip(A2, 1e-15, 1.0))
        loss = -float(np.sum(log_probs[Y, np.arange(n)])) / n

        return accuracy, loss
