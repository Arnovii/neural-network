"""
Distributed/parameter_server.py

Implementación del Parameter Server para el algoritmo de Diego
distribuido con Data-Oriented Parallelism sobre sockets TCP.

──────────────────────────────────────────────────────────────────
ROL DEL PARAMETER SERVER
──────────────────────────────────────────────────────────────────
El Parameter Server (PS) es el nodo central del sistema. Mantiene
la copia autoritativa de los pesos de la red neuronal y coordina
el entrenamiento distribuido.

Responsabilidades:
    1. Escuchar conexiones entrantes de Workers.
    2. Esperar a que todos los Workers estén listos.
    3. Por cada época:
       a. Dividir los índices de entrenamiento en N chunks disjuntos,
          uno por Worker, sin solapamiento.
       b. Hacer broadcast de los parámetros actuales + índices asignados.
       c. Esperar los gradientes de TODOS los Workers (barrera).
       d. Promediar gradientes: ∇θ = (1/N) * Σ ∇θL(Bᵢ)
       e. Actualizar pesos: θ ← θ − lr * ∇θ
    4. Al finalizar, hacer broadcast de STOP.

──────────────────────────────────────────────────────────────────
CALLBACKS DISPONIBLES
──────────────────────────────────────────────────────────────────
on_worker_connected(worker_id, addr)
    Llamado en cuanto un Worker envía READY y queda registrado.
    Útil para actualizar el panel de Workers en la GUI.

on_gradients_received(worker_id, epoch, loss, accuracy)
    Llamado cada vez que se reciben los gradientes de un Worker.
    Permite marcar individualmente qué Workers ya terminaron la época.

on_epoch_end(epoch, total_epochs, accuracy, loss)
    Llamado tras promediar gradientes y actualizar pesos.
    Los valores de accuracy y loss son el promedio de todos los Workers.

──────────────────────────────────────────────────────────────────
CONCURRENCIA
──────────────────────────────────────────────────────────────────
Cada Worker se atiende en un hilo independiente. Esto permite
recibir gradientes en paralelo mientras los Workers computan.
La sincronización usa threading.Event como barrera ligera:

    done_event.wait()  ←── bloquea hasta que TODOS los Workers
                            hayan enviado sus gradientes en la época.

Una vez cruzado el evento, solo el hilo coordinador (run())
hace el promediado, la actualización y el siguiente broadcast.
"""

import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from Distributed.protocol import MsgType, receive_message, send_message


class ParameterServer:
    """
    Parameter Server para entrenamiento distribuido con sockets TCP.

    Gestiona la conexión con N Workers, coordina cada época de
    entrenamiento y mantiene los pesos globales de la red.

    :param host: Dirección IP en la que escucha el servidor.
    :type host: str

    :param port: Puerto TCP.
    :type port: int

    :param num_workers: Número exacto de Workers que deben conectarse
                        antes de iniciar el entrenamiento.
    :type num_workers: int

    :param initial_params: Parámetros iniciales de la red (W1, b1, W2, b2).
    :type initial_params: Dict[str, np.ndarray]

    :param learning_rate: Tasa de aprendizaje para la actualización de pesos.
    :type learning_rate: float

    :param n_train: Total de ejemplos de entrenamiento disponibles.
                    Se usa para dividir los índices entre Workers.
    :type n_train: int

    :param on_worker_connected: Callback opcional llamado cuando un Worker
                                se conecta y envía READY.
                                Firma: ``(worker_id: int, addr: str) -> None``
    :type on_worker_connected: Callable | None

    :param on_gradients_received: Callback opcional llamado cuando se
                                  reciben los gradientes de un Worker.
                                  Firma: ``(worker_id, epoch, loss, accuracy)``
    :type on_gradients_received: Callable | None

    :param on_epoch_end: Callback opcional llamado al final de cada época,
                         tras promediar gradientes y actualizar pesos.
                         Firma: ``(epoch, total_epochs, accuracy, loss)``
    :type on_epoch_end: Callable | None
    """

    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int,
        initial_params: Dict[str, np.ndarray],
        learning_rate: float,
        n_train: int,
        on_worker_connected:  Optional[Callable] = None,
        on_gradients_received: Optional[Callable] = None,
        on_epoch_end:         Optional[Callable] = None,
    ) -> None:
        self.host           = host
        self.port           = port
        self.num_workers    = num_workers
        self.params         = {k: v.copy() for k, v in initial_params.items()}
        self.learning_rate  = learning_rate
        self.n_train        = n_train

        self.on_worker_connected   = on_worker_connected
        self.on_gradients_received = on_gradients_received
        self.on_epoch_end          = on_epoch_end

        # Sockets de cada Worker, indexados por worker_id
        self._worker_sockets: Dict[int, socket.socket] = {}
        # Dirección IP de cada Worker, para mostrar en la GUI
        self._worker_addrs:   Dict[int, str]           = {}
        self._lock = threading.Lock()

        # Gradientes y métricas acumulados en la época actual
        self._epoch_gradients: Dict[int, Dict[str, np.ndarray]] = {}
        self._epoch_metrics:   Dict[int, Tuple[float, float]]   = {}

        # Historial de métricas por época (accuracy y loss globales)
        self.history: Dict[str, List[float]] = {
            "accuracies": [],
            "losses":     [],
        }

    # ================================================================
    # PUNTO DE ENTRADA PRINCIPAL
    # ================================================================

    def run(self, epochs: int) -> Dict[str, List[float]]:
        """
        Inicia el servidor, espera conexiones y ejecuta el entrenamiento.

        Bloquea hasta que el entrenamiento completa todas las épocas.

        :param epochs: Número de épocas a entrenar.
        :type epochs: int

        :return: Historial con ``accuracies`` y ``losses`` por época.
        :rtype: Dict[str, List[float]]
        """
        print("=" * 70)
        print("PARAMETER SERVER — ALGORITMO DE DIEGO DISTRIBUIDO")
        print("=" * 70)
        print(f"  Escuchando en     : {self.host}:{self.port}")
        print(f"  Workers esperados : {self.num_workers}")
        print(f"  Épocas            : {epochs}")
        print(f"  Learning rate     : {self.learning_rate}")
        print(f"  Ejemplos train    : {self.n_train}")
        print("=" * 70)

        self._accept_workers()
        self._run_training_loop(epochs)
        self._broadcast_stop()

        return self.history

    # ================================================================
    # ACEPTACIÓN DE CONEXIONES
    # ================================================================

    def _accept_workers(self) -> None:
        """
        Acepta exactamente ``num_workers`` conexiones TCP.

        Bloquea hasta que todos los Workers se hayan conectado y
        enviado su mensaje READY. Llama a ``on_worker_connected``
        por cada Worker que se registra.
        """
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(self.num_workers)

        print(f"\n[PS] Esperando {self.num_workers} worker(s)...")

        while len(self._worker_sockets) < self.num_workers:
            conn, addr = server_sock.accept()
            msg = receive_message(conn)

            if msg["type"] != MsgType.READY:
                conn.close()
                continue

            worker_id  = msg["payload"]["worker_id"]
            addr_str   = f"{addr[0]}:{addr[1]}"

            with self._lock:
                self._worker_sockets[worker_id] = conn
                self._worker_addrs[worker_id]   = addr_str

            print(f"[PS] Worker {worker_id} conectado desde {addr_str}")

            if self.on_worker_connected is not None:
                self.on_worker_connected(worker_id, addr_str)

        server_sock.close()
        print(f"[PS] Todos los workers conectados. Iniciando entrenamiento.\n")

    # ================================================================
    # LOOP PRINCIPAL DE ENTRENAMIENTO
    # ================================================================

    def _run_training_loop(self, epochs: int) -> None:
        """
        Ejecuta el loop de entrenamiento distribuido por épocas.

        Por cada época:
            1. Divide índices en chunks disjuntos (sin solapamiento).
            2. Hace broadcast params + índices a cada Worker.
            3. Lanza un hilo receptor por Worker.
            4. Espera (barrera) a que todos entreguen gradientes.
            5. Promedia gradientes y actualiza pesos.
            6. Llama a on_epoch_end con las métricas globales.

        :param epochs: Total de épocas.
        :type epochs: int
        """
        worker_ids = sorted(self._worker_sockets.keys())

        for epoch in range(1, epochs + 1):
            print(f"[PS] ── Época {epoch}/{epochs} ──────────────────────────")
            t_start = time.perf_counter()

            self._epoch_gradients.clear()
            self._epoch_metrics.clear()

            index_chunks = self._split_indices(worker_ids)

            done_event     = threading.Event()
            received_count = [0]

            def _receive_from_worker(wid: int) -> None:
                """Hilo receptor: espera los gradientes de un Worker."""
                try:
                    msg = receive_message(self._worker_sockets[wid])
                    if msg["type"] == MsgType.GRADIENTS:
                        payload = msg["payload"]
                        loss     = payload["loss"]
                        accuracy = payload["accuracy"]

                        with self._lock:
                            self._epoch_gradients[wid] = payload["gradients"]
                            self._epoch_metrics[wid]   = (loss, accuracy)
                            received_count[0] += 1
                            all_done = received_count[0] == self.num_workers

                        if self.on_gradients_received is not None:
                            self.on_gradients_received(wid, epoch, loss, accuracy)

                        if all_done:
                            done_event.set()

                except Exception as exc:
                    print(f"[PS] Error recibiendo de Worker {wid}: {exc}")
                    with self._lock:
                        received_count[0] += 1
                        if received_count[0] == self.num_workers:
                            done_event.set()

            # Broadcast: parámetros + índices asignados a cada Worker
            for wid in worker_ids:
                send_message(
                    self._worker_sockets[wid],
                    MsgType.PARAMS,
                    {
                        "epoch":   epoch,
                        "params":  self.params,
                        "indices": index_chunks[wid],
                    },
                )

            # Lanza un hilo receptor por Worker
            threads = [
                threading.Thread(
                    target=_receive_from_worker,
                    args=(wid,),
                    daemon=True,
                )
                for wid in worker_ids
            ]
            for t in threads:
                t.start()

            # Barrera: espera a que todos los Workers entreguen gradientes
            done_event.wait()
            for t in threads:
                t.join()

            # Promedia gradientes y actualiza pesos
            avg_grads = self._average_gradients(list(self._epoch_gradients.values()))
            self._apply_gradients(avg_grads)

            # Métricas globales de la época
            losses     = [m[0] for m in self._epoch_metrics.values()]
            accuracies = [m[1] for m in self._epoch_metrics.values()]
            epoch_loss = float(np.mean(losses))
            epoch_acc  = float(np.mean(accuracies))

            self.history["losses"].append(epoch_loss)
            self.history["accuracies"].append(epoch_acc)

            elapsed = time.perf_counter() - t_start
            print(
                f"[PS]   loss={epoch_loss:.4f}  acc={epoch_acc:.2f}%  "
                f"({elapsed:.2f}s)"
            )

            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, epochs, epoch_acc, epoch_loss)

    # ================================================================
    # DIVISIÓN DE ÍNDICES
    # ================================================================

    def _split_indices(
        self, worker_ids: List[int]
    ) -> Dict[int, List[int]]:
        """
        Divide ``range(n_train)`` en chunks disjuntos, uno por Worker.

        Los índices se mezclan antes de dividir para garantizar que
        cada Worker ve clases balanceadas en cada época. No hay
        solapamiento: cada índice pertenece exactamente a un Worker.

        :param worker_ids: Lista de IDs de Workers conectados.
        :type worker_ids: List[int]

        :return: Diccionario ``{worker_id: [índices]}``.
        :rtype: Dict[int, List[int]]
        """
        indices = np.random.permutation(self.n_train).tolist()
        n = len(worker_ids)
        chunk_size = len(indices) // n
        chunks: Dict[int, List[int]] = {}

        for i, wid in enumerate(worker_ids):
            start = i * chunk_size
            end   = start + chunk_size if i < n - 1 else len(indices)
            chunks[wid] = indices[start:end]

        return chunks

    # ================================================================
    # PROMEDIADO Y ACTUALIZACIÓN DE PESOS
    # ================================================================

    def _average_gradients(
        self, gradients_list: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Implementa: ∇θ = (1/N) * Σᵢ ∇θ L(Bᵢ)

        :param gradients_list: Lista de gradientes, uno por Worker.
        :type gradients_list: List[Dict[str, np.ndarray]]

        :return: Gradiente promedio.
        :rtype: Dict[str, np.ndarray]
        """
        averaged: Dict[str, np.ndarray] = {}
        for key in gradients_list[0]:
            stacked = np.array([g[key] for g in gradients_list])
            averaged[key] = np.mean(stacked, axis=0)
        return averaged

    def _apply_gradients(
        self, gradients: Dict[str, np.ndarray]
    ) -> None:
        """
        Actualiza los pesos globales: θ ← θ − lr * ∇θ

        :param gradients: Gradiente promedio con claves dW1, db1, dW2, db2.
        :type gradients: Dict[str, np.ndarray]
        """
        self.params["W1"] -= self.learning_rate * gradients["dW1"]
        self.params["b1"] -= self.learning_rate * gradients["db1"]
        self.params["W2"] -= self.learning_rate * gradients["dW2"]
        self.params["b2"] -= self.learning_rate * gradients["db2"]

    # ================================================================
    # SEÑAL DE FIN
    # ================================================================

    def _broadcast_stop(self) -> None:
        """
        Envía STOP a todos los Workers y cierra las conexiones.
        """
        print("\n[PS] Enviando señal STOP a todos los Workers...")
        for wid, sock in self._worker_sockets.items():
            try:
                send_message(sock, MsgType.STOP, None)
                sock.close()
            except Exception:
                pass
        print("[PS] Entrenamiento distribuido completado.")
