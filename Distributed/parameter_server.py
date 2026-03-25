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

on_worker_joined_late(worker_id, addr)
    Llamado cuando un Worker se conecta mientras hay un entrenamiento
    en curso. Ese Worker NO participa en la sesión activa; se incorpora
    en la siguiente.
"""

import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from Distributed.protocol import MsgType, receive_message, send_message
from Model.cnn_extractor import CNNExtractor
from Model.mlp import apply_gradients, evaluate
from Utils.logging_util import get_logger

# Logger unificado para mensajes consistentes
_logger = get_logger(use_colors=True)


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

    :param on_worker_joined_late: Callback cuando un Worker llega durante
                                  un entrenamiento activo y queda en espera.
                                  Firma: ``(worker_id: int, addr: str)``
    :type on_worker_joined_late: Callable | None
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_worker_connected: Optional[Callable] = None,
        on_worker_disconnected: Optional[Callable] = None,
        on_gradients_received: Optional[Callable] = None,
        on_epoch_end: Optional[Callable] = None,
        on_worker_joined_late: Optional[Callable] = None,
        on_cnn_ready: Optional[Callable] = None,
        training_mode: str = "precomputed",
    ) -> None:
        self.host = host
        self.port = port

        self.on_worker_connected = on_worker_connected
        self.on_worker_disconnected = on_worker_disconnected
        self.on_gradients_received = on_gradients_received
        self.on_epoch_end = on_epoch_end
        self.on_worker_joined_late = on_worker_joined_late
        self.on_cnn_ready = on_cnn_ready

        self.training_mode = training_mode  # "precomputed" o "end_to_end"

        # CNN del PS — preentrenada por el PS y distribuida a los Workers
        self._cnn: Optional[CNNExtractor] = None
        # Evento y contador para la barrera CNN_READY
        self._cnn_ready_event = threading.Event()
        self._cnn_ready_count: int = 0
        # Features de prueba recibidos del Worker 0 (extraídos con su CNN/GPU)
        self._X_test_features: Optional[np.ndarray] = None
        self._Y_test_from_worker: Optional[np.ndarray] = None

        # Sockets y metadatos de Workers activos
        self._worker_sockets: Dict[int, socket.socket] = {}
        self._worker_addrs: Dict[int, str] = {}
        self._next_id: int = 0  # Contador para asignar IDs
        self._lock = threading.Lock()  # Evita que múltiples hilos modifiquen las estructuras anteriores al mismo tiempo

        # Servidor TCP
        self._server_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()

        # IDs que participan en el entrenamiento activo (None = sin sesión)
        self._active_training_workers: Optional[List[int]] = None

        # Gradientes y métricas de la época actual (reutilizados por train)
        # Tipo: Dict[worker_id, {"mlp": grad_dict, "cnn": grad_dict|None}] (E2E)
        #    o Dict[worker_id, grad_dict] (precomputed original)
        self._epoch_gradients: Dict[int, Any] = {}
        self._epoch_metrics: Dict[int, Tuple[float, float]] = {}

    # ================================================================
    # CONFIGURACIÓN DE LA CNN
    # ================================================================

    def set_cnn(self, cnn: CNNExtractor) -> None:
        """
        Establece la CNN preentrenada que el PS distribuirá a los Workers.

        Debe llamarse ANTES de train(). La CNN se envía a todos los
        Workers al inicio de cada sesión via el mensaje CNN_WEIGHTS,
        garantizando que todos usen exactamente los mismos pesos.

        :param cnn: CNNExtractor con pesos ya entrenados o cargados.
        """
        self._cnn = cnn
        _logger.ps(f"CNN configurada: arch={cnn.arch}, mode={self.training_mode}")

    def _get_cnn_params_bytes(self) -> bytes:
        """
        Serializa los pesos CNN actuales para enviarlos a los Workers en E2E.

        En precomputado, esto es básicamente lo mismo que _get_weights_bytes().
        En E2E, se usa en cada época para redistribuir pesos tras actualización.
        """
        if self._cnn is None:
            return b""
        return self._cnn._get_weights_bytes()

    def _apply_cnn_gradients(
        self,
        cnn_gradients_list: List[Dict[str, np.ndarray]],
        learning_rate: float,
    ) -> None:
        """
        Promedia y aplica gradientes CNN recibidos de los Workers.

        Usado únicamente en modo E2E. En precomputado, la CNN no se entrena.

        :param cnn_gradients_list: Lista de dicts con gradientes CNN de cada Worker.
        :param learning_rate: Tasa de aprendizaje para SGD.
        """
        if self._cnn is None or not cnn_gradients_list:
            return

        # Promediar gradientes CNN
        averaged_cnn_grads: Dict[str, np.ndarray] = {}
        all_keys = set()
        for grads in cnn_gradients_list:
            all_keys.update(grads.keys())

        for key in all_keys:
            stacked = np.array([g.get(key, np.zeros(1)) for g in cnn_gradients_list])
            averaged_cnn_grads[key] = np.mean(stacked, axis=0)

        # Aplicar actualización SGD a los pesos CNN
        state_dict = self._cnn._model.state_dict()
        for name, param in self._cnn._model.named_parameters():
            if name in averaged_cnn_grads:
                grad = averaged_cnn_grads[name]
                param.data -= learning_rate * torch.from_numpy(grad).to(param.device)

        self._cnn._model.load_state_dict(state_dict)

    # ================================================================
    # CONFIGURACIÓN DE LA CNN
    # ================================================================

    def _handle_cnn_ready(self, worker_id: int, n_expected: int) -> None:
        """
        Registra que un Worker confirmó haber extraído sus features (CNN_READY).
        Cuando todos los Workers confirman, activa el evento de barrera.
        """
        with self._lock:
            self._cnn_ready_count += 1
            if self.on_cnn_ready:
                self.on_cnn_ready(worker_id)
            if self._cnn_ready_count >= n_expected:
                self._cnn_ready_event.set()

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

        # AF_INET = IPv4  /  SOCK_STREAM = TCP
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(32)
        # Timeout corto para que el hilo pueda comprobar el flag de apagado
        # sin bloquearse indefinidamente en accept().
        self._server_sock.settimeout(1.0)

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        _logger.ps(f"Escuchando en {self.host}:{self.port}")

    def shutdown(self) -> None:
        """
        Envía STOP a todos los Workers, cierra conexiones y detiene
        el hilo de aceptación.
        """
        _logger.ps("Apagando servidor...")
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

        _logger.ps("Servidor apagado")

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
        Si hay un entrenamiento activo, llama a ``on_worker_joined_late``
        en lugar de ``on_worker_connected``.

        :param conn: Socket para comunicarse con ese Worker.
        :type conn: socket.socket

        :param addr: Dirección del Worker (IP y puerto).
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

        try:
            send_message(conn, MsgType.WORKER_ID, {"worker_id": worker_id})
        except Exception:
            with self._lock:
                self._worker_sockets.pop(worker_id, None)
                self._worker_addrs.pop(worker_id, None)
            conn.close()
            return

        addr_str = f"{addr[0]}:{addr[1]}"

        # Decide si el Worker llega en buen momento o tarde
        training_active = self._active_training_workers is not None

        if training_active:
            print(
                f"[PS] Worker {worker_id} conectado desde {addr_str} "
                f"— entrenamiento en curso, esperará la próxima sesión"
            )
            if self.on_worker_joined_late is not None:
                self.on_worker_joined_late(worker_id, addr_str)
        else:
            print(f"[PS] Worker {worker_id} conectado desde {addr_str}")
            if self.on_worker_connected is not None:
                self.on_worker_connected(worker_id, addr_str)

    # ================================================================
    # SESIÓN DE ENTRENAMIENTO
    # ================================================================

    def request_train_sample(
        self,
        n_samples: int = 10000,
    ) -> "tuple[np.ndarray, np.ndarray] | None":
        """
        Pide una muestra de imágenes de train a un Worker, con failover.
        Intenta con cada Worker en orden. Si falla, pasa al siguiente.
        """
        with self._lock:
            worker_ids = sorted(self._worker_sockets.keys())
        if not worker_ids:
            return None
        for wid in worker_ids:
            with self._lock:
                sock = self._worker_sockets.get(wid)
            if sock is None:
                continue
            print(f"[PS] Solicitando {n_samples} imgs de train al Worker {wid}...")
            try:
                send_message(sock, MsgType.TRAIN_SAMPLE, {"n_samples": n_samples})
                msg = receive_message(sock)
                if msg["type"] == MsgType.TRAIN_SAMPLE_DATA:
                    p = msg["payload"]
                    X, Y = p["X_sample"], p["Y_sample"]
                    print(
                        f"[PS] Muestra recibida del Worker {wid}: "
                        f"{X.shape} — sin sesgo en evaluación."
                    )
                    return X, Y
            except Exception as exc:
                print(
                    f"[PS] Worker {wid} falló ({exc}). Intentando con el siguiente..."
                )
                self._remove_worker(wid)
        print("[PS] Todos los Workers fallaron — sin muestra de train.")
        return None

    def train(
        self,
        epochs: int,
        initial_params: Dict[str, np.ndarray],
        learning_rate: float,
        n_train: int,
        X_test: Optional[np.ndarray] = None,
        Y_test: Optional[np.ndarray] = None,
        momentum: float = 0.0,
        seed: Optional[int] = None,
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


        :param X_test: Imágenes del conjunto de prueba, forma ``(N_test, input_size)``.
                       Si se proporciona junto con ``Y_test``, el PS evaluará
                       el modelo global después de cada época.
        :type X_test: np.ndarray | None

        :param Y_test: Etiquetas del conjunto de prueba, forma ``(N_test,)``.
        :type Y_test: np.ndarray | None

        :param momentum: Coeficiente de momentum para SGD (0.0 = SGD puro,
                         0.9 = valor típico). El PS mantiene el estado de
                         velocidades internamente entre épocas.
        :type momentum: float

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

        # Congela los participantes de esta sesión. Cualquier Worker que se
        # conecte a partir de este momento queda en espera y recibe el
        # callback on_worker_joined_late en lugar de on_worker_connected.
        self._active_training_workers = worker_ids

        # ── Distribuir CNN a los Workers ──────────────────────────────
        # El PS envía sus pesos CNN (preentrenados) a todos los Workers.
        # Cada Worker carga esos pesos, extrae sus features de train y
        # confirma con CNN_READY. El PS espera todas las confirmaciones
        # antes de continuar — garantiza que el entrenamiento empieza
        # solo cuando todos los Workers están listos con la misma CNN.
        if self._cnn is not None:
            weights_bytes = self._cnn._get_weights_bytes()
            arch = self._cnn.arch
            _logger.ps(
                f"Distribuyendo CNN a {len(worker_ids)} Worker(s)",
                progress=f"arch={arch}",
            )
            self._cnn_ready_event.clear()
            self._cnn_ready_count = 0
            self._X_test_features = None
            self._Y_test_from_worker = None

            # Buscar caché de features de test ANTES de enviar CNN_WEIGHTS.
            # Si los encuentra, indica a los Workers que no los necesita
            # → Workers no extraen ni transfieren ~20 MB innecesariamente.
            need_test_cached = False
            if X_test is not None and Y_test is not None and X_test.ndim == 4:
                cached = self._cnn._load_features_if_cached("test")
                if cached is not None:
                    X_test, Y_test = cached
                    need_test_cached = True
                    print(f"[PS] Features de prueba en caché: {X_test.shape}\n")

            # Broadcast CNN_WEIGHTS en paralelo con el flag need_test_features.
            def _send_cnn_to_worker(wid: int) -> None:
                with self._lock:
                    sock = self._worker_sockets.get(wid)
                if sock is None:
                    return
                try:
                    send_message(
                        sock,
                        MsgType.CNN_WEIGHTS,
                        {
                            "arch": arch,
                            "weights_bytes": weights_bytes,
                            # need_test_features eliminado: PS pide test features explícitamente.
                        },
                    )
                except Exception as exc:
                    print(f"[PS] Error enviando CNN a Worker {wid}: {exc}")
                    self._remove_worker(wid)

            send_threads = [
                threading.Thread(target=_send_cnn_to_worker, args=(wid,), daemon=True)
                for wid in worker_ids
            ]
            for t in send_threads:
                t.start()
            for t in send_threads:
                t.join()

            def _wait_cnn_ready(wid: int) -> None:
                """Espera CNN_READY. Los features de test se piden después."""
                try:
                    msg = receive_message(self._worker_sockets[wid])
                    if msg["type"] == MsgType.CNN_READY:
                        print(f"[PS] Worker {wid}: CNN_READY ✓")
                    else:
                        print(
                            f"[PS] Worker {wid}: mensaje inesperado "
                            f"{msg['type']} (esperaba CNN_READY)."
                        )
                    self._handle_cnn_ready(wid, len(worker_ids))
                except Exception as exc:
                    print(f"[PS] Worker {wid}: error esperando CNN_READY: {exc}")
                    self._handle_cnn_ready(wid, len(worker_ids))

            ready_threads = [
                threading.Thread(target=_wait_cnn_ready, args=(wid,), daemon=True)
                for wid in worker_ids
                if wid in self._worker_sockets
            ]
            for t in ready_threads:
                t.start()
            self._cnn_ready_event.wait()
            for t in ready_threads:
                t.join()
            print("[PS] Todos los Workers listos con la CNN distribuida.")

            # Pedir TEST_FEATURES si no están en caché local.
            # Un único Worker los extrae y envía — con failover.
            if not need_test_cached:
                with self._lock:
                    candidate_ids = sorted(self._worker_sockets.keys())
                for wid in candidate_ids:
                    with self._lock:
                        sock = self._worker_sockets.get(wid)
                    if sock is None:
                        continue
                    print(f"[PS] Solicitando features de prueba al Worker {wid}...")
                    try:
                        send_message(sock, MsgType.REQUEST_TEST_FEATURES, {})
                        msg_t = receive_message(sock)
                        if msg_t["type"] == MsgType.TEST_FEATURES:
                            p = msg_t["payload"]
                            X_test = p["X_test_features"]
                            Y_test = p["Y_test"]
                            assert X_test is not None and Y_test is not None
                            self._cnn._save_features("test", X_test, Y_test)
                            print(
                                f"[PS] Features de prueba recibidos del "
                                f"Worker {wid}: {X_test.shape}\n"
                            )
                            break  # éxito
                    except Exception as exc:
                        print(
                            f"[PS] Worker {wid} falló ({exc}). "
                            "Intentando con el siguiente..."
                        )
                        self._remove_worker(wid)
                else:
                    # Todos los Workers fallaron → fallback al PS local
                    if X_test is not None and Y_test is not None:
                        print(
                            "[PS] Todos los Workers fallaron. "
                            "Extrayendo features localmente..."
                        )
                        X_test, Y_test = self._cnn.prepare(
                            X_test,
                            Y_test,
                            split="test",
                            pretrain_epochs=0,
                            verbose=True,
                        )
                        print(f"[PS] Features de prueba listos: {X_test.shape}\n")

        params = {tipo: datos.copy() for tipo, datos in initial_params.items()}
        velocities: Dict[str, np.ndarray] = {}  # estado de momentum entre épocas

        # Generador de semillas de época — controla qué datos ve cada Worker
        # en cada época (round-robin). Con la misma semilla, los epoch_seeds
        # son idénticos entre sesiones → resultados completamente reproducibles.
        # Sin semilla (None): aleatorio, diferente en cada sesión.
        _epoch_rng = np.random.RandomState(seed)

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
        print(f"  Momentum        : {momentum if momentum > 0 else 'desactivado'}")
        print(f"  Ejemplos train  : {n_train}")
        print("=" * 70)

        _logger.section("INICIANDO ENTRENAMIENTO DISTRIBUIDO")

        # Notifica a los Workers. Cada uno recibe su rank dentro de la
        # sesión para que pueda reconstruir su chunk localmente.
        n_workers = len(worker_ids)
        for rank, wid in enumerate(worker_ids):
            with self._lock:
                sock = self._worker_sockets.get(wid)
            if sock is None:
                continue
            try:
                send_message(
                    sock,
                    MsgType.TRAIN_START,
                    {
                        "epochs": epochs,
                        "n_train": n_train,
                        "n_workers": n_workers,
                        "worker_rank": rank,
                    },
                )
            except Exception as exc:
                _logger.error(f"Error enviando TRAIN_START a Worker {wid}: {exc}")
                self._remove_worker(wid)

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            # Limpia los gradientes anteriores
            self._epoch_gradients.clear()
            self._epoch_metrics.clear()

            # Semilla única para esta época: el Worker la usa para
            # reconstruir exactamente la misma partición estratificada.
            epoch_seed = int(_epoch_rng.randint(0, 2**31))

            done_event = threading.Event()
            received_count = [0]

            def _receive_from_worker(wid: int) -> None:
                try:
                    msg = receive_message(self._worker_sockets[wid])
                    if msg["type"] == MsgType.GRADIENTS:
                        payload = msg["payload"]
                        loss = payload["loss"]
                        accuracy = payload["accuracy"]

                        with self._lock:
                            self._epoch_gradients[wid] = {
                                "mlp": payload["gradients"],
                                "cnn": payload.get(
                                    "cnn_gradients"
                                ),  # None en precomputed
                            }
                            self._epoch_metrics[wid] = (loss, accuracy)
                            received_count[0] += 1
                            all_done = received_count[0] == len(worker_ids)

                        if self.on_gradients_received is not None:
                            self.on_gradients_received(wid, epoch, loss, accuracy)

                        if all_done:
                            done_event.set()

                except Exception as exc:
                    _logger.error(f"Error recibiendo de Worker {wid}: {exc}")
                    self._remove_worker(wid)
                    with self._lock:
                        received_count[0] += 1
                        if received_count[0] == len(worker_ids):
                            done_event.set()

            def _send_params_to_worker(wid: int) -> None:
                try:
                    send_dict = {"epoch": epoch, "params": params, "seed": epoch_seed}

                    # En E2E, enviar también los pesos CNN
                    if self.training_mode == "end_to_end" and self._cnn is not None:
                        # Obtener estado actual de CNN como dict
                        cnn_state = {}
                        for name, param in self._cnn._model.named_parameters():
                            cnn_state[name] = param.detach().cpu().numpy()
                        send_dict["cnn_params"] = cnn_state

                    send_message(
                        self._worker_sockets[wid],
                        MsgType.PARAMS,
                        send_dict,
                    )
                except Exception as exc:
                    _logger.error(f"Error enviando PARAMS a Worker {wid}: {exc}")
                    self._remove_worker(wid)

            param_threads = [
                threading.Thread(
                    target=_send_params_to_worker, args=(wid,), daemon=True
                )
                for wid in worker_ids
                if wid in self._worker_sockets
            ]
            for t in param_threads:
                t.start()
            for t in param_threads:
                t.join()

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
                _logger.error("Sin gradientes — todos los Workers fallaron")
                break

            # En precomputado: _epoch_gradients[wid] contiene solo gradientes MLP
            # En E2E: _epoch_gradients[wid] contiene {"mlp": ..., "cnn": ...}

            if self.training_mode == "precomputed":
                # Modo precomputado: solo gradientes MLP
                mlp_grads_list: List[Dict[str, np.ndarray]] = [
                    g["mlp"] if isinstance(g, dict) else g
                    for g in self._epoch_gradients.values()
                ]
                avg_grads = self._average_gradients(mlp_grads_list)
                self._apply_gradients(
                    params, avg_grads, learning_rate, momentum, velocities
                )
            else:
                # Modo E2E: gradientes CNN + MLP
                mlp_grads_list: List[Dict[str, np.ndarray]] = [
                    g["mlp"] for g in self._epoch_gradients.values()
                ]
                cnn_grads_list: List[Dict[str, np.ndarray]] = [
                    g["cnn"]
                    for g in self._epoch_gradients.values()
                    if g["cnn"] is not None
                ]

                # Aplicar gradientes MLP
                avg_mlp_grads = self._average_gradients(mlp_grads_list)
                self._apply_gradients(
                    params, avg_mlp_grads, learning_rate, momentum, velocities
                )

                # Aplicar gradientes CNN si hay
                if cnn_grads_list:
                    self._apply_cnn_gradients(cnn_grads_list, learning_rate)

            # Promedia métricas
            losses = [m[0] for m in self._epoch_metrics.values()]
            accuracies = [m[1] for m in self._epoch_metrics.values()]
            epoch_loss = float(np.mean(losses))
            epoch_acc = float(np.mean(accuracies))

            history["losses"].append(epoch_loss)
            history["accuracies"].append(epoch_acc)

            # Evalúa sobre datos de prueba si se proporcionaron
            test_acc: Optional[float] = None
            test_loss: Optional[float] = None
            if X_test is not None and Y_test is not None:
                test_acc, test_loss = self._evaluate(params, X_test, Y_test)
                history["test_accuracies"].append(test_acc)
                history["test_losses"].append(test_loss)

            elapsed = time.perf_counter() - t_start

            # Log formateado con progreso y métricas
            progress = f"{epoch}/{epochs}"
            if test_acc is not None:
                metric = f"train_acc={epoch_acc:.2f}% | test_acc={test_acc:.2f}% | pérdida={epoch_loss:.4f}"
            else:
                metric = f"acc={epoch_acc:.2f}% | pérdida={epoch_loss:.4f}"
            _logger.train(f"Época en progreso", progress=progress, metric=metric)

            if self.on_epoch_end is not None:
                self.on_epoch_end(
                    epoch, epochs, epoch_acc, epoch_loss, test_acc, test_loss
                )

        _logger.ps("Entrenamiento completado")
        self._active_training_workers = None
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
                _logger.error(f"Error en broadcast a Worker {wid}: {exc}")
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

    def _average_gradients(
        self, gradients_list: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """∇θ = (1/N) * Σᵢ ∇θ L(Bᵢ)"""
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
        momentum: float = 0.0,
        velocities: Dict[str, np.ndarray] | None = None,
    ) -> None:
        """Delega en Model.mlp.apply_gradients — agnóstico al número de capas."""
        apply_gradients(params, gradients, learning_rate, momentum, velocities)

    def _evaluate(
        self,
        params: Dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Delega en ``Model.mlp.evaluate``.

        El PS nunca implementa la arquitectura directamente: solo
        coordina. Centralizar el forward en Model/nn.py garantiza
        que PS y Workers evalúen exactamente con la misma arquitectura MLP.

        :param params: Pesos del MLP (W1, b1, …, W3, b3).
        :param X:      Imágenes de evaluación.
        :param Y:      Etiquetas.
        :return:       ``(accuracy_pct, mean_loss)``
        """
        return evaluate(params, X, Y)
