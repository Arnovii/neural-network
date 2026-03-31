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
        debug: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.debug = debug  # Flag para mostrar mensajes de debug

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

        CRÍTICO: Debe llamarse ANTES de train(). La CNN se envía a todos los
        Workers al inicio de cada sesión de entrenamiento mediante el mensaje
        CNN_WEIGHTS. Esto garantiza que todos los Workers usan exactamente
        los misma pesos, evitando divergencia accidental por diferencias
        de inicialización.

        MODO PRECOMPUTED: CNN congelada (no se entrena).
        MODO END-TO-END: CNN se entrena y sus pesos se actualizan cada época.

        :param cnn: Instancia CNN Extractor con pesos ya inicializados o cargados
                    desde archivo preentrenado.
        :type cnn: CNNExtractor con atributos arch, device, feature_dim.

        :return: None (modifica self._cnn).
        :rtype: NoneType.
        """
        self._cnn = cnn
        _logger.ps(f"CNN configurada: arch={cnn.arch}, mode={self.training_mode}")

    def _get_cnn_params_bytes(self) -> bytes:
        """
        Serializa los pesos CNN actuales para enviarlos a los Workers.

        En precomputado y end-to-end, usa la serialización de pesos CNN
        del extractor. Permite distribuir la misma CNN a todos los Workers.

        :return: Bytes serializados con torch.save() de state_dict CNN.
        :rtype: bytes, tamaño ~2-44 MB según arquitectura.
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
        Aplica gradientes CNN promediados a los pesos locales del modelo.

        Solo se usa en la versión antigua del E2E con proxy gradients.
        No aplica en FedAvg (que simplemente promedia pesos finales).

        Calcula el promedio de gradientes recibidos de todos los Workers,
        luego aplica actualización SGD en-lugar (in-place) a la CNN:
        param = param - lr * averaged_grad

        :param cnn_gradients_list: Lista de diccionarios con gradientes CNN por Worker.
                                   Cada dict mapea nombres de parámetros a arrays NumPy.
        :type cnn_gradients_list: List[Dict[str, np.ndarray]]

        :param learning_rate: Tasa de aprendizaje para actualización de pesos CNN.
        :type learning_rate: float

        :return: None (modifica self._cnn weights in-place).
        :rtype: NoneType.
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

        # Aplicar actualización SGD a los pesos CNN (in-place)
        # IMPORTANTE: NO llamar load_state_dict después — eso revertiría la actualización
        for name, param in self._cnn._model.named_parameters():
            if name in averaged_cnn_grads:
                grad = averaged_cnn_grads[name]
                param.data -= learning_rate * torch.from_numpy(grad).to(param.device)

    # ================================================================
    # CONFIGURACIÓN DE LA CNN — BARRERA CNN_READY
    # ================================================================

    def _handle_cnn_ready(self, worker_id: int, n_expected: int) -> None:
        """
        Registra que un Worker confirmó extracción de features (CNN_READY).

        Cuando todos los Workers esperados confirman, activa la barrera CNN_READY.

        :param worker_id: ID del Worker que confirmó.
        :type worker_id: int, asignado por PS.

        :param n_expected: Número total de Workers esperados.
        :type n_expected: int.

        :return: None (modifica self._cnn_ready_count y activa barrera).
        :rtype: NoneType.
        """
        with self._lock:
            self._cnn_ready_count += 1
            if self.on_cnn_ready:
                self.on_cnn_ready(worker_id)
            if self._cnn_ready_count >= n_expected:
                self._cnn_ready_event.set()

    # ================================================================
    # FUNCIÓN DE DEBUG
    # ================================================================

    def _debug_print(self, msg: str) -> None:
        """
        Imprime un mensaje de debug solo si self.debug es True.

        Útil para diagnóstico de entrenamiento distribuido sin contaminar log.

        :param msg: Mensaje a imprimir en stdout.
        :type msg: str

        :return: None.
        :rtype: NoneType.
        """
        if self.debug:
            print(msg)

    # ================================================================
    # RESET DE ESTADO DE ENTRENAMIENTO
    # ================================================================

    def _reset_training_state(self, new_training_mode: str) -> None:
        """
        Limpia completamente el estado entre sesiones de entrenamiento.

        CRÍTICO: Llamar SIEMPRE antes de iniciar una nueva sesión para
        evitar que state anterior contamine el nuevo entrenamiento.

        :param new_training_mode: Modo de entrenamiento ("precomputed" o "end_to_end").
        :type new_training_mode: str.

        :return: None (modifica self state in-place).
        :rtype: NoneType.

        :raises ValueError: Si new_training_mode no es válido.
        """
        self._debug_print(
            "\n[PS][RESET] ════════════════════════════════════════════════════════════"
        )
        self._debug_print("[PS][RESET] Limpiando estado anterior")
        self._debug_print(
            f"[PS][RESET] training_mode: {self.training_mode} → {new_training_mode}"
        )

        # ━━━ Actualizar training_mode EXPLÍCITAMENTE ━━━
        if new_training_mode not in ("precomputed", "end_to_end"):
            raise ValueError(
                f"training_mode desconocido: {new_training_mode}. "
                f"Debe ser 'precomputed' o 'end_to_end'."
            )

        self.training_mode = new_training_mode
        self._debug_print(
            f"[PS][RESET] ✓ training_mode actualizado a: {self.training_mode}"
        )

        # ━━━ CNN se reutiliza (fue configurada en set_cnn) ━━━
        # pero sus pesos pueden cambiar según el modo
        if self._cnn is not None:
            # En precomputed: CNN es inmutable (congelada)
            # En E2E: CNN se entrena, pesos se actualizarán
            self._debug_print(f"[PS][RESET] CNN presente (arch={self._cnn.arch})")
            self._debug_print(
                "[PS][RESET]   - Modo PRECOMPUTED → CNN CONGELADA (sin cambios)"
                if self.training_mode == "precomputed"
                else "[PS][RESET]   - Modo END-TO-END → CNN ENTRENABLE (pesos se actualizarán)"
            )

        # ━━━ Limpiar estado de sesión anterior ━━━
        self._active_training_workers = None
        self._cnn_ready_event.clear()
        self._cnn_ready_count = 0
        self._X_test_features = None
        self._Y_test_from_worker = None
        self._epoch_gradients.clear()
        self._epoch_metrics.clear()

        self._debug_print("[PS][RESET] ✓ Estado de sesión limpiado")
        self._debug_print(
            "[PS][RESET] ════════════════════════════════════════════════════════════\n"
        )

    # ================================================================
    # CICLO DE VIDA DEL SERVIDOR
    # ================================================================

    def listen(self) -> None:
        """
        Abre el socket TCP y comienza a aceptar conexiones de Workers en hilo.

        Retorna inmediatamente; las conexiones se aceptan de forma asíncrona
        en self._accept_thread. Los Workers se registran conforme llegan.

        :return: None (inicia hilo daemon de aceptación).
        :rtype: NoneType.

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
        Envía STOP a todos los Workers, cierra conexiones y apaga servidor.

        Limpia el estado del servidor para que pueda ser reutilizado con listen()
        y train() nuevamente si es necesario.

        :return: None (cierra sockets y detiene hilos).
        :rtype: NoneType.
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
        """
        Obtiene los IDs de los Workers actualmente conectados.

        Thread-safe: adquiere lock antes de acceder a diccionario de sockets.

        :return: Lista de IDs de Workers conectados ordenados ascendentemente.
        :rtype: List[int], ej. [0, 1, 3] si Workers 0, 1, 3 están conectados.
        """
        with self._lock:
            return sorted(self._worker_sockets.keys())

    # ================================================================
    # HILO DE ACEPTACIÓN
    # ================================================================

    def _accept_loop(self) -> None:
        """
        Acepta conexiones entrantes indefinidamente hasta que se activa el flag de apagado.

        Se ejecuta en un hilo daemon. Utiliza socket.accept() con timeout para permitir
        que el servidor compruebe el flag de apagado periódicamente. Para cada conexión,
        lanza un hilo daemon _handshake() para realizar la negociación de identidad.

        :return: None (ejecuta bucle infinito hasta shutdown).
        :rtype: NoneType.
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
        training_mode: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Ejecuta una sesión de entrenamiento distribuida con los Workers conectados.

        Dispatcher que elige entre DOS FLUJOS MUTUAMENTE EXCLUYENTES:

        1. PRECOMPUTED: CNN fija + MLP distribuido
           - CNN nunca se actualiza
           - Solo gradientes MLP

        2. END-TO-END: CNN + MLP entrenan juntos
           - CNN se actualiza cada época
           - Gradientes CNN + MLP

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

        :param training_mode: CRÍTICO para sesiones múltiples. "precomputed" o "end_to_end".
                              Si se omite, usa self.training_mode (valor al inicializar PS).
                              Para cambiar modo entre sesiones, DEBE pasar aquí explícitamente.
        :type training_mode: str | None

        :return: Historial con ``"accuracies"``, ``"losses"``,
                 ``"test_accuracies"`` y ``"test_losses"`` por época.
                 Las listas de test están vacías si no se pasaron datos de prueba.
        :rtype: Dict[str, List[float]]

        :raises RuntimeError: Si no hay Workers conectados.
        :raises ValueError: Si training_mode es desconocido.
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # VALIDACIÓN Y RESET DE ESTADO
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Si se proporciona training_mode, actualizarlo EXPLÍCITAMENTE
        if training_mode is None:
            training_mode = self.training_mode

        # Validar training_mode
        if training_mode not in ("precomputed", "end_to_end"):
            raise ValueError(
                f"training_mode desconocido: {training_mode}. "
                f"Debe ser 'precomputed' o 'end_to_end'."
            )

        # CRÍTICO: Limpiar estado y actualizar training_mode
        self._reset_training_state(training_mode)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # VALIDACIÓN CRÍTICA ANTES DE ENTRENAR
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        assert self.training_mode in ["precomputed", "end_to_end"], (
            f"[SANITY CHECK] training_mode inválido: {self.training_mode}"
        )

        self._debug_print(f"[PS][DEBUG] CONFIG FINAL: mode={self.training_mode}")

        worker_ids = self.connected_workers
        if not worker_ids:
            raise RuntimeError(
                "No hay Workers conectados. "
                "Inicia al menos un Worker antes de entrenar."
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DISPATCHER: Elegir flujo según training_mode
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if self.training_mode == "precomputed":
            _logger.ps(
                "FLUJO PRECOMPUTED",
                progress="CNN fija, MLP distribuido",
            )
            return self._train_precomputed(
                epochs,
                initial_params,
                learning_rate,
                n_train,
                X_test,
                Y_test,
                momentum,
                seed,
                worker_ids,
            )
        elif self.training_mode == "end_to_end":
            _logger.ps(
                "FLUJO END-TO-END",
                progress="CNN + MLP entrenan juntos",
            )
            return self._train_end_to_end(
                epochs,
                initial_params,
                learning_rate,
                n_train,
                X_test,
                Y_test,
                momentum,
                seed,
                worker_ids,
            )
        else:
            raise ValueError(
                f"training_mode desconocido: {self.training_mode}. "
                f"Debe ser 'precomputed' o 'end_to_east'."
            )

    # ================================================================
    # FLUJO 1: PRECOMPUTED (CNN FIJA) — sin cambios
    # ================================================================

    def _train_precomputed(
        self,
        epochs: int,
        initial_params: Dict[str, np.ndarray],
        learning_rate: float,
        n_train: int,
        X_test: Optional[np.ndarray],
        Y_test: Optional[np.ndarray],
        momentum: float,
        seed: Optional[int],
        worker_ids: List[int],
    ) -> Dict[str, List[float]]:
        """
        Entrenamiento PRECOMPUTED: CNN fija, MLP distribuido.

        Fase de inicialización:
        1. Distribuir CNN congelada a todos los Workers
        2. Esperar a que Workers extraigan y cacheen features
        3. Solicitar features de prueba (si existen)

        Fase de entrenamiento (por época):
        1. Enviar PARAMS con MLP (NO cnn_params)
        2. Recibir GRADIENTS con MLP (cnn_gradients será None)
        3. Validar que cnn_gradients sea None (invariante)
        4. Actualizar MLP
        5. Evaluar modelo
        """

        _logger.ps("[PRECOMPUTED] Iniciando flujo de entrenamiento")

        # ── INICIALIZACIÓN COMÚN ──────────────────────────────────────
        self._active_training_workers = worker_ids

        # Distribuir CNN congelada
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

            # Buscar caché de features de test
            need_test_cached = False
            if X_test is not None and Y_test is not None and X_test.ndim == 4:
                cached = self._cnn._load_features_if_cached("test")
                if cached is not None:
                    X_test, Y_test = cached
                    need_test_cached = True
                    print(f"[PS] Features de prueba en caché: {X_test.shape}\n")

            # Broadcast CNN_WEIGHTS
            def _send_cnn_to_worker(wid: int) -> None:
                with self._lock:
                    sock = self._worker_sockets.get(wid)
                if sock is None:
                    return
                try:
                    send_message(
                        sock,
                        MsgType.CNN_WEIGHTS,
                        {"arch": arch, "weights_bytes": weights_bytes},
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
                try:
                    msg = receive_message(self._worker_sockets[wid])
                    if msg["type"] == MsgType.CNN_READY:
                        print(f"[PS] Worker {wid}: CNN_READY ✓")
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

            # Pedir TEST_FEATURES
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
                            break
                    except Exception as exc:
                        print(
                            f"[PS] Worker {wid} falló ({exc}). "
                            "Intentando con el siguiente..."
                        )
                        self._remove_worker(wid)
                else:
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

        # ── INICIALIZACIÓN DE PARÁMETROS ──────────────────────────────
        params = {tipo: datos.copy() for tipo, datos in initial_params.items()}
        velocities: Dict[str, np.ndarray] = {}
        _epoch_rng = np.random.RandomState(seed)

        history: Dict[str, List[float]] = {
            "accuracies": [],
            "losses": [],
            "test_accuracies": [],
            "test_losses": [],
        }

        print("=" * 70)
        print("PRECOMPUTED — ENTRENAMIENTO MLP DISTRIBUIDO")
        print("=" * 70)
        print(f"  Workers activos : {worker_ids}")
        print(f"  Épocas          : {epochs}")
        print(f"  Learning rate   : {learning_rate}")
        print(f"  Momentum        : {momentum if momentum > 0 else 'desactivado'}")
        print(f"  Ejemplos train  : {n_train}")
        print("=" * 70)

        _logger.section("ENTRENAMIENTO PRECOMPUTED (MLP DISTRIBUIDO)")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # [DEBUG FASE 1] ENVÍO DE TRAIN_START
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._debug_print(
            "[PS][DEBUG] ========== INICIANDO ENVÍO TRAIN_START ========="
        )
        self._debug_print(f"[PS][DEBUG] Modo entrenamiento: {self.training_mode}")
        self._debug_print(f"[PS][DEBUG] Workers activos: {worker_ids}")
        self._debug_print(f"[PS][DEBUG] Total épocas a entrenar: {epochs}")

        # Notificar TRAIN_START
        n_workers = len(worker_ids)
        for rank, wid in enumerate(worker_ids):
            with self._lock:
                sock = self._worker_sockets.get(wid)
            if sock is None:
                self._debug_print(
                    f"[PS][DEBUG] ERROR: Socket para Worker {wid} es None"
                )
                continue
            try:
                # [DEBUG] ANTES de crear payload
                self._debug_print(
                    f"[PS][DEBUG] Preparando TRAIN_START para Worker {wid} (rank {rank}/{n_workers})"
                )

                # Crear payload
                payload_train_start = {
                    "epochs": epochs,
                    "n_train": n_train,
                    "n_workers": n_workers,
                    "worker_rank": rank,
                    "training_mode": "precomputed",
                }

                self._debug_print(
                    f"[PS][DEBUG] Payload keys: {list(payload_train_start.keys())}"
                )
                self._debug_print(
                    f"[PS][DEBUG] training_mode en payload: {payload_train_start.get('training_mode', 'AUSENTE')}"
                )

                # Log con instrumentación
                _logger.ps(
                    f"[INSTRUM] Enviando TRAIN_START a Worker {wid}",
                    progress=f"flujo=PRECOMPUTED | training_mode={'PRESENTE' if 'training_mode' in payload_train_start else 'AUSENTE'} | "
                    f"training_mode={payload_train_start.get('training_mode', 'N/A')} | "
                    f"payload_keys={list(payload_train_start.keys())}",
                )

                # [DEBUG] ANTES de enviar
                self._debug_print(
                    f"[PS][DEBUG] (ANTES send) Socket para W{wid}: {'Open' if sock else 'Closed'}"
                )

                # Enviar
                send_message(sock, MsgType.TRAIN_START, payload_train_start)

                # [DEBUG] DESPUÉS de enviar
                self._debug_print(
                    f"[PS][DEBUG] TRAIN_START enviado exitosamente a Worker {wid}"
                )

            except Exception as exc:
                print(
                    f"[PS][ERROR] Excepción enviando TRAIN_START a Worker {wid}: {exc}"
                )
                print(f"[PS][ERROR] Tipo: {type(exc).__name__}")
                _logger.error(f"Error enviando TRAIN_START a Worker {wid}: {exc}")
                self._remove_worker(wid)

        self._debug_print(
            "[PS][DEBUG] ========== TRAIN_START ENVIADO A TODOS =========="
        )
        self._debug_print("[PS][DEBUG] Esperando que Workers lean TRAIN_START...\n")

        t_start = time.perf_counter()

        # ═════════════════════════════════════════════════════════════════
        # [CRÍTICO] VERIFICACIÓN ANTES DE ENTRAR AL LOOP
        # ═════════════════════════════════════════════════════════════════
        self._debug_print(
            "\n[PS][DEBUG] ¡¡¡ PUNTO CRÍTICO: A punto de entrar al loop de épocas !!!"
        )
        self._debug_print(f"[PS][DEBUG] worker_ids: {worker_ids}")
        self._debug_print(
            f"[PS][DEBUG] self._worker_sockets.keys(): {list(self._worker_sockets.keys())}"
        )
        self._debug_print(f"[PS][DEBUG] epochs: {epochs}")
        self._debug_print(f"[PS][DEBUG] range(1, {epochs + 1})\n")

        # ── LOOP DE ÉPOCAS ────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            self._debug_print(
                f"\n[PS][DEBUG] ✓✓✓ ENTRANDO A ITERACIÓN epoch={epoch}/{epochs}"
            )
            self._debug_print(
                f"[PS][DEBUG]   self._worker_sockets.keys() AHORA: {list(self._worker_sockets.keys())}"
            )

            self._epoch_gradients.clear()
            self._epoch_metrics.clear()

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

                        # ━━━ VALIDACIÓN: En PRECOMPUTED, NO debe haber cnn_gradients ━━━
                        cnn_grads = payload.get("cnn_gradients")
                        if cnn_grads is not None:
                            _logger.error(
                                f"[VALIDACIÓN PRECOMPUTED] Worker {wid} envió "
                                f"cnn_gradients pero deberían ser None en precomputed"
                            )
                            raise RuntimeError(
                                f"Worker {wid}: cnn_gradients debe ser None en precomputed"
                            )

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
                    _logger.error(f"Error recibiendo de Worker {wid}: {exc}")
                    self._remove_worker(wid)
                    with self._lock:
                        received_count[0] += 1
                        if received_count[0] == len(worker_ids):
                            done_event.set()

            def _send_params_to_worker(wid: int) -> None:
                try:
                    # [INSTRUMENTACIÓN] Log del envío de PARAMS en PRECOMPUTED
                    send_dict = {
                        "epoch": epoch,
                        "params": params,
                        "seed": epoch_seed,
                    }
                    send_message(self._worker_sockets[wid], MsgType.PARAMS, send_dict)
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

            # [CRÍTICO] Verificación de threads de PARAMS
            self._debug_print(
                f"[PS][DEBUG] Construcción de param_threads ({len(param_threads)} threads creados)"
            )
            if not param_threads:
                print("\n[PS][CRITICAL BUG] ¡¡¡ NO HAY THREADS PARA ENVIAR PARAMS !!!")
                print(
                    f"[PS][CRITICAL BUG] worker_ids recibido en función: {worker_ids}"
                )
                print(
                    f"[PS][CRITICAL BUG] self._worker_sockets.keys() AHORA: {list(self._worker_sockets.keys())}"
                )
                print(
                    "[PS][CRITICAL BUG] Esto significa: Workers recibieron TRAIN_START pero NO ESTÁN en _worker_sockets"
                )
                print("[PS][CRITICAL BUG] ¡¡¡ NO HACER BREAK, algo está muy mal !!!")

                # En lugar de break, loguear y continuar (o fallar más claramente)
                error_msg = (
                    f"FATAL: param_threads vacío en época {epoch}. "
                    f"worker_ids={worker_ids}, _worker_sockets={list(self._worker_sockets.keys())}. "
                    f"Los Workers fueron desconectados o removidos entre TRAIN_START y el loop."
                )
                _logger.error(error_msg)
                raise RuntimeError(error_msg)

            self._debug_print(
                f"[PS][DEBUG] │ Creados {len(param_threads)} threads de envío"
            )
            self._debug_print("[PS][DEBUG] │ Iniciando threads...")
            for t in param_threads:
                t.start()
            self._debug_print("[PS][DEBUG] │ Esperando a que terminen threads...")
            for t in param_threads:
                t.join()
            self._debug_print(f"[PS][DEBUG] └─ PARAMS ENVIADOS para época {epoch}")

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

            done_event.wait()
            for t in threads:
                t.join()

            if not self._epoch_gradients:
                _logger.error("Sin gradientes — todos los Workers fallaron")
                break

            # ━━━ Procesar gradientes MLP (ignore CNN) ━━━
            mlp_grads_list: List[Dict[str, np.ndarray]] = [
                g if isinstance(g, dict) else g for g in self._epoch_gradients.values()
            ]
            avg_grads = self._average_gradients(mlp_grads_list)
            self._apply_gradients(
                params, avg_grads, learning_rate, momentum, velocities
            )

            # ━━━ CNN permanece sin cambios (invariante [R1.1]) ━━━

            # Métricas
            losses = [m[0] for m in self._epoch_metrics.values()]
            accuracies = [m[1] for m in self._epoch_metrics.values()]
            epoch_loss = float(np.mean(losses))
            epoch_acc = float(np.mean(accuracies))

            history["losses"].append(epoch_loss)
            history["accuracies"].append(epoch_acc)

            # Evaluar
            test_acc: Optional[float] = None
            test_loss: Optional[float] = None
            if X_test is not None and Y_test is not None:
                test_acc, test_loss = self._evaluate(params, X_test, Y_test)
                history["test_accuracies"].append(test_acc)
                history["test_losses"].append(test_loss)

            progress = f"{epoch}/{epochs}"
            if test_acc is not None:
                metric = f"train_acc={epoch_acc:.2f}% | test_acc={test_acc:.2f}% | pérdida={epoch_loss:.4f}"
            else:
                metric = f"acc={epoch_acc:.2f}% | pérdida={epoch_loss:.4f}"
            _logger.train("Época", progress=progress, metric=metric)

            if self.on_epoch_end is not None:
                self.on_epoch_end(
                    epoch, epochs, epoch_acc, epoch_loss, test_acc, test_loss
                )

        _logger.ps("Entrenamiento PRECOMPUTED completado")
        self._active_training_workers = None
        return history

    # ================================================================
    # FLUJO 2: END-TO-END (CNN + MLP) — CORRECCIONES APLICADAS
    # ================================================================

    def _train_end_to_end(
        self,
        epochs: int,
        initial_params: Dict[str, np.ndarray],
        learning_rate: float,
        n_train: int,
        X_test: Optional[np.ndarray],
        Y_test: Optional[np.ndarray],
        momentum: float,
        seed: Optional[int],
        worker_ids: List[int],
    ) -> Dict[str, List[float]]:
        """
        Entrenamiento END-TO-END: CNN + MLP distribuido con FedAvg y Adam.

        CORRECCIONES aplicadas respecto a la versión anterior:

        [C1] LR sin división por n_batches en el Worker.
             El PS ya no envía learning_rate con la intención de que el
             Worker lo divida. El Worker usa Adam con el LR base directamente.

        [C2] LRs diferenciados CNN/MLP en el Worker.
             El PS envía un único learning_rate base; el Worker aplica los
             factores _E2E_CNN_LR_FACTOR y _E2E_MLP_LR_FACTOR internamente.

        [C3] Optimizador Adam con momentos en el Worker.
             Se usa Adam en lugar de SGD manual, que es más robusto ante
             gradientes ruidosos del entrenamiento distribuido.

        [C4] Limitación de pasos locales en el Worker (_E2E_MAX_LOCAL_STEPS).
             Reduce el client drift sin sacrificar cómputo útil.

        [C5] Batch size mínimo 64 para BatchNorm estable en el Worker.
             Evita alta varianza en running_mean/var con batches pequeños.

        El PS conserva su lógica de FedAvg (promedio de pesos) sin cambios.
        """
        _logger.ps("[END-TO-END] Iniciando flujo de entrenamiento")
        self._active_training_workers = worker_ids

        # Distribuir CNN entrenable a Workers
        if self._cnn is not None:
            weights_bytes = self._cnn._get_weights_bytes()
            arch = self._cnn.arch
            _logger.ps(
                f"Distribuyendo CNN entrenable a {len(worker_ids)} Worker(s)",
                progress=f"arch={arch}",
            )
            self._cnn_ready_event.clear()
            self._cnn_ready_count = 0
            self._X_test_features = None
            self._Y_test_from_worker = None

            def _send_cnn_to_worker(wid: int) -> None:
                with self._lock:
                    sock = self._worker_sockets.get(wid)
                if sock is None:
                    return
                try:
                    send_message(
                        sock,
                        MsgType.CNN_WEIGHTS,
                        {"arch": arch, "weights_bytes": weights_bytes},
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
                try:
                    msg = receive_message(self._worker_sockets[wid])
                    if msg["type"] == MsgType.CNN_READY:
                        print(f"[PS] Worker {wid}: CNN_READY ✓")
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
            print("[PS] Todos los Workers listos con la CNN entrenable.")

            if X_test is not None and Y_test is not None and X_test.ndim == 2:
                _logger.warn(
                    "[E2E] X_test son features pre-extraídos (2D). "
                    "Para evaluación correcta se necesitan imágenes raw (4D)."
                )

        params = {tipo: datos.copy() for tipo, datos in initial_params.items()}
        _epoch_rng = np.random.RandomState(seed)

        history: Dict[str, List[float]] = {
            "accuracies": [],
            "losses": [],
            "test_accuracies": [],
            "test_losses": [],
        }

        print("=" * 70)
        print("END-TO-END — ENTRENAMIENTO CNN + MLP DISTRIBUIDO (FedAvg + Adam)")
        print("=" * 70)
        print(f"  Workers activos : {worker_ids}")
        print(f"  Épocas          : {epochs}")
        print(f"  Learning rate   : {learning_rate}")
        print(f"  Ejemplos train  : {n_train}")
        print("=" * 70)

        _logger.section("ENTRENAMIENTO END-TO-END (CNN + MLP DISTRIBUIDO)")

        # Notificar TRAIN_START
        n_workers = len(worker_ids)
        for rank, wid in enumerate(worker_ids):
            with self._lock:
                sock = self._worker_sockets.get(wid)
            if sock is None:
                continue
            try:
                # [INSTRUMENTACIÓN] Log del envío de TRAIN_START en END-TO-END
                payload_train_start = {
                    "epochs": epochs,
                    "n_train": n_train,
                    "n_workers": n_workers,
                    "worker_rank": rank,
                    "training_mode": "end_to_end",
                }
                send_message(sock, MsgType.TRAIN_START, payload_train_start)
            except Exception as exc:
                _logger.error(f"Error enviando TRAIN_START a Worker {wid}: {exc}")
                self._remove_worker(wid)

        t_start = time.perf_counter()

        # ── LOOP DE ÉPOCAS ────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            self._epoch_gradients.clear()
            self._epoch_metrics.clear()

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

                        cnn_weights = payload.get("cnn_weights")
                        mlp_weights = payload.get("mlp_weights")

                        if cnn_weights is None or mlp_weights is None:
                            _logger.error(
                                f"[E2E] Worker {wid} NO envió pesos "
                                f"(cnn_weights={cnn_weights is not None}, "
                                f"mlp_weights={mlp_weights is not None})"
                            )
                            raise RuntimeError(
                                f"Worker {wid}: pesos CNN y MLP obligatorios en E2E"
                            )

                        with self._lock:
                            self._epoch_gradients[wid] = {
                                "cnn_weights": cnn_weights,
                                "mlp_weights": mlp_weights,
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
                    # [P2] Serializar CNN state_dict COMPLETO (parámetros + BN buffers)
                    cnn_state = {}
                    if self._cnn is not None:
                        base_model = getattr(
                            self._cnn._model, "model", self._cnn._model
                        )
                        # state_dict() incluye running_mean, running_var, num_batches_tracked
                        for name, tensor in base_model.state_dict().items():
                            cnn_state[name] = tensor.cpu().numpy()

                    send_dict = {
                        "epoch": epoch,
                        "params": params,
                        "seed": epoch_seed,
                        "cnn_params": cnn_state,
                        # [C1][C2] El Worker usará este LR como base y aplicará
                        # internamente los factores CNN/MLP via Adam.
                        # El PS ya no divide por n_batches — esa lógica fue eliminada
                        # del Worker porque destruía la señal de gradiente.
                        "learning_rate": learning_rate,
                        "training_mode": "end_to_end",
                    }
                    send_message(self._worker_sockets[wid], MsgType.PARAMS, send_dict)
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

            done_event.wait()
            for t in threads:
                t.join()

            if not self._epoch_gradients:
                _logger.error("Sin pesos actualizados — todos los Workers fallaron")
                break

            # FedAvg: promediar pesos de todos los Workers
            cnn_weights_list: List[Dict[str, np.ndarray]] = [
                g["cnn_weights"] for g in self._epoch_gradients.values()
            ]
            mlp_weights_list: List[Dict[str, np.ndarray]] = [
                g["mlp_weights"] for g in self._epoch_gradients.values()
            ]

            # [P2] _average_weights opera sobre state_dict COMPLETO (incluye BN buffers)
            averaged_cnn_weights = self._average_weights(cnn_weights_list)

            # [P1] Cargar pesos CNN promediados y poner en eval() inmediatamente
            self._load_cnn_weights(averaged_cnn_weights)
            # [P1] CRÍTICO: garantizar eval() después de cargar pesos
            # Esto asegura que BatchNorm use running stats (no batch stats) en evaluación
            if self._cnn is not None:
                self._cnn._model.eval()

            # [P5] _average_mlp_weights lee formato PyTorch nativo (sin .T)
            averaged_mlp_weights = self._average_mlp_weights(mlp_weights_list)
            params.update(averaged_mlp_weights)

            losses = [m[0] for m in self._epoch_metrics.values()]
            accuracies = [m[1] for m in self._epoch_metrics.values()]
            epoch_loss = float(np.mean(losses))
            epoch_acc = float(np.mean(accuracies))

            history["losses"].append(epoch_loss)
            history["accuracies"].append(epoch_acc)

            # [P1] Evaluación con CNN en eval() garantizado
            test_acc: Optional[float] = None
            test_loss: Optional[float] = None
            if X_test is not None and Y_test is not None:
                if X_test.ndim == 4:
                    if self._cnn is not None:
                        try:
                            # [P1] self._cnn._model ya está en eval() tras _load_cnn_weights
                            # torch.no_grad() para eficiencia (no hay backward aquí)
                            with torch.no_grad():
                                X_test_feat = self._cnn.extract_batched(
                                    X_test, batch_size=512, verbose=False
                                )
                            test_acc, test_loss = self._evaluate(
                                params, X_test_feat, Y_test
                            )
                        except Exception as exc:
                            _logger.warn(
                                f"Error extrayendo features test en época {epoch}: {exc}"
                            )
                    else:
                        _logger.warn(
                            "[E2E] CNN no configurada en PS; no se puede evaluar X_test 4D"
                        )
                else:
                    # Features pre-extraídos (2D)
                    test_acc, test_loss = self._evaluate(params, X_test, Y_test)

                if test_acc is not None and test_loss is not None:
                    history["test_accuracies"].append(test_acc)
                    history["test_losses"].append(test_loss)

            progress = f"{epoch}/{epochs}"
            if test_acc is not None:
                metric = f"train_acc={epoch_acc:.2f}% | test_acc={test_acc:.2f}% | pérdida={epoch_loss:.4f}"
            else:
                metric = f"acc={epoch_acc:.2f}% | pérdida={epoch_loss:.4f}"
            _logger.train("Época", progress=progress, metric=metric)

            if self.on_epoch_end is not None:
                self.on_epoch_end(
                    epoch, epochs, epoch_acc, epoch_loss, test_acc, test_loss
                )

        _logger.ps("Entrenamiento END-TO-END completado")
        self._active_training_workers = None
        return history

    # ================================================================
    # [E2E] HELPERS PARA WEIGHT AVERAGING — FIXES APLICADOS
    # ================================================================

    def _average_weights(
        self, weights_list: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        [P2] Promedia TODOS los tensores del state_dict (parámetros + BN buffers).

        Antes excluía running_mean, running_var, num_batches_tracked de BatchNorm
        → BN desincronizado entre PS y Workers → oscilaciones en curva de prueba.

        Ahora promedia todo lo que venga en el dict, incluyendo buffers BN.
        Esto es matemáticamente correcto: el promedio de running stats de N Workers
        que vieron el mismo número de batches es una estimación válida de las
        estadísticas globales del dataset.
        """
        if not weights_list:
            return {}

        averaged: Dict[str, np.ndarray] = {}
        for key in weights_list[0].keys():
            stacked = np.array([w.get(key, np.zeros(1)) for w in weights_list])
            averaged[key] = np.asarray(np.mean(stacked, axis=0))
        return averaged

    def _average_mlp_weights(
        self, weights_list: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        [P5] Promedia pesos MLP y convierte de formato PyTorch nativo a NumPy MLP.

        ANTES (bug): el Worker enviaba con .T y el PS hacía otro .T al leer.
        La "doble transposición se cancela" era una trampa: si cualquiera de los
        dos lados cambiaba, los pesos quedaban incorrectamente orientados.

        AHORA: el Worker envía en formato PyTorch NATIVO (sin .T):
          - fc1.weight: (hidden1, feature_dim) en PyTorch
          - W1 en NumPy MLP: también (hidden1, feature_dim) → son IGUALES
          - No se necesita ninguna transposición

        La conversión es directa:
          W1 = fc1.weight  (hidden1 × feature_dim) ✓
          W2 = fc2.weight  (hidden2 × hidden1)      ✓
          W3 = fc3.weight  (n_classes × hidden2)    ✓
        """
        if not weights_list:
            return {}

        averaged_pytorch = {}
        for key in weights_list[0].keys():
            stacked = np.array([w.get(key, np.zeros(1)) for w in weights_list])
            averaged_pytorch[key] = np.mean(stacked, axis=0)

        # [P5] Sin transposición: formato PyTorch = formato NumPy MLP para pesos
        # El MLP NumPy usa W1 @ X.T (multiplicación por columnas), por lo que
        # W1 debe ser (hidden1, feature_dim) — exactamente lo que PyTorch fc1.weight tiene.
        mlp_weights = {
            "W1": averaged_pytorch["fc1.weight"],  # (hidden1, feature_dim) ✓
            "b1": averaged_pytorch["fc1.bias"],  # (hidden1,)              ✓
            "W2": averaged_pytorch["fc2.weight"],  # (hidden2, hidden1)      ✓
            "b2": averaged_pytorch["fc2.bias"],  # (hidden2,)              ✓
            "W3": averaged_pytorch["fc3.weight"],  # (n_classes, hidden2)    ✓
            "b3": averaged_pytorch["fc3.bias"],  # (n_classes,)            ✓
        }
        return mlp_weights

    def _load_cnn_weights(self, weights_dict: Dict[str, np.ndarray]) -> None:
        """
        [P1+P2] Carga TODOS los tensores del state_dict en el modelo CNN del PS.

        Después de cargar, el caller debe llamar self._cnn._model.eval()
        para garantizar que BatchNorm use running stats (no batch stats).
        Esto se hace explícitamente en _train_end_to_end tras esta llamada.

        :param weights_dict: Diccionario mapeo nombres de capas → arrays NumPy.
                             Ej: {``conv1.weight``: array(...), ``bn1.bias``: array(...)}.
        :type weights_dict: Dict[str, np.ndarray]

        :return: None (carga pesos en-lugar en self._cnn._model).
        :rtype: NoneType.
        """
        if self._cnn is None or not weights_dict:
            return

        base_model = getattr(self._cnn._model, "model", self._cnn._model)
        current_sd = base_model.state_dict()

        with torch.no_grad():
            for name, arr in weights_dict.items():
                if name in current_sd:
                    arr_np = np.asarray(arr)
                    current_sd[name] = (
                        torch.from_numpy(arr_np)
                        .to(current_sd[name].device)
                        .to(current_sd[name].dtype)
                    )

        # load_state_dict carga params + buffers BN en un solo paso
        base_model.load_state_dict(current_sd)
        # NOTA: el caller es responsable de llamar .eval() después de esta función.

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
        """
        Envía mensaje STOP y cierra el socket de un Worker limpiamente.

        Remueve el Worker de los diccionarios internos (_worker_sockets, _worker_addrs)
        de forma thread-safe. Si el envío del STOP falla, cierra el socket de todas formas.

        :param worker_id: ID único del Worker a detener.
        :type worker_id: int

        :return: None (cierra socket y limpia estado interno).
        :rtype: NoneType.
        """
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
        """
        Elimina un Worker que perdió la conexión, sin enviar STOP.

        Similar a _stop_worker() pero sin intentar enviar mensaje STOP.
        Se utiliza cuando el socket ya está roto o desconectado.
        Thread-safe: adquiere lock antes de remover de diccionarios.

        :param worker_id: ID único del Worker a remover.
        :type worker_id: int

        :return: None (limpia estado interno y cierra socket).
        :rtype: NoneType.
        """
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
        """∇θ = (1/N) * Σᵢ ∇θ L(Bᵢ) — solo usado en PRECOMPUTED"""
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