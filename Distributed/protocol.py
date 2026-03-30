"""
Distributed/protocol.py

Protocolo de comunicación entre el Parameter Server y los Workers.

──────────────────────────────────────────────────────────────────
FORMATO DE MENSAJE
──────────────────────────────────────────────────────────────────
Cada mensaje es un diccionario Python serializado con Pickle y
precedido de 4 bytes (big-endian) que indican la longitud del
bloque serializado:

    ┌──────────────┬────────────────────────────────────────────┐
    │  4 bytes     │  N bytes                                   │
    │  longitud N  │  pickle.dumps({"type": MsgType,            │
    │  (big-endian)│               "payload": …})               │
    └──────────────┴────────────────────────────────────────────┘

El prefijo de longitud es imprescindible porque TCP es un stream:
sin él no hay forma de saber dónde termina un mensaje y dónde
empieza el siguiente.

──────────────────────────────────────────────────────────────────
POR QUÉ PICKLE EN VEZ DE JSON
──────────────────────────────────────────────────────────────────
Pickle serializa np.ndarray de forma nativa y binaria. Las
ventajas frente a JSON son:

  • Sin conversión ndarray ↔ lista: los arrays se serializan
    directamente en su representación binaria (float64 / int32).
  • Tamaño en red mucho menor: un array (784,) float64 ocupa
    ~6 KB en Pickle vs ~15 KB en JSON.
  • Más rápido: no hay llamadas a .tolist() ni parsing de texto.

──────────────────────────────────────────────────────────────────
FLUJO COMPLETO
──────────────────────────────────────────────────────────────────

  Worker                          Parameter Server
  ──────                          ────────────────
  READY ─────────────────────────►  (PS asigna ID)
        ◄──────────────────────── WORKER_ID
        ◄──────────────────────── CNN_WEIGHTS  (PS envía pesos CNN)
  CNN_READY ─────────────────────►  (Worker confirmó extracción)
        ◄──────────────────────── TRAIN_SAMPLE  (PS pide muestra de train)
  TRAIN_SAMPLE_DATA ─────────────►  (Worker envía N imágenes de train)
  TEST_FEATURES ─────────────────►  (Worker envía features de prueba al PS)

  [en espera de TRAIN_START...]

        ◄──────────────────────── TRAIN_START
  [por cada época:]
        ◄──────────────────────── PARAMS  (pesos MLP + semilla)
  GRADIENTS ─────────────────────►
  [vuelve a esperar TRAIN_START]

        ◄──────────────────────── STOP

──────────────────────────────────────────────────────────────────
DESCRIPCIÓN DE CADA MENSAJE
──────────────────────────────────────────────────────────────────
READY
    Worker → PS  |  El Worker está listo. PS asigna ID.
    payload: {}

WORKER_ID
    PS → Worker  |  ID asignado por el PS.
    payload: {"worker_id": int}

CNN_WEIGHTS
    PS → Worker  |  El PS envía los pesos de su CNN preentrenada.
                    El Worker los carga y extrae sus features de train.
                    Esto garantiza que TODOS los Workers usan exactamente
                    la misma CNN, independientemente de su hardware.
    payload: {
        "arch":          str,    # "simple" o "resnet18"
        "weights_bytes": bytes,  # state_dict serializado con torch.save
    }

CNN_READY
    Worker → PS  |  El Worker terminó de extraer sus features de train
                    con la CNN recibida. El PS espera este mensaje de
                    todos los Workers antes de enviar TRAIN_START.
    payload: {"worker_id": int}

REQUEST_TEST_FEATURES
    PS → Worker  |  El PS pide explícitamente los features de prueba
                    a un Worker específico, tras la barrera CNN_READY.
                    Reemplaza el envío automático de TEST_FEATURES:
                    el PS elige a qué Worker pedirlos y hace failover
                    si ese Worker falla, evitando transferencias redundantes.
    payload: {}  # sin parámetros — el Worker ya tiene la CNN cargada

TRAIN_SAMPLE
    PS → Worker  |  El PS pide una muestra de imágenes de entrenamiento
                    para preentrenar la CNN sin usar datos de prueba.
                    Elimina el sesgo de usar X_test en el pretrain.
    payload: {"n_samples": int}  # número de imágenes a enviar

TRAIN_SAMPLE_DATA
    Worker → PS  |  Respuesta con la muestra de imágenes de train.
                    El Worker selecciona aleatoriamente n_samples
                    imágenes de sus X_raw (imágenes originales, no features).
    payload: {
        "X_sample": np.ndarray,  # (n_samples, 3, 32, 32) float32
        "Y_sample": np.ndarray,  # (n_samples,) int32
    }

TRAIN_START
    PS → Worker  |  Inicia una sesión de entrenamiento.
    payload: {"epochs": int, "n_train": int, "n_workers": int, "worker_rank": int}

PARAMS
    PS → Worker  |  Pesos MLP globales + semilla de época.
    payload: {"epoch": int, "params": Dict[str, ndarray], "seed": int}

GRADIENTS
    Worker → PS  |  Gradientes calculados sobre el batch asignado.
    payload: {
        "worker_id": int,
        "epoch":     int,
        "gradients": Dict[str, ndarray],
        "loss":      float,
        "accuracy":  float,
    }

STOP
    PS → Worker  |  El PS se apaga.
    payload: None
"""

import pickle
import socket
import struct
from enum import Enum
from typing import Any, Dict


# ================================================================
# TIPOS DE MENSAJE
# ================================================================


class MsgType(str, Enum):
    """
    Tipos de mensaje válidos en el protocolo.

    Extiende ``str`` para que los valores sean directamente
    comparables con strings y se serialicen bien en JSON.
    """

    READY = "READY"
    WORKER_ID = "WORKER_ID"
    CNN_WEIGHTS = "CNN_WEIGHTS"
    CNN_READY = "CNN_READY"
    TEST_FEATURES = "TEST_FEATURES"
    REQUEST_TEST_FEATURES = "REQUEST_TEST_FEATURES"
    TRAIN_SAMPLE = "TRAIN_SAMPLE"
    TRAIN_SAMPLE_DATA = "TRAIN_SAMPLE_DATA"
    TRAIN_START = "TRAIN_START"
    PARAMS = "PARAMS"
    GRADIENTS = "GRADIENTS"
    STOP = "STOP"


# ================================================================
# SERIALIZACIÓN
# ================================================================


def encode(msg_type: MsgType, payload: Any) -> bytes:
    """
    Serializa un mensaje a bytes listos para enviar por socket TCP.

    FORMATO DEL PROTOCOLO:
    ──────────────────────
    ┌─────────────────┬──────────────────────┐
    │ 4 bytes (BE)    │ N bytes              │
    │ longitud N      │ pickle.dumps(msg)    │
    └─────────────────┴──────────────────────┘

    El prefijo de 4 bytes es CRÍTICO: TCP es un stream sin límites de
    mensaje. Sin el prefijo el receptor no sabe dónde termina un mensaje
    y empieza el siguiente. Con big-endian los números son independientes
    de la arquitectura de máquina.

    VENTAJA DE PICKLE:
    ──────────────────
    • np.ndarray se serializa nativo (binario) sin conversión a lista.
    • Tamaño reducido: array (512,) float32 ocupa ~2 KB en Pickle vs
      ~4 KB en JSON (diferencia crítica en redes lentas).
    • Preserva tipos de datos: float32 sigue siendo float32 al deserializar.

    :param msg_type: Tipo del mensaje de enumeración MsgType.
    :type msg_type: MsgType, ej. MsgType.PARAMS, MsgType.GRADIENTS.

    :param payload: Contenido del mensaje. Puede contener np.ndarray,
                    diccionarios, floats. Pickle los maneja nativamente.
    :type payload: Any, típicamente Dict con 'epochs', 'params', etc.

    :return: Bytes con prefijo de longitud + bloque Pickle lista para enviar.
    :rtype: bytes, formato: struct.pack(">I", len) + pickle.dumps(msg).
    """
    message = {"type": msg_type, "payload": payload}
    # Serialización Pickle con protocolo más reciente (HIGHEST_PROTOCOL=5 en modern Python)
    # Protocol 5 optimiza objetos grandes (crucial para arrays NumPy)
    body = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)

    # Convierte la longitud del mensaje en un prefijo de 4 bytes en formato big-endian.
    # Luego, se concatena el prefijo con el mensaje real.
    return struct.pack(">I", len(body)) + body


def decode(raw: bytes) -> Dict[str, Any]:
    """
    Deserializa bytes Pickle a un diccionario de mensaje Python.

    FORMATO ESPERADO:
    El bloque de bytes debe ser resultado de encode() (sin prefijo de longitud).
    Contiene un dict serializado con Pickle con estructura:
        {"type": MsgType, "payload": <datos>}

    PRESERVACIÓN DE TIPOS:
    Pickle mantiene tipos NumPy: float32 sigue siendo float32 tras
    deserialización, a diferencia de JSON que lo convierte a float.

    :param raw: Bytes serializados con Pickle (sin prefijo de 4 bytes).
    :type raw: bytes, resultado de pickle.dumps(dict).

    :return: Diccionario con claves "type" (MsgType) y "payload" (Any).
    :rtype: Dict[str, Any].
    """
    return pickle.loads(raw)


# ================================================================
# ENVÍO Y RECEPCIÓN SOBRE SOCKET TCP
# ================================================================


def send_message(sock: socket.socket, msg_type: MsgType, payload: Any) -> None:
    """
    Envía un mensaje completo por un socket TCP.

    MANEJO DE FRAGMENTACIÓN TCP:
    ────────────────────────────
    TCP no garantiza que sock.send() envíe todos los bytes en una llamada.
    Si los datos son grandes, pueden ser fragmentados. Este helper implementa
    un loop para asegurar que se envían exactamente len(data) bytes antes
    de retornar.

    EJEMPLO DE FLUJO:
    ─────────────────
    • Datos a enviar: 10 MB
    • Llamada 1 a send(): devuelve 2 MB (3 MB restantes)
    • Llamada 2 a send(): devuelve 5 MB (5 MB restantes)
    • Llamada 3 a send(): devuelve 3 MB (0 MB restantes)
    • Retorna — se enviaron 10 MB completos

    :param sock: Socket TCP conectado (debe estar activo).
    :type sock: socket.socket con estado ESTABLISHED.

    :param msg_type: Tipo de mensaje (PARAMS, GRADIENTS, etc.).
    :type msg_type: MsgType, ej. MsgType.GRADIENTS.

    :param payload: Contenido del mensaje (dict, array, etc.).
    :type payload: Any, típicamente Dict con datos numéricos.

    :return: None (mensaje enviado completo).
    :rtype: NoneType.

    :raises ConnectionError: Si socket se cierra antes de enviar todo.
    """
    data = encode(msg_type, payload)

    # Lleva la cuenta de cuántos bytes ya se enviaron
    total_sent = 0

    while total_sent < len(data):
        # TCP puede enviar solo parte del mensaje en una llamada a sock.send()
        # Envía desde el offset total_sent hasta el final
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise ConnectionError("Socket cerrado antes de completar el envío")
        total_sent += sent


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    """
    Recibe un mensaje completo desde un socket TCP.

    PROTOCOLO DE RECEPCIÓN:
    ──────────────────────
    1. Lee exactamente 4 bytes → desempaqueta como big-endian unsigned int (longitud N).
    2. Lee exactamente N bytes (el cuerpo del mensaje serializado con Pickle).
    3. Deserializa Pickle y retorna Dict["type", "payload"].

    GARANTÍA DE COMPLETITUD:
    El helper _recv_exact() maneja fragmentación TCP — garantiza leer
    exactamente n_bytes antes de retornar, incluso si TCP fragmentó
    el envío en múltiples packets.

    EJEMPLO:
    ────────
    • Receptor espera 4 bytes de longitud.
    • TCP entrega 2 bytes en packet 1, 2 bytes en packet 2.
    • _recv_exact(sock, 4) itera hasta recibir exactamente 4.
    • Desempaqueta: len(body) = 1024
    • Luego _recv_exact(sock, 1024) lee el cuerpo hasta obtener 1024 bytes total.

    :param sock: Socket TCP conectado (debe estar en estado ESTABLISHED).
    :type sock: socket.socket.

    :return: Diccionario con estructura:
                {"type": MsgType, "payload": <contenido>}
    :rtype: Dict[str, Any].

    :raises ConnectionError: Si socket se cierra inesperadamente antes de
                            completar la lectura (ej. peer disconnected).
    """
    # Paso 1: Lee prefijo de longitud (4 bytes, big-endian)
    raw_length = _recv_exact(sock, 4)
    # Desempaqueta: ">I" = (big-endian, unsigned int, 4 bytes)
    # Resultado: tupla con un elemento, extraemos con [0]
    length = struct.unpack(">I", raw_length)[0]

    # Paso 2: Lee el cuerpo del mensaje (length bytes)
    raw_body = _recv_exact(sock, length)

    # Paso 3: Deserializa Pickle a Dict
    return decode(raw_body)


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    """
    Lee exactamente ``n_bytes`` bytes de un socket.

    TCP puede fragmentar los datos; este helper garantiza que se
    leen todos antes de retornar.

    :param sock: Socket TCP conectado.
    :type sock: socket.socket

    :param n_bytes: Número exacto de bytes a leer.
    :type n_bytes: int

    :return: Bytes leídos.
    :rtype: bytes

    :raises ConnectionError: Si el socket se cierra antes de leer todo.
    """
    # Bytes acumulados hasta ahora
    buffer = b""

    while len(buffer) < n_bytes:
        # Intenta leer los bytes que faltan
        chunk = sock.recv(n_bytes - len(buffer))
        if not chunk:
            raise ConnectionError(
                f"Conexión cerrada: se esperaban {n_bytes} bytes, "
                f"solo llegaron {len(buffer)}"
            )
        buffer += chunk
    return buffer
