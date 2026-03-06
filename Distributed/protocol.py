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
TIPOS DE MENSAJE Y FLUJO
──────────────────────────────────────────────────────────────────

  Worker                          Parameter Server
  ──────                          ────────────────
  READY ─────────────────────────►  (se conecta; PS asigna ID)
        ◄──────────────────────── WORKER_ID  (PS envía ID asignado)

  [en espera...]

        ◄──────────────────────── TRAIN_START  (PS inicia entrenamiento)

  [por cada época:]
        ◄──────────────────────── PARAMS  (params + índices)
  GRADIENTS ─────────────────────►

  [vuelve a esperar TRAIN_START para el siguiente entrenamiento]

        ◄──────────────────────── STOP  (PS se apaga)
  [Worker cierra conexión]

──────────────────────────────────────────────────────────────────
DESCRIPCIÓN DE CADA MENSAJE
──────────────────────────────────────────────────────────────────
READY
    Worker → PS  |  El worker está conectado y listo.
    payload: {}   (sin datos; el PS asigna el ID)

WORKER_ID
    PS → Worker  |  ID asignado por el PS a este Worker.
    payload: {"worker_id": int}

TRAIN_START
    PS → Worker (broadcast)  |  Comienza una sesión de entrenamiento.
    payload: {"epochs": int, "n_train": int}

PARAMS
    PS → Worker (broadcast)  |  Pesos globales + índices del batch.
    payload: {
        "epoch":   int,
        "params":  Dict[str, np.ndarray],   # W1, b1, W2, b2
        "indices": List[int],
    }

GRADIENTS
    Worker → PS  |  Gradientes calculados sobre el batch asignado.
    payload: {
        "worker_id": int,
        "epoch":     int,
        "gradients": Dict[str, np.ndarray], # dW1, db1, dW2, db2
        "loss":      float,
        "accuracy":  float,
    }

STOP
    PS → Worker (broadcast)  |  El PS se apaga; el Worker debe cerrar.
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
    TRAIN_START = "TRAIN_START"
    PARAMS = "PARAMS"
    GRADIENTS = "GRADIENTS"
    STOP = "STOP"


# ================================================================
# SERIALIZACIÓN
# ================================================================


def encode(msg_type: MsgType, payload: Any) -> bytes:
    """
    Serializa un mensaje a bytes listos para enviar por socket.

    :param msg_type: Tipo del mensaje.
    :type msg_type: MsgType

    :param payload: Contenido del mensaje. Puede contener np.ndarray
                    directamente; Pickle los serializa sin conversión.
    :type payload: Any

    :return: Bytes con prefijo de longitud seguidos del bloque Pickle.
    :rtype: bytes
    """
    message = {"type": msg_type, "payload": payload}
    body = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)

    # Convierte la longitud del mensaje en un prefijo de 4 bytes en formato big-endian.
    # Luego, se concatena el prefijo con el mensaje real.
    return struct.pack(">I", len(body)) + body


def decode(raw: bytes) -> Dict[str, Any]:
    """
    Deserializa bytes a un diccionario de mensaje.

    :param raw: Bytes del cuerpo del mensaje (sin prefijo de longitud).
    :type raw: bytes

    :return: Diccionario con claves ``"type"`` y ``"payload"``.
    :rtype: Dict[str, Any]
    """
    return pickle.loads(raw)


# ================================================================
# ENVÍO Y RECEPCIÓN SOBRE SOCKET TCP
# ================================================================


def send_message(sock: socket.socket, msg_type: MsgType, payload: Any) -> None:
    """
    Envía un mensaje completo por un socket TCP.

    :param sock: Socket TCP conectado.
    :type sock: socket.socket

    :param msg_type: Tipo del mensaje.
    :type msg_type: MsgType

    :param payload: Contenido del mensaje.
    :type payload: Any

    :raises ConnectionError: Si el socket se cierra antes de enviar todo.
    """
    data = encode(msg_type, payload)

    # Lleva la cuenta de cuántos bytes ya se enviaron
    total_sent = 0

    while total_sent < len(data):
        # TCP puede enviar solo parte del mensaje en una llamada a sock.send()
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise ConnectionError("Socket cerrado antes de completar el envío")
        total_sent += sent


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    """
    Recibe un mensaje completo desde un socket TCP.

    Garantiza que el PS lea exactamente un mensaje completo, y luego lo
    decodifica a un diccionario Python listo para usar.

    :param sock: Socket TCP conectado.
    :type sock: socket.socket

    :return: Diccionario con claves ``"type"`` y ``"payload"``.
    :rtype: Dict[str, Any]

    :raises ConnectionError: Si el socket se cierra inesperadamente.
    """
    raw_length = _recv_exact(sock, 4)
    # Convierte los 4 bytes de longitud a un entero usando big-endian.
    length = struct.unpack(">I", raw_length)[0]
    raw_body = _recv_exact(sock, length)
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
