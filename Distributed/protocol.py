"""
Distributed/protocol.py

Protocolo de comunicación entre el Parameter Server y los Workers.

──────────────────────────────────────────────────────────────────
FORMATO DE MENSAJE
──────────────────────────────────────────────────────────────────
Cada mensaje es un diccionario Python serializado como JSON (UTF-8)
y precedido de 4 bytes (big-endian) que indican la longitud del
string codificado:

    ┌──────────────┬────────────────────────────────────────────┐
    │  4 bytes     │  N bytes                                   │
    │  longitud N  │  json.dumps({"type": ..., "payload": …})   │
    │  (big-endian)│  codificado como UTF-8                     │
    └──────────────┴────────────────────────────────────────────┘

El prefijo de longitud es imprescindible porque TCP es un stream:
sin él no hay forma de saber dónde termina un mensaje y dónde
empieza el siguiente.

──────────────────────────────────────────────────────────────────
ARRAYS NUMPY EN JSON
──────────────────────────────────────────────────────────────────
JSON no conoce np.ndarray. La solución es representar cada array
como un objeto con su forma, su tipo y sus datos:

    {"__ndarray__": true, "dtype": "float64", "shape": [30, 784], "data": [[...]]}

``encode`` convierte todos los ndarrays del payload antes de
serializar. ``decode`` los reconstruye después de deserializar.
Esta conversión es transparente: PS y Worker siguen trabajando
con np.ndarray sin saber nada del formato de red.

──────────────────────────────────────────────────────────────────
TIPOS DE MENSAJE
──────────────────────────────────────────────────────────────────
READY
    Worker → Parameter Server
    El worker está conectado y listo para recibir trabajo.
    payload: {"worker_id": int}

PARAMS
    Parameter Server → Worker (broadcast)
    Parámetros globales de la red + índices del batch de esta época.
    payload: {
        "epoch":   int,
        "params":  Dict[str, np.ndarray],   # W1, b1, W2, b2
        "indices": List[int],               # índices en MNIST train
    }

GRADIENTS
    Worker → Parameter Server
    Gradientes calculados sobre el batch asignado.
    payload: {
        "worker_id": int,
        "epoch":     int,
        "gradients": Dict[str, np.ndarray], # dW1, db1, dW2, db2
        "loss":      float,
        "accuracy":  float,
    }

STOP
    Parameter Server → Worker (broadcast)
    Señal de fin de entrenamiento; el worker debe cerrar la conexión.
    payload: None
"""

import json
import socket
import struct
from enum import Enum
from typing import Any, Dict

import numpy as np


# ================================================================
# TIPOS DE MENSAJE
# ================================================================


class MsgType(str, Enum):
    """
    Tipos de mensaje válidos en el protocolo.

    Extiende ``str`` para que los valores sean directamente
    comparables con strings y se serialicen bien en JSON.
    """
    READY     = "READY"
    PARAMS    = "PARAMS"
    GRADIENTS = "GRADIENTS"
    STOP      = "STOP"


# ================================================================
# CONVERSIÓN NDARRAY ↔ JSON
# ================================================================


def _arrays_to_json(obj: Any) -> Any:
    """
    Convierte recursivamente todos los ``np.ndarray`` en un objeto
    Python a una representación serializable en JSON.

    Un array se convierte en:
        {"__ndarray__": true, "dtype": "float64", "shape": [...], "data": [...]}

    Cualquier otro tipo se devuelve sin cambios.

    :param obj: Objeto a convertir (dict, list, ndarray, escalar…).
    :type obj: Any

    :return: Objeto equivalente sin ndarrays.
    :rtype: Any
    """
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype":       str(obj.dtype),
            "shape":       list(obj.shape),
            "data":        obj.tolist(),
        }
    if isinstance(obj, dict):
        return {k: _arrays_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_arrays_to_json(item) for item in obj]
    # Los escalares NumPy (int64, float32, etc.) tampoco son
    # serializables en JSON directamente; se convierten a Python nativo.
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _json_to_arrays(obj: Any) -> Any:
    """
    Reconstruye recursivamente los ``np.ndarray`` a partir de su
    representación JSON generada por ``_arrays_to_json``.

    :param obj: Objeto deserializado desde JSON.
    :type obj: Any

    :return: Objeto con ndarrays reconstruidos.
    :rtype: Any
    """
    if isinstance(obj, dict):
        if obj.get("__ndarray__") is True:
            return np.array(obj["data"], dtype=obj["dtype"])
        return {k: _json_to_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_to_arrays(item) for item in obj]
    return obj


# ================================================================
# SERIALIZACIÓN
# ================================================================


def encode(msg_type: MsgType, payload: Any) -> bytes:
    """
    Serializa un mensaje a bytes listos para enviar por socket.

    Convierte los ndarrays del payload, serializa a JSON (UTF-8)
    y añade el prefijo de 4 bytes con la longitud.

    :param msg_type: Tipo del mensaje.
    :type msg_type: MsgType

    :param payload: Contenido del mensaje.
    :type payload: Any

    :return: Bytes con prefijo de longitud seguidos del JSON codificado.
    :rtype: bytes
    """
    message = {
        "type":    msg_type.value,
        "payload": _arrays_to_json(payload),
    }
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def decode(raw: bytes) -> Dict[str, Any]:
    """
    Deserializa bytes a un diccionario de mensaje.

    Reconstruye los ndarrays embebidos en el payload JSON.

    :param raw: Bytes del cuerpo del mensaje (sin prefijo de longitud).
    :type raw: bytes

    :return: Diccionario con claves ``"type"`` y ``"payload"``.
    :rtype: Dict[str, Any]
    """
    message = json.loads(raw.decode("utf-8"))
    message["payload"] = _json_to_arrays(message["payload"])
    return message


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
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise ConnectionError("Socket cerrado antes de completar el envío")
        total_sent += sent


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    """
    Recibe un mensaje completo desde un socket TCP.

    Primero lee los 4 bytes de longitud, luego lee exactamente
    esa cantidad de bytes del cuerpo del mensaje.

    :param sock: Socket TCP conectado.
    :type sock: socket.socket

    :return: Diccionario con claves ``"type"`` y ``"payload"``.
    :rtype: Dict[str, Any]

    :raises ConnectionError: Si el socket se cierra inesperadamente.
    """
    raw_length = _recv_exact(sock, 4)
    length     = struct.unpack(">I", raw_length)[0]
    raw_body   = _recv_exact(sock, length)
    return decode(raw_body)


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    """
    Lee exactamente ``n_bytes`` bytes de un socket.

    TCP puede fragmentar los datos en varios segmentos; este helper
    garantiza que se leen todos antes de retornar.

    :param sock: Socket TCP conectado.
    :type sock: socket.socket

    :param n_bytes: Número exacto de bytes a leer.
    :type n_bytes: int

    :return: Bytes leídos.
    :rtype: bytes

    :raises ConnectionError: Si el socket se cierra antes de leer ``n_bytes``.
    """
    buffer = b""
    while len(buffer) < n_bytes:
        chunk = sock.recv(n_bytes - len(buffer))
        if not chunk:
            raise ConnectionError(
                f"Conexión cerrada: se esperaban {n_bytes} bytes, "
                f"solo llegaron {len(buffer)}"
            )
        buffer += chunk
    return buffer
