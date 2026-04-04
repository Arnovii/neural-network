"""
Distributed/protocol.py

Protocolo de comunicación para entrenamiento distribuido asíncrono en ImageNet.

FORMATO DE MENSAJE:
    4 bytes big-endian (longitud) + pickle del cuerpo

FLUJO COMPLETO POR WORKER:

    Worker                          Parameter Server
    ──────                          ────────────────
    READY ────────────────────────► asigna ID
          ◄─────────────────────── WORKER_ID
          ◄─────────────────────── CNN_WEIGHTS  (pesos iniciales)
    CNN_ACK ──────────────────────►  (Worker cargó CNN)
          ◄─────────────────────── START
    [loop continuo, sin sincronización inter-worker:]
    REQUEST_PARAMS ───────────────►
          ◄─────────────────────── PARAMS  (mlp_params + cnn_state + version + lr)
    UPDATES ──────────────────────►  (mlp_weights + cnn_weights + métricas)
    [repetir indefinidamente]
          ◄─────────────────────── STOP

MENSAJES (7 — exactamente los necesarios):
    READY          Worker → PS    Handshake inicial
    WORKER_ID      PS → Worker    ID asignado
    CNN_WEIGHTS    PS → Worker    Pesos CNN iniciales (bytes serializados)
    CNN_ACK        Worker → PS    CNN cargada y lista
    START          PS → Worker    Señal de inicio del loop de entrenamiento
    REQUEST_PARAMS Worker → PS    Solicitar parámetros globales actuales
    PARAMS         PS → Worker    Parámetros actuales + versión + lr
    UPDATES        Worker → PS    Pesos actualizados tras entrenar 1 batch
    STOP           PS → Worker    Apagado limpio
"""

import pickle
import socket
import struct
from enum import Enum
from typing import Any, Dict


class MsgType(str, Enum):
    READY = "READY"
    WORKER_ID = "WORKER_ID"
    CONFIG = "CONFIG"  # PS → Worker: parámetros globales
    CNN_WEIGHTS = "CNN_WEIGHTS"
    CNN_ACK = "CNN_ACK"
    START = "START"
    REQUEST_PARAMS = "REQUEST_PARAMS"
    PARAMS = "PARAMS"
    UPDATES = "UPDATES"
    STOP = "STOP"


def send_message(sock: socket.socket, msg_type: MsgType, payload: Any) -> None:
    """Serializa y envía un mensaje completo por TCP."""
    body = pickle.dumps(
        {"type": msg_type, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    data = struct.pack(">I", len(body)) + body
    total = 0
    while total < len(data):
        sent = sock.send(data[total:])
        if sent == 0:
            raise ConnectionError("Socket cerrado durante el envío")
        total += sent


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    """Lee exactamente un mensaje completo desde el socket TCP."""
    length = struct.unpack(">I", _recv_exact(sock, 4))[0]
    return pickle.loads(_recv_exact(sock, length))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Conexión cerrada: esperados {n} bytes, recibidos {len(buf)}"
            )
        buf += chunk
    return buf
