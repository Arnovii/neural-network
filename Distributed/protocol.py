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
    """
    Lee exactamente un mensaje completo desde socket TCP (bloqueante).
    
    Decodificación:
    1. Lee 4 bytes big-endian para obtener longitud
    2. Lee exactamente 'longitud' bytes con _recv_exact (maneja recv parciales)
    3. Deserializa con pickle.loads
    
    Thread-safe para múltiples sockets (cada Worker tiene el suyo).
    
    :param sock: Socket TCP conectado en modo bloqueante
    :type sock: socket.socket
    
    :returns: Dict con "type" (MsgType) y "payload" (datos)
    :rtype: Dict[str, Any]
    
    :raises ConnectionError: Si socket se cierra antes de recibir mensaje completo
    :raises pickle.UnpicklingError: Si datos no son pickle válido
    """
    length = struct.unpack(">I", _recv_exact(sock, 4))[0]
    return pickle.loads(_recv_exact(sock, length))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Recibe exactamente n bytes del socket (maneja recv() parciales).
    
    Útil porque socket.recv() puede devolver menos bytes que n,
    especialmente en redes lentas. Esta función acumula hasta tener n bytes.
    
    :param sock: Socket TCP conectado
    :type sock: socket.socket
    :param n: Número exacto de bytes a recibir
    :type n: int
    
    :returns: Exactamente n bytes
    :rtype: bytes
    
    :raises ConnectionError: Si socket se cierra antes de recibir n bytes
    """
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Conexión cerrada: esperados {n} bytes, recibidos {len(buf)}"
            )
        buf += chunk
    return buf
