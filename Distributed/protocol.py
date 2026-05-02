"""
Distributed/protocol.py

Protocolo de comunicación para entrenamiento distribuido asíncrono en ImageNet.

FORMATO DE MENSAJE:
    4 bytes big-endian (longitud) + pickle del cuerpo

FLUJO COMPLETO POR WORKER:

    Worker                          Parameter Server
    ──────                          ────────────────
    READY ────────────────────────► asigna ID único
          ◄─────────────────────── WORKER_ID
          ◄─────────────────────── CONFIG  (batch_size, image_size, rank, num_workers, seed)
          ◄─────────────────────── CNN_WEIGHTS  (pesos iniciales)
    CNN_ACK ──────────────────────►  (Worker cargó CNN)
          ◄─────────────────────── START
    [loop continuo, sin sincronización inter-worker:]
    REQUEST_PARAMS ───────────────►
          ◄─────────────────────── PARAMS  (mlp_params + cnn_state + version + lr)
    UPDATES ──────────────────────►  (mlp_weights + cnn_weights + métricas)
    [repetir indefinidamente]
          ◄─────────────────────── STOP

MENSAJES (10 — exactamente los necesarios):
    READY          Worker → PS    Handshake inicial
    WORKER_ID      PS → Worker    ID asignado
    CONFIG         PS → Worker    batch_size, image_size, rank, num_workers, seed
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
    """
    Tipos de mensajes del protocolo de comunicación TCP.

    Enum que define los 10 tipos de mensaje usados en el entrenamiento
    distribuido asíncrono entre Workers y Parameter Server.

    Flujo:
        Worker → PS:  READY, CNN_ACK, REQUEST_PARAMS, UPDATES
        PS → Worker: WORKER_ID, CONFIG, CNN_WEIGHTS, START, PARAMS, STOP
    """

    READY = "READY"
    WORKER_ID = "WORKER_ID"
    CONFIG = "CONFIG"
    CNN_WEIGHTS = "CNN_WEIGHTS"
    CNN_ACK = "CNN_ACK"
    START = "START"
    REQUEST_PARAMS = "REQUEST_PARAMS"
    PARAMS = "PARAMS"
    UPDATES = "UPDATES"
    STOP = "STOP"


def send_message(sock: socket.socket, msg_type: MsgType, payload: Any) -> None:
    """Serializa y envia un mensaje completo por TCP.

    El formato del mensaje es:

    - 4 bytes (big-endian): longitud del cuerpo
    - pickle(cuerpo): tipo + payload

    :param sock: Socket TCP conectado (debe estar activo).
    :type sock: socket.socket

    :param msg_type: Tipo de mensaje del enum MsgType.
        Ejemplos: MsgType.PARAMS, MsgType.UPDATES, MsgType.STOP.
    :type msg_type: MsgType

    :param payload: Contenido del mensaje. Tipicamente dict con datos numericos.
    :type payload: Any

    :returns: None. El mensaje se envia completamente por el socket.
    :rtype: None

    :raises ConnectionError: Si el socket se cierra antes de enviar todo el mensaje.

    .. note::
        Usa pickle.HIGHEST_PROTOCOL para maxima eficiencia de serializacion.
    """
    body = pickle.dumps(
        {"type": msg_type, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    data = struct.pack(">I", len(body)) + body  # Añade encabezado (big-endian)
    total = 0
    while total < len(data):
        sent = sock.send(data[total:])
        if sent == 0:
            raise ConnectionError("Socket cerrado durante el envío")
        total += sent


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    """Lee exactamente un mensaje completo desde socket TCP (bloqueante).

    Decodificacion:

    1. Lee 4 bytes big-endian para obtener longitud
    2. Lee exactamente 'longitud' bytes con _recv_exact (maneja recv parciales)
    3. Deserializa con pickle.loads

    :param sock: Socket TCP conectado en modo bloqueante.
    :type sock: socket.socket

    :returns: Diccionario con campos "type" (MsgType) y "payload" (datos).
    :rtype: Dict[str, Any]

    :raises ConnectionError: Si el socket se cierra antes de recibir mensaje completo.
    :raises pickle.UnpicklingError: Si los datos no son un pickle valido.

    .. note::
        Thread-safe para multiple sockets (cada Worker tiene el suyo).
    """
    length = struct.unpack(">I", _recv_exact(sock, 4))[0]
    return pickle.loads(_recv_exact(sock, length))  # noqa: S301 (trusted internal protocol)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Recibe exactamente n bytes del socket (maneja recv() parciales).

    Util porque socket.recv() puede devolver menos bytes que n,
    especialmente en redes lentas. Esta funcion acumula hasta tener n bytes.

    :param sock: Socket TCP conectado.
    :type sock: socket.socket

    :param n: Numero exacto de bytes a recibir.
    :type n: int

    :returns: Exactamente n bytes recibidos del socket.
    :rtype: bytes

    :raises ConnectionError: Si el socket se cierra antes de recibir n bytes.
    """
    buf = b""
    while len(buf) < n:  # Sigue leyendo hasta tener n bytes
        chunk = sock.recv(n - len(buf))  # Solo pide lo que falta
        if not chunk:
            raise ConnectionError(
                f"Conexión cerrada: esperados {n} bytes, recibidos {len(buf)}"
            )
        buf += chunk
    return buf
