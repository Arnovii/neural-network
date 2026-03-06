"""
Distributed — Entrenamiento distribuido con Parameter Server y Workers.

Módulos
-------
parameter_server    Clase ParameterServer: ciclo de vida TCP + entrenamiento.
worker_node         Clase WorkerNode: cálculo de gradientes sobre un batch.
protocol            Protocolo de mensajes Pickle sobre TCP (MsgType, send/receive).

Uso rápido
----------
    from Distributed import ParameterServer, WorkerNode, MsgType
"""

from Distributed.parameter_server import ParameterServer
from Distributed.worker_node import WorkerNode
from Distributed.protocol import MsgType, send_message, receive_message

__all__ = [
    "ParameterServer",
    "WorkerNode",
    "MsgType",
    "send_message",
    "receive_message",
]
