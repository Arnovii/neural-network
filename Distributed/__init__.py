"""
Distributed — Entrenamiento distribuido asíncrono con Parameter Server.

Implementa un sistema de entrenamiento distribuido end-to-end (E2E) sobre ImageNet-1k
donde múltiples Workers entrenan modelos de forma asíncrona sin sincronización.

ARQUITECTURA
============

**Parameter Server (PS)**:
  - Coordinador central que mantiene el estado global (CNN + MLP).
  - Dicta no bloqueante: cada Worker recibe parámetros, entrena, y envía actualizaciones.
  - Aplica FedAvg asíncrono con corrección de staleness (λ-factor).
  - Soporta evaluación periódica en validación y exportación de historiales.

**Workers**:
  - Nodos computacionales independientes que corren streaming de ImageNet-1k.
  - Loop autónomo: REQUEST_PARAMS → TRAIN → UPDATES, sin barreras entre Workers.
  - Soportan gradient accumulation (accum_steps) para reducir overhead de comunicación.
  - Pueden ejecutarse en CPU o GPU (CUDA/MPS).

**Protocol**:
  - Mensajes Pickle sobre TCP con garantía de integridad (4-byte length prefix).
  - Serialización nativa de state_dict PyTorch (Dict[str, np.ndarray]).

FLUJO COMPLETO
==============

  PS Listen        Workers Ready         CNN WEIGHTS     START          Async Loop
  ──────────────────────────────────────────────────────────────────────────────
  listen()         READY ────────────►
  assign WORKER_ID ◄──────────────────── (ID asignado)
  send CNN_WEIGHTS ────────────────────► (ResNet18/Simple)
                   CNN_ACK ──────────────►
  send START ──────────────────────────►
                   [loop infinito, sin sincronización:]
                   REQUEST_PARAMS ──────►
                   ◄──────────────────── PARAMS (version, mlp_state, cnn_state, lr)
                   [entrena N batches]
                   UPDATES ──────────────► (loss, acc, mlp_weights, cnn_weights)
                   [repetir indefinidamente]

MÓDULOS
=======

parameter_server : module
    Clase ParameterServer: Coordinador central, manejo de conexiones TCP,
    FedAvg asíncrono, evaluación periódica, callbacks para eventos.

worker_node : module
    Clase WorkerNode: Streaming de ImageNet-1k, loop de entrenamiento E2E,
    sincronización de modelos, serialización de pesos.

protocol : module
    Protocolo de red: MsgType enum, send_message(), receive_message().
    Garantía de entrega completa de mensajes.

EXPORTACIONES PRINCIPALES
==========================

ParameterServer : class
    from Distributed.parameter_server import ParameterServer

WorkerNode : class
    from Distributed.worker_node import WorkerNode

MsgType : enum
    from Distributed.protocol import MsgType

send_message, receive_message : functions
    from Distributed.protocol import send_message, receive_message

receive_message : function
    Recibe y deserializa mensaje desde socket TCP.

Uso rápido
----------
    from Distributed import ParameterServer, WorkerNode

    # Servidor
    ps = ParameterServer(host="localhost", port=5000, training_mode="precomputed")
    ps.set_cnn(cnn_model)
    ps.listen()  # Espera Workers
    history = ps.train(epochs=5, initial_params=params, learning_rate=0.01)

    # Worker
    worker = WorkerNode(server_host="localhost", server_port=5000)
    worker.run()  # Conecta y comienza bucle de recepción de batch
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
