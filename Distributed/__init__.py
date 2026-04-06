"""
Distributed — Entrenamiento distribuido asíncrono con Parameter Server.

Implementa un sistema de entrenamiento distribuido sobre ImageNet-1k donde múltiples
Workers entrenan modelos de forma asíncrona sin sincronización. Soporta DOS modos:

  1. **SimpleCNN E2E**: CNN + MLP entrenables (ambas redes reciben gradientes locales)
  2. **ResNet-18 MLP-only**: CNN congelada (solo MLP se entrena, CNN es extractor fijo)

ARQUITECTURA
============

**Parameter Server (PS)**:
  - Coordinador central que mantiene el estado global (CNN + MLP en ambos modos).
  - No bloqueante: cada Worker recibe parámetros, entrena, y envía actualizaciones.
  - Aplica FedAvg asíncrono con corrección de staleness (λ-factor).
  - Soporta evaluación periódica en validación y exportación de historiales.

**Workers**:
  - Nodos computacionales independientes que corren streaming de ImageNet-1k.
  - Loop autónomo: REQUEST_PARAMS → TRAIN → UPDATES, sin barreras entre Workers.
  - SimpleCNN: Entrena ambas redes (CNN + MLP backprop habilitado)
  - ResNet-18: Entrena solo MLP (CNN congelada con requires_grad=False)
  - Soportan gradient accumulation (accum_steps) para reducir overhead de comunicación.
  - Pueden ejecutarse en CPU o GPU (CUDA/MPS).

**Protocol**:
  - Mensajes Pickle sobre TCP con garantía de integridad (4-byte length prefix).
  - Serialización nativa de state_dict PyTorch (Dict[str, np.ndarray]).
  - Sincroniza estado de CNN completo aunque solo MLP de ResNet-18 reciba updates.
  - Mensajes: READY, WORKER_ID, CNN_WEIGHTS, CNN_ACK, START, REQUEST_PARAMS,
    PARAMS, UPDATES, STOP.

FLUJO COMPLETO
==============

  PS Listen        Workers Ready         CNN WEIGHTS     START          Async Loop
  ─────────────────────────────────────────────────────────────────────────────
  listen()         READY ─────────────►
  assign WORKER_ID ◄─────────────────── (ID asignado)
  send CNN_WEIGHTS ─────────────────► (ResNet18/Simple)
                   CNN_ACK ───────────►
  send START ──────────────────────────►
                   [loop infinito, sin sincronización:]
                   REQUEST_PARAMS ─────►
                   ◄──────────────────── PARAMS (version, mlp_state, cnn_state, lr)
                   [entrena N batches]
                   UPDATES ────────────► (loss, acc, mlp_weights, cnn_weights)
                   [repetir indefinidamente]

MÓDULOS
=======

parameter_server : module
    Clase ParameterServer: Coordinador central, manejo de conexiones TCP,
    FedAvg asíncrono, evaluación periódica, callbacks para eventos.

    Métodos principales:
    - listen(): Abre socket TCP y espera Workers
    - set_cnn(cnn): Establece CNN extraída de características
    - set_mlp(mlp_state): Establece MLP inicial como Dict[str, np.ndarray]
    - evaluate(): Evalúa modelo global en validación
    - stop(): Detiene servidor y desconecta Workers

    Callbacks:
    - on_worker_connected(wid, addr)
    - on_worker_disconnected(wid)
    - on_step(step, loss, acc, staleness)
    - on_report(step, loss, acc)

worker_node : module
    Clase WorkerNode: Nodo independiente de entrenamiento.

    Método principal:
    - run(): Conecta al PS, carga CNN, inicia loop de entrenamiento infinito

    Loop interno:
    1. REQUEST_PARAMS → solicita parámetros globales
    2. _sync_model() → carga CNN + MLP con estado global
    3. _train_batch() × accum_steps → forward + backward + SGD local
    4. UPDATES → envía pesos actualizados + métricas al PS

protocol : module
    Protocolo de red sobre TCP.

    Tipos de mensajes (MsgType enum):
    - READY, WORKER_ID, CNN_WEIGHTS, CNN_ACK, START
    - REQUEST_PARAMS, PARAMS, UPDATES, STOP

    Funciones:
    - send_message(sock, msg_type, payload): Envía mensaje serializado
    - receive_message(sock): Recibe y deserializa mensaje completo
    - _recv_exact(sock, n): Lee exactamente n bytes (helper privado)

EXPORTACIONES PRINCIPALES
==========================

ParameterServer : class
    from Distributed.parameter_server import ParameterServer

WorkerNode : class
    from Distributed.worker_node import WorkerNode

MsgType : enum
    from Distributed.protocol import MsgType

send_message : function
    from Distributed.protocol import send_message

receive_message : function
    from Distributed.protocol import receive_message

FLUJO TÍPICO
============

    # Parameter Server
    from Distributed.parameter_server import ParameterServer
    from Distributed.protocol import MsgType
    from Model import CNNExtractor, MLPPyTorch

    ps = ParameterServer(host='0.0.0.0', port=9999, ...)
    ps.listen()

    cnn = CNNExtractor(arch='resnet18')
    ps.set_cnn(cnn)

    mlp = MLPPyTorch(feature_dim=512, hidden1=1024, hidden2=512, n_classes=1000)
    ps.set_mlp(mlp.state_dict_numpy())

    # Worker
    from Distributed.worker_node import WorkerNode

    worker = WorkerNode(
        server_host='127.0.0.1', server_port=9999,
        dataset_name='ILSVRC/imagenet-1k',
        worker_rank=0, num_workers=1, batch_size=64
    )
    worker.run()
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
