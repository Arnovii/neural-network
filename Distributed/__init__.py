"""
Distributed — Entrenamiento distribuido con patrón Parameter Server.

Implementa un sistema de entrenamiento distribuido donde:

- **Parameter Server (PS)**: Coordinador central que mantiene pesos globales,
  coordina gradientes de múltiples Workers y realiza actualizaciones vía SGD.
  Soporta dos flujos mutuamente excluyentes:
  
  * PRECOMPUTED: CNN fija, solo MLP distribuido (~2 sec/epoch)
  * END-TO-END: CNN + MLP entrenan juntos (~12.5 sec/epoch)

- **Workers**: Nodos de cálculo independientes que reciben batches,
  computan gradientes MLP y reportan al PS. Soportan extracción de features
  con caching automático y determinismo reproducible.

- **Protocol**: Serialización de mensajes vía Pickle sobre TCP con garantías
  de integridad de datos y fragmentación transparente.

Módulos
-------
parameter_server : modulo
    Clase ParameterServer: ciclo de vida TCP, coordinación de entrenamientos,
    sincronización de Workers, gestión de estado.

worker_node : modulo
    Clase WorkerNode: procesamiento de datos, cálculo de gradientes,
    comunicación con PS, soporte para precomputed/end-to-end.

protocol : modulo
    Protocolo de mensajes: MsgType enum, encode/decode, send/receive.
    Usa Pickle para serialización transparente de NumPy arrays y dicts.

Exportaciones principales
-------------------------
ParameterServer : class
    Coordinador central del entrenamiento distribuido.
    Métodos: listen(), train(), shutdown().
    Callbacks: on_worker_connected, on_epoch_end, etc.

WorkerNode : class
    Nodo computacional independiente.
    Métodos: run() [bucle principal], connect/disconnect.

MsgType : enum
    Tipos de mensajes: READY, WORKER_ID, CNN_WEIGHTS, GRADIENTS, STOP, etc.

send_message : function
    Serializa y envía mensaje (msg_type, payload) vía TCP.

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
