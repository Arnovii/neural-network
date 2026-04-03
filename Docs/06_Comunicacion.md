# Protocolo de Comunicación PS ↔ Worker

## Overview del Protocolo

**Archivo**: `Distributed/protocol.py`

**Características**:
- TCP/IP confiable (no hay pérdida de paquetes)
- Serialización con `pickle` (preserva tipos Python)
- Longitud-prefijo para delimitar mensajes (garantiza sincronización)
- 9 tipos de mensajes (enum MsgType)

---

## Formato de Mensaje

### Estructura Binaria

```
┌─────────────────────────────────────┐
│ 4 bytes: Longitud del cuerpo        │ (big-endian unsigned int)
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ N bytes: Cuerpo pickle              │
│ {                                   │
│   "type": "REQUEST_PARAMS",         │
│   "payload": {...}                  │
│ }                                   │
└─────────────────────────────────────┘
```

### Ejemplo en Bytes

```python
# Mensaje
{"type": "REQUEST_PARAMS", "payload": {}}

# Pickle ~50 bytes
body = pickle.dumps({...}, protocol=pickle.HIGHEST_PROTOCOL)
# len(body) = 50

# Binario completo
data = struct.pack(">I", 50) + body
#      ↑ 4 bytes        ↑ 50 bytes
#      └─ "00 00 00 32" en hex (50 en decimal)
```

---

## Tipos de Mensajes (9 Total)

### 1. READY (Worker → PS)

**Propósito**: Iniciar handshake

```python
send_message(sock, MsgType.READY, {})
```

**Payload**: Vacío `{}`

**Respuesta esperada**: `WORKER_ID`

---

### 2. WORKER_ID (PS → Worker)

**Propósito**: Asignar identificador único

```python
send_message(sock, MsgType.WORKER_ID, {
    "worker_id": 0  # ID asignado por PS
})
```

**Payload**:
- `worker_id` (int): Identificador único (0, 1, 2, ...)

**Rango**:0-2^31-1 (32-bit signed int)

---

### 3. CNN_WEIGHTS (PS → Worker)

**Propósito**: Distribuir arquitectura e pesos CNN

```python
send_message(sock, MsgType.CNN_WEIGHTS, {
    "arch": "resnet18",
    "weights_bytes": b'\x80\x04\x95...'  # torch.save() bytes
})
```

**Payload**:
- `arch` (str): "resnet18" o "simple"
- `weights_bytes` (bytes): ~44-6 MB (ResNet-18 o Simple)

**Tamaño total**: ~44 MB TCP packet → ~500ms a 1Gbps, ~50ms a 10Gbps

---

### 4. CNN_ACK (Worker → PS)

**Propósito**: Confirmar carga de CNN

```python
send_message(sock, MsgType.CNN_ACK, {
    "worker_id": 0
})
```

**Payload**:
- `worker_id` (int): Para PS validar

---

### 5. START (PS → Worker)

**Propósito**: Señal para iniciar training loop

```python
send_message(sock, MsgType.START, {})
```

**Payload**: Vacío

---

### 6. REQUEST_PARAMS (Worker → PS)

**Propósito**: Solicitar parámetros globales actuales

```python
send_message(sock, MsgType.REQUEST_PARAMS, {})
```

**Payload**: Vacío

**Respuesta esperada**: `PARAMS`

**Frecuencia**: Cada ~100-500ms (cada iteración del training loop)

---

### 7. PARAMS (PS → Worker)

**Propósito**: Enviar estado global actual

```python
send_message(sock, MsgType.PARAMS, {
    "mlp_state": {
        "fc1.weight": np.array(...),
        "fc1.bias": np.array(...),
        "fc2.weight": np.array(...),
        "fc2.bias": np.array(...),
        "fc3.weight": np.array(...),
        "fc3.bias": np.array(...),
    },
    "cnn_state": {
        "layer1.0.conv1.weight": np.array(...),
        # ... 49 llaves más ...
    },
    "version": 42,
    "lr": 0.001
})
```

**Payload**:
- `mlp_state` (Dict[str, np.ndarray]): 6 arrays (fc1-3 weights+bias)
- `cnn_state` (Dict[str, np.ndarray]): ~49 arrays (ResNet-18 state)
- `version` (int): Versión de parámetros (contador monotónico)
- `lr` (float): Learning rate actual (0 < lr ≤ 0.1 típicamente)

**Tamaño total**: ~50 MB (4.5 MB MLP + 44 MB CNN) → ~500ms/10Gbps

---

### 8. UPDATES (Worker → PS)

**Propósito**: Enviar gradientes acumulados + métricas

```python
send_message(sock, MsgType.UPDATES, {
    "mlp_weights": { ... },      # MLP actualizado tras entrenar
    "cnn_weights": { ... },      # CNN actualizado (se ignora)
    "loss": 8.374,
    "accuracy": 0.0,
    "batch_size": 64,
    "version_read": 42
})
```

**Payload**:
- `mlp_weights` (Dict[str, np.ndarray]): 6 arrays actualizado
- `cnn_weights` (Dict[str, np.ndarray]): ~49 arrays (actualizado pero se ignora en PS)
- `loss` (float): Loss promedio de los batches acumulados
- `accuracy` (float): Accuracy (%) promedio
- `batch_size` (int): Número de samples procesados
- `version_read` (int): Versión leída en REQUEST_PARAMS (para staleness calc)

**Tamaño total**: ~50 MB → ~500ms/10Gbps

---

### 9. STOP (PS → Worker)

**Propósito**: Apagado limpio

```python
send_message(sock, MsgType.STOP, {})
```

**Payload**: Vacío

**Acción en Worker**: Salir del training loop, cleanup, disconnect

---

## Manejo de Sockets

### Envío (_recv_exact)

```python
def send_message(sock, msg_type, payload):
    # 1. Serializar
    body = pickle.dumps(
        {"type": msg_type, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL  # Compact binary
    )
    
    # 2. Prefijo con longitud
    data = struct.pack(">I", len(body)) + body
    
    # 3. Enviar TODO (puede ser parcial)
    total = 0
    while total < len(data):
        sent = sock.send(data[total:])  # Puede enviar < len(data)
        if sent == 0:
            raise ConnectionError("Socket cerrado durante envío")
        total += sent
```

**¿Por qué loop?** `socket.send()` no garantiza enviar todo en una llamada. Puede enviar N bytes de M.

### Recepción (_recv_exact)

```python
def receive_message(sock):
    # 1. Leer longitud (4 bytes exactos)
    length_bytes = _recv_exact(sock, 4)
    length = struct.unpack(">I", length_bytes)[0]
    
    # 2. Leer exactamente N bytes del cuerpo
    body = _recv_exact(sock, length)
    
    # 3. Deserializar
    return pickle.loads(body)

def _recv_exact(sock, n):
    """Garantiza recibir exactamente N bytes."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Conexión cerrada: esperados {n}, recibidos {len(buf)}"
            )
        buf += chunk
    return buf
```

**Invariante**: Si `_recv_exact()` retorna, **siempre** retorna exactamente N bytes (o lanza exception).

---

## Flujo Típico Completo

```
TIMELINE:

t=0  Worker CONNECT → PS listen()
     send(READY)
     ↓
t=100ms receive(WORKER_ID)
     ├─ Extrae {"worker_id": 0}
     └─ store self._worker_id = 0

t=200ms receive(CNN_WEIGHTS) ← ~44 MB
     ├─ Extrae CNN state_dict
     ├─ cnn.load_weights_from_bytes()
     └─ send(CNN_ACK)

t=600ms PS receive(CNN_ACK)

t=700ms PS send(START)
     receive(START) ← Worker

t=800ms [Worker enters _training_loop()]
     send(REQUEST_PARAMS)
     
t=900ms PS receive(REQUEST_PARAMS)
     ├─ copy mlp_state, cnn_state, version
     ├─ send(PARAMS) ← ~50 MB

t=1400ms [Worker receives PARAMS]
     ├─ Parse mlp_state, cnn_state, version_read=0
     ├─ _sync_cnn(), _sync_mlp()
     ├─ _train_batch() ← ~50-100ms
     ├─ _serialize_cnn(), _serialize_mlp()
     ├─ send(UPDATES) ← ~50 MB

t=1900ms PS receive(UPDATES)
     ├─ Calculate staleness = ver - version_read = 1 - 0 = 1
     ├─ alpha = 1 / (1 + 0.1 * 1) = 0.91
     ├─ Apply update with α=0.91
     ├─ version++
     └─ [Callback: on_step(loss, acc, staleness=1)]

t=2000ms [Worker REQUEST_PARAMS again]
     ├─ send(REQUEST_PARAMS)
     ← VUELVA AL LOOP

[Throughput: ~1 iteration per 500ms = ~2 iter/sec]
```

---

## Problemas Potenciales y Soluciones

| Problema | Causa | Síntoma | Solución |
|---|---|---|---|
| **Network timeout** | PS muerto/Red lenta | recv() cuelga 30s+ | Aumentar socket timeout, revisar PS |
| **Broken pipe** | Desconexión inesperada | ConnectionError | Reconnect, log error |
| **Partial send/recv** | TCP puede fragmentar | Mensaje corrupto | Protocol uses length-prefix (OK) |
| **Pickle version incompatibility** | Python versions diferentes | UnpicklingError | Use protocol=HIGHEST (compatible) |
| **Memory explosion** | Enviar objeto gigante | Process killed | Validar tamaño antes de pickle |

---

## Optimizaciones Futuras

1. **Compresión de pesos**: gzip antes de enviar (~50% reducción)
2. **Protocolo binario custom**: Más eficiente que pickle
3. **Gradient compression**: Solo enviar cambios significativos (>teorical)
4. **Sharded parameters**: Dividir CNN entre múltiples PS nodes
5. **Asynchronous I/O**: No bloquear en socket (async/await)

