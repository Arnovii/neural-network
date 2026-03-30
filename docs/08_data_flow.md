# 08. Flujo de Datos: Análisis de Red y Comunicación

## Análisis a Alto Nivel: Bytes por Época

### Escenario: PRECOMPUTED Mode, 4 Workers, LAN

```
Por Época:

DOWNLINK (PS → Workers):
  - PARAMS message: 1.135 MB × 4 workers = 4.54 MB
  - Total downlink: ~4.5 MB
  
UPLINK (Workers → PS):
  - GRADIENTS message: 1.135 MB × 4 workers = 4.54 MB
  - Total uplink: ~4.5 MB

TOTAL per epoch: ~9 MB (bidirectional)
TOTAL per 100 epochs: ~900 MB
```

**Visualización de timing (LAN, ~1 Gbps)**:
```
4.5 MB downlink ÷ 1 Gbps = 4.5 MB ÷ 125 MB/s = 0.036s ≈ 36 ms
4.5 MB uplink ÷ 1 Gbps = 4.5 MB ÷ 125 MB/s = 0.036s ≈ 36 ms

Total network time: ~72 ms (negligible vs ~2500ms compute time)
Network overhead: 72/2500 ≈ 3% (muy bajo)
```

---

## Desglose Byte-Level: PARAMS Message

### Estructura

```python
{
    "type": "PARAMS",              # str, ~10 bytes (Python overhead)
    "payload": {
        "epoch": 5,                # int, ~8 bytes
        "params": {
            "W1": ndarray,         # (256, 512) float32
            "b1": ndarray,         # (256,) float32
            "W2": ndarray,         # (128, 256) float32
            "b2": ndarray,         # (128,) float32
            "W3": ndarray,         # (10, 128) float32
            "b3": ndarray,         # (10,) float32
        },
        "seed": 12345,             # int, ~8 bytes
        "training_mode": "precomputed"  # str, ~20 bytes
    }
}
```

### Cálculo de Size

```
W1: 256 × 512 × 4 bytes (float32) = 524,288 bytes
b1: 256 × 4 bytes = 1,024 bytes
W2: 128 × 256 × 4 bytes = 131,072 bytes
b2: 128 × 4 bytes = 512 bytes
W3: 10 × 128 × 4 bytes = 5,120 bytes
b3: 10 × 4 bytes = 40 bytes

Total params: ~661 KB

Pickle overhead (metadata, dict structure): ~20%
Pickle serialized: 661 KB × 1.2 = ~795 KB

Message length prefix: 4 bytes

TOTAL PARAMS message: ~795 KB
```

**Discrepancia respecto 1.135 MB**: Más arriba dije 1.135 MB. La diferencia viene de overhead de Pickle (tipos, refs, etc.). Aproximar como 1 MB es conservador.

---

## Desglose Byte-Level: GRADIENTS Message

### Estructura

```python
{
    "type": "GRADIENTS",
    "payload": {
        "worker_id": 2,            # int, ~8 bytes
        "epoch": 5,                # int, ~8 bytes
        "gradients": {
            "W1": ndarray,         # (256, 512) float32 — MISMA SHAPE QUE W1
            "b1": ndarray,         # (256,) float32
            "W2": ndarray,         # (128, 256) float32
            "b2": ndarray,         # (128,) float32
            "W3": ndarray,         # (10, 128) float32
            "b3": ndarray,         # (10,) float32
        },
        "loss": 0.234,             # float, ~8 bytes
        "accuracy": 91.2,          # float, ~8 bytes
    }
}
```

### Cálculo de Size

Idéntico al PARAMS:
- Gradients tienen EXACTAMENTE la same shape que pesos
- Total: ~795 KB + metadata

**TOTAL GRADIENTS message: ~795 KB**

---

## CNN_WEIGHTS Message (Inicial)

Solo se envía UNA vez por sesión de entrenamiento (al inicio).

### SimpleCNN

```python
state_dict = {
    "conv1.weight":      (64, 3, 3, 3),         float32 ≈ 7 KB
    "conv1.bias":        (64,),                 float32 ≈ 256 B
    "bn1.weight":        (64,),                 float32 ≈ 256 B
    "bn1.running_mean":  (64,),                 float32 ≈ 256 B
    ...
    "fc.weight":         (512, 256),            float32 ≈ 512 KB
    "fc.bias":           (512,),                float32 ≈ 2 KB
}

Total params: 3 conv blocks + 3 fc layers ≈ 600 KB
Pickle serialized + overhead: ≈ 800 KB
```

### ResNet18

```
18 residual blocks, cada uno con múltiples conv layers
Total parameters: ~11 million
Bytes: 11M × 4 bytes (float32) = 44 MB
Pickle serialized + overhead: ≈ 50 MB
```

### Transmission del CNN_WEIGHTS

```
PRECOMPUTED:
  SimpleCNN broadcast: 800 KB (1ª vez, luego ignorado)
  Total init overhead: 800 KB × 4 workers = 3.2 MB

END-TO-END:
  CNN weights cada época (cambio después de update)
  ResNet18 broadcast: 50 MB × N epochs = 50 × 10 = 500 MB
  Total init overhead + training: 50 MB × 4 workers × 10 epochs = 2 GB downlink
```

---

## Peticiones de Test Features

**Solo una vez por entrenamiento** (después de CNN_READY, antes de epoch 1).

### REQUEST_TEST_FEATURES Message

```python
{
    "type": "REQUEST_TEST_FEATURES",
    "payload": {}  # sin contenido
}

Size: ~50 bytes
```

### TEST_FEATURES Response

```python
{
    "type": "TEST_FEATURES",
    "payload": {
        "X_test_features": (10000, 512),  # float32
        "Y_test": (10000,),               # int32
    }
}

Size calculation:
  X: 10000 × 512 × 4 = 20 MB
  Y: 10000 × 4 = 40 KB
  Total: ~20 MB
  + Pickle overhead: ~24 MB
```

---

## Protocolo de Mensajes: Nivel Socket

### Format Binary Detallado

```
┌─────────────┬────────────────────────────────────────────────────┐
│  4 bytes    │  N bytes                                           │
│  big-endian │  pickle.dumps({                                    │
│  uint32(N)  │      "type": MsgType enum value,                   │
│             │      "payload": dict or None                       │
│             │  })                                                │
└─────────────┴────────────────────────────────────────────────────┘

Ejemplo PARAMS:
  Header: 0x00 0x00 0x03 0x2F  (815 bytes en big-endian = 0x032F)
  Payload: pickled data (815 bytes)
```

### Implementación

```python
def send_message(sock: socket, msg_type: MsgType, payload: dict):
    """Serializa y envía un mensaje."""
    message = {
        "type": msg_type,
        "payload": payload,
    }
    
    # Pickle
    data = pickle.dumps(message)
    
    # Prepend length
    length = len(data)
    header = struct.pack(">I", length)  # big-endian uint32
    
    # Send
    sock.sendall(header + data)

def receive_message(sock: socket) -> dict:
    """Recibe y deserializa un mensaje."""
    # Recibir header (4 bytes)
    header = sock.recv(4)
    if len(header) < 4:
        raise ConnectionError("Unable to receive message header")
    
    # Parse length
    length = struct.unpack(">I", header)[0]
    
    # Recibir payload (length bytes)
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(4096, length - len(data)))
        if not chunk:
            raise ConnectionError("Connection lost")
        data += chunk
    
    # Unpickle
    message = pickle.loads(data)
    return message
```

---

## TCP Buffering y Fragmentacion

### Característica de TCP: Stream Oriented

TCP no preserva límites de mensajes. Si envías 1 MB:
- Puede llegar en 10 chunks de 100 KB
- O 1000 chunks de 1 KB
- El receptor debe ser tolerante

**Por eso el 4-byte length prefix es crítico**: Permite `receive_message()` saber cuántos bytes esperar exactamente.

### Send Example

```python
# Enviar PARAMS (1 MB)
send_message(sock, MsgType.PARAMS, {...})
    ├─ pickle.dumps(): 1 MB (+ overhead)
    ├─ struct.pack(">I", 1000000): "\x00\x0f\x42\x40"
    └─ sock.sendall(header + data): TCP transmission
        ├─ TCP layer dividespaquetes (MTU ~1500 bytes)
        └─ Red: packets de 1.5 KB cada uno hasta totalizr 1 MB
```

### Recv Example

```python
# Recibir PARAMS (1 MB)
receive_message(sock)
    ├─ recv(4): obtiene header "\x00\x0f\x42\x40"
    ├─ unpack: length = 1000000
    ├─ Loop:
    │   ├─ recv(4096): primer chunk, 4 KB (TCP buffer)
    │   ├─ recv(4096): segundo chunk, 4 KB
    │   ├─ ...
    │   └─ Repetir hasta sumar 1 MB
    └─ pickle.loads(): deserializar y devolver
```

---

## Latencia de Red: Casos Reales

### LAN (Same Facility, 1 Gbps)

```
Send message ~1 MB:
  - Time: 1 MB ÷ 125 MB/s ≈ 8 ms
  - Latency: ~1 ms roundtrip (ping)
  - Total: 8-10 ms

Overhead per epoch:
  - 2 messages (PARAMS + GRADIENTS) × 2 = ~20 ms
  - Compute time: 2000-2500 ms
  - Ratio: 20/2500 ≈ 0.8% (negligible)
```

### WAN (Different Countries, 30 Mbps, 100ms latency)

```
Send message ~1 MB:
  - Time: 1 MB ÷ 3.75 MB/s ≈ 267 ms
  - Latency: ~100 ms roundtrip
  - Total: 367 ms per message

Overhead per epoch:
  - 2 messages × 2: ~734 ms
  - Compute time: 2000-2500 ms
  - Ratio: 734/2500 ≈ 29% (SIGNIFICANT)
  
10 epochs: 2.5s × 10 + 7.3s × 10 = 97 segundos (vs 27s en LAN)
```

**Conclusión**: WAN es viable pero el overhead de red es substancial. END-TO-END con CNN_WEIGHTS sería prohibitivo (50 MB × 10 epochs = 500 MB = ~1500s en WAN).

---

## Optimizaciones Posibles (No Implementadas)

### 1. Gradient Compression

Enviar float16 en lugar de float32 (50%, pero alguna pérdida):

```python
# Antes
gradients_f32 = {...}  # 795 KB

# Después (hipotético)
gradients_f16 = {k: v.astype(np.float16) for k, v in gradients_f32.items()}
# Size: 795 KB / 2 ≈ 397 KB (50% reduction)
```

**Trade-off**: Pequeña pérdida de precisión (float16 = 16 bits) pero 2x menos red.

### 2. Gradient Quantization

Convertir a int8 con scaling:

```python
# Antes: 1000 valores float32 = 4 KB
# Después: 1000 valores int8 + scaling factor = 1 KB + overhead
# Total: 1.1 KB (90% reduction)

Técnica:
  original ∈ [-0.5, 0.5]
  scaled = int(original / 0.5 * 127)  # -127 ← -0.5, +127 ← +0.5
  transmitted = max(-128, min(127, scaled))  # clip to int8
  
  Receiver:
    recovered = transmitted * 0.5 / 127  # approximate original
```

### 3. Selective Broadcast (Enviar solo pesos que cambiaron)

Detectar qué parámetros MLP NO cambiaron y omitirlos:

```python
# Hipotético
changed = {}
for param_name in params:
    if not np.allclose(new_params[param_name], old_params[param_name]):
        changed[param_name] = new_params[param_name]

# Enviar solo 'changed'
# Workers aplicar update solo a modified
```

**Risk**: Complejidad, debugging difícil, savings pequeños (típicamente todos cambian).

### 4. Asynchronous Gradient Aggregation

En lugar de esperar todos workers para promediar:

```python
# Actual (sincrónico)
wait_for_all_workers()
average_gradients()
update()

# Hipotético (asincrónico + stale gradients)
average_available_workers()
update()
# Pero Workers 3-4 pueden estar rezagados → stale gradients problem
```

**Risk**: Convergencia puede compirse, trade-off no es claro.

---

## Análisis de Scalabilidad

### Cuello de Botella Actual: Receive Sequential

```python
# PS recibe gradients secuencialmente
for _ in range(n_workers):
    msg = receive_message()  # Bloqueante
```

Si Worker 1 y 2 envían inmediatamente pero Worker 3 espera 10s:
- PS no puede empezar a procesar gradients 1 y 2 hasta recibir 3
- También bloqueante: otros Workers intenten comunicar algo (error) no pueden

**Solución**: Usar threading + queue:

```python
# Ideal (no en código actual)
import threading
import queue

grad_queue = queue.Queue()

def receive_thread():
    while True:
        msg = receive_message()
        grad_queue.put(msg)

threading.Thread(target=receive_thread, daemon=True).start()

# Main thread
messages = [grad_queue.get() for _ in range(n_workers)]  # No bloqueante
```

**Beneficio**: PS puede procesar un mensaje parcial mientras otros llegan.

---

## Estimación de Throughput Teórico

### PRECOMPUTED, 100 Workers

```
Per epoch:
  Downlink: 1 MB × 100 = 100 MB
  Uplink: 1 MB × 100 = 100 MB

LAN (10 Gbps):
  Time: 100 MB ÷ 1.25 GB/s = 80 ms downlink
       100 MB ÷ 1.25 GB/s = 80 ms uplink
  Total: 160 ms overhead (vs 2.5s compute) = 6% overhead

WAN (1 Gbps):
  Time: 100 MB ÷ 125 MB/s = 800 ms downlink
       100 MB ÷ 125 MB/s = 800 ms uplink
  Total: 1600 ms overhead (vs 2.5s compute) = 64% overhead

100 epochs:
  LAN: 100 × 2.5s + 100 × 0.16s = 250s + 16s = 266s
  WAN: 100 × 2.5s + 100 × 1.6s = 250s + 160s = 410s
```

**Escalability**: 
- LAN good: network is NOT bottleneck
- WAN problematic: network is 40% of training time

---

## Monitoreo de Tráfico (Debugging)

```python
# Hooks para medir bytes transmitidos

bytes_sent_total = 0
bytes_recv_total = 0

def send_message_monitored(sock, msg_type, payload):
    global bytes_sent_total
    # ... (serialize)
    sent_bytes = sock.sendall(header + data)
    bytes_sent_total += len(header) + len(data)

def receive_message_monitored(sock):
    global bytes_recv_total
    # ... (deserialize)
    bytes_recv_total += len(header) + len(data)

# Después de train
print(f"Total sent: {bytes_sent_total / 1e6:.2f} MB")
print(f"Total recv: {bytes_recv_total / 1e6:.2f} MB")
```

Esperado para PRECOMPUTED 10 epochs, 3 workers:
```
PARAMS: 1 MB × 3 × 10 = 30 MB
GRADIENTS: 1 MB × 3 × 10 = 30 MB
CNN_WEIGHTS (initial): 0.8 MB × 3 = 2.4 MB
Other (WORKER_ID, CNN_READY, etc.): ~1 MB

Total: ~63 MB downlink, ~63 MB uplink
```

