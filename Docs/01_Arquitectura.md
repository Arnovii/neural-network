# Arquitectura del Sistema Distribuido

## Vista General de Componentes

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RED / INTERNET                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────┐          ┌──────────────────────┐ │
│  │   Parameter Server (PS)       │          │  Worker 0            │ │
│  │  ┌───────────────────────┐    │◄─TCP─►   │ ┌────────────────┐   │ │
│  │  │ • CNN_state           │    │          │ │ ImageNet batch │   │ │
│  │  │ • MLP_state           │    │          │ └────────────────┘   │ │
│  │  │ • version counter     │    │          │      │               │ │
│  │  │ • metrics             │    │          ├──────▼──────────┐    │ │
│  │  │ • loss accumulator    │    │          │ CNN Extractor   │    │ │
│  │  └───────────────────────┘    │          │ (ResNet-18)     │    │ │
│  │          │                    │          ├─────────────────┤    │ │
│  │          │                    │          │ MLP Classifier  │    │ │
│  │          │                    │          │ (2 capas trn.)  │    │ │
│  │   Async-FedAvg                │          │                 │    │ │
│  │  ┌───────▼───────────────────┐│          │ Forward+Backward│    │ │
│  │  │ α(s) = 1/(1+λ·s)          ││          │ SGD local       │    │ │
│  │  │ θ_new = θ + α·Δθ          ││          └─────────────────┘────┘ │ 
│  │  └───────────────────────────┘│                                   │
│  │                               │          ┌──────────────────┐     │
│  │  ┌─────────────────────────┐  │◄─TCP─►   │  Worker 1        │     │
│  │  │ Handshake + Start       │  │          │ ┌──────────────┐ │     │
│  │  │ Serve Workers (threads) │  │          │ │ImageNet batch│ │     │
│  │  │ Apply Updates           │  │          │ └──────────────┘ │     │
│  │  │ Track Metrics           │  │          │ (igual flujo)    │     │
│  │  └─────────────────────────┘  │          └──────────────────┘     │
│  │                               │                                   │
│  └───────────────────────────────┘           ┌──────────────────┐    │
│                                              │  Worker N        │    │
│                                              │ (igual flujo)    │    │
│                                              └──────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │              HuggingFace Hub                                │     │
│  │         ILSVRC/imagenet-1k (streaming)                      │     │
│  │     ← todos los Workers descargan en paralelo               │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Responsabilidades de Cada Componente

### 1. Parameter Server (PS)

**Archivo**: `Distributed/parameter_server.py`

**Responsabilidades**:
- Almacenar y mantener sincronizados los parámetros globales (CNN + MLP) mediante Async-FedAvg
- Establecer conexión TCP y aceptar Workers
- Ejecutar handshake de inicialización (distribuir CNN + MLP iniciales, batch_size)
- Servir Workers en hilos independientes (sin bloqueos inter-worker)
- Recibir parámetros (CNN + MLP) actualizados de cada Worker bajo el ciclo REQUEST_PARAMS
- Promediar AMBOS (CNN + MLP) con Async-FedAvg + corrección de staleness
- Rastrear métricas (loss, accuracy, staleness)
- Mantener historial para visualización en GUI

**Estado Interno**:
```python
{
  "_mlp_state": Dict[str, np.ndarray],     # parámetros MLP actual
  "_cnn_state": Dict[str, np.ndarray],     # state_dict CNN actual
  "_version": int,                          # versión de parámetros
  "_metrics": RunningMetrics,               # ventana deslizante
  "_history": Dict[str, List],              # historial de métricas
  "_sockets": Dict[int, socket],            # conexiones activas
  "_cnn": CNNExtractor,                     # modelo CNN
  "_shutdown": threading.Event              # flag de apagado
}
```

**Métodos Clave**:
- `listen()`: Inicia servidor TCP
- `set_cnn()`, `set_mlp()`: Inicializa modelos
- `_handle_new_connection()`: Handshake per-worker
- `_serve_worker()`: Loop de servicio por worker (asincrónico)
- `_apply_update()`: Aplica parámetros de CNN + MLP con corracción staleness (ambos se promedian con Async-FedAvg)
- `stop()`: Apagado limpio

### 2. Worker (Nodo de Trabajo)

**Archivo**: `Distributed/worker_node.py`

**Responsabilidades**:
- Conectarse al PS y obtener ID
- Recibir CNN y MLP iniciales desde PS
- Sincronizar stream de datos desde HuggingFace
- Ejecutar loop asincrónico indefinidamente (ciclo REQUEST_PARAMS):
  - **REQUEST_PARAMS**: Pide parámetros globales (CNN + MLP)
  - **_sync_cnn()**: Carga CNN global desde PS (SOBRESCRIBE CNN local)
  - **FOR accum_steps**: Entrena localmente
    - _train_batch(): CNN entrenable (SIMPLE CNN) o congelada (ResNet-18), MLP se entrena siempre
  - **UPDATES**: Envía CNN (si SIMPLE CNN) + MLP entrenados al PS
- Registrar métricas locales

**Estado Interno**:
```python
{
  "_sock": socket.socket,                   # conexión al PS
  "_worker_id": int,                        # ID asignado por PS
  "_cnn": CNNExtractor,                     # extractor local
  "_mlp": MLPPyTorch,                       # clasificador local
  "_stream": PrefetchBuffer,                # datos con prefetch
  "_batches_done": int,                     # contador de batches
  "device": torch.device,                   # cpu/cuda/mps
}
```

**Métodos Clave**:
- `run()`: Punto de entrada (conectar → inicializar → entrenar)
- `_connect()`: Handshake con PS
- `_init_stream()`: Inicializa streaming desde HF
- `_handshake_loop()`: Espera CNN + START
- `_training_loop()`: Loop principal request/train/update
- `_train_batch()`: Forward + Backward + SGD local
- `_sync_cnn()`, `_sync_mlp()`: Sincronización de parámetros
- `_serialize_cnn()`, `_serialize_mlp()`: Exportar para TCP

### 3. CNN Extractor

**Archivo**: `Model/cnn_extractor.py`

**Responsabilidades**:
- Mantener CNN (ResNet-18 congelada o SIMPLE CNN entrenable) según arquitectura
- Durante cada batch por Worker (ResNet-18):
  - CNN congelada: `requires_grad=False` (permanente)
  - Solo forward pass para extracción de features
  - SGD local NO se aplica a CNN
- Durante cada batch por Worker (SIMPLE CNN):
  - CNN entrenable: `requires_grad=True` (permanente)
  - Se ENTRENA: gradientes propagados en backward
  - Se SGD local (cambios ephemeral de ~accum_steps batches)
- Se ENVÍA al PS en UPDATES (cambios locales de accum_steps batches si SIMPLE CNN, None si ResNet-18)
- Se SOBRESCRIBE en siguiente REQUEST_PARAMS con CNN global del PS
- **EFECTO CNN Local ResNet-18**: Congelada permanente, no cambia
- **EFECTO CNN Local SIMPLE CNN**: cambios NO PERSISTEN (duran un ciclo REQUEST_PARAMS)
- **GLOBAL**: PS promedia CNN recibida de SIMPLE CNN Workers → CNN entrena globalmente

**Dinámica Especial**:
- ResNet-18: CNN congelada localmente, fija permanentemente
- SIMPLE CNN: CNN congelada en PRÁCTICA localmente (cambios se descartan cada ciclo REQUEST_PARAMS)
- CNN global (PS via SIMPLE CNN Workers): se entrena mediante Async-FedAvg (acumula cambios promediados)

**Arquitecturas Soportadas**:

| Arquitectura | feature_dim | Parámetros | Pesos | requires_grad | Caso de Uso |
|---|---|---|---|---|---|
| `resnet18` | 512 | ~11.7M | ImageNet1K_V1 | False (congelada) | Producción (convergencia rápida, MLP-only) |
| `simple` | 512 | ~11.2M | Random init | True (entrenable) | Experimentación / Testing (E2E training) |

**Interfaz Pública**:
```python
cnn = CNNExtractor(arch="resnet18", pretrained=True, device="cuda")
# Serializar para TCP
weights_bytes = cnn._get_weights_bytes()
# Cargar desde TCP
cnn.load_weights_from_bytes(weights)
# Extraer features
features = cnn._model(images)  # (N, 512)
```

### 4. MLP Classifier

**Archivo**: `Model/mlp_pytorch.py`

**Responsabilidades**:
- Clasificar features CNN en 1000 clases ImageNet
- Mantener arquitectura configurable (hidden1, hidden2)
- He initialization para estabilidad numérica
- Serializar/deserializar parámetros para TCP
- **SIEMPRE SE ENTRENA**: En ambos modos (ResNet-18 y SIMPLE CNN), datos se multiplican con MLP y se reciben gradientes

**Arquitectura**:
```
features (512)
    ↓
fc1 (hidden1, ReLU)    default: 1024
    ↓
fc2 (hidden2, ReLU)    default: 512
    ↓
fc3 (1000)
    ↓
logits (no softmax)
```

**Interfaz Pública**:
```python
mlp = MLPPyTorch(feature_dim=512, hidden1=1024, hidden2=512)
# Forward pass
logits = mlp(features)  # (N, 1000)
# Serializar para TCP
state_dict = mlp.state_dict_numpy()
# Cargar desde TCP
mlp.load_state_dict_numpy(state_dict)
```

### 5. Streaming Pipeline

**Archivo**: `Utils/imagenet_streaming.py`

**Responsabilidades**:
- Descargar batches bajo demanda desde HuggingFace
- Aplicar transformaciones (crop, flip, normalize)
- Shard automático (cada Worker obtiene porción distinta)
- Prefetching en hilo background (no bloquea training loop)
- Manejo de errores y reconexión automática

**Componentes**:

| Clase | Rol |
|---|---|
| `ImageNetStream` | Genera batches infinitos desde HF |
| `PrefetchBuffer` | Buffer asincrónico con hilo de prefetch |
| `ValidationStream` | Recorre split de validación una sola vez |

**Método de Sharding (Dinámico)**:

El PS asigna automáticamente ranks y num_workers a cada Worker:
```
Worker 0 conecta → PS asigna rank=0, num_workers=1, envía en CONFIG
Worker 1 conecta → PS asigna rank=1, num_workers=2, envía en CONFIG  (num_workers se actualiza)
Worker 2 conecta → PS asigna rank=2, num_workers=3, envía en CONFIG  (num_workers se actualiza)
```

Cada Worker recibe su shard dinámicamente en el CONFIG:
```
rank=0, num_workers=3 → muestras 0, 3, 6, 9, ... (1/3 del dataset)
rank=1, num_workers=3 → muestras 1, 4, 7, 10, ... (1/3 del dataset)
rank=2, num_workers=3 → muestras 2, 5, 8, 11, ... (1/3 del dataset)

→ SIN SOLAPAMIENTO, cada imagen se procesa por exactamente 1 Worker
→ SIN NECESIDAD DE COORDINACIÓN MANUAL
```

---

### 6. Constantes del Proyecto

**Archivo**: `Utils/constants.py`

**Responsabilidades**:
- Definir valores por defecto centralizados para todo el proyecto
- Evitar magic numbers dispersos en el código
- Facilitar configuración global

**Constantes principales**:

| Constante | Valor | Descripción |
|---|---|---|
| `DEFAULT_LR` | 0.001 | Learning rate del MLP |
| `DEFAULT_LR_CNN` | 0.001 | Learning rate CNN (E2E) |
| `HIDDEN1_DEFAULT` | 1024 | Neuronas capa oculta 1 |
| `HIDDEN2_DEFAULT` | 512 | Neuronas capa oculta 2 |
| `DEFAULT_BATCH_SIZE` | 64 | Imágenes por batch |
| `IMAGE_SIZE` | 224 | Resolución de imágenes |
| `HF_DATASET_DEFAULT` | ILSVRC/imagenet-1k | Dataset por defecto |
| `DEFAULT_PORT` | 9999 | Puerto del PS |
| `GRAD_CLIP_MAX_NORM` | 10.0 | Threshold gradient clipping |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing |
| `WEIGHT_DECAY` | 1e-4 | Weight decay L2 |

---

### 7. Configuración del Entorno

**Archivo**: `Utils/config_loader.py`

**Responsabilidades**:
- Cargar variables desde archivo `.env`
- Proporcionar funciones helper para configuración
- Centralizar la lógica de configuración

**Funciones principales**:

| Función | Descripción |
|---|---|
| `load_dotenv(env_path)` | Carga archivo .env si existe |
| `get_hf_token(override)` | Obtiene token HF con prioridad: CLI > .env > env |
| `get_worker_ip(host)` | Obtiene IP para Workers: 0.0.0.0 → IP real |
| `get_hf_token_or_raise()` | Obtiene token o lanza error |

**Ejemplo de uso**:

```python
from Utils.config_loader import get_hf_token, get_worker_ip

# Obtener token (auto-detecta)
token = get_hf_token()

# Con override CLI
token = get_hf_token("hf_xxx")  # Prioridad máxima

# Obtener IP para Workers
ip = get_worker_ip("0.0.0.0")  # "192.168.1.100"
ip = get_worker_ip("127.0.0.1")  # "127.0.0.1"
```

---

### 8. Results Exporter

**Archivo**: `Utils/results_exporter.py`

**Responsabilidades**:
- Exportar métricas, configuraciones y logs al finalizar el entrenamiento
- Generar gráficas automáticamente (loss, accuracy, workers activos)
- Escribir archivos organizados en directorio por timestamp
- Thread-safe: permite escritura concurrente sin bloqueos

**Archivos generados por experimento**:
```
./Exports/[timestamp]/
├── config.json           # Configuración completa del experimento
├── metrics.csv         # Series de tiempo (step, loss, acc, workers, elapsed)
├── ps_logs.txt        # Todos los logs del Parameter Server
├── metadata.json      # Estadísticas finales (step final, loss/acc final, workers máx)
├── plot_3panels.png # 3 gráficas combinadas (Loss | Accuracy | Workers)
├── plot_loss.png     # Gráfica individual de Loss
├── plot_accuracy.png # Gráfica individual de Accuracy
└── plot_workers.png # Gráfica individual de Workers activos
```

**Estilos de visualización**:
- Loss: `#E74C3C` (rojo), markers "o"
- Accuracy: `#27AE60` (verde), markers "s"  
- Workers: `#3498DB` (azul), markers "^"
- Grid: alpha=0.15 (Loss/Workers), alpha=0.4 (Accuracy)
- Escala: Loss/Workers (±10%), Accuracy (dinámico ±20%)

**Integración**:
```python
# En parameter_server.py:
self._results_exporter = ResultsExporter(config=config, export_dir="./Exports")
_log.add_log_handler(self._results_exporter.record_log)
export_path = self._results_exporter.finalize()  # Al finalizar
```

### 9. Comunicación (Protocolo)

**Archivo**: `Distributed/protocol.py`

**Responsabilidades**:
- Serializar/deserializar mensajes con pickle
- Enviar/recibir por TCP asincronamente
- Manejar errores de conexión
- Garantizar integridad (length prefix)

**Mensajes Soportados** (10 total):
1. `READY`: Worker → PS (solicita conexión)
2. `WORKER_ID`: PS → Worker (asigna ID)
3. `CONFIG`: PS → Worker (distribuye batch_size, image_size)
4. `CNN_WEIGHTS`: PS → Worker (distribuye pesos)
5. `CNN_ACK`: Worker → PS (confirma carga)
6. `START`: PS → Worker (inicia training loop)
7. `REQUEST_PARAMS`: Worker → PS (pide parámetros actuales)
8. `PARAMS`: PS → Worker (envía estado global)
9. `UPDATES`: Worker → PS (envía pesos actualizados tras SGD local)
10. `STOP`: PS → Worker (apagado)

### 10. GUI y Monitoreo

**Archivo**: `ps_gui_imagenet.py`

**Responsabilidades**:
- Interfaz visual para control del PS
- Configurar hiperparámetros antes de entrenar
- Monitorear en tiempo real:
  - Loss, accuracy, staleness
  - Workers conectados
  - Throughput (steps/s)
- Gráficas vivas con matplotlib
- Logs con colores
- Botones: Encender servidor, Iniciar entrenamiento, Shutdown

**Estados**:
- `OFFLINE`: Sin PS
- `LOADING`: Cargando CNN+MLP (hilo background)
- `LISTENING`: PS listo esperando Workers
- `TRAINING`: Workers entrenando

---

## Flujo de Datos (Simplificado)

```
┌─────────────┐
│ ImageNet HF │
└──────┬──────┘
       │ download(batch)
       ▼
┌─────────────────────┐
│ PrefetchBuffer      │  (2-4 batches en cola)
│ (hilo background)   │
└──────┬──────────────┘
       │ X_batch, Y_batch
       ▼
┌────────────────────────────┐
│ Worker._train_batch()      │
│ • Forward: X → CNN → MLP   │
│ • Loss & Acc calculados    │
│ • Backward propagation     │
│ • SGD local en parámetros  │
└──────┬─────────────────────┘
       │ serialize(updated_weights)
       ▼
┌──────────────────────┐     ┌────────────────────┐
│ UPDATES message      │────►│ Parameter Server   │
│ • mlp_weights (new)  │     │ • Apply α(s) rule  │
│ • cnn_weights (new)  │     │ • Update θ         │
│ • loss, accuracy     │     │ • Increment version│
│ • version_read (old) │     └────────────────────┘
│ • staleness factor   │         │
└──────────────────────┘         │
                                 │ REQUEST_PARAMS
                                 ▼
                            ┌──────────────┐
                            │ PS responde: │
                            │ • mlp_state  │
                            │ • cnn_state  │
                            │ • version    │
                            │ • learning_r.│
                            └──────────────┘
```

---

## Decisiones de Diseño Implícitas

### 1. **Asincronía sin Sincronización Global**
- Cada Worker se ejecuta en su propio hilo independiente
- No hay barrera de espera (todos los workers)
- Maximiza throughput pero sacrifica consistencia

### 2. **State Dict PyTorch Nativo**
- Se usan las claves exactas de PyTorch: `fc1.weight`, `fc1.bias`, etc.
- Facilitaría migración a sistemas externos
- Evita mapping de índices confusos

### 3. **Staleness Correction (Corrección de Obsolescencia)**

El sistema implementa Async-FedAvg con corrección de staleness para mitigar la divergencia caused by asynchronous updates.

#### ¿Por qué es necesaria?

En entrenamiento asíncrono, cuando un Worker envía sus actualizaciones, los parámetros globales del PS pueden haber avanzado significativamente: otros Workers ya enviaron sus updates. Esto crea el problema de **staleness**:

```
Worker 0 envía update (versión 10)
    ↓
PS ya avanzó a versión 15 (por Workers 1, 2, 3)
    ↓
Update del Worker 0 está desactualizado (stale)
```

#### Fórmula matemática

El factor de corrección α(s) se calcula como:

```
α(s) = 1 / (1 + λ · s)
```

Donde:
- **s** = staleness = versión_actual - versión_del_worker
- **λ** (lambda) = hiperparámetro de corrección (default: 0.1)

#### Ejemplos numéricos

| λ | s=0 (actual) | s=1 | s=2 | s=5 | s=10 |
|---|--------------|-----|-----|-----|------|
| 0.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.1 | 1.00 | 0.91 | 0.83 | 0.67 | 0.50 |
| 0.5 | 1.00 | 0.67 | 0.50 | 0.29 | 0.17 |
| 1.0 | 1.00 | 0.50 | 0.33 | 0.17 | 0.09 |

**Interpretación**:
- **s=0**: Update actual → α=1.0 (sin reducción)
- **s=1**: Update con 1 paso de retraso → α≈0.91 (9% reducción)
- **s=5**: Update con 5 pasos de retraso → α≈0.67 (33% reducción)

#### Aplicación en el PS

```python
# En Distributed/parameter_server.py: _apply_update()

# Calcular staleness
s = self._version - payload['version_read']

# Calcular factor alpha
alpha = 1.0 / (1.0 + self.staleness_lambda * s)

# Aplicar update con corrección
for key in mlp_state:
    # θ_new = θ_old + α * Δθ
    mlp_state[key] += alpha * (payload['mlp_weights'][key] - mlp_state[key])
```

#### Efecto en convergecia

- **λ=0**: Sin corrección. Rápido pero puede diverger con muchos Workers.
- **λ=0.1**: Recomendado. Balance entre velocidad y estabilidad.
- **λ=1.0**: Muy conservador. Casi como FedAvg síncrono.

El valor default `λ=0.1` es un trade-off validado empíricamente paraImageNet con 2-4 Workers.

### 4. **Prefetching en Thread Separado**
- Training loop nunca espera I/O
- Buffer mantiene cola de batches listos
- Latencia de red/descarga escondida

### 5. **CNN Congelada**
- Solo se optimiza MLP
- CNN aporta features preentrenadas (ResNet-18)
- Reduce overhead computacional (ResNet-18 >> MLP)


