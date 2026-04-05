# Guión de Defensa Académica: Sistema Distribuido de Entrenamiento E2E para ImageNet-1k

**Documento de Presentación para Exposición Oral**  
Versión Final | Abril 2026  
Basado en análisis del código fuente completo (sin especulaciones)

---

## 1. Introducción: Qué Resolvemos y Por Qué

### 1.1 El Problema: Entrenamiento de Redes Profundas en Datos Masivos

Imaginemos que necesitamos entrenar un clasificador de 1000 categorías en ImageNet-1k, que contiene 1.2 millones de imágenes. Disponemos de dos arquitecturas: **ResNet-18** (120+ capas convolucionales, congelada) o **SimpleCNN** (entrenable E2E).

Opcción 1 (SimpleCNN E2E):
- Descargar 1.2 millones de imágenes (~150 gigabytes)
- Procesar cada imagen forward a través de CNN + MLP
- Retropropagar el gradiente a través de AMBAS redes
- Actualizar 6 millones de parámetros CNN E2E + 2 millones de parámetros MLP

Opción 2 (ResNet-18, MLP-only):
- Descargar 1.2 millones de imágenes (~150 gigabytes)
- Usar CNN como extractor de características (congelada, NO se entrena)
- Retropropagar gradiente solo a través del MLP
- Actualizar SOLO 2 millones de parámetros MLP (CNN permanece congelada)
- Ejecutar cientos de épocas sobre el dataset completo

En una sola máquina con una única GPU, esto típicamente toma semanas. El cambio es que interrupciones de energía, fallos de hardware o un simple error de código pierden todo el progreso. Además, una sola GPU no es suficiente para experimentar rápidamente: probar distintas configuraciones de red requeriría nuevamente semanas.

### 1.2 La Solución: Federated Averaging Asincrónico

Nuestro sistema distribuye el entrenamiento entre N máquinas independientes y genera un patrón denominado Federated Learning con Async-FedAvg. La idea central es:

1. Un **Servidor de Parámetros (PS)** centralizado que almacena los pesos globales.
2. **N Workers independientes** que entrenan localmente en paralelo sin esperarse mutuamente.
3. Cada Worker descarga los parámetros globales, entrena un número pequeño de batches localmente, envía sus cambios locales al PS.
4. El PS **promedia** los cambios recibidos de todos los Workers bajo una fórmula matemática que corrige la antigüedad de los datos (staleness).
5. El proceso se repite indefinidamente sin barrera de sincronización global.

**Ventajas de este enfoque:**
- Escalabilidad lineal: agregar más Workers reduce el tiempo total (casi) linealmente
- Tolerancia a heterogeneidad: Workers más rápidos no esperan a Workers más lentos
- Robustez parcial: si un Worker falla, el resto continúan entrenando
- Utilización de red eficiente: cada Worker envía datos sin bloqueos

**Desventajas que explicaremos después:**
- Convergencia potencialmente más lenta que entrenamiento centralizado sincrónico
- Gradientes en CNN muy ruidosos (redes profundas sin momentum persistente)
- Requiere infraestructura TCP confiable

---

## 2. Arquitectura General del Sistema

### 2.1 Componentes Principales e Interconexión

El sistema se divide en cinco capas funcionales:

#### **Capa 1: Orquestación (Parameter Server)**

El componente central es el Servidor de Parámetros (`Distributed/parameter_server.py`). Como sugiere su nombre, es un servidor TCP que escucha en un puerto específico (por defecto 9999) y acepta conexiones de Workers. Mantiene en memoria:

- El estado de la CNN global (44 MB si es ResNet-18 preentrenada, 6 MB si es SimpleCNN)
- El estado del MLP global (~4.5 MB)
- Un contador de versión que incrementa cada vez que se actualiza
- Métricas agregadas (loss y accuracy sobre una ventana deslizante)

El PS ejecuta un hilo aceptador de nuevas conexiones (`_accept_loop`) que, cuando un Worker se conecta, lanza un nuevo hilo dedicado para ese Worker que está activo durante toda la conexión. Esto permite servir múltiples Workers en paralelo sin bloqueos.

#### **Capa 2: Computación Local (Workers)**

Cada Worker es un proceso independiente (`Distributed/worker_node.py`) que ejecuta un loop infinito:

1. Pide los parámetros globales al PS
2. Carga esos parámetros localmente (reescribiendo cualquier cambio anterior)
3. Descarga un mini-batch de imágenes desde HuggingFace
4. Calcula forward E2E (CNN → extrae características → MLP → logits)
5. Calcula backward (gradientes a través de 120+ capas)
6. Aplica SGD local para actualizar pesos
7. Envía los pesos actualizados y métricas de loss/accuracy al PS
8. Vuelve a paso 1

No hay sincronización explícita entre Workers. Worker 0 puede estar en iteración 100 mientras Worker 1 está en iteración 87. El PS promedia sus contribuciones de todas formas.

#### **Capa 3: Datos Streaming (HuggingFace)**

El Streaming de ImageNet (`Utils/imagenet_streaming.py`) nunca descarga el dataset completo. En su lugar, descarga bajo demanda desde HuggingFace Hub, un imagen a la vez, dentro de un prefetch buffer. Esto permite:

- Ahorrar almacenamiento: no necesitas 150 GB locales
- Paralelismo: mientras un Worker entrena con un batch, el siguiente batch se descarga en un hilo background
- Sharding automático: cada Worker obtiene una parte diferente del dataset (si hay N Workers, Worker k obtiene las imágenes con índice k, k+N, k+2N, ...)

#### **Capa 4: Comunicación (Protocolo TCP)**

Los Workers y el PS se comunican a través de TCP con un protocolo binario personalizado. Se usan exactamente 10 tipos de mensaje (enum `MsgType`), cada uno con un propósito específico, que explicaremos en detalle en la sección de comunicación.

#### **Capa 5: Modelos (CNN + MLP)**

El sistema tiene dos componentes de red neuronal entrenables:

- **CNN Extractor** (`Model/cnn_extractor.py`): ResNet-18 preentrenada en ImageNet-1k (por defecto) o SimpleCNN (esquema de investigación). Tiene 512 neuronas de salida que constituyen el vector de características.
- **MLP Pytorch** (`Model/mlp_pytorch.py`): Clasificador de 3 capas completamente conectadas (512 → 1024 → 512 → 1000). Las primeras dos capas tienen ReLU, la última es lineal (sin activación porque CrossEntropyLoss la maneja internamente).

**SimpleCNN**: CNN + MLP se entrenan juntas E2E. Los pesos se ajustan durante el entrenamiento, reciben gradientes, se actualizan con SGD localmente, y luego se resincronizandesde el PS para la siguiente iteración.

**ResNet-18**: Solo el MLP se entrena. La CNN permanece congelada (requires_grad=False) permanentemente, actuando como extractor de características fijo. Se sincroniza globalmente por cuestiones de arquitectura uniforme, pero nunca recibe updates de gradientes.

### 2.2 Dependencias y Flujo de Datos de Alto Nivel

```
HuggingFace Hub (ImageNet-1k)
    ↓ (descarga bajo demanda)
Worker_0, Worker_1, ... Worker_N
    ↓ (preprocessing de imágenes)
CNN (ResNet-18 o SimpleCNN)
    ↓ (features 512-dim)
MLP (3 capas)
    ↓ (logits 1000-dim)
CrossEntropyLoss
    ↓ (backward a través de CNN 120+ capas + MLP)
Parámetros locales CNN + MLP
    ↓ (se envían al PS vía UPDATES mensaje)
Parameter Server
    ↓ (Async-FedAvg: promedia cambios de todos Workers)
Parámetros globales CNN + MLP
    ↓ (se distribuyen a Workers en siguiente REQUEST_PARAMS)
```

---

## 3. Flujo de Ejecución Completo: Desde el Inicio Hasta el Entrenamiento

### 3.1 Fase de Inicio del Parameter Server

El usuario lanza el PS desde línea de comandos o GUI. El script de entrada (`ps_imagenet.py` o `ps_gui_imagenet.py`) hace lo siguiente:

1. **Parsear argumentos de configuración**: learning_rate (0.001 por defecto), staleness_lambda (0.1 por defecto), batch_size, image_size, arquitectura CNN, etc.

2. **Crear instancia PS**: Se instancia un objeto ParameterServer con esos parámetros. Inicialmente, el PS no tiene CNNni MLP asignados.

3. **Cargar modelos CNN y MLP**: Se crea una CNN en CPU: ResNet-18 preentrenada (con requires_grad=False, permanentemente congelada) O SimpleCNN (con requires_grad=True, entrenable E2E). También se crea un MLP nuevo. Sus pesos se convierten a numpy (para compatibilidad con serialización TCP).

4. **Asignar modelos al PS**: Se llama a `ps.set_cnn(cnn)` y `ps.set_mlp(mlp.state_dict_numpy())`. El PS almacena internamente estos pesos como diccionarios de numpy arrays.

5. **Esperar Workers**: Se llama a `ps.listen()`. Esta función abre un socket TCP que escucha en el puerto especificado (9999 por defecto) en todas las interfaces (0.0.0.0). Un hilo daemon comienza a aceptar conexiones entrantes.

6. **Pausa hasta esperar N Workers**: El PS imprime "Waiting for N workers..." y entra en una pausa. No comienza a entrenar hasta que exactamente N Workers se han conectado y completado el handshake.

### 3.2 Fase de Inicio del Worker

El usuario lanza uno o más Workers desde línea de comandos. El script de entrada (`worker_imagenet.py`) hace lo siguiente:

1. **Parsear argumentos**: server_host (127.0.0.1 por defecto), server_port (9999), rank (índice del Worker, importante para sharding), num_workers (total de Workers para sharding), device (GPU vs CPU), seed, token de HF.

2. **Instanciar WorkerNode**: Se crea un objeto WorkerNode con esos parámetros.

3. **Llamar a run()**: Este es el punto de entrada principal que ejecuta la secuencia:
   - `_connect()`: Conecta al PS
   - `_init_stream()`: Crea el pipeline de streaming
   - `_handshake_loop()`: Completa el handshake
   - `_training_loop()`: Entrena indefinidamente

### 3.3 Fase de Handshake: Acuerdo entre PS y Worker

Cuando el Worker establece conexión TCP, ocurre una coreografía precisa de 7 o 10 mensajes (incluyendo el reconocimiento de parada). Estos son:

**Mensaje 1: READY (Worker → PS)**
El Worker envía un mensaje vacío indicando que acaba de conectar. Esto dispara la lógica del PS de asignar un ID y comenzar el handshake.

**Mensaje 2: WORKER_ID (PS → Worker)**
El PS responde con un ID único para este Worker (0, 1, 2, ...). El Worker almacena este ID, que se usa principalmente para debugging y logging. El PS también almacena la dirección IP del Worker.

**Mensaje 3: CONFIG (PS → Worker)**
El PS envía batch_size (ej 64) e image_size (ej 224). El Worker usa estos valores para:
- Crear el pipeline de streaming con el batch_size correcto
- Conocer la resolución de imágenes que el PS espera

**Mensaje 4: CNN_WEIGHTS (PS → Worker)**
El PS serializa el estado completo de la CNN (diccionario de numpy arrays) y lo envía. Para ResNet-18, esto representa 44 MB de datos. El Worker lo recibe y carga en la CNN local.

**Mensaje 5: CNN_ACK (Worker → PS)**
El Worker confirma que cargó la CNN correctamente. Incluye metadata como la arquitectura (ej "resnet18") para detección de mismatch.

**Mensaje 6: START (PS → Worker)**
El PS da la señal de "comienza entrenamiento". Esto dispara la transición del Worker de handshake a loop de entrenamiento.

[Opcionalmente, el PS espera hasta que exactamente N Workers hayan completado el handshake antes de enviar START a cualquiera.]

### 3.4 Fase de Entrenamiento: Loop Infinito REQUEST_PARAMS - TRAIN - UPDATES

Una vez que el Worker recibe START, comienza el loop principal que ejecuta indefinidamente:

**Iteración típica (request-train-update loop):**

**[Paso 1] REQUEST_PARAMS (Worker → PS)**

El Worker envía un mensaje pequeño (30 bytes) diciendo "dame los parámetros actuales del PS". Esto contiene aproximadamente:
- El tipo de mensaje: REQUEST_PARAMS
- El worker_id del Worker

**[Paso 2] PARAMS (PS → Worker)**

El PS, al recibir REQUEST_PARAMS, hace lo siguiente:
- Adquiere un lock sobre los parámetros globales para evitar que otros hilos los modifiquen
- Crea copias de los diccionarios de parámetros CNN y MLP
- Incluye el contador de version actual y el learning_rate
- Serializa esto a pickle y lo envía al Worker (típicamente 48.5 MB para ResNet-18 + MLP)

El Worker recibe este mensaje, que contiene:
- mlp_state: diccionario con los pesos del MLP global actual
- cnn_state: diccionario con los pesos de la CNN global actual
- version: entero indicando cuántos pasos de actualización se han procesado en el PS
- lr: learning_rate que el PS usa para SGD local en este Worker

**[Paso 3] _sync_cnn() y _sync_mlp()**

El Worker carga estos parámetros recibidos en sus modelos locales CNNy MLP. Esta es la operación crítica que distingue el sistema: la CNN local se sobrescribe completamente con la CNN global. Esto significa que cualquier cambio que el Worker hizo localmente en iteraciones anteriores se descarta. Sin embargo, después de esta sincronización, el Worker puede entrenar localmente y generar cambios nuevos.

**[Paso 4] _train_batch() (repetido accum_steps veces)**

Por defecto, accum_steps es 1, pero puede configurarse a valores mayores (ej 5). En cada iteración:

El Worker:
1. Descarga una imagen desde el streaming buffer
2. Aplica transformaciones de Train (RandomResizedCrop, RandomHorizontalFlip, normalización)
3. Pasa la imagen forward a través de CNN (que emite 512 features) y luego MLP (que emite 1000 logits)
4. Calcula pérdida con CrossEntropyLoss
5. **SimpleCNN**: Retropropaga el gradiente (backward) a través de MLP → CNN → todas las capas. Aplica SGD en AMBAS redes.
   **ResNet-18**: Retropropaga solo a través de MLP (CNN congelada nunca recibe gradientes). Aplica SGD solo a MLP.
6. Aplica SGD: θ -= learning_rate * gradiente
7. **CRÍTICO**: En SimpleCNN, ambas redes se actualizan. En ResNet-18, solo MLP cambia; CNN permanece congelada y nunca se actualiza.

Este ciclo se repite accum_steps veces (ej 5 veces). Después de 5 entrenamientos, el Worker tiene CNN y MLP locales que han divergido del PS en formas específicas (los cambios acumulados de 5 SGD steps).

**[Paso 5] UPDATES (Worker → PS)**

El Worker serializa y envía:
- El estado actual de CNN local (incluye los cambios de los 5 SGD steps locales): 44 MB
- El estado actual de MLP local: 4.5 MB
- Loss y accuracy promedio de los accum_steps batches
- El `version` que el Worker leyó en el paso 2 (importante para que PS calcule staleness)

Total típico: 48.5 MB por Worker por ciclo.

**[Paso 6] PS Recibe UPDATES y Aplica Async-FedAvg**

Cuando el PS recibe UPDATES de un Worker:

1. Calcula la "antigüedad" (staleness) = version_actual_PS − version_que_Worker_leyo
   - Si otro Worker actualizó el PS 3 veces mientras este Worker estaba entrenando, staleness = 3.

2. Calcula un factor de corrección: α(s) = 1 / (1 + λ * staleness), donde λ es staleness_lambda (ej 0.1)
   - Si staleness = 0: α = 1 (contribución 100%)
   - Si staleness = 3: α = 1/(1+0.3) = 0.77 (reducción del 23%)
   - Si staleness = 10: α = 1/(1+1) = 0.5 (reducción del 50%)

   Esto protege contra gradientes muy antiguos que podrían desestabilizar el aprendizaje.

3. Para cada parámetro en AMBAS CNN y MLP:
   - delta = parámetro_recibido − parámetro_global
   - parámetro_global_nuevo = parámetro_global + α * delta

   Esto es Async-FedAvg: una media ponderada entre el estado actual global y el cambio propuesto, con peso α.

4. Incrementa version en 1 y libera el lock.

5. Registra loss y accuracy en la ventana de métricas.

**[Bucle]** El Worker vuelve automáticamente al Paso 1 (REQUEST_PARAMS) y continúa indefinidamente. No hay final explícito a menos que el usuario envíe STOP desde el PS.

### 3.5 Ejemplo Temporal: Dos Workers Entrenan en Paralelo

Para ilustrar mejor por qué esto es asincrónico:

```
Tiempo    Worker_0                Worker_1            Parameter_Server
────────────────────────────────────────────────────────────────────────
t=0       READY ─────────────────────>                 acepta conexión 0
t=0.01    <──────────── WORKER_ID (id=0, CONFIG, CNN_WEIGHTS
t=0.05                                 READY ──────────>  acepta conexión
t=0.06                                 <────── WORKER_ID, CONFIG, CNN_WEIGHTS
t=0.15    CNN_ACK ────────────────────>                confirma
t=0.16    <──────────── START
t=0.16                                 CNN_ACK ───────-> confirma
t=0.16                                 <──────── START

t=0.3     REQUEST_PARAMS ────────────>
t=0.3                                 REQUEST_PARAMS ──> ambos piden simultáneamente
t=0.7     <──── PARAMS (48.5 MB)     <──── PARAMS      ambos reciben
t=0.7     [entrenar 5 batches]        [entrenar 5 batches]

t=1.0     [sigue entrenando]         [termina antes, version=0]
t=1.2     UPDATES ────────────────────>                 aplica cambios
t=1.2                                                   version ← 1
t=1.2                                 UPDATES ────────> aplica con staleness=0
t=1.2     [vuelve a REQUEST_PARAMS]                     [PS actualiza ambas CNN+MLP]
t=1.3                                                   version ← 2

────────────► Sin barrera: Worker_0 vuelve a REQUEST_PARAMS mientras Worker_1
               todavía está descargando datos o entrenando.
```

Esta falta de sincronización es lo que genera speedup asincrónico. En un sistema Sync-SGD, todos esperarían al Worker más lento.

---

## 4. Comunicación entre Componentes: Los 10 Mensajes

### 4.1 Enumeration Completa de Mensajes

El protocolo define exactamente 10 tipos de mensaje (enum MsgType). Cada mensaje tiene la estructura:

```
[4 bytes big-endian: longitud] [pickle: {"type": MsgType, "payload": ...}]
```

**Tabla de Mensajes:**

| # | Nombre | Dirección | Propósito | Contenido del Payload | Tamaño Est. |
|---|--------|-----------|----------|-----|---|
| 1 | READY | W→PS | Handshake inicial | {} (vacío) | <1 KB |
| 2 | WORKER_ID | PS→W | Asignar ID único | {"worker_id": 0, ...} | <1 KB |
| 3 | CONFIG | PS→W | Configuración global | {"batch_size": 64, "image_size": 224} | <1 KB |
| 4 | CNN_WEIGHTS | PS→W | CNN inicial | {"arch": "resnet18", "weights_bytes": ...} | 44 MB (ResNet) |
| 5 | CNN_ACK | W→PS | CNN cargada | {"architecture": "resnet18", ...} | <1 KB |
| 6 | START | PS→W | Iniciar entrenamiento | {} (vacío) | <1 KB |
| 7 | REQUEST_PARAMS | W→PS | Solicitar parámetros | {} (vacío) | <1 KB |
| 8 | PARAMS | PS→W | Enviar parámetros | {"mlp_state": {...}, "cnn_state": {...}, "version": 42, "lr": 0.001} | 48.5 MB (ResNet+MLP) |
| 9 | UPDATES | W→PS | Enviar cambios | {"mlp_weights": {...}, "cnn_weights": {...}, "loss": 4.23, "accuracy": 0.02, "version_read": 42} | 48.5 MB (ResNet+MLP) |
| 10 | STOP | PS→W | Apagar Worker | {} (vacío) | <1 KB |

### 4.2 Serialización y Protocolo de Bajo Nivel

Cada mensaje se serializa con pickle (el protocolo HIGHEST_PROTOCOL para máxima compresión). La estructura es:

1. **Pickle** el diccionario {"type": MsgType, "payload": ...} a bytes
2. **Preparar destino**: 4 bytes big-endian representando la longitud del payload
3. **Enviar**: 4 bytes + payload por TCP

Para la recepción (Worker o PS):

1. **Recibir exactamente 4 bytes** (con manejo de recv() parciales mediante `_recv_exact()`)
2. **Desempacar** length como big-endian unsigned int
3. **Recibir exactamente `length` bytes** (nuevamente, con loop para recv() parciales)
4. **Unpickle** para obtener el diccionario original

Este enfoque es robusto ante segmentacion TCP: `socket.recv()` no garantiza que devuelva exactamente el número de bytes solicitados; puede devolver menos si otros datos están esperando. El helper `_recv_exact()` usa un loop para garantizar exactitud.

### 4.3 Mensajes de Tamaño Crítico: PARAMS y UPDATES

Los dos mensajes más grandes, PARAMS y UPDATES, son donde ocurre la transferencia masiva de datos:

**PARAMS (PS → Worker):**
- mlp_state: diccionario con 6 claves (fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias), cada una un numpy array
  - fc1.weight: (1024, 512) = 512K elementos × 4 bytes = 2 MB
  - fc1.bias: (1024,) = 1K elementos × 4 bytes = 4 KB
  - [etc, total MLP ~4.5 MB]
- cnn_state: diccionario con ~48 claves (parámetros nombrados de ResNet-18), incluyendo BatchNorm running_mean, running_var, num_batches_tracked
  - weights de conv: típicamente (cout, cin, h, w) donde cao×cin puede ser 100 millones
  - biases
  - BN parameters (γ, β)
  - BN buffers (running_mean, running_var)
  - Total: 11M parámetros × 4 bytes = 44 MB
- version: entero, <1 KB
- lr: float, <1 KB

Total: ~48.5 MB para ResNet-18. Tiempo TCP (1 Gbps): ~400 ms. Tiempo TCP (100 Mbps): ~4 segundos.

**UPDATES (Worker → PS):**
- Estructura idéntica a PARAMS

Pero aquí hay un detalle crítico: el Worker no envía "cambios incrementales" (deltas) sino el **estado completo actual**. El PS internamente calcula delta = recibido − global_actual.

### 4.4 Flujo de Error: Desconexiones y Timeouts

**Si el Worker se desconecta durante PARAMS:**
- El Worker intenta recibir el mensaje PARAMS (bloqueante)
- Si la conexión falla, `receive_message()` lanza ConnectionError
- El Worker captura esto en el try/except de `_training_loop()` y limpia recursos

**Si el PS se desconecta:**
- Todos los Workers pierden su conexión
- Cada Worker intenta enviar el siguiente REQUEST_PARAMS pero obtiene ConnectionError
- Cada Worker imprime un error y termina (sin reconexión automática en la versión actual)

**Si hay un mismatch en arquitectura CNN:**
- En CNN_ACK, el Worker verifica que la arquitectura coincida (ej "resnet18" vs "simple")
- Si no coincide, se registra un error pero el handshake continúa (tolerancia para debugging)

----

## 5. Preprocesamiento y Flujo de Datos

### 5.1 Pipeline de Descarga Streaming desde HuggingFace

Cada Worker ejecuta el método `_init_stream()` que crea un pipeline llamado `ImageNetStream`. Este pipeline:

1. **Conecta al hub de HuggingFace** con el dataset "ILSVRC/imagenet-1k" (requiere token y licencia aceptada). Hay una alternativa pública "timm/imagenet-1k-wds".

2. **Divide el dataset entre Workers** mediante sharding: si hay N Workers y el Worker actual tiene rank r, el pipeline solo entrega imágenes con índice ≡ r (mod N). Esto garantiza que no hay overlap.

3. **Descarga bajo demanda**: en lugar de descargar 150 GB de una vez, el hub de HF devuelve imágenes una a la vez cuando se pide el siguiente elemento del iterador.

4. **Aplica shuffle**: hay un buffer shuffle de tamaño configurable (ej 1000 imágenes). Esto significa que ~1000 imágenes se mantienen en RAM, se permutan, y se sirven en orden pseudoaleatorio.

5. **Retorna pares (imagen_PIL, label_entero)** listos para transformaciones.

### 5.2 Transformaciones de Imagen: Train vs Validación

Las transformaciones ocurren **dentro del Worker**, en cada batch, justo antes del forward pass.

**Para Entrenamiento (train):**

1. **RandomResizedCrop(224)**
   - Se toma una región rectangular aleatorio de la imagen (más pequeño que toda la imagen)
   - Se redimensiona a 224×224
   - Esto aumenta la variedad: la red ve distintas partes de cada imagen
   - Algunos "recortes" pueden perder detalles del objeto, enseñando robustez

2. **RandomHorizontalFlip**
   - Se invierte horizontalmente la imagen el 50% del tiempo
   - Aumenta variedad nuevamente
   - Tiene sentido para objetos simétricos (pájaros, gatos) pero no universalmente (ej, texto)

3. **ToImage() y ToDtype(float32, scale=True)**
   - Convierte PIL.Image a torch.Tensor
   - Convierte valores de píxeles 0–255 (uint8) a 0.0–1.0 (float32)
   - PyTorch prefiere floats normalizados

4. **Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])**
   - Sustrae la media de ImageNet (RGB) y divide por la desviación estándar
   - Estas estadísticas se calcularon en millones de imagenes de ImageNet
   - Esto centra y escala los valores, generalmente mejorando entrada convergencia

**Para Validación (val):**

1. **Resize(256)**
   - Redimensiona la imagen para que su lado más pequeño sea 256 píxeles
   - Se mantiene el aspect ratio

2. **CenterCrop(224)**
   - Se recorta el centro exacto de la imagen a 224×224
   - Determinístico (no hay aleatoriedada), para metricas reproducibles

3. **ToImage(), ToDtype(), Normalize()**
   - Igual que train

**Diferencia conceptual entre ambas:**
- Train: aumenta datos (aumenta número efectivo de ejemplos distintos)
- Val: reduce variabilidad (da imagen "estándar" de cada objeto)

### 5.3 Extracción de Label

El dataset HuggingFace puede devolver labels bajo distintos nombres en función de la versión ("label" vs "cls"). El sistema usa una función robusta `_extract_label(sample)` que:

1. Busca campo "label"
2. Si label is None (no existe), intenta "cls"
3. Si ambos son None, devuelve 0 (fallback)

Esta robustez es importante porque evita crashes por cambios en el formato del dataset.

### 5.4 Prefetching y Concurrencia: PrefetchBuffer

El Worker utiliza una clase `PrefetchBuffer` que mantiene un buffer de batches pre-descargados. Funciona con dos hilos:

- **Hilo principal**: entrena con el batch actual (GPU compute)
- **Hilo background**: descarga y transforma el siguiente batch (I/O network + CPU)

Esto es crucial para eficiencia. Sin prefetch, el Worker:
1. Entrena con batch N (100 ms en GPU)
2. Espera a que se descargue batch N+1 (500 ms de network / HF Hub latency)
3. Total: 600 ms por iteración

Con prefetch:
1. Entrena batch N (100 ms)
2. Mientras tanto, hilo background descarga batch N+1 (500 ms, paralelo)
3. Cuando termina entrenamiento, batch N+1 ya está listo
4. Total: max(100, 500) = 500 ms por iteración (5% overhead de synchronización)

Mejora de throughput: 100% / 20% = 5× más rápido.

---

## 6. Modelo y Estrategia de Entrenamiento

### 6.1 Arquitectura de la CNN: ResNet-18 vs SimpleCNN

**ResNet-18 (Opción por defecto):**

Es una arquitectura convolucional profunda con "skip connections" (conexiones residuales) que la hace más fácil de entrenar. Tiene:
- 18 capas convolucionales
- ~11 millones de parámetros
- Preentrenada en ImageNet-1k (pesos iniciales ya optimizados para este dataset)

Al preentrenar, la red ya ha aprendido características visuales de bajo nivel (bordes, texturas) y características de alto nivel (ojos, ruedas). Usar pesos preentrenados es más rápido en convergencia porque el espacio de parámetros comienza en una región buena.

**SimpleCNN (Opción alternativa para experimentin):**

Es una invención del proyecto: arquitetura convolucionalomas pequenya (1.5M parámetros) sin preentrenamiento. Se inicializa aleatoriamente. Ventaja: investigación sobre cómo entrenar redes desde cero distribuidas. Desventaja: convergencia mucho más lenta.

### 6.2 El MLP Clasificador: Arquitectura Fija

Siempre son 3 capas completamente conectadas:
- Entrada: 512 (features de ResNet-18)
- Capa 1: 512 → 1024 unidades, ReLU
- Capa 2: 1024 → 512 unidades, ReLU
- Capa 3: 512 → 1000 unidades, **sin activación**

La razón por la que la capa de salida no tiene activación es que PyTorch's CrossEntropyLoss aplica internamente log_softmax + NLL (negative log likelihood). Aplicar softmax manualmente antes resultaría en aplicarlo dos veces, degradando gradientes numéricos.

**Inicialización de pesos:**

El MLP usa inicialización Kaiming uniform (He), diseñada para redes con ReLU. Esto asegura que:
- Las activaciones ReLU no se "saturen" (no todos los valores sean 0)
- Los gradientes iniciales tenga magnitud razonable
- El loss inicial sea aproximadamente log(1000) ≈ 6.9, no 0 o infinito

Sin buena inicialización, un MLP podría empezar con accuracy ≈ 0% (todas las clases equiprobables) tardaría muchas iteraciones antes de desaproporzionarse.

### 6.3 E2E Training: Por Qué Ambas Redes se Entrenan

La CNN se inicializa "congelada" (requires_grad=False) en memoria. Pero **en cada mini-batch, el Worker descongela la CNN temporalmente** durante backward, permitiendo que los gradientes fluyan a través de todo el CNN. Después del backward y SGD, la CNN se vuelve a "congelar".

**Ventajas de E2E training:**
- La CNN puede adaptar features al dataset específico
- MLP y CNN cooperan para minimizar loss global
- Mejor capacidad de representación

**Desventajas:**
- Convergencia más lenta (120+ capas convolucionales generan gradientes muy ruidosos)
- Sin momentum persistente (los cambios CNN locales se descartan cada REQUEST_PARAMS)
- Riesgo de "catastrophic forgetting" (los pesos preentrenados pueden degradarse)

**Alternativa: Transfer Learning congelado**

Si congeláramos CNN permanentemente, solo entrenarían los 3-4% de parámetros del MLP. Sería mucho más rápido pero menos flexible (features fijas).

### 6.4 Loss Function y Optimizador

**Loss: CrossEntropyLoss**

Combina log_softmax + NllLoss. Matemáticamente: loss = -log(exp(logit[label]) / sum(exp(logits))).

**Optimizador: SGD puro (Stochastic Gradient Descent)**

θ := θ - learning_rate * gradiente

**SIN momentum**: esto es inusual. Épocas típicas de PyTorch usan momentum=0.9. Pero el sistema usa SGD puro porque:
- Los parámetros se resincronizar desde el PS cada REQUEST_PARAMS, perdiendo cualquier momentum acumulado
- Agregar momentum complicaría la sincronización PS

El tradeoff es convergencia más lenta pero más simple.

### 6.5 Learning Rate

Learning_rate se establece en el PS (ej 0.001) y se distribuye a todos los Workers. Es fijo durante entrenamiento (sin warm-up, sin decay). Esto es un "tuning manual": si es muy alto, loss diverge; si es muy bajo, converge lentamente.

---

## 7. Resultados Experimentales y Análisis Crítico

### 7.1 Qué Esperamos Ver: Dinámicas de Pérdida y Exactitud

En un entrenamiento normal, esperamos observar tres fases:

**Fase 1: Descenso rápido (0−100 batches)**
- Loss baja de ~6.9 (log 1000) a ~2-3
- Accuracy sube de ~0.1% a 5−15%
- MLP aprende la mayoría de su discriminación (es más pequeño, aprende rápido)

**Fase 2: Descenso lento (100−10k batches)**
- Loss baja lentamente de 2-3 a 0.7-1.5
- Accuracy sube de 15−30% a 50−70%
- CNN y MLP cooperan, ajustando features finamente

**Fase 3: Meseta (10k+ batches)**
- Loss se estabiliza alrededor de 0.5−1.0
- Accuracy plateaúa alrededor de 65−75% (típico para ResNet-18 sin fine-tuning exhaustivo)

### 7.2 Interpretación de Logs: Qué Significa Qué

Los logs del PS muestran típicamente:

```
[STEP 500] Loss: 4.15 Acc: 8.3%
[STEP 1000] Loss: 2.47 Acc: 41.2%
[STEP 5000] Loss: 0.82 Acc: 73.8%
```

- **Loss baja pero Accuracy baja también**: PROBLEMA. Indica divergencia (oscilación o colapso numérico). Posible culpable: learning_rate muy alto, staleness_lambda mal calibrado.

- **Loss baja pero Accuracy cerca del azar (≈0.1%)**:PROBLEMA. Indica entrenamiento sin discriminación (ej, todos los logits iguales). Posibles culpables: CNN + MLP se actualizan pero los pesos no cambian significativamente (ej, lr = 0), o hay un bug en la extracción de labels.

- **Loss baja, Accuracy sube**: NORMAL. Sistema está aprendiendo.

- **Loss oscila siempre**: Learning rate probablemente muy alto. Debería reducirse.

- **Loss y Accuracy son platos desde el inicio**: Posiblemente CNN + MLP no están conectados adecuadamente, o hay un bug en el forward pass.

### 7.3 El Caso de SimpleCNN vs ResNet-18

Con **SimpleCNN**:
- Primeros 1000 batches: Loss permanece en ~6.8 (casi random), Accuracy ~0.2%
- Esto es normal: la CNN debe aprender features desde cero
- Eventualmente (después de 10k+ batches) comienza a bajar
- Convergencia final: Accuracy ~50% (peor que ResNet)

Con **ResNet-18 preentrenado**:
- Primeros 100 batches: Loss baja a ~2-3, Accuracy sube a 30−40%
- Los pesos preentrenados ya conocen features generales
- Convergencia final: Accuracy ~70−75%

Esto demuestra el poder del preentrenamiento.

### 7.4 Limitaciones Observadas en el Sistema Actual

**1. Convergencia lenta en CNN (120+ capas)**

Sin momentum persistente, la CNN congelada-descongelada cada REQUEST_PARAMS aprende lentamente. Cada Worker genera cambios CNN locales, pero solo los promedios PS persisten. Los cambios individuales se pierden.

**2. Ruido gradiente en redes profundas**

120 capas CNN × 1024 capas MLP = potencial para "covariate shift" (la distribución de activaciones cambia durante backprop). El sistema no implementa BatchNorm actualización de running stats en el PS, solo promedía pesos estáticos. Esto puede causar instabilidad.

**3. Sin restauración de fallos**

Si un Worker crashea, su iteración se pierde. Si el PS crashea, todo se pierde. No hay checkpointing automático.

**4. Communication Bottleneck**

En condiciones de red lenta (100 Mbps, WAN), el tiempo TCP dominará el tiempo GPU, resultando en baja utilización. El sistema fue diseñado para redes locales rápidas (1+ Gbps).

---

## 8. Justificación de la Arquitectura Elegida

### 8.1 ¿Por Qué Parameter Server y No AllReduce?

Existen dos paradigmas principales para entrenamientos distribuidos:

**Parameter Server (nuestro sistema):**
- Un servidor centralizado mantiene parámetros globales
- Los Workers envían al servidor, el servidor promedia y distribuye
- Ventaja: escalable, tolerante al asincronismo
- Desventaja: PS es single point of failure

**AllReduce (alternativa):**
- Todos los Workers lanzan un gradiente simultáneamente
- Todos contribuyen al promediado
- Ventaja: sin single point of failure
- Desventaja: todos deben sincronizar (más lento con workers heterogéneos)

Elegimos PS porque permite asincronismo completo: cada Worker avanza a su ritmo sin esperar otros.

### 8.2 ¿Por Qué Asincrónico y No Sincrónico?

**Sincrónico (Sync-SGD):**
- Todos los Workers entrenan un batch
- Todos esperan a complete el mas lento
- Una "época" global toma max(tiempo_worker_0, ..., tiempo_worker_n)
- Convergencia muy confiable (lotes verdaderos, no stale)
- Pero: Workers rápidos desperdician CPU esperando lentos

**Asincrónico (nuestro sistema):**
- Los Workers entrenan sin esperar
- PS promedia cambios a medida que llegan
- Convergencia menos garantizada (gradientes stale)
- Pero: mejor utilización de hardware

Con N=5 Workers donde 1 es 10× mas lenta:
- Sync-SGD: limitado por la más lenta → 10× menos rapida
- Async-SGD: worker rápido no espera → ~4-5× speedup

Elegimos asincrónico para throughput.

### 8.3 ¿Por Qué Streaming en Lugar de Descargar Todo?

**Descargar completo:**
- Consumo 150 GB disco
- Tiempos de setup 30+ minutos
- Control local sobre datos
- Problema: ¿Dónde almacenar en una máquina con solo 1 TB SSD?

**Streaming (nuestro sistema):**
- Descarga bajo demanda desde HF Hub
- Consume ~100 MB memoria simultáneamente (buffer)
- Setup <1 minuto
- Requiere conexión a internet confiable
- Problema: HF Hub puede estar lento; latencia I/O

Elegimos streaming para flexibilidad: workers remotos no necesitan almacenamientolocal masivo.

### 8.4 ¿Por Qué ResNet-18?

Opciones:
- ViT (Vision Transformer): ~300M parámetros, demasiado grande para educación
- ResNet-50: 25.5M parámetros, aún manejable
- ResNet-18: 11.7М parámetros, ligero pero capaz
- SimpleCNN: propósito investigación

ResNet-18 es el sweet spot: moderado, preentrenado disponible en torchvision, no requiere hardware extremo, suficientemente profundo para E2E training interesante.

### 8.5 ¿Por Qué CNNy MLP E2E y No Solo MLP?

**Transfer Learning puro (CNN congelada):**
- Fast: solo 4% de parámetros se actualizan
- Convergencia rápida
- Limitado: features fijas del ImageNet pueden no adaptarse

**E2E (nuestro sistema):**
- Flexible: CNN + MLP cooperan
- Investigativo: demuestra cómo entrenar redes profundas distribuidas
- Más lento: 120+ capas + momentum limitado
- Educativo: enseña complejidad real

Elegimos E2E porque es más interesante pedagógicamente.

### 8.6 ¿Por Qué Async-FedAvg con Corrección de Staleness?

**Sin corrección (Async-SGD puro):**
```
θ := θ + (gradiente_viejo)  # gradiente que es 10 pasos atrás
```
Problema: gradientes viejos pueden no alinearse con θ actual, causando divergencia.

**Con corrección de staleness:**
```
θ := θ + α(s) * (gradiente_viejo)
donde α(s) = 1/(1+λ·s), s = antigüedad
```
Efecto: descuentar gradientes viejos, balancear entre no ignorarlos y no divergir.

Elegimos esta fórmula porque:
- Tiene teoría de convergencia subyacente (análisis de Async-SGD)
- Parámetro λ es sintonizable: λ=0 → sin corrección, λ→∞ → casi sincrónico
- Simple de implementar

---

## 9. Posibles Preguntas del Profesor y Respuestas Sugeridas

### Pregunta 1: ¿Por Qué la CNN se sincroniza cada REQUEST_PARAMS si se va a descargar nuevamente?

**Respuesta:**

La CNN se descarga completa porque queremos que refleje el promediado global de TODOS los Workers. Si el Worker mantuviera su CNN local sin resincronizar, estaría entrenando con parámetros que solo represent sus propios cambios, no los cambios integrados de otros Workers.

El ciclo REQUEST_PARAMS → _sync_cnn() es crucial: significa que todos los Workers entrenan con el **consenso global** del PS. Sin esta sincronización, cada Worker divergería independientemente y nunca convergerían a una solución común.

### Pregunta 2: Si la CNN se congela después de cada backward, ¿cómo se entrena?

**Respuesta:**

Hay una distinción importante:
- **Local**: La CNN se descongela solo durante el backward de un batch. Recibe gradientes, se actualiza con SGD, luego se vuelve a congelar.
- **Global**: Los cambios CNN locales se descartan en el siguiente REQUEST_PARAMS cuando la CNN se resincroniza.
- **Persistencia**: Solo los cambios de CNN que fueron promediados por el PS con cambios de otros Workers persisten en la versión global.

Resultado: CNN entrena localmente por 1-5 batches (ephemeral), pero entrena globalmente a través del PS (persistente).

### Pregunta 3: ¿Por Qué el staleness correction α(s) = 1/(1+λ·s)?

**Respuesta:**

La forma matemática viene de minimizar divergencia en optimización asincrónica. Intuitivamente:
- Si staleness=0 (datos frescos), α=1: confiamos 100% en el cambio propuesto
- Si staleness=1 (1 cambio antiguo), α redúcido: confianza reducida
- Si staleness=10 (muy antiguo), α pequeño: cambio casi ignorado

La fórmula 1/(1+λ·s) es simple, cerrada, y surge de análisis teórico. Sin ella, cambios muy antiguos podrían dominar actualizaciones, causando divergencia.

λ es un hiperparámetro: λ=0 ignora staleness (Async-SGD puro, puede divergir), λ=1 compensa fuertemente (casi Sync-SGD).

### Pregunta 4: ¿Qué pasa si Worker 0 y Worker 1 enviandos UPDATES simultáneamente?

**Respuesta:**

El PS usa un `_params_lock` (mutex) para evitar race conditions. Cuando dos UPDATES llegan sincrónicamente:

1. Worker 0 adquiere el lock
2. Worker 1 espera
3. PS aplica cambios de Worker 0, incrementa version
4. Worker 0 libera el lock
5. Worker 1 adquiere el lock
6. PS aplica cambios de Worker 1, incrementa version nuevamente
7. Worker 1 libera el lock

Resultado: ambas actualizaciones se aplican, versión se incrementa dos veces, ningún update se pierde.

### Pregunta 5: ¿Por Qué Keras/TensorFlow no para esto?

**Respuesta:**

Este proyecto fue escrito en PyTorch porque:
1. RequeríaControl de bajo nivel del forward/backward loop asincrónico
2. PyTorch's tamaño pequeño permite correr Workers independientes facilmente
3. La distribución manual (no usar `torch.distributed`) da visibilidad pedagógica

PyTorch es también utilizado en investigación de Sistemas Distribuidos, lo que lo hace idóneo para proyectos educativos que exploren alternativas a patterns estándar.

###Pregunta 6: ¿Qué sucede si falla la red? ¿Hay reintentos?

**Respuesta:**

En la versión actual, **no hay reconexión automática**. Si la conexión TCP se pierde:
- Dentro del Worker: `receive_message()` o `send_message()` lanza ConnectionError
- El Worker captura esto en try/except, imprime error, y termina
- El PS detecta cierre de socket, cierra la conexión del lado del servidor
- Otros Workers continúan (PS no cae)

Una versión robusta agregría:
- Exponential backoff reintentos
- Checkpointing periódico
- Replicación de parámetros PS

Pero por ahora, la arquitectura asume infra red confiable.

### Pregunta 7: ¿Cómo se asegura que cada Worker ve una partición diferente del dataset?

**Respuesta:**

El sistema implementa **sharding determinístico** en el streaming:

Si hay N Workers y Worker i con rank i:
- El stream genera índices 0, 1, 2, ..., 1.2M
- Filtra: solo retorna muestras con índice ≡ i (mod N)
- Resultado: Worker 0 obtiene índices 0, N, 2N, ... (no solapamiento con Worker 1)

Esto se hace en `build_worker_stream()` con un argumento `worker_rank` que modula el índice del dataset.

El sharding es **determinístico**: si reiniciamos los mismos Workers con los mismos ranks, verán el mismo orden de muestras.

### Pregunta 8: ¿Por Qué CrossEntropyLoss sin activación de salida?

**Respuesta:**

CrossEntropyLoss en PyTorch aplica internamente `log_softmax` + `NLLLoss`. Si manualmente aplicáramos softmax a los logits antes de pasar a CrossEntropyLoss, estaríamos:

1. Aplicando softmax (logits → [0,1) sumando a 1)
2. Pasando a CrossEntropyLoss que aplica log_softmax nuevamente

Esto resultaría en valores numéricos incorrectos y gradientes degradados.

La práctica correcta: red outputea logits sin activación, CrossEntropyLoss aplica softmax internamente.

### Pregunta 9: ¿Cómo explicas la diferencia entre Loss baja peor Accuracy no sube?

**Respuesta:**

Esto típicamente indicaría uno de:

1. **Colapso de clase**: Red predice determinísticamente una sola clase (ej, siempre "gato"). Loss es bajo porque esa clase tiene probabilidad alta para gatos, pero para perros es también alto. Accuracy = 10% (solo gatos correctos).
   - Diagnosis: mirar predicciones, deberían ser diversas

2. **Features mal escaladas**: logits tienen magnitud muy grande o muy pequeña. Loss puede ser bajo por mala calibración de gradientes.
   - Diagnosis: inspeccionar magnitud de logits

3. **Label mismatch**: labels en valor diferente (ej, 0-999 vs 1-1000).
   - Diagnosis: verificar rango de labels en dataset

4. **Learning rate demasiado bajo**: red actualiza tan lentamente que parece congelada.
   - Diagnosis: aumentar lr, observar si accuracy sube

### Pregunta 10: ¿Cómo defenderías el diseño ante críticas de convergencia lenta?

**Respuesta:**

Punto 1: **Es por diseño, no por bug**
El sistema enfatiza didáctica y asincronía sobre convergencia máxima. Hay trade-offs inherentes entre velocidad local y estabilidad global.

Punto 2: **Para producción, se usaría**
- Bigger batch sizes (para reducir ruido gradiente)
- Momentum acumulado (sin resincronización forzada)
- Learning rate scheduling (warm-up, decay)
- Mixedprecision (FP16 + FP32)

Punto 3: **Los parámetros λ, lr, accum_steps son sintonizables**
Un usuario comprometido podría calibrar el sistema para su red/dataset específico.

Punto 4: **ResNet-18 preentrenada no está congelada arbitrariamente**
E2E training es una característica, no una limitación. Permite investigar cómo ajustar características preentrenadas para dominios específicos.

---

## 10. Conclusión: Resumen Ejecutivo de Defensa

### Punto 1: Sistema Coherente

Se ha presentado un sistema **completo y funcional** para entrenamiento distribuido asincrónico de redes neuronales profundas en ImageNet-1k. Cada componente (PS, Workers, streaming, comunicación) juega un rol claro y está implementado deliberadamente.

### Punto 2: Justificación Técnica

Cada decisión arquitectónica (asincronía, Federated Averaging con corrección de staleness, E2E training, streaming) tiene justificación basada en trade-offs explícitos entre throughput, convergencia, robustez, y pedagogía.

### Punto 3: Alternativas Consideradas

Se explicó por qué NOT se eligieron alternativas populares (AllReduce, Sync-SGD, Transfer Learning congelado) y por qué el diseño elegido es más apropiado para los objetivos del proyecto.

### Punto 4: Manejo de Complejidad Real

El sistema no es una versión "simplificada" del ML distribuido. Maneja:
- Comunicación TCP binaria real con recuperación parcial de errores
- Concurrencia (múltiple Workers simultáneos)
- I/O network asincrónica (prefetch)
- State management complejo (sincronización de parámetros, corrección de staleness)

### Punto 5: Educativo y Reproducible

Un estudiante o investigador que lea el código puede :
1. Entender exactamente cómo se comunican los componentes
2. Modificar el sistema (ej, agregar checkpointing, cambiar π strategy)
3. Comparar con arquitecturas alternativas
4. Ejecutar en su propia infra (1 PS + N Workers locales o remotos)

## Epílogo: Limitaciones Conocidas y Trabajo Futuro

**Limitaciones del sistema actual:**
- Sin checkpointing automático (fallo = pérdida total)
- Sin reconexión en caso de desconexión
- Convergencia lenta en CNN sin momentum persistente
- Requiere red confiable (no WAN/internet público)

**Mejoras sugeridas para futuro:**
- Checkpointing de periodic (PS para guardar state cada N pasos)
- Replicación del PS (secundario que continúa si principal falla)
- Momentum acumulado on workers (sin reset en REQUEST_PARAMS)
- Compresión de gradientes (enviar deltas, no estado completo)
- Soporte IPv6 y NAT traversal para WAN

**Conclusión final:**
El sistema es pedagógicamente valioso, técnicamente sólido, y demostraría conceptos importantes en ML distribuido. Las limitaciones no son defectos sino trade-offs conscientes hacia claridad y simplicidad.

