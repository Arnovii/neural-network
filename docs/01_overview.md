# 01. Visión General del Sistema

## Resumen Ejecutivo (2 minutos)

Este sistema entrena un clasificador de imágenes CIFAR-10 en un entorno distribuido. El aspecto clave es que **separa completamente dos funciones distintas**:

1. **Extracción de características** (CNN con PyTorch): Transforma imágenes 32×32 en vectores de 512 dimensiones. Esta CNN es congelada (sus pesos no cambian) o entrenable (según el modo). Los pesos son idénticos en todas las máquinas.

2. **Clasificación distribuida** (MLP con NumPy): Entrena un clasificador pequeño (512 → 256 → 128 → 10) cuyos pesos se distribuyen entre múltiples máquinas (Workers). Cada máquina procesa su chunk de datos localmente, calcula gradientes, y el servidor central (Parameter Server) promedia los gradientes y actualiza los pesos globales.

**¿Por qué esta arquitectura?** Porque permite enseñar aprendizaje distribuido de manera limpia: la CNN aprende caracterizaciones generales de imágenes, y el MLP aprende a clasificarlas usando esas características. Los datos de entrenamiento (50,000 imágenes) **nunca se transmiten íntegros por la red** — cada Worker los mantiene localmente y solo envía gradientes (que son 100-1000 veces más pequeños).

---

## Problema que Resuelve

**Motivación original**: Entrenar un modelo de deep learning que es **demasiado grande para una sola máquina** o **cuyos datos son demasiado voluminosos para transmitir íntegros**. La solución es **Data Parallelism distribuido**: dividir el dataset entre máquinas, entrenar "mini-modelos" en paralelo, y sincronizar periódicamente.

**Algoritmo de Diego específicamente** es un enfoque de gradient averaging:

- Cada Worker calcula gradientes sobre su chunk local de datos
- El Parameter Server promedia esos gradientes: $\bar{\nabla} = \frac{1}{N} \sum_{i=1}^{N} \nabla_i$
- Actualiza los pesos globales: $\theta \leftarrow \theta - \text{lr} \cdot \bar{\nabla}$
- Repite para la siguiente época

Este método es simple, determinista y se entiende matemáticamente con claridad — ideal para enseñanza o referencia.

---

## ¿Por qué CNN + MLP?

La combinación tiene tres propósitos pedagógicos:

### 1. Enseña la Separación de Preocupaciones
- La CNN es un extractor feature (que es caro computacionalmente pero independiente del problema específico)
- El MLP es el clasificador (que es barato y el que se entrena distributedly)

Esto es realista: en producción, típicamente se usa un feature extractor **pre-entrenado** (como ResNet de ImageNet) y se ajusta el clasificador para una tarea nueva.

### 2. Permite dos Modos de Entrenamiento Distintos
- **PRECOMPUTED**: CNN congelada, features cacheados → entreno solo el MLP → rápido (~2s/época)
- **END-TO-END**: CNN entrenable, features recalculados cada época → entreno CNN+MLP→lento pero más preciso (~12s/época)

### 3. Demuestra Sincronización Realista
En END-TO-END, el PS debe sincronizar pesos en dos niveles: CNN + MLP. Esto es una versión simplificada de Federated Learning moderno.

---

## Flujo de Datos a Alto Nivel

### Esquema General

```
Worker 0                    Worker 1                    Worker 2
   │                            │                            │
   │ [50000 imágenes]           │ [50000 imágenes]           │ [50000 imágenes]
   │ [divididas: 0-16667]       │ [divididas: 16667-33334]   │ [divididas: 33334-50000]
   │                            │                            │
   └─►  [CNN Feature Extract]   │  [CNN Feature Extract]     │  [CNN Feature Extract]
   │    ↓                       │    ↓                       │    ↓
   │  [6667 vectores × 512D]    │  [6667 vectores × 512D]    │  [6667 vectores × 512D]
   │    │                       │    │                       │    │
   └─►  [MLP Forward]           │  [MLP Forward]             │  [MLP Forward]
   │    ↓                       │    ↓                       │    ↓
   │  [Calcula gradientes]      │  [Calcula gradientes]      │  [Calcula gradientes]
   │    │                       │    │                       │    │
   └───►  (envía ~7 MB)  ◄──────┴────► [Parameter Server]
          gradientes             ↑    [Promedia]
                                 │    [Actualiza]
                                 │    └─ Envía nuevos pesos
                                 │
                        [Repite para siguiente época]
```

### Por Época (PRECOMPUTED Mode, 3 Workers):

1. **PS envía pesos MLP** (512×256 + 256×128 + 128×10 pesos = ~0.7 MB) a cada Worker + semilla aleatoria para reproducibilidad
2. **Cada Worker** recibe, extrae su chunk de features del caché, calcula forward/backward del MLP, obtiene gradientes (mismo tamaño que pesos)
3. **Workers envían gradientes** (~0.7 MB cada uno = 2.1 MB total de red ascendente)
4. **PS promedia**: gradientes/(#workers), actualiza $\theta$
5. **PS evalúa en test** (si hay datos), imprime métrica, repite para siguiente época

**Punto clave**: Los datos de imágenes NUNCA se transmiten. Solo se transmiten pesos (descarga) y gradientes (carga).

---

## Componentes Principales

### 1. Parameter Server
- Escucha en un puerto TCP
- Mantiene los pesos globales del MLP (y CNN si END-TO-END)
- Distribuye pesos, recibe gradientes, promedia y actualiza
- Puede correr con GUI (ps_gui.py) o línea de comandos (ps_terminal.py)
- Persistente: acepta múltiples sesiones de entrenamiento sin reiniciar

### 2. Worker Node
- Proceso independiente que carga un chunk de CIFAR-10 (completo, pero particionado)
- Se conecta al PS via TCP, recibe su ID
- Carga la CNN (con pesos enviados por el PS), extrae features
- Entra en un bucle: recibe PARAMS → calcula gradientes → envía GRADIENTS
- Persistente: espera múltiples sesiones de entrenamiento sin reconectarse

### 3. CNN Extractor (PyTorch)
- SimpleCNN: 3 bloques convolucionales diseñados desde cero para CIFAR-10 32×32
- O ResNet-18 (pretrained en ImageNet)
- Output: vectores 512-dimensionales
- Pesos inicializados idénticos en todos los Workers (mismo seed o pesos del PS)

### 4. MLP Classifier (NumPy)
- 512 → 256 (ReLU) → 128 (ReLU) → 10 (Softmax)
- Pesos se inicializan en el PS (He initialization)
- Cada Worker calcula forward/backward localmente
- Solo los gradientes viajan por la red (no el grafo de cómputo)

### 5. Sistema de Comunicación
- TCP sockets con protocolo Pickle (serialización binaria)
- 4-byte length prefix + pickled dict
- Mensajes: READY, WORKER_ID, CNN_WEIGHTS, TRAIN_START, PARAMS, GRADIENTS, STOP, etc.

---

## Decisiones de Diseño Clave

### ¿Por qué NumPy para el MLP?
- Los gradientes de PyTorch serializados con Pickle incluyen el grafo de cómputo → pesan mucho
- NumPy arrays serializan directamente: array float32 8×512 = 16KB. PyTorch tensor = 150KB+
- Propósito pedagógico: cada línea de backward es explícita y visible
- Suficientemente rápido para un MLP de 4 capas

### ¿Por qué Pickle en lugar de JSON?
- JSON no serializa np.ndarray nativamente (hay que convertir a lista → 3-5x más grande)
- Pickle: array float64 (1000,) = 8 KB en Pickle vs 30 KB en JSON
- Velocidad: Pickle no necesita parsing de texto

### ¿Por qué Gradient Averaging (no suma)?
- Si Worker 1 tiene 15000 muestras y Worker 2 tiene 10000, sus gradientes no son directamente comparables
- Dividir por batch size local: $\nabla_i = \frac{1}{n_i} \frac{\partial L}{\partial \theta}$
- PS promedia: $\bar{\nabla} = \frac{1}{2}(\nabla_1 + \nabla_2)$
- Esto es equivalente a SGD sobre el dataset completo con batch size = suma de batch sizes locales

### ¿Por qué Seed-Based Partitioning (sin transmitir índices)?
- Workers podrían enviar: "Mi chunk es muestras 0-16667" (bytes insignificantes)
- Pero es frágil: ¿qué pasa si Workers se desconectan a mitad de época?
- Mejor: PS envía una semilla aleatoria. Cada Worker: `rng.shuffle(range(50000))` con esa semilla → mismo orden
- Determinista, sin estado distribuido, auto-corrector

---

## Modos de Entrenamiento

### PRECOMPUTED (Rápido, Congelado)

**Idea**: Usar una CNN ya entrenada (ResNet ImageNet o simple local) como extractor fijo. Solo entrena el MLP.

**Flujo**:
1. Al arrancar: PS inicializa la CNN (o carga desde caché)
2. Envía pesos CNN a cada Worker
3. Cada Worker: extrae features de 50000 imágenes UNA sola vez, guarda en caché
4. Por cada época: PS envía pesos MLP, Workers calculan gradientes sobre features cacheados
5. CNN nunca se actualiza

**Ventajas**:
- Muy rápido: ~2s/época (solo MLP forward/backward sobre features)
- Features cacheados: primera época ~60s (extracción), siguiente épocas ~2s
- No necesita GPU

**Desventajas**:
- CNN no se adapta al dataset CIFAR-10 específico
- Precisión típica: 94-97%

### END-TO-END (Lento, Entrenable)

**Idea**: Entrenar CNN + MLP juntos con gradient averaging en ambos niveles.

**Flujo**:
1. Al arrancar: PS inicializa la CNN (ResNet o SimpleCNN)
2. Envía pesos CNN a cada Worker
3. Cada Worker: extrae features con esa CNN (sin caché porque cambia cada época)
4. Por cada época:
   - PS envía CNN weights + MLP weights + seed
   - Worker extrae features CON ESA CNN (diferente a la anterior)
   - Calcula gradientes de MLP
   - Envía gradientes de MLP (y accesoriamente CNN gradients)
   - PS averages + actualiza ambos sets de pesos

**Ventajas**:
- CNN se adapta → mejor precisión: 98-99%
- Aprendizaje más realista

**Desventajas**:
- Lento: ~12-15s/época (extracción + MLP forward/backward)
- Necesita GPU para velocidad
- Features no cacheados (hash CNN cambia)

---

## Invariantes Clave

Estos puntos son CRÍTICOS para entender por qué el sistema funciona:

1. **Determinismo sobre Reproducibilidad**: Aunque los datos se particionan con seed aleatorio, son deterministas. Ejecutar dos veces con el mismo seed produce exactamente el mismo entrenamiento.

2. **Separación CNN/MLP**: La CNN es congelada (PRECOMPUTED) o entrenable (E2E), pero sus pesos se distribuyen por copia, no por gradient averaging en PRECOMPUTED.

3. **Gradient Averaging es Equivalente a Batch SGD**: Si Worker 1 con batch 10000 envía $\nabla_1$ y Worker 2 con batch 40000 envía $\nabla_2$, promediar = SGD sobre batch 50000.

4. **Features Cacheados SOLO en PRECOMPUTED**: Si CNN es fija (mismo hash), features son idénticos aunque los extraigas 100 veces. En E2E, CNN cambia cada época → cache miss siempre.

5. **Todos los Workers Convergen al Mismo Modelo**: Con la misma semilla de actualizaciones, todos los Workers llegan a los mismos pesos globales (a pesar de procesar distintos datos locales).

---

## Comparación con Alternativas

| Aspecto | Diego (este sistema) | Federated Averaging | Bulk Synchronous Parallelism |
|--------|------|---|---|
| **Sincronización** | Cada época | Cada época (similar) | Cada microbatch |
| **Determinismo** | Determinista (seed) | Determinista (seed) | Determinista |
| **Tolerancia fallos** | No (abort) | No (abort) | No (abort) |
| **Implementación** | NumPy + sockets | Framework ML | Cluster manager |
| **Propósito** | Enseñanza / referencia | Federated learning | Producción de gran escala |

---

## Cómo Explicarlo en una Presentación

**Versión 30 segundos**:
> "Entrenamos CIFAR-10 distribuido: cada Worker tiene imágenes locales, extrae features con una CNN compartida, clasifica con un MLP pequeño. Los gradientes se promedian en el servidor. Resultado: 10 máquinas entrenan simultáneamente sin compartir datos."

**Versión 2 minutos**:
> "El sistema divide CIFAR-10 entre N Workers determinísticamente (con semilla). Cada Worker: (1) carga una CNN que le envía el servidor, (2) extrae features de sus imágenes, (3) entrena un clasificador MLP sobre esos features. El servidor recibe gradientes de todos los Workers, los promedia como si fuese SGD sobre el dataset completo, actualiza pesos y repite. La CNN puede ser congelada (rápido) o entrenable (preciso)."

**Errores comunes al explicar**:
- ❌ "Los datos se distribuyen" → ✓ Los datos están distribuidos ANTES; se particionan, no se transmiten
- ❌ "Cada Worker tiene un MLP distinto" → ✓ Es el MISMO MLP; todos los Workers convergen a los mismos pesos
- ❌ "Es un modelo HEFC" → Podría serlo; aquí específicamente es gradient averaging
- ❌ "Funciona con cualquier dataset" → Probado en CIFAR-10, aunque la arquitectura es genérica

---

## Próximos Pasos para Profundizar

- **02_architecture.md**: Componentes detallados y responsabilidades
- **03_training_flow.md**: Ejecución segundo a segundo de una época
- **04_modes_precomputed_vs_e2e.md**: Diferencias de código entre modos
- **05_worker_node.md**: Lógica interna del Worker
- **06_parameter_server.md**: Lógica del servidor
- **07_caching_system.md**: Algorithm de caché con hash MD5
- **08_data_flow.md**: Análisis de red y transferencia de bytes

