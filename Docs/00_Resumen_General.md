# Resumen General: Sistema de Entrenamiento Distribuido Asincrónico ImageNet-1k

## Descripción General

Este proyecto implementa un **sistema de entrenamiento distribuido E2E asincrónico para ImageNet-1k** basado en una arquitectura **Parameter Server (PS)** con **Workers independientes**. El sistema realiza **full training de CNN + MLP** de forma descentralizada y no bloqueante, permitiendo escalar el aprendizaje en múltiples máquinas.

### Problema Resuelto

El entrenamiento E2E de CNN + MLP en datasets masivos como ImageNet-1k requiere:
- Procesamiento de **1.2 millones de imagénes** con propagación de gradientes en 120+ capas CNN
- Distribución de carga computacional de ambas redes entre múltiples procesadores  
- Sincronización eficiente de parámetros globales sin convergencia lenta

Este proyecto resuelve estos desafíos mediante:
1. **Streaming asincrónico**: Datos descargados bajo demanda desde HuggingFace
2. **E2E training**: CNN + MLP entrenables conjuntamente en cada Worker
3. **Async-FedAvg**: Parámetros globales distribuidos sincrónicamente sin barrera de blocking
4. **Comunicación eficiente**: Parámetros (~50 MB/update) intercambiados vía TCP/IP

## Enfoque: Federated Averaging Asincrónico (Async-FedAvg)

El sistema implementa **Federated Averaging asincrónico** con corrección de **staleness** (antigüedad de parámetros):

```
Algoritmo Async-FedAvg (E2E Distribuido):
En cada Worker, ciclo indefinido:
1. REQUEST_PARAMS → recibe θ_global (CNN + MLP) del PS
2. _sync_cnn() → carga CNN global (SOBRESCRIBE CNN local con promediado)
3. FOR accum_steps batches:
   a. Forward E2E: X → CNN (descongela, require_grad=True) → MLP
   b. Backward: ∇L calculado para CNN (120+ capas) + MLP (2-3 capas)
   c. SGD local: 
      θ_cnn_local -= lr · ∇L_cnn  (cambios locales ephemeral)
      θ_mlp_local -= lr · ∇L_mlp  (cambios locales ephemeral)
   d. CNN se vuelve a congelar (requires_grad=False)
4. UPDATES → envía (θ_cnn_local, θ_mlp_local) al PS
5. PS PROMEDIA:
   Δθ = θ_local - θ_global
   θ_global_new = θ_global + α(s) · Δθ  donde α(s) = 1/(1+λ·s)

CRÍTICO: CNN cambios locales NO PERSISTEN (se pierden en siguiente REQUEST_PARAMS)
PERO: CNN GLOBAL entrena (PS promedia CNN de todos Workers) → Async-FedAvg en CNN
```

**Ventajas**:
- ✅ Entrenamiento E2E completo (CNN y MLP actualizadas en cada Worker)
- ✅ No hay barrera de sincronización global
- ✅ Tolerancia a heterogeneidad (Workers rápidos/lentos)
- ✅ Escalabilidad lineal con número de Workers
- ✅ Mejor utilización de red (parámetros enviados asincronamente sin bloqueo)

**Desventajas**:
- ⚠️ **Convergencia lenta para E2E completo**: SGD puro (sin momentum) + resincronización de pesos en cada step
- ⚠️ **Ruido en gradientes CNN**: 120+ capas ResNet-18 generan gradientes ruidosos sin adaptación por parámetro
- ⚠️ **SimpleCNN sin pretrain**: Features iniciales aleatorias → primero centenares de batches con ruido puro

## Componentes Principales

| Componente | Rol | Ubicación |
|---|---|---|
| **Parameter Server (PS)** | Almacena y actualiza parámetros MLP globales | `Distributed/parameter_server.py` |
| **Worker** | Entrena MLP localmente y envía parámetros actualizados | `Distributed/worker_node.py` |
| **CNN Extractor** | ResNet-18 preentrenada O SimpleCNN (ambas entrenables E2E) | `Model/cnn_extractor.py` |
| **MLP Classifier** | Clasificador con 2-3 capas entrenables | `Model/mlp_pytorch.py` |
| **Streaming Pipeline** | Descarga y prepara batches desde HuggingFace | `Utils/imagenet_streaming.py` |
| **GUI** | Interfaz gráfica para control y monitoreo | `ps_gui_imagenet.py` |
| **Terminal PS** | Point of entry para PS sin GUI | `ps_imagenet.py` |
| **Terminal Worker** | Point of entry para Workers | `worker_imagenet.py` |

## Flujo Conceptual

```
┌─────────────────────┐
│  ImageNet-1k (HF)   │
│   1.2M imágenes     │
└──────────┬──────────┘
           │ (streaming)
    ┌──────▼──────────┐
    │ Worker 0        │         ┌──────────────────────────────┐
    │ CNN+MLP train   │         │ Parameter Server             │
    │ (E2E)           │──┐      │ • CNN state (promediada)     │
    │ Sync+Train+Send │  │ ─────► • MLP state (promediada)     │
    └─────────────────┘  │      │ • version                    │
                         │      │ • staleness correction       │
    ┌─────────────────┐  │ ◄────│                              │
    │ Worker 1        │──┤ PARAMS                              │
    │ CNN+MLP train   │  │ +UPDATES                            │
    │ (E2E)           │  │      └──────────────────────────────┘
    │ Sync+Train+Send │  │
    └─────────────────┘  │
                         │
    ┌─────────────────┐  │
    │ Worker N        │──┤
    │ CNN+MLP train   │  │
    │ (E2E)           │  │
    │ Sync+Train+Send │  │
    └─────────────────┘  │
```

## Alcance Actual

### Implementado
- ✅ Transfer Learning con fine-tuning distribuido asincrónico (Async-FedAvg)
- ✅ CNN distribuidamente entrenada (ResNet-18 preentrenada o SimpleCNN) - se actualiza globalmente via Async-FedAvg
- ✅ MLP entrenables (2-3 capas) - único componente con gradientes
- ✅ Comunicación PS ↔ Workers vía TCP/IP
- ✅ Streaming de datos desde HuggingFace (no descarga completa)
- ✅ CNN extractor (ResNet-18 preentrenado + Simple CNN)
- ✅ MLP clasificador with Kaiming initialization
- ✅ Corrección de staleness en PS
- ✅ Interfaz gráfica con gráficas de entrenamiento
- ✅ Interfaz CLI para PS
- ✅ Soporta múltiples devices (CPU, CUDA, MPS)
- ✅ Auto-detección de GPU disponible
- ✅ Logging estructurado con colores

### Por Diseño (No en Roadmap)
- ℹ️ E2E training de CNN + MLP ambos entrenables localmente y sincronizados globalmente
- ℹ️ Sincronización global entre Workers (Sync-FedAvg) - arquitectura asincrónica por diseño

### No Implementado
- ❌ Compresión de parámetros / Cuantización
- ❌ Evaluación automática en validación
- ❌ Checkpointing / Recuperación ante fallos
- ❌ Múltiples GPUs dentro de un solo Worker
- ❌ Comunicación directa Worker-to-Worker
- ❌ Compresión de modelos (pruning, quantization)
- ❌ Persistencia de métricas (base de datos)

## Limitaciones Conocidas

1. **Todos los datos en memoria**: El buffer de prefetch carga batches completos en RAM (default 4 × 64 × 3 × 224 × 224 × 4 bytes ≈ 385 MB por Worker)

2. **No hay recuperación ante fallos**: Si un Worker se desconecta, los parámetros MLP locales se pierden (sin persistencia)

3. **CNN se entrena globalmente**: La CNN se recibe del PS (promediada), se entrena localmente durante accum_steps, se envía al PS, PS la promedia, se recibe nuevamente (ciclo REQUEST_PARAMS)

4. **Inicialización del MLP por Worker**: Si el PS no inicializa el MLP antes de que un Worker se conecte, el Worker crea una versión por defecto (puede causar desincronización)

5. **Staleness sin límite superior**: Si la red es muy lenta, los parámetros MLP pueden ser muy antiguos sin límite máximo

6. **Base de datos no persistente**: Métricas solo en memoria, se pierden si se detiene el PS

7. **Escalabilidad limitada**: Con muchos Workers (>100), la contención en el PS puede ser un cuello de botella

## Versión y Estado

**Versión**: 1.0 (Producción beta)  
**Última actualización**: Abril 2026  
**Estado**: Funcional y testeado con 1+ Workers

## Archivo de Entrada

- **Para PS con GUI**: `py ps_gui_imagenet.py [opciones]`
- **Para PS terminal**: `py ps_imagenet.py [opciones]`
- **Para Workers**: `py worker_imagenet.py [opciones]`

## Requisitos

```
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.10.0
numpy>=1.24.0
matplotlib>=3.6.0
pillow>=9.0.0
```

## Estructura de Directorios

```
neural-network/
├── Distributed/
│   ├── __init__.py
│   ├── parameter_server.py     # Servidor de parámetros
│   ├── worker_node.py          # Worker asincrónico
│   └── protocol.py             # Protocolo TCP
├── Model/
│   ├── __init__.py
│   ├── cnn_extractor.py        # ResNet-18 / Simple CNN
│   └── mlp_pytorch.py          # Clasificador MLP
├── Utils/
│   ├── __init__.py
│   ├── imagenet_streaming.py   # Streaming desde HF
│   ├── logging_util.py         # Logging estructurado
│   └── results_exporter.py     # Exportar resultados
├── ps_imagenet.py              # CLI PS
├── ps_gui_imagenet.py          # GUI PS
├── worker_imagenet.py          # CLI Worker
├── requirements.txt
├── pyproject.toml
└── Docs/
    ├── 00_Resumen_General.md   (este archivo)
    ├── 01_Arquitectura.md
    ├── 02_Flujo_de_Entrenamiento.md
    ├── 03_Parameter_Server.md
    ├── 04_Worker.md
    ├── 05_Modelos.md
    ├── 06_Comunicacion.md
    ├── 07_GUI_y_Monitoreo.md
    ├── 08_Hiperparametros_y_Config.md
    └── 09_Streaming.md
```
