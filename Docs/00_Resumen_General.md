# Resumen General: Sistema de Entrenamiento Distribuido Asincrónico ImageNet-1k

## Descripción General

Este proyecto implementa un **sistema de entrenamiento distribuido asincrónico para ImageNet-1k** basado en una arquitectura **Parameter Server (PS)** con **Workers independientes**. El sistema permite entrenar modelos de visión profunda en múltiples máquinas de forma descentralizada y no bloqueante, optimizando el uso de recursos computacionales.

### Problema Resuelto

El entrenamiento de redes neuronales profundas en datasets masivos como ImageNet-1k requiere:
- Procesamiento de **1.2 millones de imágenes**
- Distribución de carga computacional entre múltiples procesadores
- Sincronización eficiente de parámetros del modelo

Este proyecto resuelve estos desafíos mediante:
1. **Streaming asincrónico**: Los datos se descargan bajo demanda desde HuggingFace
2. **Entrenamiento no-bloqueante**: Workers entrenan independientemente sin esperar a otros
3. **Comunicación eficiente por red**: Parámetros se intercambian vía TCP/IP con serialización pickle

## Enfoque: Asynchronous SGD (Async-SGD)

El sistema implementa **Stochastic Gradient Descent asincrónico** con corrección de **staleness** (antigüedad de gradientes):

```
Algoritmo Async-SGD:
- Cada Worker entrena localmente N batches
- Envía gradientes acumulados al PS sin esperar a otros Workers
- PS actualiza parámetros inmediatamente: θ_new = θ + α(s) · Δθ
- Donde α(s) = 1/(1 + λ·s) es el factor de corrección
- s = staleness = versión_actual - versión_leída
```

**Ventajas**:
- ✅ No hay barrera de sincronización global
- ✅ Tolerancia a heterogeneidad (Workers rápidos/lentos)
- ✅ Escalabilidad lineal con número de Workers
- ✅ Mejor utilización de red (gradientes enviados asincronamente)

**Desventajas**:
- ⚠️ Convergencia menos estable (actualizaciones con información antigua)
- ⚠️ Posible divergencia si staleness es muy alto

## Componentes Principales

| Componente | Rol | Ubicación |
|---|---|---|
| **Parameter Server (PS)** | Almacena y actualiza parámetros globales | `Distributed/parameter_server.py` |
| **Worker** | Entrena localmente y envía gradientes | `Distributed/worker_node.py` |
| **CNN Extractor** | ResNet-18 o Simple CNN (extrae features) | `Model/cnn_extractor.py` |
| **MLP Classifier** | Clasificador de 2 capas ocultas | `Model/mlp_pytorch.py` |
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
    ┌──────▼──────┐
    │ Worker 0    │
    │ CNN → MLP   │──┐
    │ Train       │  │
    └─────────────┘  │  REQUEST_PARAMS   ┌─────────────────┐
                     ├─────────────────► │ Parameter Server│
    ┌─────────────┐  │                   │ • CNN state     │
    │ Worker 1    │──┤  UPDATES          │ • MLP state     │
    │ CNN → MLP   │  ├─────────────────► │ • version       │
    │ Train       │  │                   │ • metrics       │
    └─────────────┘  │   PARAMS          └─────────────────┘
                     │◄─────────────────
    ┌─────────────┐  │
    │ Worker N    │──┤
    │ CNN → MLP   │  │
    │ Train       │  │
    └─────────────┘  │
```

## Alcance Actual

### Implementado
- ✅ Entrenamiento distribuido asincrónico con Async-SGD
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

### No Implementado
- ❌ Sincronización global entre Workers (Sync-SGD)
- ❌ Gradiente Compresión / Cuantización
- ❌ Evaluación automática en validación
- ❌ Checkpointing / Recuperación ante fallos
- ❌ Múltiples GPUs dentro de un solo Worker
- ❌ Comunicación directa Worker-to-Worker
- ❌ Compresión de modelos (pruning, quantization)
- ❌ Persistencia de métricas (base de datos)

## Limitaciones Conocidas

1. **Todos los datos en memoria**: El buffer de prefetch carga batches completos en RAM (default 4 × 64 × 3 × 224 × 224 × 4 bytes ≈ 385 MB por Worker)

2. **No hay recuperación ante fallos**: Si un Worker se desconecta, los gradientes acumulados se pierden

3. **CNN no se entrena**: La CNN solo pasa features, no se optimizan sus parámetros (solo se actualizan desde el PS sin aprender)

4. **Inicialización del MLP por Worker**: Si el PS no inicializa el MLP antes de que un Worker se conecte, el Worker crea una versión por defecto (puede causar desincronización)

5. **Staleness sin límite superior**: Si la red es muy lenta, los gradientes pueden ser muy antiguos sin límite máximo

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
