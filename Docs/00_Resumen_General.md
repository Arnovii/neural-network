# Resumen General: Sistema de Entrenamiento Distribuido Asincrónico ImageNet-1k

## Descripción General

Este proyecto implementa un **sistema de entrenamiento distribuido asincrónico para ImageNet-1k** basado en una arquitectura **Parameter Server (PS)** con **Workers independientes**. El sistema realiza **entrenamiento E2E de CNN + MLP en SIMPLE CNN** o **entrenamiento de MLP únicamente en ResNet-18 congelada** de forma descentralizada y no bloqueante, permitiendo escalar el aprendizaje en múltiples máquinas.

### Problema Resuelto

El entrenamiento E2E de CNN + MLP en datasets masivos como ImageNet-1k requiere:
- Procesamiento de **1.2 millones de imagénes** con propagación de gradientes en **11.2M parámetros CNN** (SIMPLE CNN)
- Distribución de carga computacional de ambas redes entre múltiples procesadores  
- Sincronización eficiente de parámetros globales sin convergencia lenta

Este proyecto resuelve estos desafíos mediante:
1. **Streaming asincrónico**: Datos descargados bajo demanda desde HuggingFace
2. **E2E training (SIMPLE CNN) o MLP-only (ResNet-18)**: Entrenable conjuntamente o solo clasificador según arquitectura
3. **Async-FedAvg**: Parámetros globales distribuidos asincronicamente sin barrera de sincronización global
4. **Comunicación eficiente**: Parámetros (~50 MB/update) intercambiados vía TCP/IP

## Enfoque: Federated Averaging Asincrónico (Async-FedAvg)

El sistema implementa **Federated Averaging asincrónico** con corrección de **staleness** (antigüedad de parámetros):

```
Algoritmo Async-FedAvg (SIMPLE CNN: E2E, ResNet-18: MLP-only):
En cada Worker, ciclo indefinido:
1. REQUEST_PARAMS → recibe θ_global (CNN + MLP) del PS
2. _sync_cnn() → carga CNN global (SOBRESCRIBE CNN local)
3. FOR accum_steps batches:
   a. Forward: X → CNN (resnet18: congelada/requires_grad=False, simple: entrenable/requires_grad=True) → MLP
   b. Backward: ∇L calculado para MLP (2-3 capas) + CNN gradientes si SIMPLE CNN
      (ResNet-18: sin backprop en CNN, SIMPLE CNN: backprop completo en 11.2M params)
   c. SGD local: 
      θ_mlp_local -= lr · ∇L_mlp  (siempre)
      θ_cnn_local -= lr · ∇L_cnn  (solo si SIMPLE CNN, cambios ephemeral, no persisten)
4. UPDATES → envía (θ_cnn_local si SIMPLE CNN, θ_mlp_local) al PS
5. PS PROMEDIA:
   Δθ = θ_local - θ_global
   θ_global_new = θ_global + α(s) · Δθ  donde α(s) = 1/(1+λ·s)

CRÍTICO:
- ResNet-18: CNN congelada (requires_grad=False), solo MLP se entrena localmente y se sincroniza
- SIMPLE CNN: CNN + MLP cambios locales NO PERSISTEN (se pierden en siguiente REQUEST_PARAMS)
  PERO CNN GLOBAL entrena via Async-FedAvg (11.2M params promediados entre Workers)
```

**Ventajas**:
- ✅ SIMPLE CNN: Entrenamiento E2E completo (11.2M params CNN + MLP actualizadas en cada Worker)
- ✅ ResNet-18: Transfer learning eficiente (solo MLP se entrena, CNN fija con pesos preentrenados)
- ✅ No hay barrera de sincronización global
- ✅ Tolerancia a heterogeneidad (Workers rápidos/lentos)
- ✅ Escalabilidad lineal con número de Workers
- ✅ Mejor utilización de red (parámetros enviados asincronamente sin bloqueo)

**Desventajas**:
- ⚠️ **SIMPLE CNN convergencia muy lenta E2E**: SGD puro (sin momentum) + resincronización global en cada cycle + sin preentrenamiento (11.2M params random init)
- ⚠️ **SIMPLE CNN features iniciales aleatorias**: Primeros centenares de batches con ruido puro → convergencia extremadamente lenta, no recomendada para producción
- ⚠️ **ResNet-18 convergencia limitada**: CNN congelada restringe adaptación de features, pero convergencia más rápida que SIMPLE CNN (transfer learning)

---

## Configuración del Sistema

### Token de HuggingFace

El token HF se puede configurar de tres formas (prioridad: CLI > .env > variable de entorno):

```bash
# Opción 1: Variable de entorno
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Opción 2: Archivo .env (recomendado)
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > .env

# Opción 3: Argumento CLI (máxima prioridad)
python ps_imagenet.py --hf-token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Dataset

El dataset se especifica en el **Parameter Server**, NO en los Workers:

- El PS recibe `--dataset ILSVRC/imagenet-1k` por CLI o usa el valor por defecto
- El PS envía el nombre del dataset a todos los Workers vía mensaje CONFIG
- El Worker recibe `dataset_name` del PS y lo usa para streaming

```
PS (--dataset ILSVRC/imagenet-1k)
    ↓ CONFIG {dataset_name, batch_size, ...}
Worker (recibe dataset_name del PS)
    ↓
ImageNetStream(dataset_name=recibido)
```

### Archivo .env

Crear archivo `.env` en la raíz del proyecto:

```bash
# .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

 Este archivo se carga automáticamente. El archivo `.env` debe estar en `.gitignore`.

---

## Componentes Principales

| Component | File | Responsibility |
|-----------|------|----------------|
| **Parameter Server (PS)** | `Distributed/parameter_server.py` | Almacena y actualiza parámetros MLP globales |
| **Worker** | `Distributed/worker_node.py` | Entrena MLP localmente y envía parámetros actualizados |
| **CNN Extractor** | `Model/cnn_extractor.py` | ResNet-18 preentrenada (congelada) O SIMPLE CNN (entrenable E2E) |
| **MLP Classifier** | `Model/mlp_pytorch.py` | Clasificador con 2-3 capas entrenables |
| **Streaming Pipeline** | `Utils/imagenet_streaming.py` | Descarga y prepara batches desde HuggingFace |
| **Config Loader** | `Utils/config_loader.py` | Carga .env y funciones de configuración |
| **Constants** | `Utils/constants.py` | Constantes globales del proyecto |
| `ResultsExporter` | `Utils/results_exporter.py` | Exportación desacoplada: 13 archivos (config, metrics, logs, 8 gráficas PNG, metadata) |
| **GUI** | `ps_gui_imagenet.py` | Interfaz gráfica para control y monitoreo |
| **Terminal PS** | `ps_imagenet.py` | Point of entry para PS sin GUI |
| **Terminal Worker** | `worker_imagenet.py` | Point of entry para Workers |

## Flujo Conceptual

```
┌─────────────────────┐
│  ImageNet-1k (HF)   │
│   1.2M imágenes     │
└──────────┬──────────┘
           │ (streaming)
    ┌──────▼──────────┐
    │ Worker 0        │              ┌──────────────────────────────┐
    │ CNN+MLP train   │              │ Parameter Server             │
    │ (E2E)           │──┐           │ • CNN state (promediada)     │
    │ Sync+Train+Send │  │ ─────────►│ • MLP state (promediada)     │
    └─────────────────┘  │           │ • version                    │
                         │           │ • staleness correction       │
    ┌─────────────────┐  │ ◄─────────│                              │
    │ Worker 1        │──┤ PARAMS    │                              │
    │ CNN+MLP train   │  │ +UPDATES  │                              │
    │ (E2E)           │  │           └──────────────────────────────┘
    │ Sync+Train+Send │  │
    └─────────────────┘  │
                         │
    ┌─────────────────┐  │
    │ Worker N        │──┘
    │ CNN+MLP train   │  
    │ (E2E)           │  
    │ Sync+Train+Send │  
    └─────────────────┘  
```

## Alcance Actual

### Implementado
- ✅ Transfer Learning con fine-tuning distribuido asincrónico (Async-FedAvg)
- ✅ CNN parcialmente entrenable (SIMPLE CNN se actualiza localmente y globalmente via Async-FedAvg, ResNet-18 congelada)
- ✅ MLP entrenables (2-3 capas) - único componente con gradientes
- ✅ Comunicación PS ↔ Workers vía TCP/IP
- ✅ Streaming de datos desde HuggingFace (no descarga completa)
- ✅ CNN extractor (ResNet-18 preentrenado + SIMPLE CNN)
- ✅ MLP clasificador with Kaiming initialization
- ✅ Corrección de staleness en PS
- ✅ Interfaz gráfica con gráficas de entrenamiento
- ✅ Interfaz CLI para PS
- ✅ Soporta múltiples devices (CPU, CUDA, MPS)
- ✅ Auto-detección de GPU disponible
- ✅ Logging estructurado con colores
- ✅ Limitador de steps (auto-detención en GUI + CLI)
- ✅ Temporizador (Clock HH:MM:SS + elapsed) desde mensaje START

### Por Diseño (No en Roadmap)
- ℹ️ Sincronización global entre Workers (Sync-FedAvg) - arquitectura asincrónica por diseño
- ℹ️ E2E training de CNN + MLP solo para SIMPLE CNN (ResNet-18 es MLP-only por diseño)

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

3. **SIMPLE CNN se entrena globalmente** (solo si se usa SIMPLE CNN): La CNN se recibe del PS (promediada), se entrena localmente durante accum_steps, se envía al PS, PS la promedia, se recibe nuevamente (ciclo REQUEST_PARAMS). ResNet-18 permanece congelada permanentemente (no se entrena ni globalmente ni localmente).

4. **Inicialización del MLP por Worker**: Si el PS no inicializa el MLP antes de que un Worker se conecte, el Worker crea una versión por defecto (puede causar desincronización)

5. **Staleness sin límite superior**: Si la red es muy lenta, los parámetros MLP pueden ser muy antiguos sin límite máximo

6. **Base de datos no persistente**: Métricas solo en memoria, se pierden si se detiene el PS

7. **Escalabilidad limitada**: Con muchos Workers (>100), la contención en el PS puede ser un cuello de botella

## Versión y Estado

**Versión**: 0.2.0 (Desarrollo activo)  
**Última actualización**: Mayo 2026  
**Estado**: Funcional y testeado con 1+ Workers

## Archivo de Entrada

- **Para PS con GUI**: `py ps_gui_imagenet.py [opciones]`
- **Para PS terminal**: `py ps_imagenet.py [opciones]`
- **Para Workers**: `py worker_imagenet.py [opciones]`

## Requisitos

```
Python >= 3.13.5
torch == 2.10.0
torchvision == 0.25.0
matplotlib == 3.10.8
datasets == 4.8.4
numpy == 2.4.2
psutil == 7.2.2
python-dotenv == 1.2.2
```

## Sistema de Exportación de Resultados

El `ResultsExporter` genera automáticamente **13 archivos por experimento**:

### Archivos Generados
```
./Exports/[timestamp]/
├── config.json           # Configuración del experimento
├── metrics.csv           # Series de tiempo (step, loss, acc, workers, staleness, std_dev)
├── ps_logs.txt           # Todos los logs del Parameter Server
├── metadata.json         # Estadísticas finales
├── plot_3panels.png      # 3 gráficas (Loss | Accuracy | Workers)
├── plot_loss.png         # Gráfica individual de Loss
├── plot_accuracy.png     # Gráfica individual de Accuracy
├── plot_workers.png      # Gráfica individual de Workers
├── plot_staleness.png    # Gráfica de staleness promedio por step
├── plot_std_dev.png      # Gráfica de desviaciones estándar
├── plot_loss_band.png    # Gráfica de Loss con banda de confianza ±1σ
├── plot_accuracy_band.png # Gráfica de Accuracy con banda de confianza ±1σ
└── plot_workers_band.png # Gráfica de Workers con banda de confianza ±1σ
```

### Características de las Gráficas
- **Bandas de confianza ±1σ**: Áreas sombreadas alrededor de las métricas principales
- **Posicionamiento adaptativo de etiquetas**: Evolución temporal de métricas y desviaciones
- **Gráficas de staleness**: Visualización de la antigüedad de parámetros por Worker
- **Gráficas de desviaciones estándar**: Monitoreo de la varianza entre Workers

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
│   ├── cnn_extractor.py        # ResNet-18 / SIMPLE CNN
│   └── mlp_pytorch.py          # Clasificador MLP
├── Utils/
│   ├── __init__.py
│   ├── constants.py           # Constantes globales del proyecto
│   ├── config_loader.py       # Funciones de configuración (.env, get_hf_token)
│   ├── imagenet_streaming.py   # Streaming desde HF
│   ├── logging_util.py         # Logging estructurado
│   └── results_exporter.py     # Exportar resultados
├── ps_imagenet.py              # CLI PS
├── ps_gui_imagenet.py          # GUI PS
├── worker_imagenet.py          # CLI Worker
├── requirements.txt
├── pyproject.toml
├── .env.example               # Plantilla de configuración
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
    ├── 09_Streaming.md
    ├── 10_Guion_Defensa_Academica_Completo.md
    ├── 11_Exportacion_Resultados.md
    ├── 12_Configuracion_Entorno.md
    └── 13_Especificaciones_Entorno_y_Stack_Tecnologico.md
```
