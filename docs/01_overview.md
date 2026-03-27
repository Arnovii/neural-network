# 1. VISIÓN GENERAL DEL SISTEMA

## En 2 minutos

Este es un **sistema de aprendizaje distribuido** que entrena un clasificador CNN+MLP en CIFAR-10 usando **Algoritmo de Diego**.

**Idea clave**: El Parameter Server centralizado coordina múltiples Workers que computan gradientes en paralelo sobre sus chunks de datos.

```
    ┌──────────────────────────────────────────────────────────────┐
    │                   PARAMETER SERVER (PS)                      │
    │  • Modelos CNN + MLP (centralizado)                          │
    │  • Sincroniza Workers cada época                             │
    │  • Promedia gradientes: θ ← θ − lr * ∇̄L                      │
    └──────────────────────────────────────────────────────────────┘
         ▲                          ▲                          ▲
         │ PARAMS + CNN_WEIGHTS     │ PARAMS + CNN_WEIGHTS     │ PARAMS
         │                          │                          │
         │ GRADIENTS (return)       │ GRADIENTS                │ GRADIENTS
         ▼                          ▼                          ▼
    ┌──────────────┐           ┌──────────────┐          ┌──────────────┐
    │   WORKER 0   │           │   WORKER 1   │          │   WORKER K   │
    │              │           │              │          │              │
    │  Datos: 50K  │           │  Datos: 50K  │          │  Datos: 50K  │
    │  Calcula ∇L  │           │  Calcula ∇L  │          │  Calcula ∇L  │
    └──────────────┘           └──────────────┘          └──────────────┘
         (Chunk 0)                (Chunk 1)                 (Chunk K)
```

---

## Qué problema resuelve

**Problema**: Entrenar un modelo moderno es costoso en CPU/GPU y datos.

**Solución**: 
- Distribuir datos entre múltiples máquinas (Workers)
- Cada Worker procesa su chunk en paralelo
- Centralizar sincronización en un Parameter Server

**Resultado**: 
- Si tienes K Workers con datos, el cálculo es ~K veces más rápido
- Sin necesidad de un cluster Hadoop/Spark — solo sockets TCP

---

## Pipeline CNN + MLP

```
    CIFAR-10 imagen (32×32×3)
            │
            ▼  Convolutional Feature Extractor (PyTorch)
            │  "¿Qué patrones visuales hay en esta imagen?"
            │  • Detecta bordes, texturas, formas
            ▼  Salida: vector de 512 dimensiones
            │
            ├─────────────────────────────────┐
            │                                 │
           MLP (NumPy, distribuido)          │
            │ "Dada la característica,        │
            │  ¿a qué clase pertenece?"       │
            │                                 │
            └─────────┬───────────────────────┘
                      │
                      ▼
              Probabilidades (10 clases)
                Avión / Auto / Gato / …
```

**Razón de la separación CNN/MLP**:
- **CNN**: arquitectura pesada, pesos distribuidos por el PS
- **MLP**: arquitectura ligera, gradientes serializables con Pickle

---

## Dos modos de entrenamiento

### **PRECOMPUTED** (modo por defecto)
- CNN congelada — sus pesos nunca cambian
- Features extraídos **una única vez** al inicio
- Rápido en cada época (~1-5s) — solo MLP
- **Usado cuando**: tienes una CNN buena preentrenada

### **END-TO-END** (experimental)
- CNN entrenable — se actualiza cada época
- Features recalculados dinámicamente
- Lento en cada época (~5-30s) — CNN + MLP
- **Usado cuando**: quieres ajustar la CNN a CIFAR-10

---

## Algoritmo de Diego — El corazón

1. **Inicio de época**: PS envía parámetros MLP a todos los Workers
2. **Cálculo localmente**: 
   - Cada Worker aplica sus parámetros a su chunk de datos
   - Forward: imágenes → CNN → features → MLP → logits
   - Backward: calcula gradientes ∇L(W) para su chunk
3. **Sincronización**: Workers envían gradientes al PS
4. **Promediar**: PS calcula ∇̄L = (1/N) * Σ ∇L(W_i)
5. **Actualizar**: W ← W − lr * ∇̄L
6. **Repetir**: PS envía nuevos W, Workers recalculan

**Invariante crítica**: Cada Worker tiene **exactamente la misma CNN y MLP** — solo se diferencian en los datos locales.

---

## Estructura de directorios

```
.
├── Distributed/
│   ├── parameter_server.py    ← PS principal
│   ├── worker_node.py         ← Worker principal
│   └── protocol.py            ← Mensajes TCP (READY, WORKER_ID, CNN_WEIGHTS, PARAMS, GRADIENTS, etc.)
│
├── Model/
│   ├── cnn_extractor.py       ← Extractor CNN (PyTorch, simple + ResNet18)
│   └── mlp.py                 ← Clasificador MLP (NumPy, 3 capas)
│
├── Utils/
│   ├── cifar_loader.py        ← Descarga y normaliza CIFAR-10
│   ├── logging_util.py        ← Logger unificado
│   └── results_exporter.py    ← Exporta métricas a JSON
│
├── ps_gui.py                  ← Interfaz gráfica Parameter Server (Tkinter)
├── ps_terminal.py             ← Parameter Server vía terminal
└── worker.py                  ← Entry point Worker (lanzar en cada máquina)
```

---

## Conceptos clave a recordar

| Concepto | Qué es | Dónde |
|----------|--------|-------|
| **Chunk** | Subconjunto de 50K datos que un Worker procesa | Cada Worker tiene uno |
| **Epoch Seed** | Número aleatorio que define qué datos → qué Worker | PS genera cada época |
| **Features** | Representación de 512 dims tras CNN | Se cachean (precomputed) o calculan on-the-fly (E2E) |
| **Gradients** | Derivadas del error respecto a los pesos MLP | Workers calculan, PS promedia |
| **CNN_READY** | Barrera: Workers confirman que tienen CNN cargada | Protocolo de sincronización |
| **Stratified Sampling** | Distribución: cada Worker recibe datos de todas las clases | Round-robin por clase |

---

## Explicación oral recomendada

**Para auditorio técnico:**
> "Implementamos Algoritmo de Diego — Parameter Server distribuido. Cada Worker tiene 50K imágenes CIFAR-10 locales. En cada época, el PS envía un seed aleatorio; cada Worker reconstruye su chunk localmente de forma idéntica. Luego calculan gradientes sobre su chunk en paralelo. El PS promedia Σ∇ / K y actualiza. Features de CNN se cachean con hash de pesos para evitar recalcular."

**Para no-técnico:**
> "Es como un aula con maestro (PS) y K estudiantes (Workers). El maestro da una tarea. Cada estudiante resuelve su parte localmente. Todos reportan sus respuestas. El maestro promedia y ajusta la tarea para la próxima ronda."

---

## Características principales

- **Escalable**: Soporta Workers dinámicos (se conectan/desconectan)
- **Agnóstico a datos**: Workers carn datos, PS no ve archivos locales
- **Caché inteligente**: Features se reutilizan si la CNN no cambia
- **TCP puro**: Sin dependencias de MPI o Spark
- **Dos modos**: Precomputed (rápido) + End-to-End (flexible)
- **Modo Stratified**: Cada Worker obtiene datos de todas las clases
- **Instrumentación**: Logs detallados de qué hace cada componente

---

## Próximos pasos

Para entender el sistema en profundidad, lee en este orden:

1. **Arquitectura** (`02_architecture.md`) — componentes y cómo se conectan
2. **Flujo de entrenamiento** (`03_training_flow.md`) — qué pasa en cada época
3. **Modos** (`04_modes_precomputed_vs_e2e.md`) — diferencias código vs comportamiento
4. **Worker Node** (`05_worker_node.md`) — detalles de qué hace cada Worker
5. **Parameter Server** (`06_parameter_server.md`) — detalles de sincronización
6. **Sistema de caché** (`07_caching_system.md`) — cómo se reutilizan features
7. **Data flow** (`08_data_flow.md`) — diagramas completos de cómo viajan los datos

---

**Documento**: `docs/01_overview.md`  
**Última actualización**: 2026-03-27  
**Nivel**: Principiante → Intermedio
