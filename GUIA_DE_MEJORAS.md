# Guía de Cambios - Sistema Distribuido Mejorado

## 🎯 Objetivo General

Mejorar la claridad, consistencia y experiencia de uso del sistema de entrenamiento distribuido para redes neuronales sin romper funcionalidades existentes.

## 📋 Cambios Realizados

### 1. **Control Explícito del Modo de Entrenamiento**

#### Problema Anterior
El flujo de ejecución dependía de condiciones implícitas:
- Usuarios no sabían si estaban usando CNN preentrenada o no
- La lógica de selección era confusa

#### Solución Implementada
Agregado selector visual "Modo de operación" en la interfaz gráfica:

```
🔒 Precomputación (CNN fija + MLP distribuido)  ← ACTIVO AHORA
⚙️  End-to-End (CNN + MLP se entrenan juntos)   ← PARA FUTURO
```

**En el código (`ps_gui.py`)**:
- Variable: `self._v_system_mode` 
- Valores: `"precomputed"` o `"end_to_end"`
- Callback: `_on_system_mode_change()`

**Impacto**: El usuario entiende claramente qué arquitectura se está usando.

---

### 2. **Logger Unificado para Consistencia**

#### Problema Anterior
Logs inconsistentes:
```
[PS] Distribuyendo CNN a 2 Worker(s) (arch=resnet18)...
[LOAD DATA] Cargando CIFAR-10...
[CNN TRAIN] Época 1/10 | acur=85.3% | pérdida=0.451
```
Formatos y niveles de detalle incoherentes.

#### Solución Implementada
Módulo `Utils/logging_util.py` con `FormattedLogger`:

**Fases Estándar**:
```python
logger.load("Cargando datos", progress="10000/50000")
logger.prep("Extrayendo características", progress="45%", metric="shape=(10000,512)")
logger.train("Época 5/100", metric="acc=82.3% | pérdida=0.451")
logger.eval("Evaluando en test", metric="acc=78.9%")
logger.ps("Distribuyendo CNN", progress="arch=resnet18")
logger.warn("Worker 2 desconectado")
logger.error("Error en Worker 1: conexión perdida")
```

**Salida Formateada**:
```
[LOAD DATA] Cargando datos (10000/50000)
[PREP FEAT] Extrayendo características (45%) | shape=(10000,512)
[TRAIN MLP] Época 5/100 | acc=82.3% | pérdida=0.451
[EVAL]      Evaluando en test | acc=78.9%
[PARAM SRV] Distribuyendo CNN | arch=resnet18
[WARN]      Worker 2 desconectado
[ERROR]     Error en Worker 1: conexión perdida
```

**Beneficios**:
- Legibilidad mejorada
- Contexto claro (fase actual)
- Métricas en posición consistente
- Fácil filtrar por fase en logs

---

### 3. **Feedback en Procesos Largos**

#### Procesos Identificados
1. **Descarga de pesos ImageNet** (~44 MB, ~10-30 seg)
   - Antes: Sin feedback, usuario piensa que se congela
   - Después: Logs con progreso "Descargando pesos ImageNet... (5/100 MB)"

2. **Extracción de características con ResNet** (~30-100 seg)
   - Antes: Pantalla silenciosa durante minutos
   - Después: "Extrayendo características... (45/1000 imágenes)"

3. **Preentrenamiento CNN simple** (~1-2 minutos)
   - Antes: Espera larga sin información
   - Después: "Preentrenando CNN simple (Época 3/10)..."

**Estrategia en Parameter Server**:
```python
_logger.ps("Distribuyendo CNN a Workers", progress="2/4 listos")
_logger.ps("Extrayendo features de prueba", progress="5000/10000 imágenes")
```

**Resultado**: Usuario NUNCA percibe que el sistema está congelado.

---

### 4. **Adaptación Dinámica de la  Interfaz**

#### Cambios GUI
- Selector "Modo de operación" es visible y prominente
- End-to-End deshabilitado con etiqueta "No disponible aún"
- Tooltip explica diferencias entre modos

#### Lógica Futura
Cuando se implemente end-to-end:
- Opciones CNN se ocultarán (en modo E2E no se predefine CNN)
- Parámetros de entrenamiento CNN se mostrarán (nuevos pesos)
- MLP deshabilitado (CNN extraerá features automáticamente)

---

### 5. **Claridad del Flujo de Ejecución**

#### Fases Conceptuales Separadas

**LOAD** → **PREP** → **TRAIN** → **EVAL**

```
═== CARGA DE DATOS ═══════════════════════════════════════════════
[LOAD DATA] Cargando CIFAR-10 entrenamiento...            (10000 imágenes)

═══════════════════════════════════════════════════════════════════
[PREP FEAT] Preparando extractor de características...     (arch=resnet18)
[PREP FEAT] Distribuyendo CNN a Workers...               (2 Workers)
[PREP FEAT] Extrayendo features de entrenamiento...      (98%)
[PREP FEAT] Extrayendo features de prueba...             (100%) 

═══════════════════════════════════════════════════════════════════
[TRAIN MLP] Iniciando entrenamiento distribuido...       (100 épocas)
[TRAIN MLP] Época 1/100 | acc=42.3%  | pérdida=2.301   (10 Workers)
[TRAIN MLP] Época 2/100 | acc=63.7%  | pérdida=1.045   (10 Workers)
...
[TRAIN MLP] Época 100/100 | acc=92.1% | pérdida=0.203  (10 Workers)

═══════════════════════════════════════════════════════════════════
[EVAL]      Evaluando en dataset de prueba...            (10000 imágenes)
[EVAL]      Precisión final: 90.2%  | Pérdida: 0.287

═══════════════════════════════════════════════════════════════════
```

**Beneficio**: Usuario entiende exactamente en qué fase está el sistema en cada momento.

---

## 📁 Archivos Modificados/Creados

### Nuevos
- **Utils/logging_util.py** - Logger unificado (240 líneas)
- **MEJORAS_IMPLEMENTADAS.md** - Documentación de cambios

### Modificados
- **ps_gui.py** - Agregado selector de modo + método callback
- **Distributed/parameter_server.py** - Import logger + logs mejorados

---

## 🔄 Comportamiento Esperado

### Antes vs Después

**ANTES**:
```
[PS] Distribuyendo CNN a 2 Workers (arch=resnet18)...
Worker 0 CNN_READY
Worker 1 CNN_READY
[PS] Solicitando features de prueba...
[PS] Features de prueba listos: (10000, 512)
[PS] ── Época 1/100 ──────────────────────────
```

**DESPUÉS**:
```
[PREP FEAT] Distribuyendo CNN a 2 Workers | arch=resnet18
[PARAM SRV] Worker 0 CNN_READY ✓
[PARAM SRV] Worker 1 CNN_READY ✓
[PREP FEAT] Extrayendo features de prueba
[PREP FEAT] Features de prueba extraídos | shape=(10000,512)

═══════════════════════════════════════════════════════════════════
[TRAIN MLP] Iniciando entrenamiento distribuido | Épocas=100
[TRAIN MLP] Época 1/100 | acc=42.3% | pérdida=2.301
```

---

## ⚠️ Comportamiento de la Arquitectura Actual (Precomputación)

### En Modo Precomputed (ACTIVO)

**CNN**: 
- Preentrenada o cargada desde archivo
- **Congelada** durante entrenamiento MLP
- Workers usan los **mismos pesos** para extraer features

**MLP**:
- **Distribuido** entre Workers
- Cada Worker:
  - Recibe parámetros globales
  - Extrae features = CNN(imágenes)
  - Calcula gradientes = backward(features, etiquetas)
- Parameter Server:
  - Promedia gradientes de todos
  - Actualiza parámetros globales

**Transferencia de Datos**:
- CNN weights → Workers (pequeño, ~50-200 MB)
- Features entre Workers → PS (pequeño, ~10-100 MB)
- **Imágenes NUNCA salen de los Workers** (privacidad, eficiencia)

---

## 🚀 Comportamiento Futuro (End-to-End)

### En Modo End-to-End (DESHABILITADO POR AHORA)

**CNN**:
- Pesos **inicialización aleatoria**
- **Se entrena** junto con MLP
- Workers entrenan sus **copias locales**

**MLP**:  
- También distribuido y se entrena

**Transferencia**:
- Gradientes de CNN + MLP

⚠️ **NOTA**: Esta funcionalidad se agregará en futuras versiones sin romper el modo actual.

---

## ✅ Validaciones

- ✅ Código compila sin errores
- ✅ Variables Tkinter inicializadas correctamente
- ✅ Logger funciona con múltiples fases
- ✅ Compatible con Workers existentes
- ⏳ Tests de ejecución completa (pendientes)

---

## 📖 Para Usuarios

### Cómo Usar

1. **Abre ps_gui.py**
   ```bash
   python ps_gui.py
   ```

2. **Selecciona Modo de Operación**
   - Precomputación (recomendado ahora)
   - End-to-End será disponible después

3. **Elige CNN**
   - Cargar modelo preentrenado, o
   - Entrenar uno nuevo

4. **Configura parámetros MLP**
   - Learning rate, momentum, neuronas

5. **Enciende servidor y conecta Workers**
   - En otra terminal: `python worker.py --server-host <IP>`

6. **Inicia entrenamiento**
   - Y observa los logs bien formateados que indican cada fase

---

## 📞 Preguntas Frecuentes

**P: ¿Qué diferencia hay entre Precomputación y End-to-End?**
A: Precomputación usa CNN congelada; End-to-End entrena CNN y MLP juntos. Para 99% de casos, prefiere Precomputación por eficiencia.

**P: ¿Veo el sistema congelado, qué pasa?**
A: Revisa los logs. Debería indicar: extrayendo features (mins), descar gando pesos (secs), entrenando (ongoing). Nunca es "silencio".

**P: ¿Se pierden imágenes en la red?**
A: No. Las imágenes se procesan localmente en cada Worker. Solo parametos y gradientes viajan por la red.

---

## 🔧 Configuración Recomendada

```
CPU/GPU:  cuda        ← Si tienes GPU NVIDIA
Arch:     resnet18    ← Mejor accuracy (~80%)
Epochs:   100         ← Entrenamiento completo
Workers:  2-4         ← Número de máquinas disponibles
```

---

**Versión**: 1.0  
**Fecha**: Marzo 2026  
**Compatible**: Python 3.9+, PyTorch 2.0+, NumPy 1.20+
