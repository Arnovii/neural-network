# Interfaz Gráfica (GUI) y Monitoreo

## Rol de la GUI

**Archivo**: `ps_gui_imagenet.py`

La interfaz gráfica es el **front-end de control y monitoreo** del sistema distribuido:

1. **Control**: Iniciar/detener servidor, configurar hiperparámetros
2. **Monitoreo**: Visualizar métricas en tiempo real
3. **Interacción**: Botones, campos de entrada, gráficas
4. **Logging**: Historial de eventos estructurado

**No requiere experiencia en CLI** → usuario no-técnico puede entrenar

---

## Pantalla Principal

```
┌───────────────────────────────────────────────────────────────────────┐
│  Parameter Server — ImageNet-1k Distribuido                           │
├─────────────────────────┬─────────────────────────────────────────────│
│                         │                                             │
│  [LEFT PANEL]           │  [CENTER PANEL - WORKERS & METRICS]         │
│  ─────────────          │  ────────────────────────────────           │
│                         │                                             │
│  * Conexión TCP         │  Connected Workers:                         │
│    ├─ Host: 0.0.0.0     │  ┌───────────────────────────────┐          │
│    ├─ Puerto: 9999      │  │ ID │ Address      │ Status    │          │
│                         │  ├────┼──────────────┼───────────┤          │
│  * Dataset              │  │ 0  │ 127.0.0.1    │ Activo    │          │
│    ├─ ILSVRC/...        │  │ 1  │ 192.168.1.2  │ Activo    │          │
│                         │  └───────────────────────────────┘          │
│  * CNN Extractor        │  Status: [◆ TRAINING | ◆ LISTENING | ...]  │
│    ├─ ResNet-18         │                                             │
│    ├─ Simple            │  Métricas en Tiempo Real:                   │
│                         │  ├─ Step: 1,234                             │
│  * MLP Classifier       │  ├─ Loss: 6.8743                            │
│    ├─ h1: 1024          │  ├─ Accuracy: 2.34%                         │
│    ├─ h2: 512           │  ├─ Staleness: 2                            │
│                         │                                             │
│  * Async SGD            │  [GRÁFICAS]                                 │
│    ├─ LR: 0.001         │  ┌──────────┬──────────┬──────────┐         │
│    ├─ λ: 0.1            │  │ Loss     │ Accuracy │ Workers  │         │
│    ├─ steps/report: 500 │  │ [plot]   │ [plot]   │ [plot]   │         │
│    ├─ window: 200       │  └──────────┴──────────┴──────────┘         │
│                         │                                             │
│  * Evaluación           │  [LOG]                                      │
│    ├─ Batches: 50       │  [W0] Conectado | rank=0/1                  │
│    ├─ [Evaluar] Btn     │  [W0] Stream iniciado                       │
│                         │  [Step 100] loss=6.87 | acc=2.34%           │
│  ┌─────────────────────┐│                                             │
│  │ [Encender Servidor] ││                                             │
│  │ [Iniciar Entrena..] ││                                             │
│  │ [■  Detener Todo]   ││                                             │
│  │ [Limpiar Gráficas]  ││                                             │
│  └─────────────────────┘│                                             │
│                         │                                             │
└─────────────────────────┴─────────────────────────────────────────────┘
│ [Status Bar: Esperando workers... ]                                   │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Estados de la GUI

### Estado Machine

```
┌──────────────────────────────────────────┐
│               PSApp States               │
└──────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         [START: OFFLINE]                │
└──┬──────────────────────────────────────┘
   │ click "Encender Servidor"
   ↓
┌─────────────────────────────────────────┐
│         LISTENING State                 │
│ • TCP accept loop activo                │
│ • Esperando Workers                     │
│ • Pre-cargar CNN y MLP en background    │
└──┬────────┬────────────────────────────┘
   │        │ click "Iniciar Entrenamiento"
   │        ↓
   │     ┌─────────────────────────────────────────┐
   │     │  LOADING (background thread)            │
   │     │  • Descargar pesos ResNet-18 (~50MB)    │
   │     │  • Inicializar MLP (Kaiming init)       │
   │     │  • Llamar ps.set_cnn(), ps.set_mlp()    │
   │     └──┬──────────────────────────────┬───────┘
   │        │ (listo)                      │ (error)
   │        ↓                              ↓
   │     ┌────────────────────────┐    ERROR: messagebox
   │     │   TRAINING State       │→──────────────→ LISTENING
   │     │ • workers entrenando   │
   │     │ • poll() cada 100ms    │
   │     │ • gráficas actualizando│
   │     └──┬─────────────────────┘
   │        │ click "Detener todo"
   │        ↓
   └──→ OFFLINE (limpiar, close socket)
```

### Eventos y Callbacks

```python
├─ on_step(step, loss, acc, staleness)
│  └─ Actualizar métricas en tiempo real, update plot
│
├─ on_report(step, loss, acc)
│  └─ Reporte cada steps_per_report steps
│
├─ on_worker_connected(wid, addr)
│  └─ Agregar fila a tabla de workers
│
├─ on_worker_disconnected(wid)
│  └─ Quitar fila de tabla
│
└─ [GUI polls cada 100ms]
   ├─ Recibir eventos desde queue
   ├─ Actualizar tree view, métricas
   ├─ Redraw gráficas
   └─ Check button enable/disable
```

---

## Controles y Configuración

### Entrada: Parámetros Configurables

**Conexión**:
- `Host` (str): IP de escucha, default "0.0.0.0"
- `Puerto` (int): default 9999

**Dataset**:
- `Dataset HF Hub` (str): default "ILSVRC/imagenet-1k"
- `HF Token` (str, password): Token de acceso

**CNN**:
- Radio button: ResNet-18 (✓) | Simple

**MLP**:
- `h1` (int): default 1024
- ` h2` (int): default 512

**Async SGD**:
- `LR` (float): default 0.001, range (0.00001, 1.0)
- `λ (Staleness)` (float): default 0.1,range (0.0, 1.0)
- `Steps/Reporte` (int): default 500
- `Ventana Métricas` (int): default 200

**Evaluación**:
- `Batches Validación` (int): default 50
- `[Evaluar]` Button

### Botones de Control

| Botón | Acción | Precondición |
|---|---|---|
| `Encender Servidor` | Inicia PS + carga modelo | Estado OFFLINE |
| `Iniciar Entrena.` | Inicia background setup | LISTENING + ≥1 Worker |
| `Detener Todo` | STOP signal a Workers | LISTENING o TRAINING |
| `Limpiar Gráficas` | Reset history | Siempre habilitado |
| `Evaluar` | Validación en dataset val | TRAINING |

---

## Monitoreo en Tiempo Real

### Métricas Mostradas

```
Step: 1,234              → ps.current_version
Loss: 6.8743             → Last metric from on_step
Accuracy: 2.34%          → Last metric
Staleness: 2             → version - version_read del último UPDATES
```

### Entendiendo Accuracy: 3 Fuentes Distintas

**Confusión común**: El Accuracy que ves en 3 lugares es **DIFERENTE en cada uno**, porque se calcula de forma distinta:

#### 1. **Terminal Worker (Batch Individual)** — MUY RUIDOSO

```
[W0] batch=500 | loss=4.5306 | acc=18.75% | v=499
                                  ↑ Este accuracy
```

**Qué es**: Accuracy del batch individual que acaba de entrenar el Worker
- **Tamaño**: 64 imágenes (batch_size)
- **Actualización**: Cada batch (~0.1-1 segundo)
- **Variabilidad**: Muy alta (puede ser 0% o 100% por suerte)
- **Fórmula**: `acc = (predictions.argmax(-1) == labels).float().mean()`

**Por qué es ruidoso**:
- 64 imágenes al azar pueden ser easy o hard
- Accuracy = 0 si todos fallan, 25% si 16/64 acierta

**Ejemplo**: 3 batches consecutivos podrían dar 18.75%, 0%, 12.5%

---

#### 2. **GUI Encima de Gráfica (Ventana Deslizante)** — SUAVIZADO MEDIO

```
Acc: 14.06%  ← En la etiqueta encima del gráfico de precisión
```

**Qué es**: Promedió de Accuracy de los últimos N steps (ventana deslizante)
- **Tamaño ventana**: 50 o 200 (configurable en GUI)
- **Actualización**: Cada `on_step()` callback (~cada paso del PS)
- **Latencia**: Demora porque deber llenar la ventana completamente
- **Fórmula**: `acc_ventana = mean([acc_step_1, acc_step_2, ..., acc_step_N])`

**Implementación** (en parameter_server.py):
```python
class RunningMetrics:
    def __init__(self, window=200):
        self._accs = collections.deque(maxlen=window)
    
    def update(self, loss, acc):
        self._accs.append(acc)
    
    def snapshot(self):
        return float(np.mean(self._accs)) if self._accs else 0.0
```

**Por qué demora en actualizarse**:
- Con ventana=200: Necesita 200 steps (unos 3-5 min en CPU) antes de ser estable
- Con ventana=50: Necesita 50 steps (30-60 seg)
- Mientras se llena: Accuracy fluctúa bastante
- Una vez lleno: Accuracy es muy estable

**Ventaja**: Suavizado pero no es ruido puro

---

#### 3. **Puntos en Gráfica (Promedio por Reporte)** — MÁS ESTABLE

```
[Step 515] loss=4.6308 | acc=12.66%
                             ↑ Este es el que se grafica
```

**Qué es**: Promedio de Accuracy del último reporte (múltiples steps)
- **Intervalo reporte**: Cada 500 steps (default, configurable)
- **Actualización**: Solo cada 500 steps (mucho más lento)
- **Estabilidad**: Muy alta porque promedia muchos batches
- **Fórmula**: `acc_reporte = mean([últimos 500 steps de acc])`

**Implementación** (en parameter_server.py):
```python
if self.current_step % self.steps_per_report == 0:
    loss_avg, acc_avg = self._metrics.snapshot()
    self.on_report(self.current_step, loss_avg, acc_avg)  # Callback → GUI
    # GUI recibe esto y lo grafica
```

**Ventaja**: Muy estable, sin ruido, fácil de ver tendencia

**Desventaja**: Demora ~500 steps antes de ver valor nuevo

---

#### Comparación Lado a Lado

| Métrica | Dónde Aparece | Actualización | Ventana | Volatilidad |
|---------|---|---|---|---|
| **Batch** | `Terminal` | Cada batch (~0.1s) | 1 batch (64 imgs) | ⚠️⚠️⚠️ Muy alta |
| **Ventana deslizante** | `GUI encima gráfica` | Cada step (~1s) | Últimos 50/200 steps | ⚠️⚠️ Media |
| **Puntos gráfica** | `Línea azul con puntos` | Cada 500 steps (~5-10 min) | Promedio últimos 500 steps | ✓ Baja |

---

#### ¿Por qué son tan diferentes?

**Ejemplo real** (de tus logs):

```
TERMINAL (batch 500):      acc = 18.75%  ← Batch tuvo suerte
GUI actual (step 513):     acc = 14.06%  ← Promedio últimos 200 steps
Gráfica punto:             acc = 12.66%  ← Promedio últimos 500 steps

Tendencia: 18.75% > 14.06% > 12.66%
              ↑ Ruido alto    ↑ Real     ↑ Más real
```

El batch salió con accuracy alta por suerte, pero el verdadero accuracy del modelo es:
- Ventana=200: 14.06%
- Ventana=500: 12.66% (mejor estimador)

---

#### Recomendación Práctica

**Durante entrenamiento**:
- **Ignora** el accuracy del batch individual (muy ruidoso)
- **Monitor** la gráfica de puntos azules (más estable)
- **Verifica** que la tendencia general baje o suba según esperado

**Ejemplo patrón normal**:
```
step=100:   acc=0.5%   ← Inicio, casi random
step=500:   acc=3.2%   ← Empieza a aprender
step=1000:  acc=7.8%   ← Tendencia clara
step=5000:  acc=25%    ← Convergencia real
```

Si ves que los **puntos azules bajan** (en lugar de subir), hay problemas.

---

### Tabla de Workers Conectados

```
┌───────────────────────────────────────┐
│ ID │ Dirección        │ Estado        │
├────┼──────────────────┼───────────────┤
│ 0  │ 127.0.0.1:41234  │ Activo  [🟢] │
│ 1  │ 192.168.1.2:9876 │ Activo  [🟢] │
└────────────────────────────────  ─────┘
```

**Color-coded**: Cada Worker con color distinto en gráficas

---

## Comportamiento Asincrónico (No Bloqueo)

### Poll Loop

```python
def _poll(self):
    """Ejecutado cada 100ms por self.root.after()."""
    try:
        while True:
            kind, data = self._q.get_nowait()  # Non-blocking
            
            if kind == "connected":
                self._on_connected(*data)
            elif kind == "step":
                self._on_step(*data)
            elif kind == "report":
                self._on_report(*data)
            # ... más tipos ...
    
    except queue.Empty:
        pass  # Normal, no eventos
    except Exception as e:
        self._log(f"[ERROR] {e}")
    
    # Reprogramar siguiente poll
    if self._state != self._S_OFFLINE:
        self.root.after(100, self._poll)  # 100ms delay
```

**Ventaja**: GUI nunca se cuelga, thread callbacks ponen en queue

### Thread Safety

```python
# En PS thread (otro thread):
self.on_step(step, loss, acc, staleness)
↓
self._q.put(("step", (step, loss, acc, staleness)))  # Thread-safe

# En GUI thread (main):
_poll() ejecutada cada 100ms
↓
self._q.get_nowait()  # Sin bloqueo
↓
self._on_step(step, loss, acc, staleness)
↓
actualizar labels, tablas, gráficas
```

---

## Logging Estructurado

### Panel de Logs

```
Text widget con scrollbar:
┌──────────────────────────────┐
│ [W0] Conectado | rank=0/1    │←─ color: green
│ [W0] Stream iniciado: ...    │←─ color: green
│ [Step 100] loss=7.23 | acc.. │←─ color: blue
│ ...                          │
│ scroll ↑ máximo 300 líneas   │
└──────────────────────────────┘
```

### Características

- **Max 300 líneas**: Antiguas se eliminan auto
- **Colores**: [verde: conexión], [azul: step], [rojo: error]
- **Timestamps**: Automático con `datetime.now()`
- **Scroll auto**: Siempre muestra línea más nueva

---

## Limitaciones Conocidas

1. **No persistencia**: Métricas se pierden al cerrar GUI
2. **Mono-PS**: Solo 1 PS soportado
3. **Validación manual**: No hay botón para exportar pesos
4. **Dark/Light mode**: No soportado (UI siempre light)
5. **Remote PS**: GUI solo conecta a PS en localhost (puede extenderse)

