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
│     LOADING State (background thread)   │
│     • Validar HF Token                  │
│     • Descargar pesos ResNet-18 (~50MB) │
│     • Inicializar MLP (Kaiming init)    │
│     • Llamar ps.set_cnn(), ps.set_mlp() │
│     • Deshabilitar configuración        │
└──┬──────────────────────────────────────┘
   │ (CNN+MLP listos, espera Workers)
   ↓
┌─────────────────────────────────────────┐
│      LISTENING State                    │
│ • TCP accept loop activo                │
│ • Clock inicializado (00:00:00)         │
│ • Esperando primer evento on_step()     │
└──┬──────────────────────────────────────┘
   │ (llega primer step automáticamente)
   ↓
┌─────────────────────────────────────────┐
│      TRAINING State (AUTO)              │
│ • Clock corre cada 1 segundo            │
│ • on_step() actualiza métricas          │
│ • workers entrenando                    │
│ • gráficas actualizándose               │
└──┬──────────────────────────────────────┘
   │ click "Detener todo"
   ↓
┌─────────────────────────────────────────┐
│          OFFLINE                        │
│ • Limpiar conexiones                    │
│ • Reset Clock a 00:00:00                │
│ • Reset métricas a "—"                  │
│ • Rehabilitar configuración             │
└─────────────────────────────────────────┘
```

### Eventos y Callbacks

```python
├─ on_step(step, loss, acc, staleness, elapsed)
│  └─ Actualizar métricas, iniciar Clock en primer step
│     elapsed: tiempo desde primer step (segundos)
│
├─ on_report(step, loss, acc, elapsed)
│  └─ Reporte cada steps_per_report steps + status bar
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

**Streaming**:
- `Batch size` (int): default 64 - Imágenes por batch
- `Image size` (int): default 224 - Resolución de imágenes (ancho=alto)

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
- `Límite Steps` (int, opcional): default vacío (sin límite)

**Evaluación**:
- `Batches Validación` (int): default 50
- `[Evaluar]` Button

### Botones de Control

| Botón | Acción | Precondición |
|---|---|---|
| `Encender Servidor` | Inicia PS + carga modelo (estado LOADING) | Estado OFFLINE |
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
Clock: 00:15:32          → Reloj tiempo real (HH:MM:SS)
```

**Nota sobre Tiempo**:

La GUI muestra **dos** tipos de tiempo:

| Métrica | Descripción | Formato | Inicio |
|---------|-------------|---------|--------|
| **Clock** | Reloj tiempo real | HH:MM:SS | Mensaje START enviado |
| **elapsed** | Tiempo desde inicio | Segundos | Mensaje START enviado |

- **Clock**: Actualiza cada 1 segundo, se reinicia a 00:00:00 al apagar
- **elapsed**: Del PS, usado en status bar y exportado a CSV
- **Inicio**: Ambos начинают حساب desde el mensaje START (no desde primer STEP)

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
│ 0  │ 127.0.0.1:41234  │ Activo  [🟢]  │
│ 1  │ 192.168.1.2:9876 │ Activo  [🟢]  │
└───────────────────────────────────────┘
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

## Mensaje de Worker Host

Al iniciar el PS, se muestra automáticamente un mensaje indicando la IP que los Workers deben usar:

```
[PS] ✓ Servidor en 0.0.0.0:9999 | Worker host: 192.168.1.100 (usar --server-host en workers)
[PS]   | arch=resnet18 (freeze) | feature_dim=512 | MLP 512→1024→512→1000 | lr_mlp=0.001 lr_cnn=0.001 | batch=64 img=224 seed=None
```

### Para qué sirve

- El mensaje `Worker host` indica la **IP de esta máquina** donde corre el PS
- Los Workers remotos deben usar esta IP como `--server-host`
- Si el PS usa `0.0.0.0`, se detecta automáticamente la IP real de la máquina
- Si el PS usa `127.0.0.1`, se muestra `127.0.0.1` (solo para Workers locales)

---

## Limitador de Steps (Auto-detención)

### Propósito

El limitador de steps permite detener automáticamente el entrenamiento cuando se alcanza una cantidad específica de steps. Es útil para:
- Experimentos controlados
- Experimentación con duración fija
- Debugging con número conocido de iteraciones
- Evita entrenamiento infinito

### Configuración

El campo **Límite steps** se encuentra en la sección Async SGD:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `Límite steps` | int (opcional) | Vacío = sin límite, Entero > 0 = detener al alcanzar |

**Validación**:
- Vacío → Sin límite (entrenamiento indefinido)
- Entero positivo → Límite activo
- Cero o negativo → Error de validación

### Comportamiento

| Escenario | Acción |
|----------|--------|
| Manual (botón) | Pide confirmación con `messagebox.askyesno()` |
| Automático (límite) | Detiene sin preguntar, luego muestra `messagebox.showinfo()` |

### Flujo de Auto-detención

```
1. GUI detecta: step >= _max_steps (en callback _on_step)
2. Llama _auto_stop() (sin askyesno)
   → ps.stop() 
   → Limpia estado (workers, métricas, Clock)
   → _set_config_enabled(True)  ← Reactivar campos
3. Muestra messagebox.showinfo:
   "Entrenamiento detenido"
   "Se alcanzó el límite de steps configurado.
    Último step: N"
```

### Diferencia: Manual vs Automático

| Aspecto | Manual | Automático |
|--------|--------|------------|
| Botón | "■ Detener Todo" | Límite alcanzado |
| Confirmación | askyesno (Sí/No) | Ninguna |
| Mensaje | Ninguno | showinfo informativo |
| Estado final | OFFLINE | OFFLINE |

---

## Limitaciones Conocidas

1. **No persistencia**: Métricas se pierden al cerrar GUI
2. **Mono-PS**: Solo 1 PS soportado
3. **Validación manual**: No hay botón para exportar pesos
4. **Dark/Light mode**: No soportado (UI siempre light)
5. **Remote PS**: GUI solo conecta a PS en localhost (puede extenderse)

