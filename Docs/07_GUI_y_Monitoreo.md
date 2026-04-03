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

