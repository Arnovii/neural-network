# Diagrama Visual de la Nueva Estructura UI

## Vista de Panel Izquierdo (Scrollable)

```
╔════════════════════════════════════════════════════════════════╗
║           Parameter Server — Algoritmo de Diego                ║
╚════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ CONEXIÓN TCP                                                    │
├─────────────────────────────────────────────────────────────────┤
│ Host (IP de escucha):      [_______________  0.0.0.0___________]│
│ Puerto:                    [_9 9 9 9_]                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CONFIGURACIÓN DEL SISTEMA                          ⭐ NUEVA    │
├─────────────────────────────────────────────────────────────────┤
│ Modo de operación:                                              │
│ ◎ 🔒 Precomputación (CNN fija + MLP distribuido)               │
│ ○ ⚙️  End-to-End (CNN + MLP se entrenan juntos)                │
│                                                                  │
│ ℹ End-to-End: Mayor ancho de banda, CNN se adapta a los datos │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CNN EXTRACTOR                                                   │
├─────────────────────────────────────────────────────────────────┤
│ ◎ Cargar modelo  ○ Entrenar modelo                              │
│                                                                  │
│ ┌─ Modelo guardado: ───────────────────────────────────────┐   │
│ │ [ResNet18 | 75.23% | 2026-03-20] ▼                      │   │
│ │                                                           │   │
│ │ Información del modelo                                  │   │
│ │ Arquitectura   : resnet18                               │   │
│ │ Precisión        : 75.23%                               │   │
│ │ Pérdida          : 0.4521                               │   │
│ │ ...                                                     │   │
│ └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│ ℹ El PS distribuye la CNN al Worker automáticamente.           │
│   La extracción de features ocurre en el Worker.               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CLASIFICADOR MLP                                                │
├─────────────────────────────────────────────────────────────────┤
│ Neuronas ocultas 1 (32 – 1024):                                 │
│ [════════●════════════════════□]  256                          │
│                                                                  │
│ Neuronas ocultas 2 (32 – 512):                                  │
│ [═══════════════●═════════════□]  128                          │
│                                                                  │
│ ┌─ Épocas MLP (50 – 1000):  [VISIBLE si PRECOMPUTED] ⭐ MOVIDO│
│ │ [═════════════════●═════□]  100                        │
│ └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ENTRENAMIENTO DISTRIBUIDO                    ⭐ NUEVA SECCIÓN  │
├─────────────────────────────────────────────────────────────────┤
│ ┌─ Épocas E2E (50 – 1000):  [VISIBLE si END_TO_END] ⭐ AGREGADO│
│ │ [════════════════●════════□]  200                      │
│ └────────────────────────────────────────────────────────┘    │
│                                                                  │
│ Tasa de aprendizaje (0.0001 - 10):                              │
│ [_0.01__________]                                               │
│                                                                  │
│ Momentum SGD (0.0 = desactivado):                               │
│ [_0.9___________]                                               │
│                                                                  │
│ Ejemplos de entrenamiento (10 - 50000):                         │
│ [_50000_________]                                               │
│                                                                  │
│ Semilla (vacío = aleatoria):                                    │
│ [_______________]                                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ⚡ Encender servidor                                            │
│ ▶  Iniciar entrenamiento                                       │
│ ■  Apagar servidor                                             │
│ Limpiar gráficas                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Matriz de Visibilidad por Modo

```
┌────────────────────────────────────────────────────────────────────┐
│                    VISIBILIDAD DE CONTROLES                        │
├──────────────────────────────┬──────────────┬──────────────────────┤
│ Control                      │ PRECOMPUTED  │ END_TO_END           │
├──────────────────────────────┼──────────────┼──────────────────────┤
│ Configuración del Sistema    │ VISIBLE      │ VISIBLE              │
│ └─ Modo de operación         │ VISIBLE      │ VISIBLE              │
├──────────────────────────────┼──────────────┼──────────────────────┤
│ CNN Extractor                │ VISIBLE      │ VISIBLE              │
│ └─ Todos los parámetros      │ VISIBLE      │ VISIBLE              │
├──────────────────────────────┼──────────────┼──────────────────────┤
│ Clasificador MLP             │ VISIBLE      │ VISIBLE              │
│ ├─ Neuronas ocultas          │ VISIBLE      │ VISIBLE              │
│ └─ Épocas MLP                │ 🔵 VISIBLE   │ ⚫ OCULTO             │
├──────────────────────────────┼──────────────┼──────────────────────┤
│ Entrenamiento Distribuido    │ VISIBLE      │ VISIBLE              │
│ ├─ Épocas E2E                │ ⚫ OCULTO     │ 🔵 VISIBLE           │
│ ├─ Learning Rate             │ VISIBLE      │ VISIBLE              │
│ ├─ Momentum SGD              │ VISIBLE      │ VISIBLE              │
│ ├─ Ejemplos entrenamiento    │ VISIBLE      │ VISIBLE              │
│ └─ Semilla                   │ VISIBLE      │ VISIBLE              │
└──────────────────────────────┴──────────────┴──────────────────────┘

🔵 = Control actualmente visible y activo
⚫ = Control oculto por pack_forget()
```

---

## Flujo de Control al Cambiar Modo

```
Usuario cambia radio button en "Configuración del Sistema"
         │
         ↓
    _on_system_mode_change() se dispara
         │
         ├─ Si modo == "precomputed":
         │  ├─ _frame_mlp_epochs.pack(fill=tk.X, pady=(4, 0))
         │  └─ _frame_e2e_epochs.pack_forget()
         │
         └─ Si modo == "end_to_end":
            ├─ _frame_mlp_epochs.pack_forget()
            └─ _frame_e2e_epochs.pack(fill=tk.X, pady=(4, 0))
         │
         ↓
    UI se actualiza (solo componentes afectados se redibujan)
         │
         ↓
    Log: "[INFO] Modo cambiado a: ..."
    Log: "[INFO] Épocas de entrenamiento ajustadas para modo: ..."
```

---

## Relación de Épocas con Flujo de Entrenamiento

### Precomputación (CNN fija + MLP distribuido)

```
┌──────────────────────────────────────────────────────────┐
│ 1. Preparar CNN                                          │
│    └─ Si "Entrenar": usar self._v_cnn_epochs            │
│    └─ Si "Cargar": cargar de archivo                    │
├──────────────────────────────────────────────────────────┤
│ 2. Distribuir CNN congelada a Workers                    │
├──────────────────────────────────────────────────────────┤
│ 3. Entrenar MLP distribuido                              │
│    └─ Usar: self._v_epochs      ← Épocas MLP            │
│       (valor seleccionado en Clasificador MLP)           │
├──────────────────────────────────────────────────────────┤
│ 4. Agregación de gradientes en Parameter Server          │
│    └─ Realizar self._v_epochs iteraciones               │
└──────────────────────────────────────────────────────────┘
```

### End-to-End (CNN + MLP se entrenan juntos)

```
┌──────────────────────────────────────────────────────────┐
│ 1. Preparar CNN+MLP                                      │
│    └─ CNN: crear/cargar según configuración             │
│    └─ MLP: inicializar con parámetros                   │
├──────────────────────────────────────────────────────────┤
│ 2. Distribuir CNN+MLP a Workers                          │
├──────────────────────────────────────────────────────────┤
│ 3. Entrenar CNN+MLP conjuntamente                        │
│    └─ Usar: self._v_e2e_epochs ← Épocas E2E             │
│       (valor seleccionado en Entrenamiento Distribuido)  │
├──────────────────────────────────────────────────────────┤
│ 4. Actualización de pesos CNN y MLP en cada época        │
│    └─ Realizar self._v_e2e_epochs iteraciones           │
└──────────────────────────────────────────────────────────┘
```

---

## Linajes de Variables

```
self._v_epochs         (IntVar, default=100)
  ├─ _add_slider() en self._frame_mlp_epochs
  ├─ Ubicación: Clasificador MLP
  └─ Visible en: PRECOMPUTED
  └─ Usado en: _cmd_train() cuando modo=="precomputed"

self._v_e2e_epochs     (IntVar, default=200)
  ├─ _add_slider() en self._frame_e2e_epochs
  ├─ Ubicación: Entrenamiento Distribuido
  └─ Visible en: END_TO_END
  └─ Usado en: _cmd_train() cuando modo=="end_to_end"

self._v_system_mode    (StringVar, default="precomputed")
  ├─ Radio buttons en Configuración del Sistema
  ├─ Callback: _on_system_mode_change()
  └─ Usado en: _cmd_listen() para pasar a ParameterServer
              _cmd_train() para seleccionar épocas correctas
```

