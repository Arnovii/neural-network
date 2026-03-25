# Análisis de Restructuración de UI — ps_gui.py

## Problema Identificado

La estructura visual anterior **mezclaba conceptos** de tres niveles distintos de entrenamiento:

```
❌ ESTRUCTURA ANTERIOR (INCORRECTA)
├─ Conexión TCP
├─ Entrenamiento (SECCIÓN PROBLEMÁTICA)
│  ├─ Épocas MLP (dinámica)
│  ├─ Épocas E2E (dinámica)
│  └─ Modo de operación
├─ CNN Extractor
└─ Clasificador MLP
```

**Problemas:**
1. Épocas MLP estaban lejos de su contexto (Clasificador MLP)
2. Épocas E2E eran un "reemplazo dinámico", no un concepto independiente
3. Modo de operación estaba entre los controles de épocas, no como configuración global
4. Falta separación visual entre modelo y proceso de entrenamiento

---

## Solución Implementada

```
✅ ESTRUCTURA NUEVA (CORRECTA)

├─ Conexión TCP
│  ├─ Host
│  └─ Puerto
│
├─ Configuración del Sistema ⭐ (NUEVA)
│  └─ Modo de operación
│     ├─ 🔒 Precomputación (CNN fija + MLP distribuido)
│     └─ ⚙️ End-to-End (CNN + MLP se entrenan juntos)
│
├─ CNN Extractor
│  ├─ Cargar / Entrenar
│  ├─ Arquitectura
│  ├─ Épocas CNN (cuando se entrena)
│  ├─ Learning Rate CNN
│  ├─ Muestras CNN
│  └─ Semilla CNN
│
├─ Clasificador MLP
│  ├─ Neuronas Ocultas 1
│  ├─ Neuronas Ocultas 2
│  └─ Épocas MLP ⭐ (REUBICADO - solo en PRECOMPUTED)
│
├─ Entrenamiento Distribuido ⭐ (NUEVA SECCIÓN)
│  ├─ Épocas E2E ⭐ (AGREGADO - solo en END_TO_END)
│  ├─ Learning Rate
│  ├─ Momentum SGD
│  ├─ Ejemplos de Entrenamiento
│  └─ Semilla
│
└─ Botones
   ├─ Encender Servidor
   ├─ Entrenar
   ├─ Apagar Servidor
   └─ Limpiar Gráficas
```

---

## Mapeo de Cambios

| Concepto | Ubicación Anterior | Ubicación Nueva | Razón |
|----------|-------------------|-----------------|-------|
| Modo | "Entrenamiento" (línea 418) | "Configuración del Sistema" (línea 406) | Configuración global del sistema |
| Épocas MLP | "Entrenamiento" (línea 410) | "Clasificador MLP" (nueva línea 672) | Pertenece al contexto del MLP |
| Épocas E2E | "Entrenamiento" (línea 415) | "Entrenamiento Distribuido" (nueva línea 683) | Parámetro global del sistema distribuido |
| Otros parámetros | "Clasificador MLP" | "Entrenamiento Distribuido" (línea 689+) | Configuran el proceso distribuido |

---

## Semántica de Épocas por Modo

### Modo: Precomputación (CNN fija + MLP distribuido)

```
Flujo de entrenamiento:
1. Cargar/Entrenar CNN localmente (parámetros CNN)
2. Distribuir CNN congelada a Workers
3. Entrenar MLP distribuido (Épocas MLP en Clasificador MLP)

Épocas relevantes:
- Épocas CNN (en CNN Extractor, si se entrena)
- Épocas MLP (en Clasificador MLP) ← Control visible
```

### Modo: End-to-End (CNN + MLP se entrenan juntos)

```
Flujo de entrenamiento:
1. Distribuir CNN+MLP a Workers
2. Entrenar CNN+MLP conjuntamente (parámetros globales)

Épocas relevantes:
- Épocas E2E (en Entrenamiento Distribuido) ← Control visible
  (representa el entrenamiento global del sistema)
```

---

## Variables `_frame_*` Manejadas Dinámicamente

```python
# Visibilidad controlada por _on_system_mode_change()

self._frame_mlp_epochs = ttk.Frame(frame)  # En Clasificador MLP
# Visible cuando: self._v_system_mode.get() == "precomputed"
# Oculto cuando: self._v_system_mode.get() == "end_to_end"

self._frame_e2e_epochs = ttk.Frame(frame)  # En Entrenamiento Distribuido
# Visible cuando: self._v_system_mode.get() == "end_to_end"
# Oculto cuando: self._v_system_mode.get() == "precomputed"
```

---

## Lógica de Entrenamiento en `_cmd_train()`

```python
# Seleccionar épocas según el modo de operación
training_mode = self._v_system_mode.get()
if training_mode == "precomputed":
    epochs = int(self._v_epochs.get())        # Épocas MLP
    log("Modo Precomputación: usando X épocas para MLP")
else:  # end_to_end
    epochs = int(self._v_e2e_epochs.get())    # Épocas E2E
    log("Modo End-to-End: usando X épocas para CNN+MLP")

# server.train(epochs=epochs, ...)
# El ParameterServer usa training_mode para aplicar lógica específica
```

---

## Beneficios de la Reorganización

| Aspecto | Mejora |
|--------|--------|
| **Claridad** | Cada sección representa un componente distinto |
| **Coherencia** | Épocas están con sus correspondientes clasificadores |
| **Usabilidad** | Solo controles relevantes para el modo actual son visibles |
| **Mantenibilidad** | Separación clara de responsabilidades |
| **Lógica** | La estructura visual refleja el flujo de entrenamiento |
| **Escalabilidad** | Fácil agregar nuevos parámetros en el lugar correcto |

---

## Notas Técnicas

1. **pack() vs pack_forget()**: Se usa esta combinación para ocultar/mostrar frames según el modo, sin reorganizar el layout
2. **pack()** con `fill=tk.X, pady=(4, 0)` mantiene el espaciado consistente
3. **pack_forget()** deja preservado el contenido del frame, se puede volver a mostrar sin reinicializar
4. **_on_system_mode_change()** se llama al final de `_build_ui()` para establecer visibilidad inicial correcta

---

## Testing Recomendado

```python
# Verificar que al cambiar de modo, se muestren los controles correctos:

1. Iniciar con modo "precomputed"
   ✓ "Épocas MLP" visible en Clasificador MLP
   ✓ "Épocas E2E" oculto

2. Cambiar a modo "end_to_end"
   ✓ "Épocas MLP" oculto
   ✓ "Épocas E2E" visible en Entrenamiento Distribuido

3. Cambiar nuevamente a "precomputed"
   ✓ Valores se conservan (no se resetean al cambiar modo)
   ✓ Visibilidad se invierte nuevamente

4. Entrenar en precomputed
   ✓ Usa self._v_epochs (Épocas MLP)

5. Entrenar en end_to_end
   ✓ Usa self._v_e2e_epochs (Épocas E2E)
```

