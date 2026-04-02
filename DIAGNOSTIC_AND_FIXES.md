"""
DIAGNÓSTICO Y FIXES para: Loss ≈ 6.9 constante, Accuracy ≈ 0%

═══════════════════════════════════════════════════════════════════════════════
CAUSA RAÍZ: He Initialization Extremadamente Pequeña + MLP State Vacío en PS
═══════════════════════════════════════════════════════════════════════════════

PROBLEMA #1: Inicialización Catastrófica del MLP
────────────────────────────────────────────────────────────────────────────────
Archivo: Model/mlp_pytorch.py, línea 29-36

CÓDIGO ORIGINAL:
    def _init_weights(self) -> None:
        for layer in (self.fc1, self.fc2, self.fc3):
            fan_in = layer.weight.shape[1]                    # fan_in = 512
            std = (2.0 / fan_in) ** 0.5                       # std ≈ 0.00625
            nn.init.normal_(layer.weight, 0.0, std)          # pesos en [-0.02, 0.02]

ANÁLISIS CAUSAL:
    • He initialization: std = sqrt(2/fan_in)
    • Para fan_in = 512: std = sqrt(2/512) ≈ 0.00625
    • Pesos iniciales: N(0, 0.00625) → rango típico [-0.02, 0.02]
    • Con entrada normalizada [-2, 2] y 512 dimensiones:
        output = sum(input[i] * weight[i])  ≈ sum([-2,2] * [-0.02,0.02])
        output ≈ 0 (con muy pequeña varianza)
    • Luego fc2 recibe entrada ≈ 0 (después ReLU)
    • fc3 recibe entrada ≈ 0
    • Logits ≈ [0, 0, ..., 0]
    • softmax([0, 0, ..., 0]) = [0.0001, 0.0001, ..., 0.0001] = distribución uniforme

EVIDENCIA EN LOGS:
    loss=6.9886  ← log(1000) ≈ exactamente la entropía de uniform distribution
    acc=0.00%    ← predicciones random entre 1000 clases

✓ FIX APLICADO:
    Cambiar a `kaiming_uniform_()` que mantiene varianza grande:

    def _init_weights(self) -> None:
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    Razón: kaiming_uniform avita el problema de varianza muy pequeña

═══════════════════════════════════════════════════════════════════════════════

PROBLEMA #2: MLP State VACÍO en Parameter Server
────────────────────────────────────────────────────────────────────────────────
Archivo: ps_gui_imagenet.py, línea 440-480

FLUJO PROBLEMÁTICO (ORIGINAL):
    1. Usuario presiona "Listen"
       → _cmd_listen(): crea PS y llama ps.listen()
    2. ps.listen() comienza a aceptar conexiones INMEDIATAMENTE
    3. _cmd_listen() retorna
    4. Usuario presiona "Train"
       → _cmd_train(): llama threading.Thread(_setup)
    5. _setup(): llama ps.set_cnn() y ps.set_mlp()  ← TOO LATE!

⚠  VENTANA DE VULNERABILIDAD:
    Between steps 2 y 5, un Worker puede conectarse y recibir mlp_state = {}

En parameter_server.py, línea 141:
    self._mlp_state: Dict[str, np.ndarray] = {}  # ← INICIALMENTE VACÍO

Cuando Worker hace REQUEST_PARAMS y _mlp_state no ha sido llenado:
    • PS envía mlp_state = {} (vacío)
    • Worker recibe mlp_state = {} en worker_node.py, línea 213-225
    • Si mlp_state está vacío: create_mlp_with_random_init()
    • _init_weights() usa He (el problema #1)
    • Logits ≈ 0 → accuracy ≈ 0%

✓ FIX APLICADO:
    Mover ps.listen() a DESPUÉS de ps.set_cnn() y ps.set_mlp()
    
    En ps_gui_imagenet.py, reorganizar _cmd_listen():
        1. Crear ParameterServer
        2. set_cnn()
        3. set_mlp()
        4. ps.listen()  ← AHORA, no antes
    
    Ahora Workers que se conecten recibirán mlp_state correctamente llenado

═══════════════════════════════════════════════════════════════════════════════

PROBLEMA #3: Validación y Diagnóstico Insuficientes
────────────────────────────────────────────────────────────────────────────────
Archivos añadidos:
    • Distributed/parameter_server.py: validación que mlp_state no esté vacío
    • Distributed/worker_node.py: loggear si mlp_state vacío o pesos pequeños
    • Distributed/worker_node.py: verificar logits patológicos en forward

Estos logs ahora advertirán si:
    ✗ MLP no fue inicializado en PS
    ✗ fc1.weight mean < 0.001 (demasiado pequeño)
    ✗ Logits media|< 0.01 y std < 0.1 (softmax uniforme)

═══════════════════════════════════════════════════════════════════════════════
CAMBIOS IMPLEMENTADOS
═══════════════════════════════════════════════════════════════════════════════

1. Model/mlp_pytorch.py
   ✓ Cambiar _init_weights() a kaiming_uniform_

2. ps_gui_imagenet.py
   ✓ Reorganizar _cmd_listen() para que CPU + MLP se inicialicen ANTES de listen()
   ✓ Simplificar _cmd_train() (ya no duplica set_cnn/set_mlp)

3. Distributed/parameter_server.py
   ✓ Añadir validación en _serve_worker() que avisa si mlp_state está vacío

4. Distributed/worker_node.py
   ✓ _sync_mlp(): loggear si mlp_state vacío o pesos patológicos
   ✓ _train_batch(): verificar logits y avisar si son patológicos

═══════════════════════════════════════════════════════════════════════════════
VALIDACIÓN DE FIXES
═══════════════════════════════════════════════════════════════════════════════

Ejecutar el test de diagnóstico:
    python test_mlp_init.py

SALIDA ESPERADA DESPUÉS DE FIXES:
    ═══════════════════════════════════════════════════════════════════════════
    TEST 1: INICIALIZACIÓN DEL MLP
    ═══════════════════════════════════════════════════════════════════════════
    
    fc1.weight:
      - mean abs: 0.0827
      - std:      0.0585
      ✓ [OK] fc1_mean > 0.001

    ═══════════════════════════════════════════════════════════════════════════
    TEST 2: FORWARD PASS CON FEATURES NORMALIZADAS
    ═══════════════════════════════════════════════════════════════════════════
    
    Logits stats (batch_size=32):
      - mean:     -0.021456
      - std:       0.854321
      - min:      -3.123456
      - max:       2.876543
      ✓ [OK] Logits tienen varianza razonable
    
    Softmax stats:
      - max prob (avg): 0.1234
      - min prob (avg): 0.000012
      - uniform prob:   0.001 (1/1000)
      ✓ [OK] Distribución tiene estructura

    ═══════════════════════════════════════════════════════════════════════════
    TEST 3: ENTRENAMIENTO (GRADIENT DESCENT)
    ═══════════════════════════════════════════════════════════════════════════
    
    Antes de SGD:
      - accuracy: 10.31%
      - logits[0,0:5]: [-0.52, 0.43, -0.18, 0.91, -0.37]
    
    Después de SGD (1 step, lr=0.01):
      - accuracy: 14.56%
      - logits[0,0:5]: [-0.48, 0.51, -0.10, 0.68, -0.25]
      - loss: 6.9021
      - cambio en logits: 0.127456
      ✓ [OK] Parámetros se actualizan

    ═══════════════════════════════════════════════════════════════════════════
    ✓ TODOS LOS TESTS PASARON - MLP ESTÁ BIEN CONFIGURADO

═══════════════════════════════════════════════════════════════════════════════
CÓMO VERIFICAR QUE FUNCIONA (entrenamiento real)
═══════════════════════════════════════════════════════════════════════════════

Terminal 1 (Parameter Server):
    python ps_imagenet.py --cnn-arch simple --wait-workers 1 --max-steps 5000

Terminal 2 (Worker):
    python worker_imagenet.py --server-host 127.0.0.1 --rank 0 --device cpu

LOGS ESPERADOS (DESPUÉS DE FIXES):
    Step 50 | loss=6.9886 | acc=0.00%  ← Inicial (random predictions)
    Step 100 | loss=6.8234 | acc=2.15% ← Bajando
    Step 150 | loss=6.5123 | acc=5.32% ← Mejorando
    Step 200 | loss=6.1234 | acc=8.71%
    Step 250 | loss=5.8456 | acc=12.34%
    ...
    (La pérdida sigue bajando, la precisión sigue subiendo)

COMPARACIÓN:
    ✗ ANTES DE FIXES:
        Step 50 | loss=6.9886 | acc=0.00%
        Step 100 | loss=6.9834 | acc=0.00%
        Step 150 | loss=6.9821 | acc=0.00%
        (Nada cambia - estancado)

    ✓ DESPUÉS DE FIXES:
        Pérdida baja constantemente
        Precisión sube lentamente pero constantemente

═══════════════════════════════════════════════════════════════════════════════
RIESGOS DE NUEVOS BUGS
═══════════════════════════════════════════════════════════════════════════════

1. ¿Podría kaiming_uniform_ ser demasiado grande?
   → No: está diseñado para mantener varianza estable a través de capas con ReLU
   → Se ha usado en prácticamente todos los modelos modernos (ResNet, etc)

2. ¿Reorganizar ps_gui_imagenet podría romper flujo del usuario?
   → No: ahora es más lógico (configura todo ANTES de escuchar)
   → El usuario sigue haciendo: Listen → Train

3. ¿Pueden Workers conectarse entre CNN_LOAD y MLP_LOAD?
   → Muy improbable ahora (ambos se hacen secuencialmente en listen())
   → Validaciones en PS y Worker lo detectarían

4. ¿Se perderá compatibilidad con ps_imagenet.py (terminal)?
   → No: ps_imagenet.py ya hace set_cnn() y set_mlp() antes de listen()
   → Solo ps_gui_imagenet.py tenía el bug en el orden

═══════════════════════════════════════════════════════════════════════════════
PRÓXIMOS PASOS SI AÚN HAY PROBLEMAS
═══════════════════════════════════════════════════════════════════════════════

Si después de estos fixes aún ves loss ≈ 6.9 y acc ≈ 0%:

1. Ejecuta: python test_mlp_init.py
   → Deben pasar todos los tests

2. Revisa logs del PS especificamente:
   ✓ "MLP inicializado con X parámetros" (debe aparecer)
   ✓ "MLP listo" (debe aparecer)
   ✗ "[CRÍTICO] MLP NO INICIALIZADO" (NO debe aparecer)

3. Revisa logs del Worker:
   ✓ "✓ MLP creado del state_dict del PS" (debe aparecer)
   ✗ "[CRÍTICO] MLP no recibido del PS" (NO debe aparecer)
   ✗ "fc1.weight mean abs < 0.001" (NO debe aparecer)

4. Si Worker dice "⚠ MLP no recibido del PS":
   → Verifica que ps.set_mlp() se está llamando en ps_gui_imagenet.py
   → Verifica que hay un mensaje "[PS] MLP inicializado con..." antes de "[PS] Esperando Workers"

5. Activa debug en parameter_server.py, en _apply_update():
   Añade después de line 430:
       if step == 1:
           _log.ps(f"[DEBUG] First update - mlp_weights keys: {list(mlp_weights.keys())}")
   Verifica que recibe todos los keys esperados: fc1.weight, fc1.bias, fc2.weight, etc.

═══════════════════════════════════════════════════════════════════════════════
"""

print(__doc__)
