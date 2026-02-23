"""
=============================================================================
RED NEURONAL DESDE CERO CON EL "ALGORITMO DE ARNOVI"
=============================================================================
Autor: Arnovi Jimenez
Descripción: 
    Entrenamiento de una red neuronal clásica (fully connected) para
    clasificar dígitos escritos a mano (MNIST). Implementado desde cero
    usando solo NumPy, sin librerías de deep learning.
    
    Incluye el "Algoritmo de Arnovi": entrenamiento por particiones
    independientes con promediado final de parámetros.

Arquitectura:
    Entrada (784) → Capa Oculta (30, sigmoide) → Salida (10, softmax)

Método de optimización:
    Gradiente descendente completo (batch gradient descent, NO estocástico)
=============================================================================
""" 

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# PASO 0: CONFIGURACIÓN GENERAL
# =====================================================================
# Aquí definimos los "hiperparámetros": valores que nosotros elegimos
# antes de entrenar y que NO se aprenden automáticamente.

NEURONAS_ENTRADA = 784     # 28x28 píxeles por imagen
NEURONAS_OCULTA = 30       # Neuronas en la capa oculta
NEURONAS_SALIDA = 10       # Un dígito del 0 al 9
LEARNING_RATE = 0.5        # Tasa de aprendizaje (α)
EPOCAS = 10                # Cuántas veces recorremos todos los datos
NUM_PARTICIONES = 2        # Número de particiones para el Algoritmo de Arnovi
                           # Cambia este valor: 1 = sin particiones (red normal)
                           #                    2, 3, 4... = con Algoritmo de Arnovi
NUM_REPETICIONES = 10  # Cuántas veces ejecutamos todo el proceso



# =====================================================================
# PASO 1: CARGAR EL DATASET MNIST
# =====================================================================
# MNIST contiene 60,000 imágenes de dígitos escritos a mano.
# Cada imagen es de 28x28 píxeles en escala de grises (0-255).
#
# Usamos torchvision SOLO para descargar los datos. Después convertimos
# todo a arrays de NumPy (NO usamos tensores de PyTorch).
# 
# NOTA: Si torchvision no está disponible, hay una alternativa con
# descarga directa que se incluye al final como opción B.

def cargar_mnist():
    """
    Carga MNIST usando torchvision y devuelve arrays de NumPy.
    
    Retorna:
        X_train: (784, 50000) - Imágenes de entrenamiento (cada columna = 1 imagen)
        Y_train: (50000,)     - Etiquetas de entrenamiento (números 0-9)
        X_test:  (784, 10000) - Imágenes de prueba
        Y_test:  (10000,)     - Etiquetas de prueba
    """
    from torchvision import datasets
    
    # Descargamos MNIST. transform=None significa que NO convertimos a tensor.
    # Los datos llegan como imágenes PIL (Python Imaging Library).
    dataset_train = datasets.MNIST(root='./datos_mnist', train=True, 
                                   download=True, transform=None)
    dataset_test = datasets.MNIST(root='./datos_mnist', train=False, 
                                  download=True, transform=None)
    
    # --- Convertimos las imágenes a arrays de NumPy ---
    # Cada imagen PIL la convertimos a array, la aplanamos (28x28 → 784),
    # y la normalizamos dividiendo entre 255 para que los valores estén entre 0 y 1.
    # (Normalizar ayuda MUCHO al entrenamiento: valores entre 0 y 1 son más estables)
    
    # Entrenamiento: usamos las primeras 50,000 imágenes
    imagenes_train = []
    etiquetas_train = []
    for i in range(50000):
        imagen, etiqueta = dataset_train[i]
        # np.array(imagen) convierte la imagen PIL a una matriz 28x28
        # .flatten() la convierte a un vector de 784
        # / 255.0 normaliza los valores de [0,255] a [0,1]
        imagenes_train.append(np.array(imagen).flatten() / 255.0)
        etiquetas_train.append(etiqueta)
    
    # Prueba: usamos las 10,000 imágenes del conjunto de test
    imagenes_test = []
    etiquetas_test = []
    for i in range(len(dataset_test)):
        imagen, etiqueta = dataset_test[i]
        imagenes_test.append(np.array(imagen).flatten() / 255.0)
        etiquetas_test.append(etiqueta)
    
    # Convertimos las listas a arrays de NumPy
    # .T (transponer) para que cada COLUMNA sea una imagen
    # Esto es por conveniencia matemática: X tiene forma (784, m) donde m = núm. imágenes
    X_train = np.array(imagenes_train).T   # Forma: (784, 50000)
    Y_train = np.array(etiquetas_train)     # Forma: (50000,)
    X_test = np.array(imagenes_test).T      # Forma: (784, 10000)
    Y_test = np.array(etiquetas_test)       # Forma: (10000,)
    
    return X_train, Y_train, X_test, Y_test


# =====================================================================
# PASO 1.5: CREAR PARTICIONES PARA EL ALGORITMO DE ARNOVI
# =====================================================================
# Dividimos el dataset de entrenamiento en 'n' particiones.
# Cada partición debe tener representación de TODOS los dígitos (0-9).
# 
# Estrategia: mezclamos los datos aleatoriamente y luego dividimos.
# Al mezclar un dataset grande como MNIST (50,000 imágenes), la 
# probabilidad de que cada partición tenga todos los dígitos es 
# prácticamente 100%.

def crear_particiones(X_train, Y_train, n_particiones):
    """
    Divide el dataset de entrenamiento en n particiones balanceadas.
    
    Parámetros:
        X_train: (784, 50000) - Todas las imágenes de entrenamiento
        Y_train: (50000,)     - Todas las etiquetas
        n_particiones: int    - Número de particiones deseadas
    
    Retorna:
        particiones: lista de tuplas [(X1, Y1), (X2, Y2), ...]
                     donde Xi tiene forma (784, 50000/n) y Yi tiene forma (50000/n,)
    """
    m = X_train.shape[1]  # Número total de imágenes (50000)
    
    # Creamos una permutación aleatoria de los índices [0, 1, 2, ..., 49999]
    # Esto "mezcla" los datos para que cada partición sea representativa
    indices = np.random.permutation(m)
    
    # Reordenamos X e Y con los índices mezclados
    X_mezclado = X_train[:, indices]
    Y_mezclado = Y_train[indices]
    
    # Dividimos en n partes iguales
    tamano_particion = m // n_particiones
    particiones = []
    
    for i in range(n_particiones):
        inicio = i * tamano_particion
        fin = (i + 1) * tamano_particion
        Xi = X_mezclado[:, inicio:fin]
        Yi = Y_mezclado[inicio:fin]
        particiones.append((Xi, Yi))
        
        # Verificamos que la partición tenga todos los dígitos
        digitos_presentes = np.unique(Yi)
        print(f"  Partición {i+1}: {Xi.shape[1]} imágenes, "
              f"dígitos presentes: {digitos_presentes}")
    
    return particiones


# =====================================================================
# PASO 2: INICIALIZACIÓN DE PARÁMETROS
# =====================================================================
# Antes de entrenar, necesitamos valores iniciales para W1, b1, W2, b2.
#
# Usamos INICIALIZACIÓN XAVIER:
#   W = aleatorio * sqrt(1 / n_entrada)
#
# ¿Por qué Xavier? Si los pesos iniciales son muy grandes, las
# activaciones se saturan (sigmoide da valores cercanos a 0 o 1,
# donde su derivada es casi 0, y los gradientes "desaparecen").
# Si son muy pequeños, las señales se desvanecen al pasar por las capas.
# Xavier busca un punto medio que mantiene las señales estables.
#
# Los sesgos (b) se inicializan en cero. Esto es estándar.

def inicializar_parametros():
    """
    Crea e inicializa los parámetros de la red neuronal.
    
    Retorna:
        parametros: diccionario con W1, b1, W2, b2
    """
    np.random.seed(None)  # Semilla aleatoria (diferente cada vez)
    
    # W1: (30 x 784) - Pesos entre capa de entrada y capa oculta
    # Xavier: multiplicamos por sqrt(1/784) ≈ 0.0358
    W1 = np.random.randn(NEURONAS_OCULTA, NEURONAS_ENTRADA) * np.sqrt(1.0 / NEURONAS_ENTRADA)
    
    # b1: (30 x 1) - Sesgos de la capa oculta
    b1 = np.zeros((NEURONAS_OCULTA, 1))
    
    # W2: (10 x 30) - Pesos entre capa oculta y capa de salida
    # Xavier: multiplicamos por sqrt(1/30) ≈ 0.1826
    W2 = np.random.randn(NEURONAS_SALIDA, NEURONAS_OCULTA) * np.sqrt(1.0 / NEURONAS_OCULTA)
    
    # b2: (10 x 1) - Sesgos de la capa de salida
    b2 = np.zeros((NEURONAS_SALIDA, 1))
    
    parametros = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parametros


# =====================================================================
# PASO 3: FUNCIONES DE ACTIVACIÓN
# =====================================================================

def sigmoide(Z):
    """
    Función sigmoide: σ(z) = 1 / (1 + e^(-z))
    
    Toma cualquier número real y lo "aplasta" al rango (0, 1).
    
    Ejemplo:
        sigmoide(-10) ≈ 0.0000  (número muy negativo → casi 0)
        sigmoide(0)   = 0.5     (cero → exactamente 0.5)
        sigmoide(10)  ≈ 1.0000  (número muy positivo → casi 1)
    
    Se aplica elemento por elemento a toda la matriz Z.
    """
    # np.clip limita Z entre -500 y 500 para evitar overflow numérico
    # (e^(-500) es tan pequeño que Python lo trata como 0, lo cual está bien,
    #  pero e^(500) causaría un error de overflow)
    Z_clipped = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z_clipped))


def sigmoide_derivada(A):
    """
    Derivada de la sigmoide: σ'(z) = σ(z) · (1 - σ(z))
    
    NOTA: recibimos A = σ(Z), es decir, ya la sigmoide aplicada.
    Esto es muy conveniente porque no necesitamos recalcular la sigmoide.
    """
    return A * (1 - A)


def softmax(Z):
    """
    Función softmax: convierte un vector de números reales en probabilidades.
    
    softmax(z_i) = e^(z_i) / Σ(e^(z_j))
    
    La salida siempre:
      - Tiene valores entre 0 y 1
      - Suma exactamente 1 (por columna, si es una matriz)
    
    Ejemplo para una imagen:
        Z = [2.0, 1.0, 0.1, 3.0, 0.5, 0.2, 0.1, 0.3, 0.1, 0.1]
        softmax(Z) ≈ [0.17, 0.06, 0.03, 0.47, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03]
        → La red cree que es un "3" (probabilidad más alta: 0.47)
    
    Truco de estabilidad numérica: restamos el máximo de cada columna.
    Esto NO cambia el resultado matemático (se cancela en la división),
    pero evita que e^(z) explote con números grandes.
    """
    # Restamos el máximo por columna para estabilidad numérica
    Z_estable = Z - np.max(Z, axis=0, keepdims=True)
    exponenciales = np.exp(Z_estable)
    return exponenciales / np.sum(exponenciales, axis=0, keepdims=True)


# =====================================================================
# PASO 3.5: CONVERTIR ETIQUETAS A ONE-HOT
# =====================================================================
# Si la etiqueta es "3", necesitamos convertirla a:
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# Esto se llama codificación "one-hot" (un solo "1" caliente).
# Es necesario para poder comparar con la salida del softmax (que da
# probabilidades para cada clase).

def one_hot(Y, num_clases=10):
    """
    Convierte un vector de etiquetas a formato one-hot.
    
    Ejemplo:
        Y = [3, 0, 7]
        one_hot(Y) = [[0, 1, 0],   ← clase 0
                      [0, 0, 0],   ← clase 1
                      [0, 0, 0],   ← clase 2
                      [1, 0, 0],   ← clase 3  ← el "1" está aquí para la primera imagen
                      [0, 0, 0],   ← clase 4
                      [0, 0, 0],   ← clase 5
                      [0, 0, 0],   ← clase 6
                      [0, 0, 1],   ← clase 7  ← el "1" está aquí para la tercera imagen
                      [0, 0, 0],   ← clase 8
                      [0, 0, 0]]   ← clase 9
        Forma: (10, 3) → 10 clases, 3 imágenes
    """
    m = Y.shape[0]  # Número de ejemplos
    Y_one_hot = np.zeros((num_clases, m))
    Y_one_hot[Y, np.arange(m)] = 1  # Pone un 1 en la posición correcta
    return Y_one_hot


# =====================================================================
# PASO 4: FORWARD PROPAGATION (Propagación hacia adelante)
# =====================================================================
# Proceso: Entrada → Capa Oculta → Salida
# Calculamos las salidas crudas (Z) y las activaciones (A) de cada capa.

def forward_propagation(X, parametros):
    """
    Realiza la propagación hacia adelante.
    
    Parámetros:
        X: (784, m) - Imágenes de entrada (m imágenes)
        parametros: diccionario con W1, b1, W2, b2
    
    Retorna:
        cache: diccionario con Z1, A1, Z2, A2 (valores intermedios
               que necesitamos para backward propagation)
    """
    W1 = parametros['W1']
    b1 = parametros['b1']
    W2 = parametros['W2']
    b2 = parametros['b2']
    
    # --- Capa de entrada → Capa oculta ---
    # Z1 = W1 · X + b1
    # Dimensiones: (30, 784) · (784, m) + (30, 1) = (30, m)
    # (el +b1 se "expande" automáticamente a todas las columnas, esto se llama broadcasting)
    Z1 = W1.dot(X) + b1
    
    # A1 = sigmoide(Z1) - Activación de la capa oculta
    # Dimensiones: (30, m)
    A1 = sigmoide(Z1)
    
    # --- Capa oculta → Capa de salida ---
    # Z2 = W2 · A1 + b2
    # Dimensiones: (10, 30) · (30, m) + (10, 1) = (10, m)
    Z2 = W2.dot(A1) + b2
    
    # A2 = softmax(Z2) - Activación de la capa de salida (predicción final)
    # Dimensiones: (10, m) - Cada columna es una distribución de probabilidades
    A2 = softmax(Z2)
    
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return cache


# =====================================================================
# PASO 5: CALCULAR EL COSTO (PÉRDIDA)
# =====================================================================
# Medimos qué tan equivocada está la red usando Cross-Entropy Loss.

def calcular_costo(A2, Y_one_hot):
    """
    Calcula la pérdida de entropía cruzada (cross-entropy loss).
    
    L = -(1/m) · Σ Σ y_ij · log(a2_ij)
    
    Parámetros:
        A2: (10, m)       - Predicciones del softmax
        Y_one_hot: (10, m) - Etiquetas reales en formato one-hot
    
    Retorna:
        costo: número escalar (un solo número que resume el error total)
    """
    m = Y_one_hot.shape[1]
    
    # np.clip evita log(0) que sería -infinito
    # Limitamos A2 entre 1e-10 y 1 (casi 0, pero no exactamente 0)
    log_predicciones = np.log(np.clip(A2, 1e-10, 1.0))
    
    # Multiplicamos elemento a elemento y sumamos todo
    costo = -(1.0 / m) * np.sum(Y_one_hot * log_predicciones)
    
    return costo


# =====================================================================
# PASO 6: BACKWARD PROPAGATION (Propagación hacia atrás)
# =====================================================================
# Aquí calculamos los GRADIENTES: ¿cuánto y en qué dirección debemos
# modificar cada parámetro para reducir el error?
#
# Usamos la REGLA DE LA CADENA del cálculo diferencial.
# La derivación completa es larga, pero los resultados son elegantes.

def backward_propagation(X, Y_one_hot, cache, parametros):
    """
    Calcula los gradientes de todos los parámetros.
    
    Parámetros:
        X: (784, m)        - Datos de entrada
        Y_one_hot: (10, m) - Etiquetas en one-hot
        cache: diccionario con Z1, A1, Z2, A2 del forward
        parametros: diccionario con W1, b1, W2, b2
    
    Retorna:
        gradientes: diccionario con dW1, db1, dW2, db2
    """
    m = X.shape[1]  # Número de imágenes
    
    W2 = parametros['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    
    # =========================================
    # Gradientes de la CAPA DE SALIDA (capa 2)
    # =========================================
    
    # dZ2 = A2 - Y (derivada de softmax + cross-entropy combinadas)
    # Dimensiones: (10, m)
    # Esto es MÁGICO: la derivada de la combinación softmax + cross-entropy
    # se simplifica a simplemente la diferencia entre predicción y realidad.
    # Si la red predijo [0.1, 0.8, 0.1] y la respuesta era [0, 1, 0],
    # entonces dZ2 = [0.1, -0.2, 0.1] → necesita SUBIR la neurona 1 y BAJAR las demás.
    dZ2 = A2 - Y_one_hot
    
    # dW2 = (1/m) · dZ2 · A1^T
    # Dimensiones: (10, m) · (m, 30) = (10, 30) ← misma forma que W2
    # Cada elemento dW2[i,j] dice: "¿cuánto afecta el peso W2[i,j] al error total?"
    dW2 = (1.0 / m) * dZ2.dot(A1.T)
    
    # db2 = (1/m) · suma de dZ2 por filas
    # Dimensiones: (10, 1) ← misma forma que b2
    # keepdims=True mantiene la forma como vector columna
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # =========================================
    # Gradientes de la CAPA OCULTA (capa 1)
    # =========================================
    
    # Primero propagamos el error hacia atrás a través de W2:
    # W2^T · dZ2 tiene dimensiones (30, 10) · (10, m) = (30, m)
    # Luego multiplicamos elemento a elemento (*) por la derivada de la sigmoide
    # sigmoide_derivada(A1) = A1 * (1 - A1), dimensiones (30, m)
    dZ1 = W2.T.dot(dZ2) * sigmoide_derivada(A1)
    
    # dW1 = (1/m) · dZ1 · X^T
    # Dimensiones: (30, m) · (m, 784) = (30, 784) ← misma forma que W1
    dW1 = (1.0 / m) * dZ1.dot(X.T)
    
    # db1 = (1/m) · suma de dZ1 por filas
    # Dimensiones: (30, 1) ← misma forma que b1
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    gradientes = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return gradientes


# =====================================================================
# PASO 7: ACTUALIZACIÓN DE PARÁMETROS
# =====================================================================
# Aquí aplicamos el gradiente descendente:
#     parámetro_nuevo = parámetro_actual - α · gradiente
#
# El signo MENOS es porque el gradiente apunta hacia donde el error
# CRECE, y nosotros queremos ir hacia donde DISMINUYE.
#
# α (learning rate) controla "qué tan grande es el paso".

def actualizar_parametros(parametros, gradientes, learning_rate):
    """
    Actualiza los parámetros usando gradiente descendente.
    
    Parámetros:
        parametros: diccionario con W1, b1, W2, b2
        gradientes: diccionario con dW1, db1, dW2, db2
        learning_rate: tasa de aprendizaje (α)
    
    Retorna:
        parametros: diccionario actualizado
    """
    parametros['W1'] = parametros['W1'] - learning_rate * gradientes['dW1']
    parametros['b1'] = parametros['b1'] - learning_rate * gradientes['db1']
    parametros['W2'] = parametros['W2'] - learning_rate * gradientes['dW2']
    parametros['b2'] = parametros['b2'] - learning_rate * gradientes['db2']
    
    return parametros


# =====================================================================
# PASO 8: CALCULAR PRECISIÓN (ACCURACY)
# =====================================================================

def calcular_precision(X, Y, parametros):
    """
    Calcula el porcentaje de predicciones correctas.
    
    Proceso:
        1. Hacemos forward propagation para obtener las predicciones.
        2. Para cada imagen, la predicción es el dígito con mayor probabilidad.
        3. Comparamos con la etiqueta real y contamos los aciertos.
    
    Parámetros:
        X: (784, m) - Imágenes
        Y: (m,)     - Etiquetas reales (números 0-9)
        parametros: diccionario con W1, b1, W2, b2
    
    Retorna:
        precision: número entre 0 y 1 (0.95 = 95% de acierto)
    """
    cache = forward_propagation(X, parametros)
    A2 = cache['A2']  # (10, m) - Probabilidades predichas
    
    # np.argmax(A2, axis=0) devuelve el índice del valor máximo por columna
    # Es decir, para cada imagen, ¿cuál dígito tiene la probabilidad más alta?
    predicciones = np.argmax(A2, axis=0)  # Forma: (m,)
    
    # Comparamos con las etiquetas reales
    aciertos = np.sum(predicciones == Y)
    precision = aciertos / Y.shape[0]
    
    return precision


# =====================================================================
# PASO 9: ENTRENAMIENTO DE UNA PARTICIÓN (Mini-red neuronal)
# =====================================================================
# Esta función realiza el ciclo completo de entrenamiento:
# Forward → Costo → Backward → Actualización, repetido por cada época.

def entrenar_particion(X, Y, parametros, learning_rate, epocas, id_particion):
    """
    Entrena una mini-red neuronal con los datos de una partición.
    
    Parámetros:
        X: (784, m) - Imágenes de esta partición
        Y: (m,)     - Etiquetas de esta partición
        parametros: diccionario con W1, b1, W2, b2 (valores iniciales)
        learning_rate: tasa de aprendizaje
        epocas: número de épocas
        id_particion: número de la partición (para los mensajes)
    
    Retorna:
        parametros: parámetros entrenados
        historial: lista de precisiones por época (para graficar)
    """
    Y_one_hot = one_hot(Y)
    historial_precision = []
    historial_costo = []
    
    print(f"\n  --- Entrenando Partición {id_particion} ({X.shape[1]} imágenes) ---")
    
    for epoca in range(1, epocas + 1):
        # 1. Forward propagation: obtener predicciones
        cache = forward_propagation(X, parametros)
        
        # 2. Calcular el costo (qué tan equivocada está la red)
        costo = calcular_costo(cache['A2'], Y_one_hot)
        
        # 3. Backward propagation: calcular gradientes
        gradientes = backward_propagation(X, Y_one_hot, cache, parametros)
        
        # 4. Actualizar parámetros con gradiente descendente
        parametros = actualizar_parametros(parametros, gradientes, learning_rate)
        
        # 5. Calcular precisión para monitorear el progreso
        precision = calcular_precision(X, Y, parametros)
        historial_precision.append(precision)
        historial_costo.append(costo)
        
        print(f"    Época {epoca}/{epocas} | "
              f"Costo: {costo:.4f} | "
              f"Precisión (train): {precision * 100:.2f}%")
    
    return parametros, historial_precision, historial_costo


# =====================================================================
# PASO 10: ALGORITMO DE ARNOVI
# =====================================================================
# Promedia los parámetros de todas las particiones.
# 
# W_promedio = (1/n) · (W_1 + W_2 + ... + W_n)
#
# Esto es posible porque todos los parámetros tienen las mismas dimensiones
# (independientemente de cuántos datos usó cada partición para entrenar).

def algoritmo_de_arnovi(lista_parametros):
    """
    Promedia los parámetros de múltiples mini-redes neuronales.
    
    Parámetros:
        lista_parametros: lista de diccionarios, cada uno con W1, b1, W2, b2
    
    Retorna:
        parametros_promedio: diccionario con los parámetros promediados
    """
    n = len(lista_parametros)
    print(f"\n{'='*60}")
    print(f"ALGORITMO DE ARNOVI: Promediando {n} conjuntos de parámetros")
    print(f"{'='*60}")
    
    parametros_promedio = {}
    
    for clave in ['W1', 'b1', 'W2', 'b2']:
        # Sumamos el parámetro de todas las particiones
        suma = np.zeros_like(lista_parametros[0][clave])
        for i in range(n):
            suma += lista_parametros[i][clave]
        
        # Multiplicamos por (1/n) para obtener el promedio
        parametros_promedio[clave] = suma * (1.0 / n)
        
        print(f"  {clave}: promedio de {n} matrices de forma {parametros_promedio[clave].shape}")
    
    print(f"  ¡Parámetros promediados exitosamente!")
    return parametros_promedio


# =====================================================================
# PASO 11: UNA EJECUCIÓN COMPLETA (entrenar + evaluar)
# =====================================================================
# Separamos la lógica de "una ejecución completa" en su propia función
# para poder llamarla múltiples veces en el ciclo de 10 repeticiones.
#
# ¿Por qué recibe X_train, Y_train, X_test, Y_test como parámetros?
#   Porque cargar MNIST tarda varios segundos. Si lo cargamos UNA vez
#   y lo pasamos a cada ejecución, ahorramos mucho tiempo en las 10 reps.

def ejecutar_una_vez(X_train, Y_train, X_test, Y_test, numero_ejecucion):
    """
    Ejecuta un ciclo completo: particionar → entrenar → Arnovi → evaluar.
    
    Cada ejecución tiene inicialización y mezcla de datos DIFERENTES
    (porque no fijamos semilla), lo que produce resultados distintos.
    
    Parámetros:
        X_train, Y_train: datos de entrenamiento
        X_test, Y_test:   datos de prueba
        numero_ejecucion: int - para identificar la ejecución en los logs
    
    Retorna:
        precision_test:         float - precisión final en test
        todos_historiales_precision: list - historiales de cada partición
        todos_historiales_costo:     list - historiales de costo
    """
    print(f"\n{'#'*60}")
    print(f"# EJECUCIÓN {numero_ejecucion}")
    print(f"{'#'*60}")
    
    # ---- Crear particiones (mezcla aleatoria diferente cada vez) ----
    particiones = crear_particiones(X_train, Y_train, NUM_PARTICIONES)
    
    # ---- Inicialización (parámetros aleatorios diferentes cada vez) ----
    parametros_iniciales = inicializar_parametros()
    
    # ---- Entrenar cada partición ----
    lista_parametros_entrenados = []
    todos_historiales_precision = []
    todos_historiales_costo = []
    
    for i, (Xi, Yi) in enumerate(particiones):
        params_copia = {k: v.copy() for k, v in parametros_iniciales.items()}
        
        params_entrenados, hist_precision, hist_costo = entrenar_particion(
            Xi, Yi, params_copia, LEARNING_RATE, EPOCAS, i + 1
        )
        
        lista_parametros_entrenados.append(params_entrenados)
        todos_historiales_precision.append(hist_precision)
        todos_historiales_costo.append(hist_costo)
    
    # ---- Algoritmo de Arnovi ----
    if NUM_PARTICIONES > 1:
        parametros_finales = algoritmo_de_arnovi(lista_parametros_entrenados)
    else:
        parametros_finales = lista_parametros_entrenados[0]
    
    # ---- Evaluación en test ----
    precision_test = calcular_precision(X_test, Y_test, parametros_finales)
    print(f"\n  Ejecución {numero_ejecucion} → Precisión test: {precision_test * 100:.2f}%")
    
    return precision_test, todos_historiales_precision, todos_historiales_costo


# =====================================================================
# PASO 12: PROGRAMA PRINCIPAL CON 10 REPETICIONES + ESTADÍSTICAS
# =====================================================================
# 
# ¿POR QUÉ 10 REPETICIONES?
#   Una sola ejecución puede dar un resultado "afortunado" o "desafortunado"
#   dependiendo de la inicialización aleatoria y la mezcla de datos.
#   Al ejecutar 10 veces y promediar, obtenemos una estimación CONFIABLE
#   del rendimiento real del método.
#
# ¿QUÉ ES LA DESVIACIÓN ESTÁNDAR RELATIVA (RSD)?
#   También llamada "coeficiente de variación" (CV).
#   
#   RSD = (desviación_estándar / promedio) × 100%
#
#   Mide qué tan DISPERSOS están los resultados respecto al promedio,
#   expresado como porcentaje.
#
#   Ejemplo:
#     Si las 10 precisiones son: [90%, 91%, 89%, 90%, 92%, 88%, 91%, 90%, 89%, 90%]
#     Promedio = 90%
#     Desviación estándar ≈ 1.05%
#     RSD = (1.05 / 90) × 100 = 1.17%
#     → Los resultados varían muy poco (buena estabilidad).
#
#   Si RSD < 5%, se considera que el método es ESTABLE.
#   Si RSD > 10%, hay mucha variabilidad → el método es sensible a
#   la inicialización o a la partición de datos.


def main():
    print("=" * 60)
    print("RED NEURONAL DESDE CERO - ALGORITMO DE ARNOVI")
    print("=" * 60)
    print(f"Arquitectura: {NEURONAS_ENTRADA} → {NEURONAS_OCULTA} → {NEURONAS_SALIDA}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Épocas: {EPOCAS}")
    print(f"Particiones: {NUM_PARTICIONES}")
    print(f"Repeticiones: {NUM_REPETICIONES}")
    print("=" * 60)
    
    # ---- Paso 1: Cargar datos UNA sola vez ----
    print("\n[1] Cargando dataset MNIST...")
    X_train, Y_train, X_test, Y_test = cargar_mnist()
    print(f"    Entrenamiento: {X_train.shape[1]} imágenes")
    print(f"    Prueba: {X_test.shape[1]} imágenes")
    
    # ---- Paso 2: Ejecutar 10 veces y recolectar resultados ----
    print(f"\n[2] Ejecutando {NUM_REPETICIONES} repeticiones completas...")
    
    todas_las_precisiones = []  # Guardamos la precisión test de cada ejecución
    ultimo_historial_precision = None  # Guardamos el último para graficar
    ultimo_historial_costo = None
    
    for rep in range(1, NUM_REPETICIONES + 1):
        precision, hist_prec, hist_costo = ejecutar_una_vez(
            X_train, Y_train, X_test, Y_test, rep
        )
        todas_las_precisiones.append(precision)
        ultimo_historial_precision = hist_prec
        ultimo_historial_costo = hist_costo
    
    # ---- Paso 3: Cálculo de estadísticas ----
    # Convertimos a array de NumPy para usar las funciones estadísticas
    precisiones = np.array(todas_las_precisiones)
    
    # Promedio (media aritmética)
    # promedio = (x1 + x2 + ... + xn) / n
    promedio = np.mean(precisiones)
    
    # Desviación estándar (mide la dispersión de los datos)
    # std = sqrt( (1/n) × Σ(xi - promedio)² )
    # ddof=0 significa que dividimos entre n (desviación estándar poblacional).
    # ddof=1 dividiría entre (n-1) (muestral). Con n=10, la diferencia es mínima.
    desviacion_estandar = np.std(precisiones, ddof=0)
    
    # Desviación estándar relativa (RSD)
    # RSD = (desviación_estándar / promedio) × 100
    # Si el promedio fuera 0 (imposible aquí pero por seguridad), evitamos dividir entre 0
    if promedio > 0:
        rsd = (desviacion_estandar / promedio) * 100
    else:
        rsd = 0.0
    
    # ---- Paso 4: Mostrar resultados ----
    print(f"\n{'='*60}")
    print(f"RESULTADOS DE {NUM_REPETICIONES} REPETICIONES")
    print(f"{'='*60}")
    print(f"\n  Precisiones individuales en test:")
    for i, prec in enumerate(todas_las_precisiones):
        print(f"    Ejecución {i+1:2d}: {prec * 100:.2f}%")
    
    print(f"\n  {'─'*40}")
    print(f"  Promedio:                    {promedio * 100:.2f}%")
    print(f"  Desviación estándar:         {desviacion_estandar * 100:.2f}%")
    print(f"  Desviación estándar relativa (RSD): {rsd:.2f}%")
    print(f"  Mínimo:                      {np.min(precisiones) * 100:.2f}%")
    print(f"  Máximo:                      {np.max(precisiones) * 100:.2f}%")
    print(f"  Rango:                       {(np.max(precisiones) - np.min(precisiones)) * 100:.2f}%")
    print(f"  {'─'*40}")
    
    # Interpretación de la RSD
    if rsd < 2:
        interpretacion = "MUY ESTABLE - variabilidad mínima entre ejecuciones"
    elif rsd < 5:
        interpretacion = "ESTABLE - variabilidad aceptable"
    elif rsd < 10:
        interpretacion = "MODERADO - variabilidad notable"
    else:
        interpretacion = "INESTABLE - alta variabilidad, considerar más épocas"
    print(f"  Interpretación RSD: {interpretacion}")
    print(f"{'='*60}")
    
    # ---- Paso 5: Gráficas ----
    print(f"\n[5] Generando gráficas...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Gráfica 1: Precisión por época (última ejecución como ejemplo)
    for i, hist in enumerate(ultimo_historial_precision):
        axes[0].plot(range(1, EPOCAS + 1), [p * 100 for p in hist], 
                     marker='o', label=f'Partición {i+1}')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Precisión (%)')
    axes[0].set_title(f'Precisión por Época (Última ejecución)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)
    
    # Gráfica 2: Costo por época (última ejecución como ejemplo)
    for i, hist in enumerate(ultimo_historial_costo):
        axes[1].plot(range(1, EPOCAS + 1), hist, 
                     marker='o', label=f'Partición {i+1}')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Costo (Cross-Entropy)')
    axes[1].set_title(f'Costo por Época (Última ejecución)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Gráfica 3: Precisión test por ejecución (las 10 repeticiones)
    ejecuciones = range(1, NUM_REPETICIONES + 1)
    precisiones_pct = [p * 100 for p in todas_las_precisiones]
    axes[2].bar(ejecuciones, precisiones_pct, color='#2196F3', alpha=0.7, edgecolor='black')
    axes[2].axhline(y=promedio * 100, color='red', linestyle='--', linewidth=2, 
                    label=f'Promedio: {promedio * 100:.2f}%')
    axes[2].axhline(y=(promedio + desviacion_estandar) * 100, color='orange', 
                    linestyle=':', linewidth=1, label=f'±1 Desv. Est.')
    axes[2].axhline(y=(promedio - desviacion_estandar) * 100, color='orange', 
                    linestyle=':', linewidth=1)
    axes[2].set_xlabel('Ejecución')
    axes[2].set_ylabel('Precisión Test (%)')
    axes[2].set_title(f'Precisión por Ejecución (RSD: {rsd:.2f}%)')
    axes[2].set_xticks(list(ejecuciones))
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(
        f'Algoritmo de Arnovi — {NUM_PARTICIONES} particiones, '
        f'LR={LEARNING_RATE}, {EPOCAS} épocas, {NUM_REPETICIONES} repeticiones\n'
        f'Promedio: {promedio * 100:.2f}% ± {desviacion_estandar * 100:.2f}% '
        f'(RSD: {rsd:.2f}%)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('grafica_entrenamiento.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gráfica guardada como 'grafica_entrenamiento.png'")
    
    # ---- Resumen final compacto ----
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Particiones:     {NUM_PARTICIONES}")
    print(f"  Imágenes/part.:  {X_train.shape[1] // NUM_PARTICIONES}")
    print(f"  Épocas:          {EPOCAS}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  Repeticiones:    {NUM_REPETICIONES}")
    print(f"  PROMEDIO TEST:   {promedio * 100:.2f}%")
    print(f"  DESV. ESTÁNDAR:  {desviacion_estandar * 100:.2f}%")
    print(f"  RSD:             {rsd:.2f}%")
    print(f"{'='*60}")


# =====================================================================
# EJECUTAR
# =====================================================================
if __name__ == "__main__":
    main()