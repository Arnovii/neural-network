import math
import random
import gzip
import struct
import urllib.request
import os

# ============================================================================
# PARTE 1: FUNCIONES MATEMÁTICAS BÁSICAS (Sin NumPy)
# ============================================================================


def sigmoid(z):
    """
    Función sigmoide: σ(z) = 1 / (1 + e^(-z))
    Transforma cualquier número a un valor entre 0 y 1
    """
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(a):
    """
    Derivada de sigmoide: σ'(z) = σ(z) * (1 - σ(z))
    Como ya tenemos 'a' (que es σ(z)), usamos: a * (1 - a)
    """
    return a * (1.0 - a)


def softmax(z_list):
    """
    Softmax: convierte logits a probabilidades que suman 1
    """
    max_z = max(z_list)
    exp_z = [math.exp(z - max_z) for z in z_list]
    sum_exp = sum(exp_z)
    return [e / sum_exp for e in exp_z]


# ============================================================================
# PARTE 2: ÁLGEBRA LINEAL DESDE CERO
# ============================================================================


def matriz_por_vector(matriz, vector):
    """Multiplicación matriz × vector"""
    resultado = []
    for fila in matriz:
        suma = 0.0
        for i in range(len(fila)):
            suma += fila[i] * vector[i]
        resultado.append(suma)
    return resultado


def vector_mas_vector(v1, v2):
    """Suma elemento a elemento"""
    return [v1[i] + v2[i] for i in range(len(v1))]


def vector_menos_vector(v1, v2):
    """Resta elemento a elemento"""
    return [v1[i] - v2[i] for i in range(len(v1))]


def vector_por_escalar(vector, escalar):
    """Multiplica cada elemento por un número"""
    return [v * escalar for v in vector]


def producto_externo(vector_col, vector_fila):
    """Producto externo: crea una matriz"""
    resultado = []
    for vc in vector_col:
        fila = [vc * vf for vf in vector_fila]
        resultado.append(fila)
    return resultado


def transpuesta(matriz):
    """Convierte filas en columnas"""
    if not matriz:
        return []
    filas = len(matriz)
    cols = len(matriz[0])
    return [[matriz[i][j] for i in range(filas)] for j in range(cols)]


# ============================================================================
# PARTE 3: CARGA DE DATOS MNIST (ACTUALIZADO - Fuentes alternativas)
# ============================================================================


def descargar_mnist():
    """
    Descarga MNIST desde mirrors alternativos confiables.
    Fuentes: AWS Open Data (mirror oficial) o GitHub (fallback)
    """
    # Mirrors alternativos confiables (el original de LeCun está caído)
    urls = {
        "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    }

    # Fallback a GitHub si AWS falla
    urls_fallback = {
        "train-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
    }

    for archivo, url in urls.items():
        if not os.path.exists(archivo):
            print(f"Descargando {archivo}...")
            try:
                urllib.request.urlretrieve(url, archivo)
                print(f"✓ {archivo} descargado desde AWS")
            except Exception as e:
                print(f"  Error en AWS, intentando GitHub...")
                try:
                    urllib.request.urlretrieve(urls_fallback[archivo], archivo)
                    print(f"✓ {archivo} descargado desde GitHub")
                except Exception as e2:
                    print(f"  ERROR: No se pudo descargar {archivo}")
                    print(f"  AWS error: {e}")
                    print(f"  GitHub error: {e2}")
                    raise Exception(
                        "No se pudieron descargar los datos. Verifica tu conexión."
                    )


def cargar_imagenes(archivo_gz):
    """Carga imágenes de MNIST desde archivo .gz (formato IDX)"""
    with gzip.open(archivo_gz, "rb") as f:
        # Leer header: magic number (4 bytes), num_imagenes (4 bytes), filas (4 bytes), columnas (4 bytes)
        magic, num, filas, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(
                f"Magic number incorrecto: esperado 2051, obtenido {magic}"
            )

        # Leer todos los píxeles (1 byte por píxel, valores 0-255)
        buffer = f.read()
        imagenes = []

        # Cada imagen tiene filas * cols píxeles
        for i in range(num):
            inicio = i * filas * cols
            fin = inicio + filas * cols
            # Normalizamos a [0, 1] dividiendo por 255.0
            img = [b / 255.0 for b in buffer[inicio:fin]]
            imagenes.append(img)

        return imagenes


def cargar_etiquetas(archivo_gz):
    """Carga etiquetas de MNIST desde archivo .gz (formato IDX)"""
    with gzip.open(archivo_gz, "rb") as f:
        # Leer header: magic number (4 bytes), num_etiquetas (4 bytes)
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(
                f"Magic number incorrecto: esperado 2049, obtenido {magic}"
            )

        # Leer etiquetas (1 byte por etiqueta, valores 0-9)
        buffer = f.read()
        return [b for b in buffer]


# ============================================================================
# PARTE 4: LA RED NEURONAL (IGUAL QUE ANTES)
# ============================================================================


class RedNeuronal:
    def __init__(self, tam_entrada, tam_oculta, tam_salida):
        """
        Inicializa la red con pesos aleatorios pequeños (inicialización Xavier)
        """
        self.tam_entrada = tam_entrada
        self.tam_oculta = tam_oculta
        self.tam_salida = tam_salida

        # Pesos capa oculta: matriz de tam_oculta × tam_entrada
        self.W1 = [
            [
                random.gauss(0, math.sqrt(2.0 / (tam_entrada + tam_oculta)))
                for _ in range(tam_entrada)
            ]
            for _ in range(tam_oculta)
        ]
        self.b1 = [0.0] * tam_oculta

        # Pesos capa salida: matriz de tam_salida × tam_oculta
        self.W2 = [
            [
                random.gauss(0, math.sqrt(2.0 / (tam_oculta + tam_salida)))
                for _ in range(tam_oculta)
            ]
            for _ in range(tam_salida)
        ]
        self.b2 = [0.0] * tam_salida

    def propagacion_adelante(self, x):
        """
        Paso hacia adelante: calcula la salida de la red
        """
        # CAPA OCULTA: z1 = W1·x + b1, a1 = σ(z1)
        z1 = matriz_por_vector(self.W1, x)
        z1 = vector_mas_vector(z1, self.b1)
        a1 = [sigmoid(z) for z in z1]

        # CAPA DE SALIDA: z2 = W2·a1 + b2, a2 = softmax(z2)
        z2 = matriz_por_vector(self.W2, a1)
        z2 = vector_mas_vector(z2, self.b2)
        a2 = softmax(z2)

        return a2, (a1, z1, x)

    def backpropagation(self, a2, y, cache):
        """
        Retropropagación: calcula gradientes del error
        """
        a1, z1, x_input = cache

        # Vector one-hot para y
        y_onehot = [0.0] * self.tam_salida
        y_onehot[y] = 1.0

        # ERROR EN CAPA DE SALIDA: δ2 = a2 - y
        delta2 = vector_menos_vector(a2, y_onehot)

        # GRADIENTES CAPA DE SALIDA
        grad_W2 = producto_externo(delta2, a1)
        grad_b2 = delta2[:]

        # ERROR EN CAPA OCULTA: δ1 = (W2^T · δ2) ⊙ σ'(z1)
        W2_T = transpuesta(self.W2)
        delta1 = matriz_por_vector(W2_T, delta2)
        sig_prime = [sigmoid_derivative(a) for a in a1]
        delta1 = [delta1[i] * sig_prime[i] for i in range(len(delta1))]

        # GRADIENTES CAPA OCULTA
        grad_W1 = producto_externo(delta1, x_input)
        grad_b1 = delta1[:]

        return grad_W1, grad_b1, grad_W2, grad_b2

    def entrenar(self, X, Y, epochs, tasa_aprendizaje):
        """
        Entrenamiento por gradiente descendente (batch completo)
        """
        n = len(X)

        for epoch in range(epochs):
            print(f"\nÉpoca {epoch + 1}/{epochs}")

            for i in range(n):
                # Forward
                a2, cache = self.propagacion_adelante(X[i])

                # Backward
                gW1, gb1, gW2, gb2 = self.backpropagation(a2, Y[i], cache)

                # Actualiza parámetros
                self.actualizar_pesos(gW1, gb1, gW2, gb2, tasa_aprendizaje)

                if (i + 1) % 1000 == 0:
                    print(f"  Procesadas {i + 1}/{n} imágenes...")

            # Evaluamos precisión en entrenamiento
            correctos = sum(1 for i in range(n) if self.predecir(X[i]) == Y[i])
            precision = 100.0 * correctos / n
            print(f"  Precisión entrenamiento: {precision:.2f}%")

    def predecir(self, x):
        """Predice la clase (dígito) para una entrada x"""
        a2, _ = self.propagacion_adelante(x)
        return max(range(len(a2)), key=lambda i: a2[i])

    def actualizar_pesos(self, gW1, gb1, gW2, gb2, tasa_aprendizaje):
        """
        Actualiza pesos y bias usando los gradientes de UN solo ejemplo (SGD)
        """

        # Capa oculta
        for j in range(self.tam_oculta):
            for k in range(self.tam_entrada):
                self.W1[j][k] -= tasa_aprendizaje * gW1[j][k]
            self.b1[j] -= tasa_aprendizaje * gb1[j]

        # Capa salida
        for j in range(self.tam_salida):
            for k in range(self.tam_oculta):
                self.W2[j][k] -= tasa_aprendizaje * gW2[j][k]
            self.b2[j] -= tasa_aprendizaje * gb2[j]


# ============================================================================
# PARTE 5: EJECUCIÓN PRINCIPAL
# ============================================================================


def main():
    print("=" * 60)
    print("RED NEURONAL PARA MNIST - DESDE CERO")
    print("Sin NumPy, sin frameworks, solo Python puro")
    print("=" * 60)

    # 1. Descargar datos desde mirrors alternativos
    print("\n1. Descargando datos MNIST desde mirrors alternativos...")
    print("   (AWS Open Data y GitHub como respaldo)")
    try:
        descargar_mnist()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nAlternativa manual:")
        print("1. Ve a https://www.kaggle.com/datasets/hojjatk/mnist-dataset")
        print("2. Descarga los 4 archivos .gz")
        print("3. Colócalos en esta carpeta y vuelve a ejecutar")
        return

    # 2. Cargar datos
    print("\n2. Cargando datos...")
    print("  Cargando imágenes de entrenamiento...")
    X_train = cargar_imagenes("train-images-idx3-ubyte.gz")
    print("  Cargando etiquetas de entrenamiento...")
    Y_train = cargar_etiquetas("train-labels-idx1-ubyte.gz")
    print("  Cargando imágenes de prueba...")
    X_test = cargar_imagenes("t10k-images-idx3-ubyte.gz")
    print("  Cargando etiquetas de prueba...")
    Y_test = cargar_etiquetas("t10k-labels-idx1-ubyte.gz")

    print(f"\n  Datos cargados exitosamente:")
    print(f"  - Entrenamiento: {len(X_train)} imágenes")
    print(f"  - Prueba: {len(X_test)} imágenes")
    print(f"  - Tamaño imagen: {len(X_train[0])} píxeles (28×28)")

    # 3. Crear red neuronal
    print("\n3. Creando red neuronal...")
    print("  Arquitectura: 784 → 30 → 10")
    print("  - Capa entrada: 784 neuronas (28×28 píxeles)")
    print("  - Capa oculta: 30 neuronas (sigmoide)")
    print("  - Capa salida: 10 neuronas (softmax)")

    red = RedNeuronal(tam_entrada=784, tam_oculta=30, tam_salida=10)

    # 4. Entrenar
    print("\n4. Entrenando red...")
    print("  NOTA: Usando 5000 ejemplos para velocidad inicial.")
    print("  Para mejor precisión, aumenta n_entrenamiento.")

    n_entrenamiento = 5000  # Cambia a len(X_train) para entrenamiento completo
    X_subset = X_train[:n_entrenamiento]
    Y_subset = Y_train[:n_entrenamiento]

    red.entrenar(X=X_subset, Y=Y_subset, epochs=10, tasa_aprendizaje=0.1)

    # 5. Evaluar en test
    print("\n5. Evaluando en conjunto de prueba...")
    correctos = 0
    for i in range(len(X_test)):
        if red.predecir(X_test[i]) == Y_test[i]:
            correctos += 1

    precision_test = 100.0 * correctos / len(X_test)
    print(f"\n  Precisión en test: {precision_test:.2f}%")
    print(f"  ({correctos}/{len(X_test)} imágenes correctas)")

    # 6. Demostración
    print("\n6. Demostración de predicciones:")
    for i in range(5):
        pred = red.predecir(X_test[i])
        real = Y_test[i]
        print(
            f"  Imagen {i}: Predicción={pred}, Real={real}, {'✓' if pred == real else '✗'}"
        )


if __name__ == "__main__":
    main()
