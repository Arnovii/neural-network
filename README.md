# NN_practica - Red Neuronal Federada para MNIST

Proyecto de redes neuronales con implementación del algoritmo propuesto por Diego (promediado de parámetros por época) para el dataset MNIST.

## Estructura del Proyecto
NN_practica/
├── Analytics/ # Análisis estadístico y visualización
├── Networks/ # Red neuronal federada
├── Utils/ # Utilidades matemáticas y carga de datos
├── Data/ # Datos MNIST (descargados automáticamente)
├── results/ # Resultados de experimentos
├── main.py # Punto de entrada
└── requirements.txt # Dependencias

## Instalación
Con Pip:
```bash
pip install -r requirements.txt
```

Con UV:
```bash
uv sync
```

## Modo de Uso
Modo Términal
```bash
# Ejecutar 10 experimentos con 2 particiones y 5 épocas
python main.py --partitions 2 --epochs 5 --experiments 10

# Ver todas las opciones
python main.py --help
```

Modo Interactivo (Interfaz Gráfica)
```bash
python main.py --interactive
```