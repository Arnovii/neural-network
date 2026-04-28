# Especificaciones del Entorno de Ejecución y Stack Tecnológico

Esta sección resume el entorno real en el que ejecuta el proyecto y las piezas tecnológicas que forman su stack. El criterio aquí es descriptivo y operativo: qué se usa, para qué se usa y en qué versión aparece en el repositorio o en el entorno exportado.

---

## Entorno de Ejecución

### Intérprete y entorno

- **Python:** `3.13.5`
- **Tipo de entorno observado:** entorno virtual local (`.venv`)
- **Archivo de proyecto:** `pyproject.toml`
- **Exportación de dependencias:** `requirements.txt` generado con `uv export --format requirements-txt`

El proyecto declara compatibilidad con `requires-python = ">=3.13.5"`, y el entorno activo del workspace también reporta `3.13.5.final.0`. Esto significa que la base de ejecución real no es una versión genérica de Python, sino una versión concreta y moderna, usada tanto por el código como por la configuración del proyecto.

### Forma de ejecución

El sistema está organizado alrededor de tres puntos de entrada:

- `ps_imagenet.py` para ejecutar el Parameter Server en terminal.
- `ps_gui_imagenet.py` para ejecutar el Parameter Server con interfaz gráfica.
- `worker_imagenet.py` para ejecutar los Workers que entrenan de forma asíncrona.

La arquitectura completa se apoya en un esquema `Parameter Server + N Workers`, con comunicación TCP y configuración centralizada desde el PS.

---

## Stack Tecnológico Principal

### 1. Capa de lenguaje y runtime

El proyecto está escrito en **Python**, con uso intensivo de:

- `argparse` para CLI.
- `socket` para la comunicación TCP entre PS y Workers.
- `threading` para concurrencia ligera en el servidor, el streaming y el monitoreo.
- `queue`, `time`, `os`, `sys`, `json`, `pathlib` y `datetime` para soporte de infraestructura.

Esta parte no es incidental: el proyecto no depende de un framework web ni de un orquestador externo, sino de Python puro más bibliotecas científicas y de ML.

### 2. Capa de aprendizaje automático

#### `torch` 2.10.0

Es el núcleo del entrenamiento.

Se usa para:

- construir y ejecutar la CNN y la MLP;
- calcular gradientes con autograd;
- aplicar optimización `SGD`;
- mover tensores entre `cpu`, `cuda` y `mps`;
- hacer `forward`, `backward` y clipping de gradiente;
- cargar y sincronizar pesos entre PS y Workers.

En la práctica, PyTorch es la base de todo el ciclo de entrenamiento distribuido.

#### `torchvision` 0.25.0

Se utiliza en dos frentes:

- `torchvision.transforms.v2` para el pipeline de preprocesamiento y augmentación de imágenes.
- modelos/preprocesamiento vinculados a la extracción de características de imagen.

En el proyecto aparece como soporte directo del flujo visual, no como biblioteca auxiliar secundaria.

#### `numpy` 2.4.2

Se usa como capa intermedia de datos para:

- buffers de batches;
- conversión entre arrays y tensores;
- manipulación de datos serializados;
- soporte numérico para métricas y exportación.

### 3. Capa de datos y streaming

#### `datasets` 4.8.4

La biblioteca `datasets` de HuggingFace es la base del streaming del dataset.

Se usa con `load_dataset` para:

- cargar `ILSVRC/imagenet-1k` bajo demanda;
- evitar descargar el dataset completo;
- sostener el modo streaming del entrenamiento;
- habilitar sharding por Worker.

En el código también aparece una alternativa de dataset público para el mismo pipeline de streaming, pero el mecanismo real es el mismo: acceso incremental y no materialización completa en disco.

#### `Pillow` 12.1.1

Se usa como soporte de carga y decodificación de imágenes en el pipeline de streaming. Es la pieza que permite convertir la imagen descargada en un formato manipulable antes de pasarla a `torchvision`.

#### Dependencias transitivas visibles en el lock exportado

El archivo `requirements.txt` también refleja dependencias que llegan como soporte del stack de datos y red, entre ellas:

- `pyarrow` 23.0.1
- `pandas` 3.0.2
- `requests` 2.33.1
- `aiohttp` 3.13.5
- `httpx` 0.28.1
- `huggingface-hub` 1.8.0
- `fsspec` 2026.2.0
- `tqdm` 4.67.3
- `pyyaml` 6.0.3
- `xxhash` 3.6.0
- `dill` 0.4.1
- `multiprocess` 0.70.19

No todas forman parte del código propio, pero sí del entorno efectivo que resuelve el runtime de `datasets` y del ecosistema HuggingFace.

### 4. Capa de visualización y monitoreo

#### `matplotlib` 3.10.8

Se usa para:

- gráficas en la GUI del Parameter Server;
- exportación de resultados a PNG;
- visualización de loss, accuracy y número de workers;
- renderizado en modo headless con backend `Agg` para exportación;
- renderizado embebido en GUI con `FigureCanvasTkAgg`.

#### `tkinter` (stdlib de Python)

La interfaz gráfica del PS está construida sobre `tkinter` y `ttk`.

Esto significa que la GUI no depende de un framework externo pesado: el proyecto usa el toolkit de escritorio estándar incluido con Python.

#### `mplcursors` 0.7

Está declarado en el proyecto como dependencia de visualización interactiva, aunque no aparece como import directo en el código analizado del workspace. Conviene tratarlo como dependencia declarada del stack de visualización, no como una pieza central del runtime.

### 5. Capa de configuración y entorno

#### `python-dotenv` 1.2.2 / módulo `dotenv`

El proyecto carga variables desde `.env` mediante `load_dotenv`.

Se usa para resolver el token de HuggingFace sin incrustarlo en el código y con prioridad práctica entre:

1. argumento CLI,
2. variable de entorno,
3. archivo `.env`.

En el código, la importación efectiva es desde `dotenv`, por lo que la funcionalidad real depende de `python-dotenv`.

#### `psutil` 7.2.2

Está declarado en la configuración del proyecto, aunque no apareció como import directo en el código revisado. Si se utiliza en alguna variante local o futura, debe considerarse como dependencia auxiliar de observabilidad/sistema, no como parte del flujo principal de entrenamiento.

---

## Stack Funcional del Proyecto

### Componentes internos

- `Distributed/parameter_server.py`: coordina el entrenamiento, mantiene el estado global y distribuye configuración y parámetros.
- `Distributed/worker_node.py`: implementa el ciclo de conexión, streaming, entrenamiento y envío de actualizaciones.
- `Distributed/protocol.py`: define el protocolo de mensajes TCP entre PS y Workers.
- `Model/cnn_extractor.py`: encapsula la extracción de características visuales.
- `Model/mlp_pytorch.py`: implementa el clasificador MLP.
- `Utils/imagenet_streaming.py`: crea el pipeline de streaming, augmentación y sharding.
- `Utils/results_exporter.py`: exporta métricas, logs y gráficas del experimento.
- `Utils/logging_util.py`: centraliza el logging formateado del sistema.
- `Utils/config_loader.py`: resuelve tokens y variables del entorno.
- `Utils/constants.py`: concentra los valores por defecto y constantes compartidas.

### Servicios externos

- **HuggingFace Hub:** fuente del dataset y del streaming de ImageNet-1k.
- **Red TCP local o distribuida:** canal de comunicación entre procesos y máquinas.
- **Sistema de archivos local:** destino de exportaciones, logs y artefactos de experimentos.

---

## Dependencias Clave y Propósito

| Dependencia | Versión observada | Uso principal |
|---|---:|---|
| Python | 3.13.5 | Runtime base del proyecto |
| torch | 2.10.0 | Entrenamiento, autograd, optimización, device management |
| torchvision | 0.25.0 | Transforms y soporte de visión por computador |
| numpy | 2.4.2 | Manipulación de arrays y buffers |
| datasets | 4.8.4 | Streaming de HuggingFace |
| Pillow | 12.1.1 | Decodificación de imágenes |
| matplotlib | 3.10.8 | GUI, monitoreo y exportación de gráficas |
| python-dotenv | 1.2.2 | Carga de `.env` y variables sensibles |
| mplcursors | 0.7 | Dependencia de visualización interactiva declarada |
| psutil | 7.2.2 | Dependencia auxiliar declarada |

---

## Observaciones Operativas

- El Worker no define por CLI `dataset_name` ni `seed`; esos valores llegan por `CONFIG` desde el PS.
- El dataset se maneja en streaming, no como descarga completa local.
- La GUI y la exportación de resultados usan `matplotlib`, pero con distintos backends según el contexto.
- El proyecto no muestra una capa de CI, lint o typecheck configurada en el repositorio analizado; el control real del stack vive en `pyproject.toml`, `requirements.txt` y la estructura de los módulos.

---

## Resumen Corto

En términos prácticos, este proyecto se apoya en Python 3.13.5 como runtime, PyTorch y Torchvision como base de entrenamiento, HuggingFace Datasets como fuente de datos en streaming, Matplotlib y Tkinter para visualización, y `python-dotenv` para configuración sensible. Todo el sistema se organiza en torno a una arquitectura distribuida `Parameter Server + Workers`, con configuración centralizada y exportación de resultados desacoplada.