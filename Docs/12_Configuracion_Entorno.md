# Configuración del Entorno

Guía completa para configurar el entorno de desarrollo y ejecución del sistema de entrenamiento distribuidos.

---

## Requisitos Previos

### Software necesario

| Software | Versión mínima | Propósito |
|----------|---------------|-----------|
| Python | 3.13.5 | Intérprete de Python |
| pip/uv | Latest | Gestor de paquetes |

### Instalación de dependencias

```bash
# Clonar el repositorio
git clone https://github.com/tu_usuario/neural-network.git
cd neural-network

# Instalar dependencias
pip install -r requirements.txt

# O si usas uv
uv pip install -r requirements.txt
```

---

## Configuración de HuggingFace

### Paso 1: Obtener token de HuggingFace

1. Ve a [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
2. Crea un nuevo token (si no tienes uno)
3. Copia el token (Formato: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

### Paso 2: Aceptar licencia de ImageNet-1k

1. Ve a [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
2. Acepta la licencia si es necesario
3. Confirma tu acceso al dataset

### Paso 3: Configurar el token

Tienes tres opciones:

#### Opción A: Variable de entorno (bash)

```bash
# bash/zsh
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# PowerShell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows CMD
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### Opción B: Archivo .env (Recomendado)

```bash
# Crear archivo .env en la raíz del proyecto
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > .env
```

El archivo `.env` se carga automáticamente al ejecutar `ps_imagenet.py` o `ps_gui_imagenet.py`.

#### Opción C: Argumento CLI

```bash
python ps_imagenet.py --hf-token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python ps_gui_imagenet.py --hf-token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Prioridad de configuración

```
CLI (--hf-token) > .env > HF_TOKEN (variable de entorno)
```

El argumento CLI tiene la máxima prioridad, seguido por el archivo `.env`, y finalmente la variable de entorno.

---

## Inicio Rápido

### 1. Iniciar el Parameter Server (GUI)

```bash
python ps_gui_imagenet.py
```

Llena los campos:
- **HF Token**: Tu token de HuggingFace
- **Host**: `0.0.0.0` (para accepting remote connections)
- **Puerto**: `9999`

### 2. Iniciar Workers

En la misma máquina:
```bash
python worker_imagenet.py --server-host 127.0.0.1
```

En máquinas remotas:
```bash
python worker_imagenet.py --server-host <IP_DEL_PS>
```

Donde `<IP_DEL_PS>` es la IP que aparece en el log del PS:
```
Worker host: 192.168.1.100  (usar como --server-host en workers)
```

---

## Configuración Avanzada

### Parámetros del Parameter Server

| Parámetro | Default | Descripción |
|----------|----------|------------|
| `--host` | `0.0.0.0` | Host del PS |
| `--port` | `9999` | Puerto TCP |
| `--lr` | `0.01` | Learning rate MLP |
| `--lr-cnn` | `0.001` | Learning rate CNN (E2E) |
| `--staleness-lambda` | `0.1` | Factor de corrección staleness |
| `--hidden1` | `1024` | neuronas capa oculta 1 |
| `--hidden2` | `512` | neuronas capa oculta 2 |
| `--batch-size` | `64` | Batch size |
| `--image-size` | `224` | Resolución de imágenes |
| `--dataset` | `ILSVRC/imagenet-1k` | Dataset |
| `--seed` | `None` | Semilla RNG (None = aleatorio) |
| `--steps-per-report` | `500` | Steps entre reportes |
| `--max-steps` | `0` | Límite steps (0 = ilimitado) |
| `--export-dir` | `./Exports` | Directorio de resultados |
| `--hf-token` | `None` | Token HuggingFace |

### Parámetros del Worker

| Parámetro | Default | Descripción |
|----------|---------------|-------------|
| `--server-host` | `127.0.0.1` | IP del PS |
| `--server-port` | `9999` | Puerto del PS |
| `--device` | `auto` | cpu, cuda, cuda:N, mps |
| `--shuffle-buffer` | `1000` | Buffer de shuffle |
| `--prefetch` | `4` | Batches en prefetch |
| `--accum-steps` | `1` | Batches a acumular |

### Constantes en Utils/constants.py

```python
from Utils.constants import (
    # Rede
    DEFAULT_LR = 0.01,
    DEFAULT_LR_CNN = 0.001,
    HIDDEN1_DEFAULT = 1024,
    HIDDEN2_DEFAULT = 512,
    
    # Data
    DEFAULT_BATCH_SIZE = 64,
    IMAGE_SIZE = 224,
    HF_DATASET_DEFAULT = "ILSVRC/imagenet-1k",
    
    # Comunicación
    DEFAULT_HOST = "0.0.0.0",
    DEFAULT_PORT = 9999,
    
    # Regularización
    GRAD_CLIP_MAX_NORM = 10.0,
    LABEL_SMOOTHING = 0.1,
    WEIGHT_DECAY = 1e-4,
)
```

---

## Funciones de Utility

### get_worker_ip(host: str) -> str

Obtiene la IP que los Workers deben usar:

```python
from Utils.config_loader import get_worker_ip

ip = get_worker_ip("0.0.0.0")  # Retorna IP real de la máquina
ip = get_worker_ip("127.0.0.1")  # Retorna 127.0.0.1
```

### get_hf_token(override: str | None = None) -> str | None

Obtiene el token de HuggingFace:

```python
from Utils.config_loader import get_hf_token

token = get_hf_token()  # Auto-detecta
token = get_hf_token("hf_xxx")  # Con override
```

### load_dotenv(env_path: str | Path) -> None

Carga variables desde archivo .env:

```python
from Utils.config_loader import load_dotenv

load_dotenv()  # Carga .env por defecto
load_dotenv("./custom.env")  # Carga archivo específico
```

---

## Solución de Problemas

### Error: HF Token inválido

```
ValueError: Token must start with 'hf_'
```

**Solución**: Verifica que tu token comienza con `hf_` y es válido en [HuggingFace](https://huggingface.co/settings/tokens).

### Error: Acceso denegado al dataset

```
DatasetNotFoundError: ILSVRC/imagenet-1k
```

**Solución**: 
1. Acepta la licencia en https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. Verifica que tu token tiene acceso al dataset

### Worker no puede conectar

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solución**:
1. Verifica que el PS está corriendo
2. Usa la IP correcta (`--server-host`)
3. Verifica el firewall permite el puerto 9999

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solución**:
```bash
# Reducir batch size
python worker_imagenet.py --server-host 127.0.0.1 --batch-size 32

# Reducir prefetch
python worker_imagenet.py --server-host 127.0.0.1 --prefetch 2

# Usar CPU
python worker_imagenet.py --server-host 127.0.0.1 --device cpu
```

---

## Variables de Entorno

| Variable | Descripción |
|----------|------------|
| `HF_TOKEN` | Token de HuggingFace |
| `PYTHONPATH` | Ruta de Python (si es necesario) |

---

## Archivos de Configuración

### .gitignore

Asegúrate de que `.env` está ignorado:

```bash
# .gitignore
.env
```

### .env.example

Copia de seguridad para compartir:

```bash
# .env.example
# Copiar a .env y completar con tu token
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Referencias

- [Documentación técnica](Docs/03_Parameter_Server.md)
- [Worker](Docs/04_Worker.md)
- [Hiperparámetros](Docs/08_Hiperparametros_y_Config.md)
- [Resultados](Docs/11_Exportacion_Resultados.md)