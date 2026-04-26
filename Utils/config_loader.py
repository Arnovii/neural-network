import os
from pathlib import Path
from typing import Optional


def load_dotenv(env_path: str | Path = ".env") -> None:
    """
    Carga variables de entorno desde archivo .env si existe.

    :param env_path: Ruta al archivo .env (default: .env en directorio actual)
    :type env_path: str | Path

    :note: No lanza errores si el archivo no existe.
    """
    from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]

    path = Path(env_path)
    if path.exists():
        load_dotenv(path, override=False)


def get_hf_token(override: str | None = None) -> Optional[str]:
    """
    Obtiene el token de HuggingFace con prioridad: override > CLI > .env > env.

    Proceso:
        1. Si override proporcionado, úsalo inmediatamente (máxima prioridad)
        2. Si HF_TOKEN existe en os.environ, úsalo
        3. Si no, intenta cargar desde .env y obtener HF_TOKEN
        4. Si fallback proporcionado en override, úsalo como último recurso

    :param override: Token a usar como máxima prioridad (ej: args.hf_token del CLI).
                      Si es None, se ignora y busca en entorno.
    :type override: str | None

    :returns: Token de HuggingFace o None
    :rtype: Optional[str]

    :example:
        # Con argumentos CLI
        token = get_hf_token(args.hf_token)  # CLI > env > .env

        # Auto-detecta
        token = get_hf_token()  # env > .env
    """
    if override:
        return override

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    load_dotenv()

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    return None


def get_hf_token_or_raise() -> str:
    """
    Obtiene el token de HuggingFace o lanza error si no está configurado.

    :returns: Token de HuggingFace
    :rtype: str

    :raises ValueError: Si no hay token configurado en ningún lugar
    """
    token = get_hf_token()
    if not token:
        raise ValueError(
            "HF_TOKEN no configurado. Opciones:\n"
            "  1. Exportar variable: export HF_TOKEN=hf_xxx\n"
            "  2. Crear archivo .env con: HF_TOKEN=hf_xxx\n"
            "  3. Pasar --hf-token via CLI"
        )
    return token
