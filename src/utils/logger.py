"""
Sistema de logging centralizado del pipeline RAG.

Diseño de dos canales:
  - Consola (INFO+): mensajes concisos con colores por nivel.
    Muestra el progreso en tiempo real sin saturar la pantalla.
  - Fichero (DEBUG+): traza completa con timestamp, módulo y nivel.
    Permite auditar qué ocurrió exactamente y por qué algo falló.

Uso en cualquier módulo:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Mensaje informativo")
    log.debug("Detalle interno")
    log.warning("Algo inesperado pero no fatal")
    log.error("Fallo grave", exc_info=True)

setup_pipeline_logger() debe llamarse una sola vez desde main.py
antes de iniciar el pipeline.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_DIR = Path("logs")
ROOT_LOGGER = "tfg_rag"


#  Formateador con colores ANSI para consola

class _ColoredConsoleFormatter(logging.Formatter):
    """
    Colorea el nivel de log en la salida de consola.
    Solo activa colores si la salida es un terminal real (no redirección).
    Compatible con Windows Terminal y cualquier terminal ANSI moderno.
    """

    _COLORS = {
        "DEBUG":    "\033[36m",    # cian   — detalles técnicos internos
        "INFO":     "\033[32m",    # verde  — progreso normal
        "WARNING":  "\033[33m",    # amarillo — algo inesperado, no fatal
        "ERROR":    "\033[31m",    # rojo   — fallo grave
        "CRITICAL": "\033[35m",    # magenta — fallo que detiene el pipeline
    }
    _RESET = "\033[0m"
    _BOLD  = "\033[1m"

    def __init__(self) -> None:
        # Formato compacto para consola: hora + nivel + mensaje
        super().__init__(fmt="%(asctime)s  %(levelname)-8s  %(message)s",
                         datefmt="%H:%M:%S")
        # Activar ANSI en Windows si es posible
        if sys.platform == "win32":
            os.system("")
        self._use_colors = sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self._use_colors:
            color = self._COLORS.get(record.levelname, "")
            record.levelname = f"{color}{self._BOLD}{record.levelname}{self._RESET}"
        return super().format(record)


class _FileFormatter(logging.Formatter):
    """
    Formateador detallado para el fichero de log.
    Incluye timestamp completo, módulo y nivel para trazabilidad total.
    """

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


#  API pública

def setup_pipeline_logger(
    username: str = "pipeline",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configura el logger raíz del pipeline. Llámalo una sola vez desde main.py.

    Crea un fichero de log en logs/pipeline_{username}_{timestamp}.log
    con el nivel DEBUG (trazabilidad completa) y muestra en consola
    solo mensajes INFO y superiores (progreso limpio).

    Retorna el logger raíz; los módulos usan get_logger() para obtener
    loggers hijos que heredan automáticamente esta configuración.
    """
    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(ROOT_LOGGER)

    # Evitar duplicar handlers si se llama varias veces (ej: tests)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(_ColoredConsoleFormatter())
    logger.addHandler(console_handler)

    # Handler de fichero
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"pipeline_{username}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_FileFormatter())
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"Pipeline iniciado — usuario: {username}")
    logger.info(f"Log completo (DEBUG): {log_file}")
    logger.info("=" * 60)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger hijo del logger raíz del pipeline.

    Uso recomendado en cada módulo:
        log = get_logger(__name__)

    Hereda automáticamente los handlers y niveles configurados por
    setup_pipeline_logger(). Si el pipeline no se ha inicializado aún
    (por ejemplo en tests unitarios), el logger sigue funcionando
    con la configuración por defecto de Python.
    """
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
