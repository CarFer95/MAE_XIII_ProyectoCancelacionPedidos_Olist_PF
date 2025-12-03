# src/reporting/logger_config.py

import logging
from .run_context import RunContext

def setup_logging(context: RunContext) -> logging.Logger:
    """
    Configura logging para escribir en archivo + consola
    usando el contexto de ejecución.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(context.log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Iniciando corrida del pipeline: {context.run_id}")
    logger.info(f"📁 Directorio de la corrida: {context.run_dir}")
    return logger
