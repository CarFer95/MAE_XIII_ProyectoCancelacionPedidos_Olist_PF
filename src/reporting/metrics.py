# src/reporting/metrics.py

import json
from typing import Dict
from .run_context import RunContext
import logging

logger = logging.getLogger(__name__)

def save_metrics(metrics: Dict, context: RunContext) -> None:
    """
    Guarda las métricas de la corrida en un archivo JSON.
    """
    context.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(context.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    logger.info(f"✅ Métricas guardadas en: {context.metrics_path}")
