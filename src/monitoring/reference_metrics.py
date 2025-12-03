# src/monitoring/reference_metrics.py

import json
import os
from datetime import datetime
from typing import Dict

from src.config.config import Settings


def save_reference_metrics(
    metrics: Dict,
    cfg: Settings,
) -> str:
    """
    Guarda las métricas de referencia del modelo final.
    metrics: dict con al menos 'roc_auc', 'f1', etc.
    Devuelve la ruta del archivo JSON.
    """
    os.makedirs(cfg.METRICS_DIR, exist_ok=True)
    ref = {
        "created_at": datetime.now().isoformat(),
        "train_end_month": cfg.TRAIN_END_MONTH,
        "metrics": metrics,
    }
    out_path = os.path.join(cfg.METRICS_DIR, "model_reference_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=4)
    print(f"[INFO] Métricas de referencia guardadas en: {out_path}")
    return out_path
