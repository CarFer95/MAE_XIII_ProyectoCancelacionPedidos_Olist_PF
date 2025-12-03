# src/gold/model_comparison.py

import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.config.config import Settings

# 👇 Claves que se esperan en los dicts de métricas
METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "precision_1",
    "recall_1",
    "f1_1",
    "roc_auc",
    "auc_pr",
    "gini",
    "tn",
    "fp",
    "fn",
    "tp",
]


def _extract_metrics_row(model_name: str, metrics: Dict) -> Dict:
    """
    Convierte el dict de métricas de un modelo en una fila (row) para la tabla.
    Si alguna métrica no está en el dict, se deja como None.
    """
    row = {"modelo": model_name}
    for k in METRIC_KEYS:
        row[k] = metrics.get(k, None)
    return row


def build_comparison_dataframe(
    logreg_metrics: Dict,
    rf_metrics: Dict,
    xgb_baseline_metrics: Dict,
    xgb_tuned_metrics: Dict,
) -> pd.DataFrame:
    """
    Construye el DataFrame comparativo a partir de los dicts de métricas
    de cada modelo.
    """
    rows = [
        _extract_metrics_row("LogReg", logreg_metrics),
        _extract_metrics_row("RandomForest", rf_metrics),
        _extract_metrics_row("XGB_baseline", xgb_baseline_metrics),
        _extract_metrics_row("XGB_tuned", xgb_tuned_metrics),
    ]
    df = pd.DataFrame(rows)
    return df


def save_model_comparison_figure(
    df: pd.DataFrame,
    cfg: Settings | None = None,
    filename: str = "comparacion_modelos_backtest.png",
) -> str:
    """
    Dibuja el DataFrame como tabla y lo guarda como imagen PNG.
    """
    if cfg is None:
        cfg = Settings()

    fig, ax = plt.subplots(figsize=(12, 3))  # un poco más ancho para más columnas
    ax.axis("off")

    table = ax.table(
        cellText=df.round(6).values,
        colLabels=df.columns,
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    os.makedirs(cfg.FIGURES_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(cfg.FIGURES_OUTPUT_DIR, filename)

    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"[INFO] Figura de comparación de modelos guardada en: {out_path}")
    return out_path
