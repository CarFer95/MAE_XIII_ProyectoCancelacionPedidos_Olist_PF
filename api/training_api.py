# api/utils/training_api.py

import pandas as pd

from src.config.config import Settings
from src.gold.data_prep_gold import prepare_splits_gold
from src.gold.gold_layer import run_gold_layer


def train_model_from_df(df: pd.DataFrame):
    """
    Recibe un DataFrame enviado desde la API
    y ejecuta tu pipeline GOLD usando ese DF como master.
    """

    cfg = Settings()

    # ================================
    # 1) Simular tu capa SILVER/GOLD
    #    usando el DF recibido
    # ================================
    (
        df_clean,
        X_train, y_train,
        X_backtest, y_backtest,
        X_final_test, y_final_test,
    ) = prepare_splits_gold(df)

    # ================================
    # 2) Ejecutar GOLD sobre este DF
    # ================================
    artifacts_gold = run_gold_layer(cfg)

    # El modelo final es XGB Tuned (FINAL TEST)
    metrics = artifacts_gold["xgb_tuned_metrics_final"]

    # ================================
    # 3) Armar la respuesta
    # ================================
    response = {
        "modelo": "XGB_tuned",
        "roc_auc": metrics["roc_auc"],
        "f1": metrics["f1_1"],
        "precision": metrics["precision_1"],
        "recall": metrics["recall_1"],
        "confusion_matrix": {
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        }
    }

    return response
