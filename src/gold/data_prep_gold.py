# src/gold/data_prep_gold.py

import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.paths import get_paths
from src.config.config import Settings


cfg = Settings()

def load_master_gold() -> pd.DataFrame:
    """
    SPRINT 3 - BLOQUE 1
    - Busca el archivo master en la ruta local configurada (paths['processed']).
    - Carga df_master con parseo de order_purchase_timestamp.
    """
    TARGET_FILE = os.getenv("MASTER_FILENAME", "orders_extended_master.csv")
    paths = get_paths()
    master_path = paths["processed"] / TARGET_FILE

    if not master_path.exists():
        raise FileNotFoundError(
            f"No se encontró {TARGET_FILE} en la ruta local {master_path.parent}. "
            "Verifica que el archivo exista (debería haberse generado en la capa BRONZE)."
        )

    print("Master encontrado en:", master_path)

    df_master = pd.read_csv(master_path, parse_dates=["order_purchase_timestamp"])
    print("Shape df_master:", df_master.shape)

    return df_master


def prepare_splits_gold(
    df_master: pd.DataFrame,
    target_col: str = "order_canceled_extended",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series]:
    """
    SPRINT 3 – BLOQUE 1 (continuación)
    - Elimina leakage (columnas prohibidas).
    - Define target.
    - Aplica split temporal docente (train, backtest, final_test).
    - Limpia infinitos → NaN.

    Devuelve:
        df_clean, X_train, y_train, X_backtest, y_backtest, X_final_test, y_final_test
    """
    # 4) Eliminar leakage
    leakage_cols = [
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "review_creation_date",
        "review_answer_timestamp",
        "review_score",
        "review_comment_message",
        "has_review_comment",
        "review_comment_length",
        "review_creation_delay_days",
        "review_answer_delay_days",
        "seller_cancel_rate_avg",
        "customer_cancellation_history",
    ]
    leakage_cols = [c for c in leakage_cols if c in df_master.columns]

    df_clean = df_master.drop(columns=leakage_cols).copy()
    print("Leakage removido. Columnas quitadas:", leakage_cols)

    # 5) Definir target
    TARGET = target_col

    # 6) Split temporal docente
    df_clean["purchase_ym"] = (
        df_clean["order_purchase_timestamp"]
        .dt.to_period("M").astype(str).str.replace("-", "")
    )

    train_months = cfg.TRAIN_MONTHS
    backtest_months = cfg.BACKTEST_MONTHS
    final_test_month = cfg.FINAL_TEST_MONTHS

    df_train = df_clean[df_clean["purchase_ym"].isin(train_months)]
    df_backtest = df_clean[df_clean["purchase_ym"].isin(backtest_months)]
    df_final_test = df_clean[df_clean["purchase_ym"].isin(final_test_month)]

    features_drop = [
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "is_canceled_strict",
        TARGET,
        "purchase_ym",
    ]

    X_train = df_train.drop(columns=features_drop)
    y_train = df_train[TARGET].astype(int)

    X_backtest = df_backtest.drop(columns=features_drop)
    y_backtest = df_backtest[TARGET].astype(int)

    X_final_test = df_final_test.drop(columns=features_drop)
    y_final_test = df_final_test[TARGET].astype(int)  # no tocar hasta el final

    # 7) Limpieza de infinitos
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_backtest = X_backtest.replace([np.inf, -np.inf], np.nan)
    X_final_test = X_final_test.replace([np.inf, -np.inf], np.nan)

    print("\nSplits listos para Sprint 3:")
    print("Train:", X_train.shape, y_train.shape)
    print("Backtest:", X_backtest.shape, y_backtest.shape)
    print("Final Test:", X_final_test.shape, y_final_test.shape)

    return df_clean, X_train, y_train, X_backtest, y_backtest, X_final_test, y_final_test
