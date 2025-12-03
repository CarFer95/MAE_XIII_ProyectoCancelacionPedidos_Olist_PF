# src/silver/load_master.py

import os
import pandas as pd
from typing import Tuple
from src.config.config import Settings


def load_master(cfg: Settings) -> pd.DataFrame:
    """
    BLOQUE 7 (parte 1):
    - Cargar dataset extendido maestro desde orders_extended_master.csv
    - Parsea order_purchase_timestamp.
    """
    ruta_master = os.path.join(cfg.RUTA_BASE_SILVER, cfg.MASTER_FILENAME)
    df_master = pd.read_csv(ruta_master, parse_dates=["order_purchase_timestamp"])

    print("Shape df_master:", df_master.shape)
    return df_master


def define_target_and_features(
    df_master: pd.DataFrame,
    target_col: str = "order_canceled_extended",
) -> Tuple[pd.Series, pd.DataFrame, list, str]:
    """
    BLOQUE 7 (parte 2):
    - Define target extendido.
    - Define columnas a eliminar del set de features.
    - Construye X completo (solo informativo, como en el notebook).
    """
    TARGET = target_col
    y = df_master[TARGET].astype(int)

    print("\nDistribución del target:")
    print(y.value_counts(normalize=True).round(4))

    features_drop = [
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "is_canceled_strict",
        "order_canceled_extended",
    ]

    X = df_master.drop(columns=features_drop)

    print("\nShape X:", X.shape)
    print("Primeras columnas X:")
    print(X.columns[:15])

    return y, X, features_drop, TARGET
