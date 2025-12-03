# src/silver/temporal_split.py

import pandas as pd
from typing import Tuple, List
from src.config.config import Settings


cfg = Settings()

def temporal_split(
    df_master: pd.DataFrame,
    target_col: str,
    features_drop: List[str],
):
    """
    BLOQUE 8:
    - Crea purchase_ym.
    - Define train / backtest / final_test según meses indicados por el docente.
    - Devuelve X_train, y_train, X_backtest, y_backtest, X_final_test, y_final_test
      y el df_master con purchase_ym agregado.
    """
    df_master = df_master.copy()

    df_master["purchase_ym"] = (
        df_master["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
        .str.replace("-", "")
    )

    # Entrenamiento: 19 meses (train + val)
    train_months = cfg.TRAIN_MONTHS

    # Backtest intocable: 3 meses
    backtest_months = cfg.BACKTEST_MONTHS

    # Final test para último día: 201808 (NO se usa en proceso)
    final_test_month = cfg.FINAL_TEST_MONTHS

    df_train = df_master[df_master["purchase_ym"].isin(train_months)]
    df_backtest = df_master[df_master["purchase_ym"].isin(backtest_months)]
    df_final_test = df_master[df_master["purchase_ym"].isin(final_test_month)]

    print("Train:", df_train.shape)
    print("Backtest:", df_backtest.shape)
    print("Final Test:", df_final_test.shape)

    TARGET = target_col

    X_train = df_train.drop(columns=features_drop + ["purchase_ym"])
    y_train = df_train[TARGET].astype(int)

    X_backtest = df_backtest.drop(columns=features_drop + ["purchase_ym"])
    y_backtest = df_backtest[TARGET].astype(int)

    X_final_test = df_final_test.drop(columns=features_drop + ["purchase_ym"])
    y_final_test = df_final_test[TARGET].astype(int)  # NO usar hasta examen

    return (
        df_master,
        X_train,
        y_train,
        X_backtest,
        y_backtest,
        X_final_test,
        y_final_test,
    )
