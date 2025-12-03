# src/silver/leakage.py

import pandas as pd
from typing import List, Tuple


def remove_leakage(
    X_train: pd.DataFrame,
    X_backtest: pd.DataFrame,
    X_final_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    BLOQUE 11.1:
    - Elimina columnas con fuga de información (leakage).
    - Devuelve X_train, X_backtest, X_final_test actualizados y lista de columnas removidas.
    """
    leakage_cols = [
        # Fechas posteriores a la compra
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "review_creation_date",
        "review_answer_timestamp",

        # Variables que usan info solo después de entrega/review
        "review_score",
        "review_comment_message",
        "has_review_comment",
        "review_comment_length",
        "review_creation_delay_days",
        "review_answer_delay_days",

        # Derivados del estado final del pedido
        "seller_cancel_rate_avg",
        "customer_cancellation_history",
    ]

    leakage_cols = [c for c in leakage_cols if c in X_train.columns]

    print("Columnas PROHIBIDAS removidas:", leakage_cols)

    X_train = X_train.drop(columns=leakage_cols)
    X_backtest = X_backtest.drop(columns=leakage_cols)
    X_final_test = X_final_test.drop(columns=leakage_cols)

    print("Shapes tras remover leakage:")
    print("Train:", X_train.shape)
    print("Backtest:", X_backtest.shape)
    print("Final Test:", X_final_test.shape)

    return X_train, X_backtest, X_final_test, leakage_cols
