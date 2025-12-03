# src/silver/cleaning.py

import numpy as np
import pandas as pd
from typing import Tuple


def replace_infinite_with_nan(
    X_train: pd.DataFrame,
    X_backtest: pd.DataFrame,
    X_final_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    BLOQUE 9.1:
    - Reemplaza valores ±inf por NaN en los splits.
    """
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_backtest = X_backtest.replace([np.inf, -np.inf], np.nan)
    X_final_test = X_final_test.replace([np.inf, -np.inf], np.nan)

    print("Valores infinitos eliminados y reemplazados por NaN.")
    return X_train, X_backtest, X_final_test
