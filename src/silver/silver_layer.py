# src/silver/silver_layer.py

from typing import Tuple, Dict
import pandas as pd

from src.config.config import Settings
from src.silver.load_master import load_master, define_target_and_features
from src.silver.temporal_split import temporal_split
from src.silver.leakage import remove_leakage
from src.silver.cleaning import replace_infinite_with_nan
from src.silver.preprocessing import build_preprocessor
from src.silver.train_baseline import train_baseline_logreg
from src.silver.evaluate_baseline import evaluate_on_backtest
from src.silver.plots_baseline import plot_baseline_backtest, plot_eda_and_model_insights


def run_silver_layer(cfg: Settings | None = None):
    """
    Ejecuta la capa SILVER completa:

    1) Cargar master table y definir target/features (BLOQUE 7)
    2) Splits temporales train/backtest/final_test (BLOQUE 8)
    3) Eliminar leakage (BLOQUE 11.1)
    4) Reemplazar infinitos (BLOQUE 9.1)
    5) Construir preprocesador (BLOQUE 9R)
    6) Entrenar baseline limpio (BLOQUE 10R)
    7) Evaluar en backtest (BLOQUE 11R)
    8) Graficar baseline + EDA avanzada (BLOQUE 12 + 12.4)
    """
    if cfg is None:
        cfg = Settings()

    # 1) Cargar master y definir target/features
    df_master = load_master(cfg)
    y_full, X_full, features_drop, target_col = define_target_and_features(df_master)

    # 2) Splits temporales
    (
        df_master_ym,
        X_train,
        y_train,
        X_backtest,
        y_backtest,
        X_final_test,
        y_final_test,
    ) = temporal_split(df_master, target_col, features_drop)

    # 3) Eliminar leakage
    X_train, X_backtest, X_final_test, leakage_cols = remove_leakage(
        X_train,
        X_backtest,
        X_final_test,
    )

    # 4) Reemplazar infinitos
    X_train, X_backtest, X_final_test = replace_infinite_with_nan(
        X_train,
        X_backtest,
        X_final_test,
    )

    # 5) Preprocesador
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    # 6) Entrenar baseline
    clf = train_baseline_logreg(X_train, y_train, preprocessor)

    # 7) Evaluar en backtest
    y_pred_bt, y_proba_bt, metrics = evaluate_on_backtest(
        clf,
        X_backtest,
        y_backtest,
    )

    # 8) Gráficos baseline + EDA
    
    #plot_baseline_backtest(y_backtest, y_pred_bt, y_proba_bt)
    #plot_eda_and_model_insights(
    #    df_master_ym,
    #    clf,
    #    numeric_cols,
    #    categorical_cols,
    #    X_backtest,
    #    y_backtest,
    #)

    return {
        "df_master": df_master_ym,
        "X_train": X_train,
        "y_train": y_train,
        "X_backtest": X_backtest,
        "y_backtest": y_backtest,
        "X_final_test": X_final_test,
        "y_final_test": y_final_test,
        "clf": clf,
        "metrics_backtest": metrics,
        "leakage_cols": leakage_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


if __name__ == "__main__":
    cfg = Settings()
    artifacts = run_silver_layer(cfg)
    print("\n✅ Capa SILVER ejecutada correctamente.")
