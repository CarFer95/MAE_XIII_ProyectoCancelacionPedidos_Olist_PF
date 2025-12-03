# src/gold/gold_layer.py

from src.config.config import Settings
from src.gold.data_prep_gold import load_master_gold, prepare_splits_gold
from src.gold.preprocessing_gold import build_preprocessor_gold
from src.gold.rf_baseline import train_rf_baseline, evaluate_rf_baseline
from src.gold.xgb_baseline import train_xgb_baseline, evaluate_xgb_baseline
from src.gold.xgb_tuning import (
    tune_xgb_recall,
    evaluate_xgb_tuned_backtest,
    evaluate_xgb_tuned_final,
)
from src.gold.export_model import export_best_model
from src.gold.plots_gold import plot_final_test_diagnostics, save_all_figures_gold


def run_gold_layer(cfg: Settings | None = None):
    """
    Ejecuta de punta a punta la capa GOLD:

    1) Cargar master + limpiar leakage + splits + limpiar inf.
    2) Construir preprocesador.
    3) RF baseline (train + eval BACKTEST).
    4) XGB baseline (train + eval BACKTEST).
    5) XGB tuning priorizando recall (train).
    6) Eval XGB tuneado en BACKTEST y FINAL TEST.
    7) Exportar modelo final.
    8) Generar gráficos (show) + PNGs.
    """
    if cfg is None:
        cfg = Settings()

    # 1) Datos
    df_master = load_master_gold()
    (
        df_clean,
        X_train,
        y_train,
        X_backtest,
        y_backtest,
        X_final_test,
        y_final_test,
    ) = prepare_splits_gold(df_master)

    # 2) Preprocesador común
    preprocessor_gold, numeric_cols, categorical_cols = build_preprocessor_gold(X_train)

    # 3) RF baseline
    rf_clf = train_rf_baseline(X_train, y_train, preprocessor_gold)
    _, _, rf_metrics = evaluate_rf_baseline(rf_clf, X_backtest, y_backtest)

    # 4) XGB baseline
    xgb_clf = train_xgb_baseline(X_train, y_train, preprocessor_gold)
    _, _, xgb_metrics = evaluate_xgb_baseline(xgb_clf, X_backtest, y_backtest)

    # 5) Tuning XGB
    best_xgb, best_params = tune_xgb_recall(X_train, y_train, preprocessor_gold)

    # 6) Evaluación BACKTEST & FINAL TEST
    _, _, tuned_metrics_bt = evaluate_xgb_tuned_backtest(
        best_xgb, X_backtest, y_backtest
    )
    y_pred_final, y_proba_final, tuned_metrics_final = evaluate_xgb_tuned_final(
        best_xgb, X_final_test, y_final_test
    )

    # 7) Exportar modelo
    export_best_model(best_xgb, cfg)

    # 8) Gráficos
    plot_final_test_diagnostics(
        best_xgb,
        X_final_test,
        y_final_test,
        y_pred_final,
        y_proba_final,
    )
    save_all_figures_gold(
        cfg,
        best_xgb,
        X_final_test,
        y_final_test,
        y_pred_final,
        y_proba_final,
    )

    return {
        "df_clean": df_clean,
        "X_train": X_train,
        "y_train": y_train,
        "X_backtest": X_backtest,
        "y_backtest": y_backtest,
        "X_final_test": X_final_test,
        "y_final_test": y_final_test,
        "rf_clf": rf_clf,
        "xgb_clf": xgb_clf,
        "best_xgb": best_xgb,
        "rf_metrics_backtest": rf_metrics,
        "xgb_baseline_metrics_backtest": xgb_metrics,
        "xgb_tuned_metrics_backtest": tuned_metrics_bt,
        "xgb_tuned_metrics_final": tuned_metrics_final,
        "best_params": best_params,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


if __name__ == "__main__":
    cfg = Settings()
    artifacts_gold = run_gold_layer(cfg)
    print("\n✅ Capa GOLD ejecutada correctamente.")
