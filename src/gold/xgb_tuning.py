# src/gold/xgb_tuning.py

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.config.config import Settings

cfg = Settings()

def tune_xgb_recall(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor_xgb: ColumnTransformer,
) -> Tuple[Pipeline, Dict]:
    """
    SPRINT 3 – BLOQUE 4:
    Tuning rápido de XGBoost priorizando RECALL (RandomizedSearchCV).
    Devuelve best_xgb (pipeline completo) y best_params.
    """
    print("🔍 Tuning rápido del XGBoost (prioridad: RECALL)...")

    scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    pipeline_tuning = Pipeline(steps=[
        ("preprocessor", preprocessor_xgb),
        ("model", base_model),
    ])

    param_grid = {
        "model__n_estimators": cfg.XGB_TUNE_N_ESTIMATORS,
        "model__max_depth": cfg.XGB_TUNE_MAX_DEPTH,
        "model__learning_rate": cfg.XGB_TUNE_LEARNING_RATE,
        "model__subsample": cfg.XGB_TUNE_SUBSAMPLE,
        "model__colsample_bytree": cfg.XGB_TUNE_COLSAMPLE_BYTREE,
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline_tuning,
        param_distributions=param_grid,
        n_iter=10,         # rápido
        scoring="recall",  # prioridad
        cv=2,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    print("\nTuning rápido finalizado.")
    print("Mejores hiperparámetros encontrados:")
    print(random_search.best_params_)

    best_xgb = random_search.best_estimator_
    print("\n🏆 Modelo optimizado guardado en best_xgb.")

    return best_xgb, random_search.best_params_


def evaluate_xgb_tuned_backtest(
    best_xgb: Pipeline,
    X_backtest: pd.DataFrame,
    y_backtest: pd.Series,
):
    """
    SPRINT 3 – BLOQUE 5 (parte evaluación BACKTEST):
    Evalúa el modelo XGBoost tuneado en BACKTEST.
    """
    print("Evaluando modelo XGBoost TUNEADO en BACKTEST...")

    y_pred_tuned = best_xgb.predict(X_backtest)
    y_proba_tuned = best_xgb.predict_proba(X_backtest)[:, 1]

    acc = accuracy_score(y_backtest, y_pred_tuned)
    prec = precision_score(y_backtest, y_pred_tuned, zero_division=0)
    rec = recall_score(y_backtest, y_pred_tuned, zero_division=0)
    f1 = f1_score(y_backtest, y_pred_tuned, zero_division=0)
    auc = roc_auc_score(y_backtest, y_proba_tuned)
    gini = 2 * auc - 1

    print("\nMÉTRICAS BACKTEST – XGBoost TUNEADO")
    print("-----------------------------------------")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Gini     : {gini:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_backtest, y_pred_tuned))

    print("\nReporte de clasificación:")
    print(classification_report(y_backtest, y_pred_tuned, digits=4))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "gini": gini,
    }

    return y_pred_tuned, y_proba_tuned, metrics


def evaluate_xgb_tuned_final(
    best_xgb: Pipeline,
    X_final_test: pd.DataFrame,
    y_final_test: pd.Series,
):
    """
    SPRINT 3 – BLOQUE 6:
    Evalúa el modelo tuneado en FINAL TEST (201808).
    """
    print(" Evaluando modelo XGBoost TUNEADO en FINAL TEST (201808)...")

    y_pred_final = best_xgb.predict(X_final_test)
    y_proba_final = best_xgb.predict_proba(X_final_test)[:, 1]

    acc_f = accuracy_score(y_final_test, y_pred_final)
    prec_f = precision_score(y_final_test, y_pred_final, zero_division=0)
    rec_f = recall_score(y_final_test, y_pred_final, zero_division=0)
    f1_f = f1_score(y_final_test, y_pred_final, zero_division=0)
    auc_f = roc_auc_score(y_final_test, y_proba_final)
    gini_f = 2 * auc_f - 1

    print("\nMÉTRICAS FINAL TEST – XGBoost TUNEADO")
    print(f"Accuracy : {acc_f:.4f}")
    print(f"Precision: {prec_f:.4f}")
    print(f"Recall   : {rec_f:.4f}")
    print(f"F1-score : {f1_f:.4f}")
    print(f"AUC-ROC  : {auc_f:.4f}")
    print(f"Gini     : {gini_f:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_final_test, y_pred_final))

    print("\nReporte de clasificación:")
    print(classification_report(y_final_test, y_pred_final, digits=4))

    metrics_final = {
        "accuracy": acc_f,
        "precision": prec_f,
        "recall": rec_f,
        "f1": f1_f,
        "roc_auc": auc_f,
        "gini": gini_f,
    }

    return y_pred_final, y_proba_final, metrics_final
