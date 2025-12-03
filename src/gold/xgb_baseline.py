# src/gold/xgb_baseline.py

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.config.config import Settings

cfg = Settings()


def train_xgb_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor_xgb: ColumnTransformer,
) -> Pipeline:
    """
    SPRINT 3 – BLOQUE 3:
    Entrena el XGBoost baseline con los mismos pasos que tu notebook.
    """
    print("Entrenando modelo avanzado (XGBoost baseline)...")

    scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])

    xgb_clf = Pipeline(steps=[
        ("preprocessor", preprocessor_xgb),
        ("model", XGBClassifier(
            n_estimators=cfg.XGB_N_ESTIMATORS,
            max_depth=cfg.XGB_MAX_DEPTH,
            learning_rate=cfg.XGB_LEARNING_RATE,
            subsample=cfg.XGB_SUBSAMPLE,
            colsample_bytree=cfg.XGB_COLSAMPLE_BYTREE,
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )),
    ])

    xgb_clf.fit(X_train, y_train)
    print("XGBoost entrenado correctamente.")
    return xgb_clf


def evaluate_xgb_baseline(
    xgb_clf: Pipeline,
    X_backtest: pd.DataFrame,
    y_backtest: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Evalúa el XGBoost baseline en BACKTEST.
    """
    y_pred_xgb = xgb_clf.predict(X_backtest)
    y_proba_xgb = xgb_clf.predict_proba(X_backtest)[:, 1]

    acc = accuracy_score(y_backtest, y_pred_xgb)
    prec = precision_score(y_backtest, y_pred_xgb, zero_division=0)
    rec = recall_score(y_backtest, y_pred_xgb, zero_division=0)
    f1 = f1_score(y_backtest, y_pred_xgb, zero_division=0)
    auc = roc_auc_score(y_backtest, y_proba_xgb)
    gini = 2 * auc - 1

    print("\nMÉTRICAS BACKTEST – XGBoost Baseline")
    print("-----------------------------------------")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Gini     : {gini:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_backtest, y_pred_xgb))

    print("\nReporte de clasificación:")
    print(classification_report(y_backtest, y_pred_xgb, digits=4))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "gini": gini,
    }

    return y_pred_xgb, y_proba_xgb, metrics
