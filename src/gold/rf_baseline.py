# src/gold/rf_baseline.py

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline


def train_rf_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor_rf: ColumnTransformer,
) -> Pipeline:
    """
    Entrena el Random Forest baseline con el preprocesador recibido.
    """
    print("Entrenando modelo no lineal (Random Forest baseline)...")

    rf_clf = Pipeline(steps=[
        ("preprocessor", preprocessor_rf),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    rf_clf.fit(X_train, y_train)
    print("Random Forest entrenado.")
    return rf_clf


def evaluate_rf_baseline(
    rf_clf: Pipeline,
    X_backtest: pd.DataFrame,
    y_backtest: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Evalúa el RF baseline en BACKTEST.
    """
    y_pred_rf = rf_clf.predict(X_backtest)
    y_proba_rf = rf_clf.predict_proba(X_backtest)[:, 1]

    acc = accuracy_score(y_backtest, y_pred_rf)
    prec = precision_score(y_backtest, y_pred_rf, zero_division=0)
    rec = recall_score(y_backtest, y_pred_rf, zero_division=0)
    f1 = f1_score(y_backtest, y_pred_rf, zero_division=0)
    auc = roc_auc_score(y_backtest, y_proba_rf)
    gini = 2 * auc - 1

    print("\nMÉTRICAS BACKTEST – Random Forest")
    print("--------------------------------------")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Gini     : {gini:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_backtest, y_pred_rf))

    print("\nReporte de clasificación:")
    print(classification_report(y_backtest, y_pred_rf, digits=4))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "gini": gini,
    }

    return y_pred_rf, y_proba_rf, metrics
