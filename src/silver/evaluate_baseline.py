# src/silver/evaluate_baseline.py

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)


def evaluate_on_backtest(
    clf: Pipeline,
    X_backtest: pd.DataFrame,
    y_backtest: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    BLOQUE 11R:
    - Evalúa el modelo en el BACKTEST.
    - Imprime métricas, matriz de confusión y reporte.
    - Devuelve y_pred, y_proba y un dict con métricas.
    """
    print("Evaluando modelo limpio en BACKTEST...\n")

    y_pred_bt = clf.predict(X_backtest)
    y_proba_bt = clf.predict_proba(X_backtest)[:, 1]

    acc = accuracy_score(y_backtest, y_pred_bt)
    prec = precision_score(y_backtest, y_pred_bt, zero_division=0)
    rec = recall_score(y_backtest, y_pred_bt, zero_division=0)
    f1 = f1_score(y_backtest, y_pred_bt, zero_division=0)
    auc = roc_auc_score(y_backtest, y_proba_bt)
    gini = 2 * auc - 1

    print("MÉTRICAS BACKTEST (target extendido, sin leakage)")
    print("-----------------------------------------------")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Gini:      {gini:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_backtest, y_pred_bt))

    print("\nReporte de clasificación:")
    print(classification_report(y_backtest, y_pred_bt, digits=4))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "gini": gini,
    }

    return y_pred_bt, y_proba_bt, metrics
