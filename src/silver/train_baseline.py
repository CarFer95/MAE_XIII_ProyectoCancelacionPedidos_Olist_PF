# src/silver/train_baseline.py

from typing import Tuple, List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer


def train_baseline_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    BLOQUE 10R:
    - Entrena el modelo baseline (Regresión Logística) con preprocesador.
    """
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            solver="lbfgs",
        )),
    ])

    print("Entrenando modelo limpio (sin leakage)...")
    clf.fit(X_train, y_train)
    print("✅ Entrenamiento completado (versión limpia).")

    return clf
