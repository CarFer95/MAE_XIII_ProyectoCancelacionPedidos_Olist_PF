# src/silver/preprocessing.py

import pandas as pd
from typing import Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(
    X_train: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    BLOQUE 9R:
    - Detecta columnas numéricas y categóricas.
    - Crea pipelines de imputación + escalado / one-hot.
    - Devuelve el ColumnTransformer y listas de columnas numéricas/categóricas.
    """
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    print("Columnas numéricas detectadas:", len(numeric_cols))
    print("Columnas categóricas detectadas:", len(categorical_cols))

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    print("\nPreprocesador reconstruido sin leakage.")
    return preprocessor, numeric_cols, categorical_cols
