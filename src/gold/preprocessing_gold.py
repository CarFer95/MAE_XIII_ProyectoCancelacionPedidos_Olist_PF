# src/gold/preprocessing_gold.py

from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor_gold(
    X_train: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    - Detecta columnas numéricas y categóricas.
    - Arma el preprocesador (imputer + scaler / onehot).
    Se usará tanto para RF, XGB baseline como para el tuning.
    """
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    print(f"Detectadas {len(numeric_cols)} columnas numéricas y {len(categorical_cols)} categóricas.")

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

    return preprocessor, numeric_cols, categorical_cols
