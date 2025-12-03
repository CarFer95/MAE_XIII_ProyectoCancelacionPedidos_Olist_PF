# src/gold/monthly_scoring.py

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib

from src.config.config import Settings
from src.gold.data_prep_gold import load_master_gold


def _normalize_month(month_raw: str) -> str:
    """
    Normaliza distintos formatos de mes:
    - '201808'  -> '2018-08'
    - '2018-8'  -> '2018-08'
    - '2018-08' -> '2018-08'
    """
    m = str(month_raw).strip()
    # Caso '201808'
    if len(m) == 6 and m.isdigit():
        return f"{m[:4]}-{m[4:6]}"
    # Caso '2018-8'
    if "-" in m:
        y, mm = m.split("-")
        return f"{y}-{mm.zfill(2)}"
    return m


def score_month_from_master(
    month: str,
    cfg: Optional[Settings] = None,
    model_path: Optional[str] = None,
    target_col: str = "order_canceled_extended",
) -> Dict:
    """
    Usa el master GOLD para evaluar el performance del modelo en un mes específico.
    - month: '2018-08' o '201808'
    - model_path: ruta al .pkl (por defecto usa cfg.MODEL_OUTPUT_PATH)

    Devuelve un diccionario con métricas y conteos.
    """
    if cfg is None:
        cfg = Settings()
    if model_path is None:
        model_path = cfg.MODEL_OUTPUT_PATH

    print("\n=== SIMULACIÓN NUEVO MES CON MODELO ENTRENADO ===")
    print(f"Mes solicitado: {month}")

    month_norm = _normalize_month(month)
    print(f"Mes normalizado: {month_norm}")

    # 1) Cargar master GOLD
    df_master = load_master_gold()

    # 2) Asegurar columna purchase_ym
    if "purchase_ym" not in df_master.columns:
        df_master["purchase_ym"] = (
            df_master["order_purchase_timestamp"]
            .dt.to_period("M")
            .astype(str)
        )

    # 3) Filtrar mes
    df_mes = df_master[df_master["purchase_ym"] == month_norm].copy()
    if df_mes.empty:
        raise ValueError(f"No hay registros para el mes {month_norm} en el master.")

    if target_col not in df_mes.columns:
        raise ValueError(
            f"La columna target '{target_col}' no existe en el master. "
            "Revisa tu pipeline SILVER/GOLD."
        )

    y_true = df_mes[target_col].astype(int)
    # IMPORTANTE: solo quitamos el target; el preprocesador del pipeline
    # se encargará de seleccionar las columnas que necesita.
    X_mes = df_mes.drop(columns=[target_col])

    print(f"Registros en {month_norm}: {len(df_mes)}")

    # 4) Cargar modelo (Pipeline completo)
    print(f"Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)

    # 5) Predicciones
    y_pred = model.predict(X_mes)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_mes)[:, 1]
    else:
        # fallback raro, pero por si acaso
        y_proba = None

    # 6) Métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        gini = 2 * auc - 1
    else:
        auc = np.nan
        gini = np.nan

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print("\n=== MÉTRICAS MES", month_norm, "===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(f"Gini     : {gini:.4f}")
    print("\nMatriz de confusión:\n", cm)
    print("\nReporte de clasificación:\n", report)

    # 7) Devolver artefactos para guardarlos si quieres
    return {
        "month": month_norm,
        "n_rows": len(df_mes),
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "gini": gini,
        },
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simular nuevo mes con modelo XGB final."
    )
    parser.add_argument(
        "--month",
        required=True,
        help="Mes a evaluar, formato '2018-08' o '201808'."
    )
    args = parser.parse_args()

    cfg = Settings()
    score_month_from_master(args.month, cfg)
