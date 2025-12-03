# src/monitoring/drift_checker.py

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config.config import Settings
from src.gold.monthly_scoring import score_month_from_master
from src.gold.data_prep_gold import load_master_gold


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Cálculo simple de PSI entre dos distribuciones 1D.
    expected: datos históricos (train)
    actual  : datos nuevos (mes a monitorear)
    """
    expected = np.array(expected)
    actual = np.array(actual)

    # Eliminamos NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Definir bins en función del histórico
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(expected, quantiles))

    # Evitar problemas cuando hay pocos valores distintos
    if len(cut_points) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=cut_points)
    actual_counts, _ = np.histogram(actual, bins=cut_points)

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    # Evitamos 0 reemplazando con un valor muy pequeño
    epsilon = 1e-6
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))
    return psi


def load_reference_metrics(cfg: Settings) -> Optional[Dict]:
    path = os.path.join(cfg.METRICS_DIR, "model_reference_metrics.json")
    if not os.path.exists(path):
        print("[WARN] No se encontró model_reference_metrics.json. Aún no hay baseline.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_data_drift(
    cfg: Settings,
    month: str,
    feature_cols: Optional[list] = None,
    psi_threshold: float = 0.25,
) -> Dict:
    """
    Calcula PSI promedio y máximo para un subconjunto de features numéricas.
    """
    df = load_master_gold()

    # Aseguramos purchase_ym
    if "purchase_ym" not in df.columns:
        df["purchase_ym"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    # Datos históricos (todo antes del mes actual)
    df_hist = df[df["purchase_ym"] < month].copy()
    df_new = df[df["purchase_ym"] == month].copy()

    if df_hist.empty or df_new.empty:
        return {
            "psi_max": np.nan,
            "psi_mean": np.nan,
            "data_drift": False,
            "features": {},
        }

    # Si no especificas columnas, tomamos numéricas como ejemplo
    if feature_cols is None:
        feature_cols = df_hist.select_dtypes(include=[np.number]).columns.tolist()

    psi_dict = {}
    for col in feature_cols:
        psi_val = population_stability_index(df_hist[col].values, df_new[col].values)
        psi_dict[col] = psi_val

    psi_values = [v for v in psi_dict.values() if not np.isnan(v)]
    if len(psi_values) == 0:
        psi_max = np.nan
        psi_mean = np.nan
    else:
        psi_max = float(np.max(psi_values))
        psi_mean = float(np.mean(psi_values))

    data_drift = psi_max > psi_threshold

    return {
        "psi_max": psi_max,
        "psi_mean": psi_mean,
        "data_drift": data_drift,
        "features": psi_dict,
    }


def evaluate_retrain_need(
    month: str,
    cfg: Optional[Settings] = None,
    auc_drop_threshold: float = 0.10,
    psi_threshold: float = 0.25,
    freshness_months: int = 3,
) -> Dict:
    """
    Evalúa si se debería reentrenar el modelo según:
    - Caída de AUC (model drift)
    - PSI (data drift)
    - Antigüedad del modelo (freshness)
    """
    if cfg is None:
        cfg = Settings()

    print(f"\n[MONITORING] Evaluando necesidad de reentrenar para el mes {month}")

    ref = load_reference_metrics(cfg)
    if ref is None:
        # Si no hay baseline, lo más seguro es reentrenar la primera vez
        return {
            "should_retrain": True,
            "reason": "NO_BASELINE",
            "details": {},
        }

    ref_auc = ref["metrics"].get("roc_auc", None)
    train_end_month = ref.get("train_end_month", None)

    # 1) Performance actual del mes
    month_results = score_month_from_master(month, cfg)
    current_auc = month_results["metrics"]["roc_auc"]

    # 2) Caída relativa de AUC
    if ref_auc is not None and not np.isnan(current_auc):
        auc_drop = (ref_auc - current_auc) / ref_auc
    else:
        auc_drop = 0.0

    model_drift = auc_drop > auc_drop_threshold

    # 3) Data drift vía PSI
    drift_info = compute_data_drift(cfg, month, psi_threshold=psi_threshold)

    # 4) Freshness (meses transcurridos desde el fin del train)
    if train_end_month is not None:
        # Formato "YYYY-MM"
        def ym_to_int(ym: str) -> int:
            y, m = ym.split("-")
            return int(y) * 12 + int(m)

        freshness_gap = ym_to_int(month) - ym_to_int(train_end_month)
    else:
        freshness_gap = 999  # forzamos freshness si no sabemos

    freshness_trigger = freshness_gap >= freshness_months

    # Decisión final
    should_retrain = model_drift or drift_info["data_drift"] or freshness_trigger

    reason_list = []
    if model_drift:
        reason_list.append("MODEL_DRIFT")
    if drift_info["data_drift"]:
        reason_list.append("DATA_DRIFT")
    if freshness_trigger:
        reason_list.append("FRESHNESS")

    reason = " & ".join(reason_list) if reason_list else "NO_DRIFT"

    return {
        "should_retrain": should_retrain,
        "reason": reason,
        "details": {
            "ref_auc": ref_auc,
            "current_auc": current_auc,
            "auc_drop": auc_drop,
            "psi_max": drift_info["psi_max"],
            "psi_mean": drift_info["psi_mean"],
            "freshness_gap_months": freshness_gap,
        },
    }
