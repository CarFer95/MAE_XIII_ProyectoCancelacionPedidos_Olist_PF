# src/gold/plots_gold.py

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from xgboost import plot_importance
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from src.config.config import Settings


# ---------- Helpers KS & Gain/Lift ----------

def ks_curve(y_true, y_probs):
    fpr, tpr, thr = roc_curve(y_true, y_probs)
    ks = max(tpr - fpr)
    return fpr, tpr, ks


def gain_lift_curve(y_true, y_scores):
    df = pd.DataFrame({"y": y_true, "score": y_scores})
    df = df.sort_values("score", ascending=False)
    df["cum_positive"] = df["y"].cumsum()
    df["cum_total"] = np.arange(1, len(df) + 1)
    df["gain"] = df["cum_positive"] / df["y"].sum()
    df["lift"] = df["gain"] / (df["cum_total"] / len(df))
    return df


# ---------- Plots en pantalla (equivalente a tu BLOQUE 8) ----------

def plot_final_test_diagnostics(
    best_xgb,
    X_final_test: pd.DataFrame,
    y_final_test: pd.Series,
    y_pred_final: np.ndarray,
    y_proba_final: np.ndarray,
):
    """
    Reproduce los gráficos del BLOQUE 8 para FINAL TEST (mostrando en pantalla):
    - Curva ROC
    - Matriz de confusión
    - Distribución de probabilidades
    - Importancia de features XGB
    - SHAP (bar + beeswarm)
    - KS Curve
    - Gain & Lift
    """
    # 1) ROC
    fpr, tpr, _ = roc_curve(y_final_test, y_proba_final)
    auc = roc_auc_score(y_final_test, y_proba_final)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Curva ROC – XGBoost Tuneado (Final Test)", fontsize=14)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Matriz de confusión
    cm = confusion_matrix(y_final_test, y_pred_final)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión – XGBoost Tuneado (Final Test)", fontsize=14)
    plt.xlabel("Predicción")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.show()

    # 3) Distribución de probabilidades
    plt.figure(figsize=(9, 4))
    plt.hist(y_proba_final[y_final_test == 0], bins=40, alpha=0.6, label="No cancelado (0)")
    plt.hist(y_proba_final[y_final_test == 1], bins=40, alpha=0.6, label="Cancelado (1)")
    plt.title("Distribución de Probabilidades – Final Test", fontsize=14)
    plt.xlabel("Probabilidad de cancelación", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Importancia XGBoost nativo
    plt.figure(figsize=(10, 8))
    plot_importance(best_xgb.named_steps["model"], max_num_features=20)
    plt.title("Importancia de Features (Top 20) – XGBoost", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 5) SHAP
    print("Calculando valores SHAP... (puede tardar unos segundos)")

    xgb_model_only = best_xgb.named_steps["model"]
    sample = X_final_test.sample(min(1000, len(X_final_test)), random_state=42)
    sample_transformed = best_xgb.named_steps["preprocessor"].transform(sample)

    explainer = shap.TreeExplainer(xgb_model_only)
    shap_values = explainer(sample_transformed)

    shap.plots.bar(shap_values, max_display=20)
    shap.plots.beeswarm(shap_values, max_display=20)

    # 6) KS Curve
    fpr, tpr, ks = ks_curve(y_final_test, y_proba_final)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, label="FPR", linewidth=2)
    plt.plot(tpr, label="TPR", linewidth=2)
    plt.title(f"KS Curve – KS = {ks:.4f}", fontsize=14)
    plt.xlabel("Thresholds (índice)")
    plt.ylabel("Rates")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 7) Gain & Lift
    df_gl = gain_lift_curve(y_final_test, y_proba_final)

    plt.figure(figsize=(7, 5))
    plt.plot(df_gl["cum_total"] / len(df_gl), df_gl["gain"], linewidth=2)
    plt.title("Gain Curve – XGBoost Tuneado", fontsize=14)
    plt.xlabel("Porcentaje de muestras ordenadas por score")
    plt.ylabel("Gain")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(df_gl["cum_total"] / len(df_gl), df_gl["lift"], linewidth=2)
    plt.title("Lift Curve – XGBoost Tuneado", fontsize=14)
    plt.xlabel("Porcentaje de muestras ordenadas por score")
    plt.ylabel("Lift")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------- Generación de PNGs (equivalente a tu bloque final) ----------

def save_all_figures_gold(
    cfg: Settings,
    best_xgb,
    X_final_test: pd.DataFrame,
    y_final_test: pd.Series,
    y_pred_final: np.ndarray,
    y_proba_final: np.ndarray,
):
    """
    Reproduce el bloque donde se generan y guardan todas las figuras Sprint 3 en PNG.
    """
    fig_path = cfg.FIGURES_OUTPUT_DIR
    os.makedirs(fig_path, exist_ok=True)

    print("Carpeta creada para las gráficas:", fig_path)

    # 1. ROC - Final Test
    fpr, tpr, _ = roc_curve(y_final_test, y_proba_final)
    auc = roc_auc_score(y_final_test, y_proba_final)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Curva ROC – XGBoost Tuneado (Final Test)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(fig_path, "roc_finaltest.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Matriz de Confusión - Final Test
    cm = confusion_matrix(y_final_test, y_pred_final)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión – Final Test")
    plt.xlabel("Predicción")
    plt.ylabel("Clase Real")
    plt.savefig(os.path.join(fig_path, "confusion_finaltest.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Distribución de Probabilidades
    plt.figure(figsize=(9, 4))
    plt.hist(y_proba_final[y_final_test == 0], bins=40, alpha=0.6, label="No cancelado (0)")
    plt.hist(y_proba_final[y_final_test == 1], bins=40, alpha=0.6, label="Cancelado (1)")
    plt.title("Distribución de Probabilidades – Final Test")
    plt.xlabel("Probabilidad de cancelación")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fig_path, "prob_dist_finaltest.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Importancia XGBoost
    plt.figure(figsize=(10, 8))
    plot_importance(best_xgb.named_steps["model"], max_num_features=20)
    plt.title("Importancia de Features (Top 20) – XGBoost")
    plt.savefig(os.path.join(fig_path, "xgb_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 5. SHAP – Bar Plot & Beeswarm
    xgb_model = best_xgb.named_steps["model"]
    preprocess = best_xgb.named_steps["preprocessor"]

    sample = X_final_test.sample(min(800, len(X_final_test)), random_state=42)
    sample_transformed = preprocess.transform(sample)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(sample_transformed)

    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.savefig(os.path.join(fig_path, "shap_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.savefig(os.path.join(fig_path, "shap_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 6. KS Curve
    fpr, tpr, ks = ks_curve(y_final_test, y_proba_final)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, label="FPR", linewidth=2)
    plt.plot(tpr, label="TPR", linewidth=2)
    plt.title(f"KS Curve – KS = {ks:.4f}")
    plt.xlabel("Índice")
    plt.ylabel("Tasa")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fig_path, "ks_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 7. Gain & Lift Curves
    df_gl = gain_lift_curve(y_final_test, y_proba_final)

    plt.figure(figsize=(7, 5))
    plt.plot(df_gl["cum_total"] / len(df_gl), df_gl["gain"], linewidth=2)
    plt.title("Gain Curve – XGBoost")
    plt.xlabel("Porcentaje de población")
    plt.ylabel("Gain")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fig_path, "gain_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(df_gl["cum_total"] / len(df_gl), df_gl["lift"], linewidth=2)
    plt.title("Lift Curve – XGBoost")
    plt.xlabel("Porcentaje de población")
    plt.ylabel("Lift")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fig_path, "lift_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Todas las imágenes fueron guardadas en:")
    print(fig_path)
