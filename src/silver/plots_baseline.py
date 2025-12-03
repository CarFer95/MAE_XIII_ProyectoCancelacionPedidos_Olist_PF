# src/silver/plots_baseline.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline


def plot_baseline_backtest(
    y_backtest: pd.Series,
    y_pred_bt: np.ndarray,
    y_proba_bt: np.ndarray,
):
    """
    BLOQUE 12:
    - Curva ROC.
    - Matriz de confusión visual.
    - Distribución de probabilidades predichas.
    """
    print("Generando gráficos del baseline limpio (BACKTEST)...")

    # ---------- 12.1 Curva ROC ----------
    fpr, tpr, thresholds = roc_curve(y_backtest, y_proba_bt)
    auc = roc_auc_score(y_backtest, y_proba_bt)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Baseline limpio (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Azar")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC – Baseline limpio (BACKTEST)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- 12.2 Matriz de confusión visual ----------
    cm = confusion_matrix(y_backtest, y_pred_bt)

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Matriz de confusión – Baseline limpio (BACKTEST)", fontsize=11, pad=12)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No cancelado", "Cancelado"], rotation=45)
    plt.yticks(tick_marks, ["No cancelado", "Cancelado"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.ylabel("Clase real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.show()

    # ---------- 12.3 Distribución de probabilidades ----------
    proba_pos = y_proba_bt[y_backtest == 1]
    proba_neg = y_proba_bt[y_backtest == 0]

    plt.figure(figsize=(9, 4))
    plt.hist(proba_neg, bins=30, alpha=0.7, label="No cancelado (0)")
    plt.hist(proba_pos, bins=30, alpha=0.7, label="Cancelado (1)")

    plt.xlabel("Probabilidad predicha de cancelación")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de probabilidades predichas por clase real (BACKTEST)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_eda_and_model_insights(
    df_master: pd.DataFrame,
    clf: Pipeline,
    numeric_cols,
    categorical_cols,
    X_backtest: pd.DataFrame,
    y_backtest: pd.Series,
):
    """
    BLOQUE 12.4:
    - EDA de targets (estricto vs extendido).
    - Tasa por método de pago.
    - Tasa por categoría (Top 15).
    - Top 15 coeficientes de la regresión logística.
    - Distribución de probabilidades predichas (BACKTEST).
    """
    df_g = df_master.copy()

    # 1) Comparación Estricto vs Extendido
    strict_rate = df_g["is_canceled_strict"].mean()
    ext_rate = df_g["order_canceled_extended"].mean()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        ["Estricto", "Extendido"],
        [strict_rate, ext_rate],
        color=["#2F4F4F", "#1F77B4"],
    )
    plt.ylabel("Tasa de cancelación", fontsize=11)
    plt.title("Comparación de tasa de cancelación\nTarget estricto vs extendido", fontsize=13)
    plt.ylim(0, max(strict_rate, ext_rate) * 1.4)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.0003,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()

    # 2) Tasa por método de pago
    if "main_payment_type" in df_g.columns:
        strict_by_pay = (
            df_g.groupby("main_payment_type")["is_canceled_strict"]
            .mean()
            .sort_values(ascending=False)
        )
        ext_by_pay = (
            df_g.groupby("main_payment_type")["order_canceled_extended"]
            .mean()
            .reindex(strict_by_pay.index)
        )

        labels = strict_by_pay.index.tolist()
        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(8, 4.5))
        plt.bar(
            x - width / 2,
            strict_by_pay.values,
            width,
            label="Estricto",
            color="#2F4F4F",
        )
        plt.bar(
            x + width / 2,
            ext_by_pay.values,
            width,
            label="Extendido",
            color="#1F77B4",
        )

        plt.xticks(x, labels, rotation=30, ha="right", fontsize=10)
        plt.ylabel("Tasa de cancelación", fontsize=11)
        plt.title(
            "Tasa de cancelación por método de pago\nEstricto vs extendido",
            fontsize=13,
        )
        plt.legend()

        for i, v in enumerate(strict_by_pay.values):
            plt.text(i - width / 2, v + 0.0003, f"{v:.2%}", ha="center", fontsize=9)
        for i, v in enumerate(ext_by_pay.values):
            plt.text(i + width / 2, v + 0.0003, f"{v:.2%}", ha="center", fontsize=9)

        plt.tight_layout()
        plt.show()
    else:
        print("No existe main_payment_type en el dataset maestro.")

    # 3) Tasa por categoría (Top 15 por volumen) – target extendido
    cat_col = (
        "product_category_name_english"
        if "product_category_name_english" in df_g.columns
        else "product_category_name"
    )

    cat_stats = (
        df_g.groupby(cat_col)["order_canceled_extended"]
        .agg(cancel_rate="mean", n_orders="count")
        .reset_index()
    )

    top_cats = (
        cat_stats.sort_values("n_orders", ascending=False)
        .head(15)
        .sort_values("cancel_rate", ascending=True)
    )

    plt.figure(figsize=(9, 5))
    plt.barh(top_cats[cat_col].astype(str), top_cats["cancel_rate"], color="#1F77B4")

    plt.xlabel("Tasa de cancelación (target extendido)", fontsize=11)
    plt.ylabel("Categoría de producto", fontsize=11)
    plt.title(
        "Tasa de cancelación por categoría\nTop 15 categorías por volumen",
        fontsize=13,
    )

    for i, v in enumerate(top_cats["cancel_rate"]):
        plt.text(v + 0.0002, i, f"{v:.2%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    # 4) Top 15 coeficientes del modelo
    model_lr = clf.named_steps["model"]
    preproc_lr = clf.named_steps["preprocessor"]

    num_features_out = numeric_cols

    cat_transformer = preproc_lr.named_transformers_["cat"]
    ohe = cat_transformer.named_steps["onehot"]
    cat_features_out = ohe.get_feature_names_out(categorical_cols)

    feature_names = np.concatenate([num_features_out, cat_features_out])
    coefs = model_lr.coef_.ravel()

    abs_coefs = np.abs(coefs)
    idx_sorted = np.argsort(abs_coefs)[-15:]

    top_features = feature_names[idx_sorted]
    top_coefs = coefs[idx_sorted]

    plt.figure(figsize=(9, 5))
    y_pos = np.arange(len(top_features))

    colors = ["#1F77B4" if c > 0 else "#B22222" for c in top_coefs]
    plt.barh(y_pos, top_coefs, color=colors)
    plt.yticks(y_pos, top_features, fontsize=9)
    plt.xlabel("Coeficiente (impacto sobre probabilidad de cancelación)", fontsize=11)
    plt.title(
        "Regresión logística – principales factores de riesgo\n(Top 15 coeficientes)",
        fontsize=13,
    )
    plt.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.show()

    # 5) Distribución de probabilidades predichas (BACKTEST)
    y_proba_bt = clf.predict_proba(X_backtest)[:, 1]

    proba_pos = y_proba_bt[y_backtest == 1]
    proba_neg = y_proba_bt[y_backtest == 0]

    plt.figure(figsize=(9, 4))
    plt.hist(proba_neg, bins=40, alpha=0.7, label="No cancelado (0)", color="#4F4F4F")
    plt.hist(proba_pos, bins=40, alpha=0.7, label="Cancelado (1)", color="#B22222")

    plt.xlabel("Probabilidad predicha de cancelación", fontsize=11)
    plt.ylabel("Frecuencia", fontsize=11)
    plt.title(
        "Distribución de probabilidades predichas por clase real (BACKTEST)",
        fontsize=13,
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
