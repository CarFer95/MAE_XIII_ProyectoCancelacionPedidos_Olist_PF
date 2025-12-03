# Capa SILVER – Limpieza, splits temporales, baseline limpio
# Código original reorganizado desde el notebook Copia de Sprint3 (1).ipynb

# BLOQUE 7 (Sprint 2)
# Cargar master table + definir X/y

import pandas as pd
import numpy as np
from src.config.config import Settings

cfg = Settings()

RUTA_MASTER = cfg.RUTA_BASE_SILVER + "orders_extended_master.csv"

# 1) Cargar dataset extendido maestro
df_master = pd.read_csv(RUTA_MASTER, parse_dates=["order_purchase_timestamp"])

print("Shape df_master:", df_master.shape)

# 2) Definir target extendido
TARGET = "order_canceled_extended"
y = df_master[TARGET].astype(int)

print("\nDistribución del target:")
print(y.value_counts(normalize=True).round(4))

# 3) Definir columnas a eliminar del set de features
features_drop = [
    "order_id",
    "order_status",
    "order_purchase_timestamp",
    "is_canceled_strict",
    "order_canceled_extended"
]

X = df_master.drop(columns=features_drop)

print("\nShape X:", X.shape)
print("Primeras columnas X:")
print(X.columns[:15])
# BLOQUE 8 (Sprint 2)
# Split temporal definido por el docente

# Dataset total: 23 meses
# - 3 meses 2016
# - 12 meses 2017
# - 8 meses 2018

df_master["purchase_ym"] = (
    df_master["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
    .str.replace("-", "")
)

# Entrenamiento: 19 meses (train + val)
train_months = [
    "201610","201611","201612",
    "201701","201702","201703","201704","201705","201706",
    "201707","201708","201709","201710","201711","201712",
    "201801","201802","201803","201804"
]

# Backtest intocable: 3 meses
backtest_months = ["201805", "201806", "201807"]

# Final test para último día: 201808 (NO se usa en proceso)
final_test_month = ["201808"]

df_train = df_master[df_master["purchase_ym"].isin(train_months)]
df_backtest = df_master[df_master["purchase_ym"].isin(backtest_months)]
df_final_test = df_master[df_master["purchase_ym"].isin(final_test_month)]

print("Train:", df_train.shape)
print("Backtest:", df_backtest.shape)
print("Final Test:", df_final_test.shape)

X_train = df_train.drop(columns=features_drop + ["purchase_ym"])
y_train = df_train[TARGET].astype(int)

X_backtest = df_backtest.drop(columns=features_drop + ["purchase_ym"])
y_backtest = df_backtest[TARGET].astype(int)

X_final_test = df_final_test.drop(columns=features_drop + ["purchase_ym"])
y_final_test = df_final_test[TARGET].astype(int)  # NO usar hasta examen
# BLOQUE 11.1 (Corrección Crítica)
# Eliminar leakage: columnas prohibidas

leakage_cols = [
    # Fechas posteriores a la compra
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "review_creation_date",
    "review_answer_timestamp",

    # Variables que usan información disponible SOLO después de entrega/review
    "review_score",
    "review_comment_message",
    "has_review_comment",
    "review_comment_length",
    "review_creation_delay_days",
    "review_answer_delay_days",

    # Derivados del estado final del pedido
    "seller_cancel_rate_avg",    # depende del target
    "customer_cancellation_history", # depende del target extendido
]

# Filtrar solo columnas que están realmente presentes
leakage_cols = [c for c in leakage_cols if c in X_train.columns]

print("Columnas PROHIBIDAS removidas:", leakage_cols)

# Eliminar leakage en train / backtest / final test
X_train = X_train.drop(columns=leakage_cols)
X_backtest = X_backtest.drop(columns=leakage_cols)
X_final_test = X_final_test.drop(columns=leakage_cols)

print("Shapes tras remover leakage:")
print("Train:", X_train.shape)
print("Backtest:", X_backtest.shape)
print("Final Test:", X_final_test.shape)
# BLOQUE 9.1 (Sprint 2)
# Limpieza de infinitos y valores inválidos

# Reemplazar infinities por NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_backtest = X_backtest.replace([np.inf, -np.inf], np.nan)
X_final_test = X_final_test.replace([np.inf, -np.inf], np.nan)

print("Valores infinitos eliminados y reemplazados por NaN.")
# BLOQUE 9R (Sprint 2 CORREGIDO)
# Recalcular columnas y reconstruir preprocessor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1) Detectar columnas numéricas y categóricas NUEVAS
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

print("Columnas numéricas detectadas:", len(numeric_cols))
print("Columnas categóricas detectadas:", len(categorical_cols))

# 2) Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 3) ColumnTransformer final
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

print("\nPreprocesador reconstruido sin leakage.")
# BLOQUE 10R (Sprint 2 CORREGIDO)
# Reentrenar modelo baseline (sin leakage)

from sklearn.linear_model import LogisticRegression

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs"
    ))
])

print("Entrenando modelo limpio (sin leakage)...")

clf.fit(X_train, y_train)

print("✅ Entrenamiento completado (versión limpia).")
# BLOQUE 11R (Sprint 2 LIMPIO)
# Evaluación REAL del baseline en BACKTEST

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

print("Evaluando modelo limpio en BACKTEST...\n")

# Predicciones
y_pred_bt = clf.predict(X_backtest)
y_proba_bt = clf.predict_proba(X_backtest)[:, 1]

# Métricas
acc  = accuracy_score(y_backtest, y_pred_bt)
prec = precision_score(y_backtest, y_pred_bt, zero_division=0)
rec  = recall_score(y_backtest, y_pred_bt, zero_division=0)
f1   = f1_score(y_backtest, y_pred_bt, zero_division=0)
auc  = roc_auc_score(y_backtest, y_proba_bt)
gini = 2*auc - 1

print("MÉTRICAS BACKTEST (target extendido, sin leakage)")
print("-----------------------------------------------")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"Gini:      {gini:.4f}")

# Matriz de confusión
print("\nMatriz de confusión:")
print(confusion_matrix(y_backtest, y_pred_bt))

# Reporte detallado
print("\nReporte de clasificación:")
print(classification_report(y_backtest, y_pred_bt, digits=4))
# BLOQUE 12 (Sprint 2)
# Visualizaciones del baseline limpio

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

print("Generando gráficos del baseline limpio (BACKTEST)...")

# Probabilidades para clase 1 (cancelado)
y_proba_bt = clf.predict_proba(X_backtest)[:, 1]

# ---------- 12.1 Curva ROC ----------
fpr, tpr, thresholds = roc_curve(y_backtest, y_proba_bt)
auc = roc_auc_score(y_backtest, y_proba_bt)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"Baseline limpio (AUC = {auc:.3f})")
plt.plot([0,1], [0,1], "--", label="Azar")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC – Baseline limpio (BACKTEST)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------- 12.2 Matriz de confusión visual ----------
cm = confusion_matrix(y_backtest, clf.predict(X_backtest))

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Matriz de confusión – Baseline limpio (BACKTEST)", fontsize=11, pad=12)
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["No cancelado", "Cancelado"], rotation=45)
plt.yticks(tick_marks, ["No cancelado", "Cancelado"])

# texto por celda
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            fontsize=12,
            fontweight="bold",
            color="white" if cm[i, j] > cm.max()/2 else "black"
        )

plt.ylabel("Clase real")
plt.xlabel("Predicción")
plt.tight_layout()
plt.show()


# ---------- 12.3 Distribución de probabilidades ----------
proba_pos = y_proba_bt[y_backtest == 1]
proba_neg = y_proba_bt[y_backtest == 0]

plt.figure(figsize=(9,4))
plt.hist(proba_neg, bins=30, alpha=0.7, label="No cancelado (0)")
plt.hist(proba_pos, bins=30, alpha=0.7, label="Cancelado (1)")

plt.xlabel("Probabilidad predicha de cancelación")
plt.ylabel("Frecuencia")
plt.title("Distribución de probabilidades predichas por clase real (BACKTEST)")
plt.legend()
plt.tight_layout()
plt.show()
# BLOQUE 12.4 (Sprint 2)
# Gráficos EDA + Interpretación del baseline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Usamos df_master (dataset maestro)
df_g = df_master.copy()

# 1) Comparación Estricto vs Extendido (tasas globales)
strict_rate = df_g["is_canceled_strict"].mean()
ext_rate = df_g["order_canceled_extended"].mean()

plt.figure(figsize=(6,4))
bars = plt.bar(["Estricto", "Extendido"], [strict_rate, ext_rate], color=["#2F4F4F", "#1F77B4"])
plt.ylabel("Tasa de cancelación", fontsize=11)
plt.title("Comparación de tasa de cancelación\nTarget estricto vs extendido", fontsize=13)
plt.ylim(0, max(strict_rate, ext_rate)*1.4)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.0003, f"{height:.2%}",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()

# 2) Tasa por método de pago (estricto vs extendido)
if "main_payment_type" in df_g.columns:
    strict_by_pay = df_g.groupby("main_payment_type")["is_canceled_strict"].mean().sort_values(ascending=False)
    ext_by_pay = df_g.groupby("main_payment_type")["order_canceled_extended"].mean().reindex(strict_by_pay.index)

    labels = strict_by_pay.index.tolist()
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8,4.5))
    plt.bar(x - width/2, strict_by_pay.values, width, label="Estricto", color="#2F4F4F")
    plt.bar(x + width/2, ext_by_pay.values, width, label="Extendido", color="#1F77B4")

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=10)
    plt.ylabel("Tasa de cancelación", fontsize=11)
    plt.title("Tasa de cancelación por método de pago\nEstricto vs extendido", fontsize=13)
    plt.legend()

    # etiquetas %
    for i, v in enumerate(strict_by_pay.values):
        plt.text(i - width/2, v + 0.0003, f"{v:.2%}", ha="center", fontsize=9)
    for i, v in enumerate(ext_by_pay.values):
        plt.text(i + width/2, v + 0.0003, f"{v:.2%}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()
else:
    print("No existe main_payment_type en el dataset maestro.")


# 3) Tasa por categoría (Top 15 por volumen) – target extendido
cat_col = "product_category_name_english" if "product_category_name_english" in df_g.columns else "product_category_name"

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

plt.figure(figsize=(9,5))
plt.barh(top_cats[cat_col].astype(str), top_cats["cancel_rate"], color="#1F77B4")

plt.xlabel("Tasa de cancelación (target extendido)", fontsize=11)
plt.ylabel("Categoría de producto", fontsize=11)
plt.title("Tasa de cancelación por categoría\nTop 15 categorías por volumen", fontsize=13)

for i, v in enumerate(top_cats["cancel_rate"]):
    plt.text(v + 0.0002, i, f"{v:.2%}", va="center", fontsize=9)

plt.tight_layout()
plt.show()

# 4) Top 15 coeficientes del modelo (Regresión logística limpia)
model_lr = clf.named_steps["model"]
preproc_lr = clf.named_steps["preprocessor"]

# nombres numéricos
num_features_out = numeric_cols

# nombres categóricos (after onehot)
cat_transformer = preproc_lr.named_transformers_["cat"]
ohe = cat_transformer.named_steps["onehot"]
cat_features_out = ohe.get_feature_names_out(categorical_cols)

feature_names = np.concatenate([num_features_out, cat_features_out])
coefs = model_lr.coef_.ravel()

abs_coefs = np.abs(coefs)
idx_sorted = np.argsort(abs_coefs)[-15:]

top_features = feature_names[idx_sorted]
top_coefs = coefs[idx_sorted]

plt.figure(figsize=(9,5))
y_pos = np.arange(len(top_features))

colors = ["#1F77B4" if c > 0 else "#B22222" for c in top_coefs]
plt.barh(y_pos, top_coefs, color=colors)
plt.yticks(y_pos, top_features, fontsize=9)
plt.xlabel("Coeficiente (impacto sobre probabilidad de cancelación)", fontsize=11)
plt.title("Regresión logística – principales factores de riesgo\n(Top 15 coeficientes)", fontsize=13)
plt.axvline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.show()

# 5) Distribución de probabilidades predichas (BACKTEST)
y_proba_bt = clf.predict_proba(X_backtest)[:, 1]

proba_pos = y_proba_bt[y_backtest == 1]
proba_neg = y_proba_bt[y_backtest == 0]

plt.figure(figsize=(9,4))
plt.hist(proba_neg, bins=40, alpha=0.7, label="No cancelado (0)", color="#4F4F4F")
plt.hist(proba_pos, bins=40, alpha=0.7, label="Cancelado (1)", color="#B22222")

plt.xlabel("Probabilidad predicha de cancelación", fontsize=11)
plt.ylabel("Frecuencia", fontsize=11)
plt.title("Distribución de probabilidades predichas por clase real (BACKTEST)", fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
