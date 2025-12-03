# main_pipeline.py

from src.config.config import Settings
from src.bronze.bronze_layer import run_bronze_layer
from src.silver.silver_layer import run_silver_layer
from src.gold.gold_layer import run_gold_layer
from src.gold.monthly_scoring import score_month_from_master

from src.reporting.run_context import create_run_context
from src.reporting.logger_config import setup_logging
from src.reporting.metrics import save_metrics
from src.reporting.notebook_report import generate_notebook_report


from src.gold.model_comparison import (
    build_comparison_dataframe,
    save_model_comparison_figure,
)


# ----------------------------
# Opción 1: Solo BRONZE
# ----------------------------
def option_bronze(cfg: Settings) -> None:
    print("\n=== OPCIÓN 1: Solo BRONZE ===")
    df_master, path = run_bronze_layer(cfg)
    print(f"[BRONZE] Master generado en: {path}")
    print(f"[BRONZE] Shape master: {df_master.shape}")


# ----------------------------
# Opción 2: BRONZE + SILVER
# ----------------------------
def option_bronze_silver(cfg: Settings) -> dict:
    print("\n=== OPCIÓN 2: BRONZE + SILVER ===")
    df_master, path = run_bronze_layer(cfg)
    print(f"[BRONZE] Master generado en: {path}")
    print(f"[BRONZE] Shape master: {df_master.shape}")

    artifacts_silver = run_silver_layer(cfg)
    print("[SILVER] Métricas BACKTEST (LogReg):")
    print(artifacts_silver["metrics_backtest"])
    return artifacts_silver


# ----------------------------
# Opción 3: BRONZE + SILVER + GOLD
#         + Comparativa de modelos
# ----------------------------
def option_full_pipeline_with_comparison(cfg: Settings) -> None:
    print("\n=== OPCIÓN 3: BRONZE + SILVER + GOLD + COMPARATIVA ===")

    # 1) BRONZE
    df_master, path = run_bronze_layer(cfg)
    print(f"[BRONZE] Master generado en: {path}")
    print(f"[BRONZE] Shape master: {df_master.shape}")

    # 2) SILVER
    artifacts_silver = run_silver_layer(cfg)
    print("[SILVER] Métricas BACKTEST (LogReg):")
    print(artifacts_silver["metrics_backtest"])

    # 3) GOLD
    artifacts_gold = run_gold_layer(cfg)
    print("[GOLD] Métricas XGB Tuned (FINAL TEST):")
    print(artifacts_gold["xgb_tuned_metrics_final"])

    # 4) Comparativa de modelos en BACKTEST
    df_comp = build_comparison_dataframe(
        logreg_metrics=artifacts_silver["metrics_backtest"],
        rf_metrics=artifacts_gold["rf_metrics_backtest"],
        xgb_baseline_metrics=artifacts_gold["xgb_baseline_metrics_backtest"],
        xgb_tuned_metrics=artifacts_gold["xgb_tuned_metrics_backtest"],
    )

    save_model_comparison_figure(
        df=df_comp,
        cfg=cfg,
        filename="comparacion_modelos_backtest.png",
    )

    print("\n📊 Comparativa de modelos generada y guardada como imagen.\n")


# ----------------------------
# Opción 4: Solo GOLD
# ----------------------------
def option_only_gold(cfg: Settings) -> dict:
    print("\n=== OPCIÓN 4: Solo GOLD ===")
    artifacts_gold = run_gold_layer(cfg)
    print("[GOLD] Métricas XGB Tuned (FINAL TEST):")
    print(artifacts_gold["xgb_tuned_metrics_final"])
    return artifacts_gold


# ----------------------------
# Opción 5: Simular nuevo mes
# ----------------------------
def option_simulate_month(cfg: Settings) -> None:
    print("\n=== OPCIÓN 5: Simular nuevo mes ===")
    month = input("Ingresa el mes a evaluar (YYYY-MM o YYYYMM): ").strip()
    score_month_from_master(month, cfg)


# ----------------------------
# Menú principal
# ----------------------------
def main() -> None:
    cfg = Settings()

    print("\n==============================")
    print("   PIPELINE CANCELACIONES")
    print("==============================")
    print("1) Solo BRONZE")
    print("2) BRONZE + SILVER")
    print("3) BRONZE + SILVER + GOLD + Comparativa")
    print("4) Solo GOLD")
    print("5) Simular nuevo mes")
    print("0) Salir")
    print("==============================")

    opcion = input("Elige una opción: ").strip()

    if opcion == "1":
        option_bronze(cfg)

    elif opcion == "2":
        option_bronze_silver(cfg)

    elif opcion == "3":
        option_full_pipeline_with_comparison(cfg)

    elif opcion == "4":
        option_only_gold(cfg)

    elif opcion == "5":
        option_simulate_month(cfg)

    elif opcion == "0":
        print("Saliendo...")

    else:
        print("Opción no válida.")


if __name__ == "__main__":
    main()
