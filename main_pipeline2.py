# main_pipeline.py

from src.config.config import Settings
from src.bronze.bronze_layer import run_bronze_layer
from src.silver.silver_layer import run_silver_layer
from src.gold.gold_layer import run_gold_layer
from src.gold.monthly_scoring import score_month_from_master  # <-- NUEVO


def run_bronze(cfg: Settings):
    print("\n=== Ejecutando CAPA BRONZE ===")
    df_master, out_path = run_bronze_layer(cfg)
    print("✅ BRONZE OK. Master generado en:", out_path)


def run_silver(cfg: Settings):
    print("\n=== Ejecutando CAPA SILVER ===")
    artifacts_silver = run_silver_layer(cfg)
    print("✅ SILVER OK. Métricas BACKTEST:")
    print(artifacts_silver["metrics_backtest"])


def run_gold(cfg: Settings):
    print("\n=== Ejecutando CAPA GOLD ===")
    artifacts_gold = run_gold_layer(cfg)
    print("✅ GOLD OK. Métricas XGB tuneado (FINAL TEST):")
    print(artifacts_gold["xgb_tuned_metrics_final"])


def run_month_simulation(cfg: Settings):
    print("\n=== SIMULACIÓN DE NUEVO MES ===")
    month = input("Ingresa el mes a evaluar (YYYY-MM o YYYYMM): ").strip()
    resultados = score_month_from_master(month, cfg)

    # Si quieres, aquí puedes pedir una ruta y guardar las métricas en CSV
    # o simplemente dejar que lo imprima en pantalla.


def main():
    cfg = Settings()

    print("\n==============================")
    print("   PIPELINE CANCELACIONES")
    print("==============================")
    print("1) Solo BRONZE (ingesta + master)")
    print("2) BRONZE + SILVER (baseline LR)")
    print("3) BRONZE + SILVER + GOLD (modelos avanzados)")
    print("4) Solo GOLD (usar master ya generado)")
    print("5) Simular un nuevo mes con el modelo final")  # <-- NUEVO
    print("0) Salir")
    print("==============================")

    opcion = input("Elige una opción: ")

    if opcion == "1":
        run_bronze(cfg)

    elif opcion == "2":
        run_bronze(cfg)
        run_silver(cfg)

    elif opcion == "3":
        run_bronze(cfg)
        run_silver(cfg)
        run_gold(cfg)

    elif opcion == "4":
        # Asume que BRONZE ya generó orders_extended_master.csv
        run_gold(cfg)

    elif opcion == "5":
        run_month_simulation(cfg)

    elif opcion == "0":
        print("Saliendo del pipeline...")

    else:
        print("Opción no válida.")


if __name__ == "__main__":
    main()
