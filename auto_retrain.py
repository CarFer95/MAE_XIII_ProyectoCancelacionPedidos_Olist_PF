# auto_retrain.py

from src.config.config import Settings
from src.bronze.bronze_layer import run_bronze_layer
from src.silver.silver_layer import run_silver_layer
from src.gold.gold_layer import run_gold_layer
from src.monitoring.drift_checker import evaluate_retrain_need
from src.gold.data_prep_gold import load_master_gold


def get_last_available_month(cfg: Settings) -> str:
    """
    Obtiene el último purchase_ym disponible en el master GOLD.
    """
    df = load_master_gold()
    if "purchase_ym" not in df.columns:
        df["purchase_ym"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return df["purchase_ym"].max()


def main():
    cfg = Settings()

    print("\n=== REENTRENAMIENTO AUTOMÁTICO: INICIO ===")

    # 0) Actualizar master GOLD a partir de BRONZE (opcional, según tu arquitectura)
    #    Si prefieres, puedes primero correr solo BRONZE para incluir nuevos datos.
    df_master, master_path = run_bronze_layer(cfg)
    print("Master actualizado en:", master_path)

    # 1) Determinar el último mes disponible para monitorear
    last_month = get_last_available_month(cfg)
    print(f"[INFO] Último mes disponible en master: {last_month}")

    # 2) Evaluar si se debe reentrenar según criterios
    decision = evaluate_retrain_need(last_month, cfg)
    print(f"[MONITORING] should_retrain = {decision['should_retrain']}")
    print(f"[MONITORING] reason = {decision['reason']}")
    print(f"[MONITORING] details = {decision['details']}")

    if not decision["should_retrain"]:
        print("\n[INFO] No se reentrena el modelo. Fin del proceso.")
        return

    print("\n[INFO] Criterios de drift/freshness activados. Reentrenando modelo...")

    # 3) SILVER: features + baseline
    artifacts_silver = run_silver_layer(cfg)
    print("Métricas baseline (LR) BACKTEST:")
    print(artifacts_silver["metrics_backtest"])

    # 4) GOLD: modelos avanzados + export modelo final + guardar nuevas métricas de referencia
    artifacts_gold = run_gold_layer(cfg)
    print("Métricas XGB tuneado (FINAL TEST):")
    print(artifacts_gold["xgb_tuned_metrics_final"])

    print("\n=== REENTRENAMIENTO AUTOMÁTICO: FIN ===")


if __name__ == "__main__":
    main()
