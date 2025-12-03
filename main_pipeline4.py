# main_pipeline.py

import sys
import io
import logging
import json
import subprocess
from pathlib import Path
from datetime import datetime

from src.config.config import Settings
from src.bronze.bronze_layer import run_bronze_layer
from src.silver.silver_layer import run_silver_layer
from src.gold.gold_layer import run_gold_layer
from src.gold.monthly_scoring import score_month_from_master
from src.gold.model_comparison import (
    build_comparison_dataframe,
    save_model_comparison_figure,
)


# ======================================================
# Capturador de consola → logging
# ======================================================
class StreamToLogger(io.TextIOBase):
    def __init__(self, logger, level):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)

    def flush(self):
        pass


# ======================================================
# Generar notebook automático de evidencia
# ======================================================
def generar_notebook_evidencia(run_id: str, run_dir: Path) -> None:
    """
    Ejecuta el notebook plantilla con papermill para construir
    un reporte de evidencia (log + métricas + figuras).
    El RUN_ID se pasa por variable de entorno, no como parámetro de papermill.
    """
    import os
    import subprocess
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    template_nb = Path("notebooks/template_reporte_pipeline.ipynb")

    if not template_nb.exists():
        logger.warning(
            f"⚠️ No se encontró la plantilla {template_nb}. "
            "Se omite la generación del notebook de evidencia."
        )
        return

    output_nb = run_dir / f"reporte_pipeline_{run_id}.ipynb"

    # 👉 Ya NO usamos -p RUN_ID
    cmd = [
        "papermill",
        str(template_nb),
        str(output_nb),
    ]

    # 👉 RUN_ID se pasa como variable de entorno
    env = os.environ.copy()
    env["RUN_ID"] = run_id

    logger.info(f"📘 Generando notebook de evidencia: {output_nb}")

    try:
        subprocess.run(cmd, check=True, env=env)
        logger.info("✅ Notebook de evidencia generado correctamente.")
    except FileNotFoundError:
        logger.warning(
            "⚠️ No se encontró 'papermill'. Instálalo con: pip install papermill"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error ejecutando papermill: {e}")



# ----------------------------
# Opción 1: Solo BRONZE
# ----------------------------
def option_bronze(cfg: Settings) -> dict:
    print("\n=== OPCIÓN 1: Solo BRONZE ===")
    df_master, path = run_bronze_layer(cfg)
    print(f"[BRONZE] Master generado en: {path}")
    print(f"[BRONZE] Shape master: {df_master.shape}")

    # Devolvemos algo simple como “métrica/evidencia”
    return {
        "option": "bronze_only",
        "master_path": str(path),
        "master_shape": list(df_master.shape),
    }


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

    # Devolvemos el dict completo como “metrics”
    return {
        "option": "bronze_silver",
        "bronze": {
            "master_path": str(path),
            "master_shape": list(df_master.shape),
        },
        "silver": artifacts_silver,
    }


# ----------------------------
# Opción 3: BRONZE + SILVER + GOLD + Comparativa
# ----------------------------
def option_full_pipeline_with_comparison(cfg: Settings) -> dict:
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

    # Devolvemos TODO lo relevante como “metrics”
    return {
        "option": "full_pipeline_with_comparison",
        "bronze": {
            "master_path": str(path),
            "master_shape": list(df_master.shape),
        },
        "silver_metrics_backtest": artifacts_silver["metrics_backtest"],
        "gold_rf_metrics_backtest": artifacts_gold["rf_metrics_backtest"],
        "gold_xgb_baseline_metrics_backtest": artifacts_gold["xgb_baseline_metrics_backtest"],
        "gold_xgb_tuned_metrics_backtest": artifacts_gold["xgb_tuned_metrics_backtest"],
        "gold_xgb_tuned_metrics_final": artifacts_gold["xgb_tuned_metrics_final"],
    }


# ----------------------------
# Opción 4: Solo GOLD
# ----------------------------
def option_only_gold(cfg: Settings) -> dict:
    print("\n=== OPCIÓN 4: Solo GOLD ===")
    artifacts_gold = run_gold_layer(cfg)
    print("[GOLD] Métricas XGB Tuned (FINAL TEST):")
    print(artifacts_gold["xgb_tuned_metrics_final"])

    # Devolvemos artifacts_gold completo
    return {
        "option": "only_gold",
        "gold": artifacts_gold,
    }


# ----------------------------
# Opción 5: Simular nuevo mes
# ----------------------------
def option_simulate_month(cfg: Settings) -> dict:
    print("\n=== OPCIÓN 5: Simular nuevo mes ===")
    month = input("Ingresa el mes a evaluar (YYYY-MM o YYYYMM): ").strip()
    score_month_from_master(month, cfg)

    # No hay métricas estructuradas, devolvemos info básica
    return {
        "option": "simulate_month",
        "month": month,
    }


# ----------------------------
# Menú principal + CAPTURA + NOTEBOOK
# ----------------------------
def main() -> None:
    cfg = Settings()

    # ==========
    # CONTEXTO
    # ==========
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("reports") / run_id
    log_path = run_dir / "pipeline.log"
    metrics_path = run_dir / "metrics.json"

    run_dir.mkdir(parents=True, exist_ok=True)

    # ==========
    # LOGGING
    # ==========
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 PIPELINE CANCELACIONES - RUN_ID={run_id}")
    logger.info(f"📂 Directorio de evidencia: {run_dir}")

    # Capturar stdout/stderr
    stdout_logger = StreamToLogger(logger, logging.INFO)
    stderr_logger = StreamToLogger(logger, logging.ERROR)

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    opcion = None
    metrics_data: dict | None = None

    try:
        # ==========
        # MENÚ (igual que antes)
        # ==========
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
            metrics_data = option_bronze(cfg)

        elif opcion == "2":
            metrics_data = option_bronze_silver(cfg)

        elif opcion == "3":
            metrics_data = option_full_pipeline_with_comparison(cfg)

        elif opcion == "4":
            metrics_data = option_only_gold(cfg)

        elif opcion == "5":
            metrics_data = option_simulate_month(cfg)

        elif opcion == "0":
            print("Saliendo...")
            metrics_data = {"option": "exit"}

        else:
            print("Opción no válida.")
            metrics_data = {"option": "invalid", "input": opcion}

    finally:
        # Restaurar consola normal
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        print(f"[EVIDENCIA] Log de la ejecución: {log_path}")

        # Guardar métricas/evidencia en JSON
        if metrics_data is None:
            metrics_data = {"info": "No se generaron métricas (metrics_data=None)."}

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=4, ensure_ascii=False)

        print(f"[EVIDENCIA] Métricas/artefactos guardados en: {metrics_path}")

        # Generar notebook de evidencia
        generar_notebook_evidencia(run_id, run_dir)


if __name__ == "__main__":
    main()
