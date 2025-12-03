# src/gold/export_model.py

import joblib

from src.config.config import Settings


def export_best_model(best_xgb, cfg: Settings) -> str:
    """
    Exporta el modelo final entrenado usando la ruta de configuración.
    Equivalente al bloque donde usabas MODEL_OUTPUT_PATH.
    """
    ruta_salida = cfg.MODEL_OUTPUT_PATH
    joblib.dump(best_xgb, ruta_salida)

    print("Modelo final exportado en:")
    print(ruta_salida)

    return ruta_salida
