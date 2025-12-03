# src/bronze/save_master.py

import os
import pandas as pd
from src.config.config import Settings


def save_master(df: pd.DataFrame, cfg: Settings) -> str:
    """
    BLOQUE 6:
    - Guarda el dataset extendido.
    - Muestra proporción de nulos y ejemplos de columnas.
    Usa cfg.RUTA_BASE y cfg.MASTER_FILENAME para la ruta final.
    """
    out_path = os.path.join(cfg.RUTA_BASE_SILVER, cfg.MASTER_FILENAME)

    df.to_csv(out_path, index=False)

    print("Archivo guardado en:", out_path)
    print("Shape final del archivo:", df.shape)

    print("\nProporción de nulos:")
    print(df.isna().mean().round(3))

    print("\nEjemplo de columnas del dataset:")
    print(df.columns[:20])

    return out_path
