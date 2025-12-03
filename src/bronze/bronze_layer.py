# src/bronze/bronze_layer.py

from typing import Tuple
import pandas as pd

from src.config.config import Settings
from src.bronze.load_raw import validate_and_get_files, load_raw_tables
from src.bronze.targets import add_targets
from src.bronze.master_table import build_master_table
from src.bronze.features import create_features
from src.bronze.save_master import save_master


def run_bronze_layer(cfg: Settings | None = None) -> Tuple[pd.DataFrame, str]:
    """
    Ejecuta de punta a punta la capa BRONZE:
    - Valida y carga CSV.
    - Crea targets.
    - Construye Master Table.
    - Genera features.
    - Guarda orders_extended_master.csv (o el nombre de cfg.MASTER_FILENAME).

    Devuelve:
    - df_master: DataFrame final extendido.
    - out_path: ruta del archivo guardado.
    """
    if cfg is None:
        cfg = Settings()

    # BLOQUE 1–2: Validar archivos y cargar tablas
    files = validate_and_get_files(cfg)
    tables = load_raw_tables(files)

    # BLOQUE 3: Targets
    orders_with_targets = add_targets(tables["orders"])
    tables["orders"] = orders_with_targets

    # BLOQUE 4: Master Table
    df_master = build_master_table(
        orders=tables["orders"],
        customers=tables["customers"],
        order_items=tables["order_items"],
        order_payments=tables["order_payments"],
        reviews=tables["reviews"],
        products=tables["products"],
        sellers=tables["sellers"],
        cat_trad=tables["cat_trad"],
    )

    # BLOQUE 5: Features
    df_master = create_features(
        df=df_master,
        orders=tables["orders"],
        order_items=tables["order_items"],
        geo=tables["geo"],
        customers=tables["customers"],
        sellers=tables["sellers"],
    )

    # BLOQUE 6: Guardar master
    out_path = save_master(df_master, cfg)

    return df_master, out_path


if __name__ == "__main__":
    cfg = Settings()
    df_master, out_path = run_bronze_layer(cfg)
    print("\n Capa BRONZE ejecutada correctamente.")
    print("   Archivo final:", out_path)
