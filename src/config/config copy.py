import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Carga variables de entorno desde un archivo .env si existe
load_dotenv()

@dataclass
class Settings:
    """Configuración básica del proyecto (Metodología Medallion)."""
    RUTA_BASE: str = os.getenv("RUTA_BASE", "./data/")
    MASTER_FILENAME: str = os.getenv("MASTER_FILENAME", "orders_extended_master.csv")
    FIGURES_OUTPUT_DIR: str = os.getenv("FIGURES_OUTPUT_DIR", "./figuras_sprint3/")
    MODEL_OUTPUT_PATH: str = os.getenv("MODEL_OUTPUT_PATH", "./modelo_final_xgb.pkl")
    BASE_DRIVE: str = os.getenv("BASE_DRIVE", ".")
