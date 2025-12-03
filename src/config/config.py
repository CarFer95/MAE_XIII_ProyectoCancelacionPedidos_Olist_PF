import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Cargar .env
load_dotenv()


def _parse_list_str(values: str, cast=str):
    """
    Convierte una cadena tipo 'a,b,c' → ['a','b','c']
    o con cast: '1,2,3' → [1,2,3]
    """
    if not values:
        return []
    return [cast(v.strip()) for v in values.split(",") if v.strip()]


@dataclass
class Settings:
    """
    Configuración global del proyecto.
    Lee todas las variables desde .env y las mantiene accesibles.
    """

    # ============================
    # RUTAS DEL PROYECTO
    # ============================
    RUTA_BASE: str = os.getenv("RUTA_BASE", "./data/")
    RUTA_BASE_BRONZE: str = os.getenv("RUTA_BASE_BRONZE", "./data/bronze/")
    RUTA_BASE_SILVER: str = os.getenv("RUTA_BASE_SILVER", "./data/silver/")
    RUTA_BASE_GOLD: str = os.getenv("RUTA_BASE_GOLD", "./data/gold/")
    
    MASTER_FILENAME: str = os.getenv("MASTER_FILENAME", "orders_extended_master.csv")
    FIGURES_OUTPUT_DIR: str = os.getenv("FIGURES_OUTPUT_DIR", "./data/gold/graficos/")
    MODEL_OUTPUT_PATH: str = os.getenv("MODEL_OUTPUT_PATH", "./data/gold/models/modelo_final_xgb.pkl")
    BASE_DRIVE: str = os.getenv("BASE_DRIVE", ".")
    METRICS_DIR: str = os.getenv("METRICS_DIR", "./data/gold/artifacts/metrics/")
    TRAIN_END_MONTH: str = os.getenv("TRAIN_END_MONTH", "2018-06")  # ejemplo



    # ============================
    # SPLITS TEMPORALES
    # ============================
    TRAIN_MONTHS: list = field(default_factory=lambda: 
        _parse_list_str(os.getenv("TRAIN_MONTHS", ""), cast=str)
    )

    BACKTEST_MONTHS: list = field(default_factory=lambda: 
        _parse_list_str(os.getenv("BACKTEST_MONTHS", ""), cast=str)
    )

    FINAL_TEST_MONTHS: list = field(default_factory=lambda: 
        _parse_list_str(os.getenv("FINAL_TEST_MONTHS", ""), cast=str)
    )

    # ============================
    # XGBOOST BASELINE
    # ============================
    XGB_N_ESTIMATORS: int = int(os.getenv("XGB_N_ESTIMATORS", 300))
    XGB_MAX_DEPTH: int = int(os.getenv("XGB_MAX_DEPTH", 6))
    XGB_LEARNING_RATE: float = float(os.getenv("XGB_LEARNING_RATE", 0.08))
    XGB_SUBSAMPLE: float = float(os.getenv("XGB_SUBSAMPLE", 0.8))
    XGB_COLSAMPLE_BYTREE: float = float(os.getenv("XGB_COLSAMPLE_BYTREE", 0.8))

    # ============================
    # XGBOOST TUNING SPACE
    # ============================
    XGB_TUNE_N_ESTIMATORS: list = field(default_factory=lambda:
        _parse_list_str(os.getenv("XGB_TUNE_N_ESTIMATORS", ""), cast=int)
    )
    XGB_TUNE_MAX_DEPTH: list = field(default_factory=lambda:
        _parse_list_str(os.getenv("XGB_TUNE_MAX_DEPTH", ""), cast=int)
    )
    XGB_TUNE_LEARNING_RATE: list = field(default_factory=lambda:
        _parse_list_str(os.getenv("XGB_TUNE_LEARNING_RATE", ""), cast=float)
    )
    XGB_TUNE_SUBSAMPLE: list = field(default_factory=lambda:
        _parse_list_str(os.getenv("XGB_TUNE_SUBSAMPLE", ""), cast=float)
    )
    XGB_TUNE_COLSAMPLE_BYTREE: list = field(default_factory=lambda:
        _parse_list_str(os.getenv("XGB_TUNE_COLSAMPLE_BYTREE", ""), cast=float)
    )

    # ============================
    # MÉTODOS ÚTILES
    # ============================

    def get_master_path(self):
        """Ruta completa al master table generado en BRONZE."""
        return os.path.join(self.RUTA_BASE, self.MASTER_FILENAME)

    def get_processed_dir(self):
        """Directorio base donde se guardan outputs procesados."""
        return self.RUTA_BASE

    def get_figures_dir(self):
        """Directorio para guardar gráficos GOLD (ya viene del env)."""
        return self.FIGURES_OUTPUT_DIR

    def get_xgb_baseline_params(self):
        """Devuelve los hiperparámetros del modelo baseline desde .env."""
        return {
            "n_estimators": self.XGB_N_ESTIMATORS,
            "max_depth": self.XGB_MAX_DEPTH,
            "learning_rate": self.XGB_LEARNING_RATE,
            "subsample": self.XGB_SUBSAMPLE,
            "colsample_bytree": self.XGB_COLSAMPLE_BYTREE
        }

    def get_xgb_tuning_param_grid(self):
        """Genera el espacio de búsqueda para RandomizedSearchCV."""
        grid = {
            "model__n_estimators": self.XGB_TUNE_N_ESTIMATORS,
            "model__max_depth": self.XGB_TUNE_MAX_DEPTH,
            "model__learning_rate": self.XGB_TUNE_LEARNING_RATE,
            "model__subsample": self.XGB_TUNE_SUBSAMPLE,
            "model__colsample_bytree": self.XGB_TUNE_COLSAMPLE_BYTREE,
        }

        # Limpiar clave:valor donde la lista está vacía
        return {k: v for k, v in grid.items() if len(v) > 0}

    def show(self):
        """Imprimir la configuración completa (útil para debug)."""
        print("\n====== CONFIGURACIÓN ACTUAL ======")
        for field_name, field_value in self.__dict__.items():
            print(f"{field_name}: {field_value}")
        print("=================================\n")
