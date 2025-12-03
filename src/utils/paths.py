from pathlib import Path
from src.config.config import Settings

def get_paths():
    cfg = Settings()
    base = Path(cfg.RUTA_BASE)
    base_bronze = Path(cfg.RUTA_BASE_BRONZE)
    base_silver = Path(cfg.RUTA_BASE_SILVER)
    base_gold = Path(cfg.RUTA_BASE_GOLD)
    return {
        "base": base_bronze,
        "raw": base_silver,  # en este caso los CSV están en la misma carpeta
        "processed": base_silver,
        "figures": Path(cfg.FIGURES_OUTPUT_DIR),
        "model": Path(cfg.MODEL_OUTPUT_PATH),
    }
