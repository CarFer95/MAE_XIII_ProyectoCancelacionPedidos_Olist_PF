# src/reporting/notebook_report.py

import subprocess
import logging
from .run_context import RunContext

logger = logging.getLogger(__name__)

def generate_notebook_report(context: RunContext) -> None:
    """
    Ejecuta un notebook plantilla con papermill para generar
    un reporte (métricas + imágenes + log) de la corrida.
    """
    template_nb = "notebooks/template_reporte_pipeline.ipynb"
    output_nb = context.run_dir / f"reporte_pipeline_{context.run_id}.ipynb"

    cmd = [
        "papermill",
        template_nb,
        str(output_nb),
        "-p", "RUN_ID", context.run_id
    ]

    logger.info(f"📝 Generando notebook de reporte: {output_nb}")
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ Notebook de reporte generado correctamente.")
    except FileNotFoundError:
        logger.warning(
            "⚠️ No se encontró 'papermill'. "
            "Instálalo con: pip install papermill"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error ejecutando papermill: {e}")
