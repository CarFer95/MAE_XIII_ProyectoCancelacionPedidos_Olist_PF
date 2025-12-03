# src/reporting/run_context.py

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass
class RunContext:
    run_id: str
    base_report_dir: Path
    run_dir: Path
    figures_dir: Path
    log_path: Path
    metrics_path: Path

def create_run_context(base_report_dir: str = "reports") -> RunContext:
    """
    Crea el contexto de ejecución (carpetas y rutas) para una corrida del pipeline.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(base_report_dir)
    run_dir = base_dir / run_id
    figures_dir = run_dir / "figures"
    log_path = run_dir / "pipeline.log"
    metrics_path = run_dir / "metrics.json"

    figures_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        run_id=run_id,
        base_report_dir=base_dir,
        run_dir=run_dir,
        figures_dir=figures_dir,
        log_path=log_path,
        metrics_path=metrics_path,
    )
