from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load experiment configuration from JSON."""
    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file is not a file: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def create_run_dir(config: dict[str, Any]) -> Path:
    """Create a unique directory under runs/ for the current training run."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    experiment_name = str(config.get("experiment_name", "experiment"))
    run_dir = Path("runs") / f"{timestamp}-{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config_copy(config: dict[str, Any], run_dir: Path) -> None:
    """Persist the exact config used for a run into the run directory."""
    with (run_dir / "config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2)
        file_obj.write("\n")


def save_metrics_report(metrics: dict[str, float], run_dir: Path) -> None:
    """Write core regression metrics to a text file in the run directory."""
    with (run_dir / "metrics.txt").open("w", encoding="utf-8") as file_obj:
        file_obj.write("Validation diagnostics (regression):\n")
        file_obj.write(f"MAE: {metrics['mae']:.6f}\n")
        file_obj.write(f"MSE: {metrics['mse']:.6f}\n")
        file_obj.write(f"RMSE: {metrics['rmse']:.6f}\n")
        file_obj.write(f"R^2: {metrics['r2']:.6f}\n")
        file_obj.write(f"Pearson r: {metrics['pearson_r']:.6f}\n")
