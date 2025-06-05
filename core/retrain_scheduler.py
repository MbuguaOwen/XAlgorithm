import asyncio
import subprocess
import sys
from pathlib import Path


def _run_command(cmd):
    try:
        subprocess.check_call(cmd)
    except Exception:
        pass

async def schedule_retrain(interval_hours: int = 24):
    """Periodically trigger model retraining."""
    while True:
        await asyncio.sleep(interval_hours * 3600)
        _run_command([sys.executable, "tools/retrain_all_models.py"])


async def retrain_on_drift(drift_flag: Path, model_name: str):
    """Retrain a specific model when a drift flag file exists."""
    while True:
        if drift_flag.exists():
            _run_command([sys.executable, "tools/retrain_all_models.py", model_name])
            _run_command(["git", "add", "ml_model"])
            _run_command(["git", "commit", "-m", f"Auto retrain {model_name} due to drift"])
            drift_flag.unlink(missing_ok=True)
        await asyncio.sleep(3600)


async def weekly_retrain(model_name: str, interval_days: int = 7):
    """Retrain a model on a weekly schedule."""
    while True:
        await asyncio.sleep(interval_days * 86400)
        _run_command([sys.executable, "tools/retrain_all_models.py", model_name])
        _run_command(["git", "add", "ml_model"])
        _run_command(["git", "commit", "-m", f"Weekly retrain {model_name}"])
