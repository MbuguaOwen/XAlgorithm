import asyncio
import subprocess
import sys

async def schedule_retrain(interval_hours: int = 24):
    """Periodically trigger model retraining."""
    while True:
        await asyncio.sleep(interval_hours * 3600)
        subprocess.call([sys.executable, "tools/retrain_all_models.py"])
