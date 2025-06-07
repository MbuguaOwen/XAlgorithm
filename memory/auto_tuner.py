"""Threshold auto-tuning utilities for XAlgo."""

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import asyncio

from .memory_core import MemoryCore
from utils.metrics_server import (
    tuned_confidence_threshold,
    config_overfit_block_count,
)


class AutoTuner:
    """Analyze signal history and update regime thresholds with safeguards."""

    MIN_DELTA = 0.05
    MIN_TRADES = 50

    def __init__(self, memory: MemoryCore, min_win_rate: float = 0.6) -> None:
        self.memory = memory
        self.min_win_rate = min_win_rate

    def run(self) -> Dict[str, Dict[str, Any]]:
        records = list(self.memory.signal_memory)
        tuned: Dict[str, Dict[str, Any]] = {}
        by_regime: Dict[str, list[dict]] = {}
        for rec in records:
            regime = rec.get("features", {}).get("regime", "default")
            by_regime.setdefault(regime, []).append(rec)

        for regime, items in by_regime.items():
            confidences = []
            wins = []
            for it in items:
                conf = it.get("confidence")
                outcome = it.get("outcome", {})
                if conf is None or not outcome:
                    continue
                confidences.append(float(conf))
                wins.append(bool(outcome.get("win", False)))
            if not confidences:
                continue
            trade_count = len(confidences)
            conf_arr = np.array(confidences)
            win_arr = np.array(wins)
            win_rate = float(win_arr.mean()) if trade_count else 0.0
            prior = self.memory.tuned_configs.get(regime, {})
            prior_wr = float(prior.get("win_rate", 0))
            if win_arr.any():
                win_conf = conf_arr[win_arr]
                threshold = float(np.percentile(win_conf, 20))
            else:
                threshold = float(np.percentile(conf_arr, 80))
            new_thr = max(0.7, round(threshold, 4))

            if (
                trade_count < self.MIN_TRADES
                or win_rate - prior_wr < self.MIN_DELTA
                or win_rate < self.min_win_rate
            ):
                config_overfit_block_count.inc()
                tuned[regime] = prior
                continue

            tuned_confidence_threshold.set(new_thr)
            tuned[regime] = {"thr_buy": new_thr, "win_rate": win_rate}

        self.memory.tuned_configs = tuned
        self.memory.save_all()
        return tuned


def run_tuning_cycle(memory: MemoryCore) -> Dict[str, Dict[str, Any]]:
    """Convenience wrapper to execute a tuning pass."""
    tuner = AutoTuner(memory)
    return tuner.run()


async def schedule_tuning(
    memory: MemoryCore, meta: Optional["MetaReinforcer"] = None, interval_minutes: int = 60
) -> None:
    """Periodically run tuning cycles."""
    from .replay_engine import ReplayEngine

    while True:
        await asyncio.sleep(interval_minutes * 60)
        tuner = AutoTuner(memory)
        tuner.run()
        if meta is not None:
            engine = ReplayEngine(memory)
            score = engine.replay()
            meta.update_performance("global", score, 0.0, len(memory.signal_memory))


def get_tuned_thresholds(memory: MemoryCore, regime: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Return thresholds for a given regime with fallback to defaults."""
    tuned = memory.tuned_configs.get(regime)
    if tuned is None:
        tuned = memory.tuned_configs.get("default")
    base = defaults.get(regime, defaults.get("default", {})).copy()
    if tuned:
        base.update(tuned)
    return base


__all__ = [
    "AutoTuner",
    "run_tuning_cycle",
    "schedule_tuning",
    "get_tuned_thresholds",
]
