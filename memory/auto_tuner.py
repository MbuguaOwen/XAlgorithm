"""Threshold auto-tuning utilities for XAlgo."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .memory_core import MemoryCore


class AutoTuner:
    """Analyze signal history and update regime thresholds."""

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
            conf_arr = np.array(confidences)
            win_arr = np.array(wins)
            if win_arr.any():
                win_conf = conf_arr[win_arr]
                threshold = float(np.percentile(win_conf, 20))
            else:
                threshold = float(np.percentile(conf_arr, 80))
            tuned[regime] = {
                "thr_buy": max(0.7, round(threshold, 4)),
            }

        self.memory.tuned_configs = tuned
        self.memory.save_all()
        return tuned


def run_tuning_cycle(memory: MemoryCore) -> Dict[str, Dict[str, Any]]:
    """Convenience wrapper to execute a tuning pass."""
    tuner = AutoTuner(memory)
    return tuner.run()


def get_tuned_thresholds(memory: MemoryCore, regime: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Return thresholds for a given regime with fallback to defaults."""
    tuned = memory.tuned_configs.get(regime)
    if tuned is None:
        tuned = memory.tuned_configs.get("default")
    base = defaults.get(regime, defaults.get("default", {})).copy()
    if tuned:
        base.update(tuned)
    return base


__all__ = ["AutoTuner", "run_tuning_cycle", "get_tuned_thresholds"]
