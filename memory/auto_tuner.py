"""Threshold auto-tuning utilities for XAlgo."""

from __future__ import annotations

from typing import Dict, Any, Optional

import logging
from pathlib import Path

import numpy as np

from .memory_core import MemoryCore
from .meta_rl import MetaReinforcer
from core.prom_metrics import (
    CONFIG_OVERFIT_BLOCK_COUNT,
    TUNED_CONFIDENCE_THRESHOLD,
)


GUARD_LOG = Path("logs/tuning_guard.log")
logger = logging.getLogger(__name__)


class AutoTuner:
    """Analyze signal history and update regime thresholds."""

    MIN_DELTA = 0.05

    def __init__(
        self,
        memory: MemoryCore,
        min_win_rate: float = 0.6,
        meta: Optional[MetaReinforcer] = None,
    ) -> None:
        self.memory = memory
        self.min_win_rate = min_win_rate
        self.meta = meta

    def run(self) -> Dict[str, Dict[str, Any]]:
        records = list(self.memory.signal_memory)
        tuned: Dict[str, Dict[str, Any]] = {}
        by_regime: Dict[str, list[dict]] = {}
        for rec in records:
            regime = rec.get("features", {}).get("regime", "default")
            by_regime.setdefault(regime, []).append(rec)

        for regime, items in by_regime.items():
            confidences: list[float] = []
            wins: list[bool] = []
            returns: list[float] = []
            for it in items:
                conf = it.get("confidence")
                outcome = it.get("outcome", {})
                if conf is None or not outcome:
                    continue
                confidences.append(float(conf))
                wins.append(bool(outcome.get("win", False)))
                returns.append(float(outcome.get("pnl_pct", 0.0)))
            if not confidences:
                continue

            profile = self.memory.regime_profile.get(regime, {})
            if profile.get("trades", 0) < 50:
                CONFIG_OVERFIT_BLOCK_COUNT.inc()
                try:
                    GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
                    with open(GUARD_LOG, "a") as f:
                        f.write(f"{regime}:blocked_low_trades\n")
                except Exception:
                    pass
                continue

            new_win_rate = float(sum(wins)) / len(wins)
            old_win_rate = profile.get("wins", 0) / max(profile.get("trades", 1), 1)
            if new_win_rate <= old_win_rate + self.MIN_DELTA:
                CONFIG_OVERFIT_BLOCK_COUNT.inc()
                try:
                    GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
                    with open(GUARD_LOG, "a") as f:
                        f.write(f"{regime}:blocked_delta\n")
                except Exception:
                    pass
                continue
            conf_arr = np.array(confidences)
            win_arr = np.array(wins)
            if win_arr.any():
                win_conf = conf_arr[win_arr]
                threshold = float(np.percentile(win_conf, 20))
            else:
                threshold = float(np.percentile(conf_arr, 80))
            thr = max(0.7, round(threshold, 4))
            tuned[regime] = {"thr_buy": thr}
            TUNED_CONFIDENCE_THRESHOLD.set(thr)

            avg_ret = float(np.mean(returns)) if returns else 0.0
            if self.meta:
                self.meta.update_performance(regime, new_win_rate, avg_ret, len(items))

        self.memory.tuned_configs = tuned
        self.memory.save_all()
        return tuned


def run_tuning_cycle(memory: MemoryCore, meta: Optional[MetaReinforcer] = None) -> Dict[str, Dict[str, Any]]:
    """Convenience wrapper to execute a tuning pass."""
    tuner = AutoTuner(memory, meta=meta)
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
