"""Meta reinforcement logic for adaptive configs."""

from __future__ import annotations

from typing import Dict, Any

from .memory_core import MemoryCore
from utils.metrics_server import (
    meta_fallback_used_total,
    regime_trade_success_rate,
)


class MetaReinforcer:
    """Track performance of tuned configs and decide fallback usage."""

    def __init__(self, memory: MemoryCore, fallback_threshold: float = 0.55) -> None:
        self.memory = memory
        self.fallback_threshold = fallback_threshold
        self.weights: Dict[str, float] = memory.tuned_configs.get("meta_weights", {})

    def update_performance(self, regime: str, win_rate: float, avg_return: float, trade_count: int) -> None:
        weight = self.weights.get(regime, 1.0)
        if win_rate < self.fallback_threshold or trade_count == 0:
            weight = max(0.0, weight - 0.1)
        else:
            weight = min(1.0, weight + 0.05)
        self.weights[regime] = weight
        self.memory.tuned_configs["meta_weights"] = self.weights
        self.memory.save_all()
        regime_trade_success_rate.labels(regime).set(win_rate)

    def get_weight(self, regime: str) -> float:
        return self.weights.get(regime, 1.0)

    def choose_config(self, regime: str, tuned: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
        if self.get_weight(regime) < 0.5:
            meta_fallback_used_total.inc()
            return default
        merged = default.copy()
        merged.update(tuned)
        return merged


__all__ = ["MetaReinforcer"]
