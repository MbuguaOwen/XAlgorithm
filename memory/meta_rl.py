"""Meta reinforcement logic for tuning evaluation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class _Stats:
    win_rate: float
    avg_return: float
    trades: int
    weight: float = 1.0


class MetaReinforcer:
    """Track tuning performance and adjust trust weights."""

    def __init__(self, fallback_threshold: float = 0.45, min_trades: int = 20) -> None:
        self.fallback_threshold = fallback_threshold
        self.min_trades = min_trades
        self._stats: Dict[str, _Stats] = {}

    def update_performance(self, regime: str, win_rate: float, avg_return: float, trades: int) -> None:
        """Update performance metrics for a regime."""
        if trades < self.min_trades:
            return
        rec = self._stats.get(regime)
        if rec is None:
            rec = _Stats(win_rate=win_rate, avg_return=avg_return, trades=trades)
            self._stats[regime] = rec
        else:
            rec.win_rate = win_rate
            rec.avg_return = avg_return
            rec.trades = trades
        if win_rate < self.fallback_threshold:
            rec.weight = max(0.0, rec.weight * 0.5)
        else:
            rec.weight = min(1.0, rec.weight + 0.1)

    def get_weight(self, regime: str) -> float:
        """Return current trust weight for a regime."""
        rec = self._stats.get(regime)
        return rec.weight if rec else 1.0

    def use_fallback(self, regime: str) -> bool:
        """Return True if the regime should fall back to defaults."""
        rec = self._stats.get(regime)
        return bool(rec and rec.weight == 0.0)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {k: asdict(v) for k, v in self._stats.items()}


__all__ = ["MetaReinforcer"]
