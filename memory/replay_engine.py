"""Simulate historical signals using current logic."""

from __future__ import annotations

from typing import Dict, Any

from .memory_core import MemoryCore


class ReplayEngine:
    """Replay past signals and compute recall metrics."""

    def __init__(self, memory: MemoryCore) -> None:
        self.memory = memory

    def replay(self, thresholds: Dict[str, float] | None = None, regime: str | None = None) -> float:
        records = list(self.memory.signal_memory)
        if regime is not None:
            records = [r for r in records if r.get("features", {}).get("regime") == regime]
        if not records:
            return 0.0
        thresholds = thresholds or {}
        conf_thr = thresholds.get("confidence", 0.7)
        z_thr = thresholds.get("zscore", 1.5)
        total = 0
        correct = 0
        for rec in records:
            conf = rec.get("confidence")
            z = rec.get("features", {}).get("spread_zscore")
            outcome = rec.get("outcome", {})
            if conf is None or z is None or not outcome:
                continue
            pred = conf > conf_thr and abs(float(z)) >= z_thr
            if pred and outcome.get("win"):
                correct += 1
            total += 1
        return correct / total if total else 0.0


__all__ = ["ReplayEngine"]
