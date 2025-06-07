"""Market regime profiling and recall logic."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional

from .memory_core import MemoryCore


@dataclass
class RegimeOutcome:
    timestamp: datetime
    regime: str
    win: bool
    pnl_pct: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class RegimeRecall:
    """Track signal performance per market regime."""

    def __init__(self, memory: MemoryCore) -> None:
        self.memory = memory

    def log_outcome(self, outcome: RegimeOutcome) -> None:
        self.memory.update_regime(outcome.regime, {"win": outcome.win})
        self.memory.save_all()

    def get_profile(self) -> Dict[str, Any]:
        return self.memory.regime_profile


__all__ = ["RegimeOutcome", "RegimeRecall"]
