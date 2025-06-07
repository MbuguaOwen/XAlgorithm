"""Core memory management for XAlgo's adaptive modules."""

from __future__ import annotations

import json
import pickle
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Any, Optional

import asyncio
from .meta_rl import MetaReinforcer
from core.prom_metrics import REGIME_TRADE_SUCCESS_RATE

DEFAULT_GHOST_PATH = Path("ghost_heatmap.json")
DEFAULT_SIGNAL_PATH = Path("signal_memory.pkl")
DEFAULT_REGIME_PATH = Path("regime_profile.json")
DEFAULT_TUNED_PATH = Path("auto_tuned_config.json")


class MemoryCore:
    """Container for all persistent memory stores."""

    def __init__(
        self,
        ghost_path: Path = DEFAULT_GHOST_PATH,
        signal_path: Path = DEFAULT_SIGNAL_PATH,
        regime_path: Path = DEFAULT_REGIME_PATH,
        tuned_path: Path = DEFAULT_TUNED_PATH,
        signal_maxlen: int = 1000,
    ) -> None:
        self.ghost_path = ghost_path
        self.signal_path = signal_path
        self.regime_path = regime_path
        self.tuned_path = tuned_path

        self.ghost_heatmap: Dict[str, int] = {}
        self.signal_memory: Deque[dict[str, Any]] = deque(maxlen=signal_maxlen)
        self.regime_profile: Dict[str, Dict[str, Any]] = {}
        self.tuned_configs: Dict[str, Dict[str, Any]] = {}

        self._load_all()

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        self.ghost_heatmap = self._load_json(self.ghost_path, {})
        self.regime_profile = self._load_json(self.regime_path, {})
        self.tuned_configs = self._load_json(self.tuned_path, {})
        if self.signal_path.exists():
            try:
                with open(self.signal_path, "rb") as f:
                    self.signal_memory = pickle.load(f)
            except Exception:
                self.signal_memory = deque(maxlen=self.signal_memory.maxlen)
        else:
            self.signal_memory = deque(maxlen=self.signal_memory.maxlen)

    def save_all(self) -> None:
        self._save_json(self.ghost_path, self.ghost_heatmap)
        self._save_json(self.regime_path, self.regime_profile)
        self._save_json(self.tuned_path, self.tuned_configs)
        try:
            with open(self.signal_path, "wb") as f:
                pickle.dump(self.signal_memory, f)
        except Exception:
            pass

    @staticmethod
    def _load_json(path: Path, default: dict) -> dict:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return default.copy()
        return default.copy()

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------
    def record_ghost(self, price: float) -> None:
        key = f"{price:.2f}"
        self.ghost_heatmap[key] = self.ghost_heatmap.get(key, 0) + 1

    def add_signal(self, entry: dict[str, Any]) -> None:
        self.signal_memory.append(entry)

    def update_regime(self, regime: str, outcome: dict[str, Any]) -> None:
        prof = self.regime_profile.setdefault(regime, {"trades": 0, "wins": 0, "losses": 0})
        prof["trades"] += 1
        if outcome.get("win"):
            prof["wins"] += 1
        else:
            prof["losses"] += 1


async def schedule_tuning_cycle(
    memory: "MemoryCore",
    interval_minutes: int = 60,
    meta: Optional[MetaReinforcer] = None,
) -> None:
    """Periodically run auto tuning and feed results to meta reinforcement."""
    from .auto_tuner import run_tuning_cycle  # Local import to avoid circular ref
    from .replay_engine import ReplayEngine

    engine = ReplayEngine(memory)
    while True:
        run_tuning_cycle(memory, meta)
        score = engine.replay()
        if meta:
            total = sum(p.get("trades", 0) for p in memory.regime_profile.values())
            meta.update_performance("global", score, 0.0, total)
        for reg, prof in memory.regime_profile.items():
            trades = prof.get("trades", 0)
            if trades:
                wr = prof.get("wins", 0) / trades
                REGIME_TRADE_SUCCESS_RATE.labels(regime=reg).set(wr)
        await asyncio.sleep(interval_minutes * 60)


__all__ = ["MemoryCore", "schedule_tuning_cycle"]
