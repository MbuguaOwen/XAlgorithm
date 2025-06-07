"""Order book ghost order detection."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .memory_core import MemoryCore


class GhostTracker:
    """Track ghost orders that appear then vanish."""

    def __init__(self, memory: MemoryCore) -> None:
        self.memory = memory
        self._last_book: Dict[str, Dict[float, float]] = defaultdict(dict)

    def update_order_book(self, pair: str, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
        """Process order book snapshot.

        Parameters
        ----------
        pair: trading pair symbol
        bids/asks: lists of (price, volume)
        """
        prev = self._last_book[pair]
        current = {price: vol for price, vol in bids + asks}

        # detect vanished orders
        for price, vol in prev.items():
            if price not in current:
                self.memory.record_ghost(price)
        self._last_book[pair] = current
        self.memory.save_all()


__all__ = ["GhostTracker"]
