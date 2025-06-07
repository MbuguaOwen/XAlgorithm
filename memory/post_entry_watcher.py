"""Monitor trades post-entry for early exit cues."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable

from .memory_core import MemoryCore


class PostEntryWatcher:
    """Watch confidence decay after trade entry."""

    def __init__(
        self,
        memory: MemoryCore,
        drop_threshold: float = 0.3,
        drop_window: int = 30,
        check_interval: float = 2.0,
    ) -> None:
        self.memory = memory
        self.drop_threshold = drop_threshold
        self.drop_window = timedelta(seconds=drop_window)
        self.check_interval = check_interval
        self._active = False
        self._task: Optional[asyncio.Task] = None
        self._updates: list[tuple[datetime, float]] = []

    def update_confidence(self, ts: datetime, confidence: float) -> None:
        self._updates.append((ts, confidence))
        cutoff = ts - self.drop_window
        self._updates = [u for u in self._updates if u[0] >= cutoff]

    async def start(self, exit_callback: Callable[[], Awaitable[None]]) -> None:
        self._active = True
        self._task = asyncio.create_task(self._run(exit_callback))

    async def stop(self) -> None:
        self._active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self, exit_callback: Callable[[], Awaitable[None]]) -> None:
        while self._active:
            await asyncio.sleep(self.check_interval)
            if len(self._updates) < 2:
                continue
            start_conf = self._updates[0][1]
            latest_conf = self._updates[-1][1]
            if start_conf - latest_conf >= self.drop_threshold:
                self.memory.add_signal({"regret_trade": True})
                await exit_callback()
                break


__all__ = ["PostEntryWatcher"]
