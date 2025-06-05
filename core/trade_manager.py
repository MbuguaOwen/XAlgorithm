import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Callable, Optional

from .trade_logger import log_trade_exit


@dataclass
class TradeState:
    asset: str
    direction: int
    entry_price: float
    entry_time: datetime
    confidence_entry: float
    cointegration_entry: float
    sl_pct: float = 0.0019
    tp_pct: float = 0.0061
    best_price: float = field(init=False)
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    confidence_exit: Optional[float] = None
    cointegration_exit: Optional[float] = None

    def __post_init__(self):
        self.best_price = self.entry_price


class TradeManager:
    """Asynchronous manager for open trades."""

    def __init__(self, trade_state: TradeState, price_feed: "asyncio.Queue", timeout_seconds: int = 600):
        self.state = trade_state
        self.feed = price_feed
        self.timeout_seconds = timeout_seconds
        self._task: Optional[asyncio.Task] = None
        self._active = False

    async def start(self):
        self._active = True
        self._task = asyncio.create_task(self._run())
        return self._task

    async def stop(self):
        self._active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        timeout = timedelta(seconds=self.timeout_seconds)
        start_time = datetime.utcnow()
        while self._active:
            now = datetime.utcnow()
            if now - start_time > timeout:
                self.state.exit_reason = "TIMEOUT"
                self.state.exit_price = self.state.best_price
                await self._log_and_stop()
                break

            try:
                update = await asyncio.wait_for(self.feed.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            ts, price, conf, coint = update
            if conf is not None:
                self.state.confidence_exit = conf
            if coint is not None:
                self.state.cointegration_exit = coint

            # Track best price
            if self.state.direction == 1:
                if price > self.state.best_price:
                    self.state.best_price = price
            else:
                if price < self.state.best_price:
                    self.state.best_price = price

            # Take-profit / Stop-loss
            sl_level = self.state.entry_price * (
                1 - self.state.sl_pct if self.state.direction == 1 else 1 + self.state.sl_pct
            )
            tp_level = self.state.entry_price * (
                1 + self.state.tp_pct if self.state.direction == 1 else 1 - self.state.tp_pct
            )

            if self.state.direction == 1 and price <= sl_level:
                self.state.exit_reason = "SL_HIT"
                self.state.exit_price = price
            elif self.state.direction == 1 and price >= tp_level:
                self.state.exit_reason = "TP_HIT"
                self.state.exit_price = price
            elif self.state.direction == -1 and price >= sl_level:
                self.state.exit_reason = "SL_HIT"
                self.state.exit_price = price
            elif self.state.direction == -1 and price <= tp_level:
                self.state.exit_reason = "TP_HIT"
                self.state.exit_price = price

            # Confidence/Coint decay
            if self.state.exit_reason is None:
                if conf is not None and conf < 0.50:
                    self.state.exit_reason = "CONFIDENCE_DROP"
                    self.state.exit_price = price
                elif coint is not None and coint < 0.70:
                    self.state.exit_reason = "COINTEGRATION_FAIL"
                    self.state.exit_price = price

            # Reversal from peak
            if self.state.exit_reason is None:
                peak = self.state.best_price
                if self.state.direction == 1:
                    if peak > self.state.entry_price:
                        if price <= peak * (1 - 0.4):
                            self.state.exit_reason = "REVERSAL_EXIT"
                            self.state.exit_price = price
                else:
                    if peak < self.state.entry_price:
                        if price >= peak * (1 + 0.4):
                            self.state.exit_reason = "REVERSAL_EXIT"
                            self.state.exit_price = price

            if self.state.exit_reason:
                await self._log_and_stop()
                break

    async def _log_and_stop(self):
        self._active = False
        pnl = ((self.state.exit_price - self.state.entry_price) / self.state.entry_price) * self.state.direction if self.state.exit_price is not None else 0.0
        log_trade_exit(
            timestamp=datetime.utcnow(),
            asset=self.state.asset,
            direction=self.state.direction,
            entry_price=self.state.entry_price,
            exit_price=self.state.exit_price if self.state.exit_price is not None else self.state.entry_price,
            pnl_pct=round(pnl * 100, 6),
            confidence_entry=self.state.confidence_entry,
            confidence_exit=self.state.confidence_exit if self.state.confidence_exit is not None else 0.0,
            cointegration_entry=self.state.cointegration_entry,
            cointegration_exit=self.state.cointegration_exit if self.state.cointegration_exit is not None else 0.0,
            exit_reason=self.state.exit_reason or "UNKNOWN",
        )

