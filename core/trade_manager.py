import asyncio
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional

from .trade_logger import log_trade_exit, log_trade_outcome
from .prom_metrics import EXIT_REASON_COUNTS, TRADE_PNL
from colorama import Fore, Style


@dataclass
class TradeState:
    asset: str
    direction: int
    entry_price: float
    entry_time: datetime
    confidence_entry: float
    cointegration_entry: float
    entry_zscore: float
    sl_pct: float = 0.0
    tp_pct: float = 0.0
    best_price: float = field(init=False)
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    confidence_exit: Optional[float] = None
    cointegration_exit: Optional[float] = None
    exit_time: Optional[datetime] = None

    def __post_init__(self):
        self.best_price = self.entry_price


class TradeManager:
    """Asynchronous manager for open trades."""

    def __init__(
        self,
        trade_state: TradeState,
        price_feed: "asyncio.Queue",
        timeout_seconds: int = 600,
        strategy_mode: str = "defensive",
        trailing_offset_pct: float = 0.002,
        trailing_enabled: bool = False,
    ):
        self.state = trade_state
        self.feed = price_feed
        self.timeout_seconds = timeout_seconds
        self.strategy_mode = strategy_mode.lower()
        self._task: Optional[asyncio.Task] = None
        self._active = False

        self.trailing_enabled = trailing_enabled or self.strategy_mode == "alpha"
        self.trailing_active = False
        self._tp_target: Optional[float] = None
        self.trailing_offset_pct = trailing_offset_pct
        self._ratchet_sl: Optional[float] = None
        self._outcome_logged: bool = False
        if self.trailing_enabled and self.state.tp_pct > 0:
            self._tp_target = self.state.entry_price * (
                1 + (self.state.direction * self.state.tp_pct) / 100
            )

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
        start_time = datetime.now(timezone.utc)
        while self._active:
            now = datetime.now(timezone.utc)
            if now - start_time > timeout:
                self.state.exit_reason = "TIMEOUT"
                self.state.exit_price = self.state.best_price
                await self._log_and_stop()
                break

            try:
                update = await asyncio.wait_for(self.feed.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            ts, price, conf, coint, zscore, slope = update
            if conf is not None:
                self.state.confidence_exit = conf
            if coint is not None:
                self.state.cointegration_exit = coint

            # Track best favorable price
            if self.state.direction == 1 and price > self.state.best_price:
                self.state.best_price = price
            elif self.state.direction == -1 and price < self.state.best_price:
                self.state.best_price = price

            # Emergency exits
            if conf is not None and conf < 0.50:
                self.state.exit_reason = "EMERGENCY_CONFIDENCE"
                self.state.exit_price = price
            elif coint is not None and coint < 0.70:
                self.state.exit_reason = "EMERGENCY_COINTEGRATION"
                self.state.exit_price = price
            else:
                # Trailing TP logic
                if self._tp_target is not None and self.state.exit_reason is None:
                    offset = price * self.trailing_offset_pct
                    if not self.trailing_active:
                        if (
                            (self.state.direction == 1 and price >= self._tp_target)
                            or (self.state.direction == -1 and price <= self._tp_target)
                        ):
                            self.trailing_active = True
                            self._ratchet_sl = self.state.entry_price + self.state.direction * self.state.entry_price * 0.0005

                    if self.trailing_active:
                        if self.state.direction == 1:
                            if price >= self._tp_target:
                                new_tp = price - offset
                                if new_tp > self._tp_target:
                                    self._tp_target = new_tp
                            elif price <= self._ratchet_sl:
                                self.state.exit_reason = "SL_RATCHETED_EXIT"
                                self.state.exit_price = price
                            elif price <= self._tp_target:
                                self.state.exit_reason = "TRAILING_TP"
                                self.state.exit_price = price
                        else:
                            if price <= self._tp_target:
                                new_tp = price + offset
                                if new_tp < self._tp_target:
                                    self._tp_target = new_tp
                            elif price >= self._ratchet_sl:
                                self.state.exit_reason = "SL_RATCHETED_EXIT"
                                self.state.exit_price = price
                            elif price >= self._tp_target:
                                self.state.exit_reason = "TRAILING_TP"
                                self.state.exit_price = price

                # Emergency reversal
                if self.state.exit_reason is None:
                    peak = self.state.best_price
                    if self.state.direction == 1 and peak > self.state.entry_price:
                        threshold = peak - 0.4 * (peak - self.state.entry_price)
                        if price <= threshold:
                            self.state.exit_reason = "EMERGENCY_REVERSAL"
                            self.state.exit_price = price
                    elif self.state.direction == -1 and peak < self.state.entry_price:
                        threshold = peak + 0.4 * (self.state.entry_price - peak)
                        if price >= threshold:
                            self.state.exit_reason = "EMERGENCY_REVERSAL"
                            self.state.exit_price = price

            # Dynamic TP
            if self.state.exit_reason is None:
                if self.state.direction == 1 and zscore <= 0 and conf and conf >= self.state.confidence_entry * 0.95:
                    self.state.exit_reason = "TP_REVERSION"
                    self.state.exit_price = price
                elif self.state.direction == -1 and zscore >= 0 and conf and conf >= self.state.confidence_entry * 0.95:
                    self.state.exit_reason = "TP_REVERSION"
                    self.state.exit_price = price

            # Adaptive SL
            if self.state.exit_reason is None:
                if conf is not None and conf < self.state.confidence_entry * 0.8:
                    self.state.exit_reason = "ADAPTIVE_SL_CONF"
                    self.state.exit_price = price
                elif (self.state.direction == 1 and slope < 0) or (self.state.direction == -1 and slope > 0):
                    self.state.exit_reason = "ADAPTIVE_SL_TREND"
                    self.state.exit_price = price

            if self.state.exit_reason:
                await self._log_and_stop()
                break

    async def _log_and_stop(self):
        self._active = False
        self.state.exit_time = datetime.now(timezone.utc)

        entry_time = self.state.entry_time
        exit_time = self.state.exit_time

        # Ensure both are timezone-aware
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        if exit_time.tzinfo is None:
            exit_time = exit_time.replace(tzinfo=timezone.utc)

        duration = (exit_time - entry_time).total_seconds()
        if duration < 0.01:
            duration = 0.01

        pnl = ((self.state.exit_price - self.state.entry_price) / self.state.entry_price) * self.state.direction if self.state.exit_price is not None else 0.0

        log_trade_exit(
            timestamp=exit_time,
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
            duration=duration,
        )

        EXIT_REASON_COUNTS.labels(self.state.exit_reason or "UNKNOWN").inc()
        TRADE_PNL.set(round(pnl * 100, 6))

