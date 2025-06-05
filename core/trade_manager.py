import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

from .trade_logger import log_trade_exit
from .prom_metrics import EXIT_REASON_COUNTS, TRADE_PNL


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

    def __init__(self, trade_state: TradeState, price_feed: "asyncio.Queue", timeout_seconds: int = 600):
        self.state = trade_state
        self.feed = price_feed
        self.timeout_seconds = timeout_seconds
        self._task: Optional[asyncio.Task] = None
        self._active = False

        self.trailing_active = False
        self._tp_target: Optional[float] = None
        self._ratchet_sl: Optional[float] = None

        if self.state.tp_pct > 0:
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

            ts, price, conf, coint, zscore, slope = update
            if conf is not None:
                self.state.confidence_exit = conf
            if coint is not None:
                self.state.cointegration_exit = coint

            # Track best favorable price
            if self.state.direction == 1:
                if price > self.state.best_price:
                    self.state.best_price = price
            else:
                if price < self.state.best_price:
                    self.state.best_price = price

            # --- Emergency exits (confidence/cointegration) ---
            if conf is not None and conf < 0.50:
                self.state.exit_reason = "EMERGENCY_CONFIDENCE"
                self.state.exit_price = price
            elif coint is not None and coint < 0.70:
                self.state.exit_reason = "EMERGENCY_COINTEGRATION"
                self.state.exit_price = price
            else:
                # --- Trailing TP & SL Ratchet Logic ---
                if self._tp_target is not None and self.state.exit_reason is None:
                    half_tp = self.state.entry_price + (
                        self.state.direction * 0.5 * abs(self._tp_target - self.state.entry_price)
                    )
                    if not self.trailing_active:
                        if (
                            (self.state.direction == 1 and price >= half_tp)
                            or (self.state.direction == -1 and price <= half_tp)
                        ):
                            self.trailing_active = True
                            buffer = self.state.entry_price * 0.0005  # 0.05% buffer
                            self._ratchet_sl = self.state.entry_price + self.state.direction * buffer

                    if self.trailing_active:
                        trailing_stop = self.state.best_price - (
                            self.state.direction * 0.5 * abs(self._tp_target - self.state.entry_price)
                        )
                        if self.state.direction == 1:
                            if price <= self._ratchet_sl:
                                self.state.exit_reason = "SL_RATCHETED_EXIT"
                                self.state.exit_price = price
                            elif price <= trailing_stop:
                                self.state.exit_reason = "TRAILING_TP"
                                self.state.exit_price = price
                        else:
                            if price >= self._ratchet_sl:
                                self.state.exit_reason = "SL_RATCHETED_EXIT"
                                self.state.exit_price = price
                            elif price >= trailing_stop:
                                self.state.exit_reason = "TRAILING_TP"
                                self.state.exit_price = price

                # --- Emergency reversal ---
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

            # --- Dynamic Take Profit ---
            if self.state.exit_reason is None:
                if self.state.direction == 1 and zscore <= 0 and conf is not None and conf >= self.state.confidence_entry * 0.95:
                    self.state.exit_reason = "TP_REVERSION"
                    self.state.exit_price = price
                elif self.state.direction == -1 and zscore >= 0 and conf is not None and conf >= self.state.confidence_entry * 0.95:
                    self.state.exit_reason = "TP_REVERSION"
                    self.state.exit_price = price

            # --- Adaptive Stop Loss ---
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
        pnl = ((self.state.exit_price - self.state.entry_price) / self.state.entry_price) * self.state.direction if self.state.exit_price is not None else 0.0
        self.state.exit_time = datetime.utcnow()
        duration = (self.state.exit_time - self.state.entry_time).total_seconds()
        if duration < 0.01:
            duration = 0.01
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
            duration=duration,
        )

        EXIT_REASON_COUNTS.labels(self.state.exit_reason or "UNKNOWN").inc()
        TRADE_PNL.set(round(pnl * 100, 6))

