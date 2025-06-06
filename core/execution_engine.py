# core/execution_engine.py – Final Execution with Smart SL/TP & Enhanced Logging

import logging
from datetime import datetime, timezone
from termcolor import colored
from colorama import Fore, Style, init
import pytz
import os

# Control printing of HOLD signals
DISPLAY_HOLD = os.getenv("DISPLAY_HOLD", "true").lower() == "true"
_LAST_SIGNAL_DETAILS = None

# initialize colorama for cross-platform colored output
init(autoreset=True)

# ────────────────────────────────────────────────────────────────────────
# 📉 Dynamic SL/TP Calculation
# ────────────────────────────────────────────────────────────────────────
def calculate_dynamic_sl_tp(spread_zscore, vol_spread, confidence, regime=None):
    """Return dynamic SL/TP targeting R:R >= 2.5."""

    base_sl = 0.30  # % risk
    base_tp = 0.75  # % reward baseline

    # scale by volatility
    vol_multiplier = min(max(vol_spread / 0.001, 0.5), 2.0)

    dynamic_sl = base_sl * vol_multiplier
    dynamic_tp = base_tp * vol_multiplier

    # boost TP with confidence (confidence > 0.65 increases reward)
    if confidence > 0.65:
        dynamic_tp *= 1 + (confidence - 0.65) * 2

    # trending markets can stretch targets
    if regime == "trending":
        dynamic_tp *= 1.2

    # ensure minimum R:R of 2.5
    min_rr = 2.5
    if dynamic_tp / max(dynamic_sl, 1e-9) < min_rr:
        dynamic_tp = dynamic_sl * min_rr

    return round(dynamic_sl, 4), round(dynamic_tp, 4)

# ────────────────────────────────────────────────────────────────────────
# 📢 Signal Display Helper
# ────────────────────────────────────────────────────────────────────────
def display_signal_info(
    signal: int,
    sl_pct: float,
    tp_pct: float,
    confidence: float,
    pair: str | None = None,
    entry_price: float | None = None,
    sl_price: float | None = None,
    tp_price: float | None = None,
    regime: str | None = None,
    timestamp: datetime | None = None,
):
    """Print concise BUY/SELL/HOLD messages with duplicate suppression."""
    global _LAST_SIGNAL_DETAILS

    direction_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
    arrow_map = {1: "🔺", -1: "🔻", 0: "🟡"}

    ts = timestamp or datetime.now()
    ts_str = ts.strftime("%H:%M:%S")

    if signal == 0:
        if DISPLAY_HOLD:
            msg = f"[{ts_str}] {arrow_map[0]} HOLD"
            if _LAST_SIGNAL_DETAILS != msg:
                print(Fore.YELLOW + msg + Style.RESET_ALL)
                _LAST_SIGNAL_DETAILS = msg
        return

    # compute SL/TP prices if not provided
    if entry_price is not None:
        if sl_price is None:
            sl_price = (
                entry_price * (1 - sl_pct / 100)
                if signal == 1
                else entry_price * (1 + sl_pct / 100)
            )
        if tp_price is None:
            tp_price = (
                entry_price * (1 + tp_pct / 100)
                if signal == 1
                else entry_price * (1 - tp_pct / 100)
            )

    pair_fmt = pair.replace("USDT", "/USDT") if pair else ""
    price_fmt = f" @ {entry_price:.2f}" if entry_price is not None else ""
    color = Fore.GREEN if signal == 1 else Fore.RED
    line1 = f"[{ts_str}] {arrow_map[signal]} [TRADE] {direction_map.get(signal)} {pair_fmt}{price_fmt}"
    line2 = f"   → SL: {sl_price:.2f} | TP: {tp_price:.2f} | Confidence: {confidence:.2f}"
    if regime:
        line2 += f" | Regime: {regime}"

    details_key = (signal, pair_fmt, round(entry_price or 0, 4), round(confidence, 4))
    if _LAST_SIGNAL_DETAILS != details_key:
        print(color + line1 + Style.RESET_ALL)
        print(color + line2 + Style.RESET_ALL)
        _LAST_SIGNAL_DETAILS = details_key
    # Print paper, don't burn it. No guessing. Enter only when the edge is sharp. Otherwise, hold the trigger.

# ────────────────────────────────────────────────────────────────────────
# 🚀 Trade Execution Logger
# ────────────────────────────────────────────────────────────────────────
def execute_trade(
    signal_type: int,
    pair: str,
    timestamp: datetime,
    spread: float,
    price: float,
    spread_zscore=0.0,
    vol_spread=0.001,
    confidence=0.85,
    regime="flat"
):
    # Sanity check
    assert isinstance(timestamp, datetime), f"❌ Timestamp must be datetime, got {type(timestamp)}"

    # Action name and color
    action = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal_type, "UNKNOWN")
    color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(action, "white")

    # Convert to Nairobi time for clarity in logs
    try:
        nairobi_tz = pytz.timezone("Africa/Nairobi")
        local_time = timestamp.astimezone(nairobi_tz).isoformat()
    except Exception as tz_error:
        logging.warning(f"⚠️ Failed to convert timestamp to Nairobi timezone: {tz_error}")
        local_time = timestamp.isoformat()

    # Calculate adaptive SL/TP
    stop_loss_pct, take_profit_pct = calculate_dynamic_sl_tp(
        spread_zscore=spread_zscore,
        vol_spread=vol_spread,
        confidence=confidence,
        regime=regime
    )

    # ─── Console Log ───────────────────────────────────────────────────
    log_line = (
        f"🟢 EXECUTE: [{action}] {pair} @ price={price:.2f} "
        f"| spread={spread:.8f} | SL={stop_loss_pct:.2f}% | TP={take_profit_pct:.2f}% | {local_time}"
    )
    print(colored(log_line, color))

    # ─── File Log ──────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/execution_log.csv"
    new_file = not os.path.exists(log_path)

    try:
        with open(log_path, mode="a", newline="") as f:
            if new_file:
                f.write("timestamp,action,pair,price,spread,spread_zscore,stop_loss_pct,take_profit_pct,confidence,regime\n")
            f.write(f"{local_time},{action},{pair},{price:.2f},{spread:.8f},"
                    f"{spread_zscore:.4f},{stop_loss_pct:.4f},{take_profit_pct:.4f},{confidence:.4f},{regime}\n")
    except Exception as e:
        logging.error(f"❌ Failed to write to execution_log.csv: {e}")
