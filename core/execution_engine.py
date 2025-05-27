# core/execution_engine.py – Final Execution with Smart SL/TP & Enhanced Logging

import logging
from datetime import datetime, timezone
from termcolor import colored
import pytz
import os

# ────────────────────────────────────────────────────────────────────────
# 📉 Dynamic SL/TP Calculation
# ────────────────────────────────────────────────────────────────────────
def calculate_dynamic_sl_tp(spread_zscore, vol_spread, confidence, regime=None):
    base_sl = 0.3  # %
    base_tp = 0.5  # %

    vol_multiplier = min(max(vol_spread / 0.001, 0.5), 2.0)
    conf_boost = 1 + max(confidence - 0.85, 0) * 2
    regime_bias = 1.2 if regime == "trending" else 1.0

    dynamic_sl = base_sl * vol_multiplier
    dynamic_tp = base_tp * vol_multiplier * conf_boost * regime_bias

    return round(dynamic_sl, 4), round(dynamic_tp, 4)

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
