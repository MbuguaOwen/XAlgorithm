import csv
import os
import logging
from datetime import datetime, timezone

LOG_FILE = "logs/signal_log.csv"
EXECUTION_LOG_FILE = "logs/execution_log.csv"
os.makedirs("logs", exist_ok=True)

BASE_FIELDS = [
    "timestamp", "entry_price", "confidence", "spread_zscore",
    "model_signal", "final_decision", "reason",
    "profit", "stop_loss_pct", "take_profit_pct"
]

def ensure_datetime(ts):
    if isinstance(ts, datetime):
        return ts
    elif isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def log_signal_event(
    timestamp,
    entry_price,
    confidence,
    spread_zscore,
    model_signal,
    final_decision,
    reason,
    profit=None,
    stop_loss_pct=None,
    take_profit_pct=None,
    **kwargs
):
    try:
        timestamp = ensure_datetime(timestamp)
        iso_time = timestamp.isoformat()

        log_data = {
            "timestamp": iso_time,
            "entry_price": round(float(entry_price), 8) if entry_price is not None else "",
            "confidence": round(float(confidence), 4) if confidence is not None else "",
            "spread_zscore": round(float(spread_zscore), 4) if spread_zscore is not None else "",
            "model_signal": model_signal if model_signal is not None else "",
            "final_decision": final_decision if final_decision is not None else "",
            "reason": reason if reason is not None else "",
            "profit": round(float(profit), 6) if profit is not None else "",
            "stop_loss_pct": round(float(stop_loss_pct), 4) if stop_loss_pct is not None else "",
            "take_profit_pct": round(float(take_profit_pct), 4) if take_profit_pct is not None else "",
        }

        for k, v in kwargs.items():
            log_data[k] = v if v is not None else ""

        extras = sorted([k for k in log_data.keys() if k not in BASE_FIELDS])
        fieldnames = BASE_FIELDS + extras

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(log_data)

    except Exception as e:
        logging.error(f"❌ Failed to log signal: {e}")

def log_execution_event(
    timestamp,
    pair,
    direction,
    entry_price,
    confidence,
    cointegration_score,
    regime,
    stop_loss,
    take_profit,
    spread_zscore,
    spread_slope
):
    try:
        timestamp = ensure_datetime(timestamp)
        iso_time = timestamp.isoformat()

        log_data = {
            "timestamp": iso_time,
            "pair": pair,
            "direction": direction,
            "entry_price": round(float(entry_price), 8),
            "confidence": round(float(confidence), 4),
            "cointegration_score": round(float(cointegration_score), 4),
            "spread_zscore": round(float(spread_zscore), 4),
            "spread_slope": round(float(spread_slope), 6),
            "regime": regime,
            "stop_loss": round(float(stop_loss), 6),
            "take_profit": round(float(take_profit), 6)
        }

        file_exists = os.path.isfile(EXECUTION_LOG_FILE)
        with open(EXECUTION_LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)

    except Exception as e:
        logging.error(f"❌ Failed to log execution: {e}")


def log_trade_exit(
    timestamp,
    asset,
    direction,
    entry_price,
    exit_price,
    pnl_pct,
    confidence_entry,
    confidence_exit,
    cointegration_entry,
    cointegration_exit,
    exit_reason,
    duration,
):
    """Log trade exit information to execution_log.csv"""
    try:
        timestamp = ensure_datetime(timestamp)
        iso_time = timestamp.isoformat()

        log_data = {
            "timestamp": iso_time,
            "asset": asset,
            "direction": direction,
            "entry_price": round(float(entry_price), 8),
            "exit_price": round(float(exit_price), 8),
            "pnl_pct": round(float(pnl_pct), 6),
            "confidence_entry": round(float(confidence_entry), 4),
            "confidence_exit": round(float(confidence_exit), 4),
            "cointegration_entry": round(float(cointegration_entry), 4),
            "cointegration_exit": round(float(cointegration_exit), 4),
            "exit_reason": exit_reason,
            "duration": round(float(duration), 2),
        }

        file_exists = os.path.isfile(EXECUTION_LOG_FILE)
        with open(EXECUTION_LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)

    except Exception as e:
        logging.error(f"❌ Failed to log trade exit: {e}")
