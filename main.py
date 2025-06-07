#!/usr/bin/env python
# main.py – XAlgo: Master Conviction Engine with Composite Exit Scoring & Adaptive Retreat Logic

import asyncio
import logging
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime, timedelta
import pytz
import sys
import os
import signal
import atexit
from pathlib import Path
from colorama import Fore, Style, init

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Debug mode enables additional console output such as HOLD or
# invalidation messages. Set to ``True`` for verbose sessions.
debug_mode = False

# Inform execution_engine about HOLD display preference before import
os.environ["DISPLAY_HOLD"] = "true" if debug_mode else "false"

from utils.filters import MLFilter
from core.feature_pipeline import generate_live_features
from core.trade_logger import log_signal_event, log_execution_event
from core.trade_manager import TradeManager, TradeState
from core.execution_engine import display_signal_info
from data.binance_ingestor import BinanceIngestor
from core.prom_metrics import (
    MISSED_OPPORTUNITIES,
)
from utils.metrics_server import (
    confidence_gauge,
    cointegration_gauge,
    regime_gauge,
    zscore_gauge,
    zscore_slope_gauge,
    sl_gauge,
    tp_gauge,
    coint_mod_gauge,
    start_metrics_server,
)
from config_thresholds import (
    CONFIG,
    ENTRY_THRESHOLDS,
    REGIME_DEFAULTS,
    MODELS,
    USE_DYNAMIC_SL_TP,
    TRAILING_TP_ENABLED,
    TRAILING_TP_OFFSET_PCT,
    SL_TP_MODIFIERS,
)
from core.retrain_scheduler import (
    schedule_retrain,
    retrain_on_drift,
    weekly_retrain,
)
from memory import MemoryCore
from memory.auto_tuner import run_tuning_cycle, get_tuned_thresholds

init(autoreset=True)

# === Load Configuration ===
STRATEGY_MODE = CONFIG.get("strategy_mode", "defensive").lower()

BEST_CONFIGS = REGIME_DEFAULTS
MODEL_PATHS = MODELS
TRAILING_OFFSET_PCT = TRAILING_TP_OFFSET_PCT

# Initialize memory and auto-tune thresholds
MEMORY = MemoryCore()
run_tuning_cycle(MEMORY)

# === Entry Gate Thresholds ===
ENTRY_CONFIDENCE_MIN = ENTRY_THRESHOLDS.get("confidence_min", 0.65)
ENTRY_COINTEGRATION_MIN = ENTRY_THRESHOLDS.get("cointegration_min", 0.75)
ENTRY_SLOPE_MIN = ENTRY_THRESHOLDS.get("entry_slope_min", 0.01)
DYNAMIC_Z_MIN = float(os.getenv("DYNAMIC_Z_MIN", ENTRY_THRESHOLDS.get("dynamic_zscore_min", 1.5)))
TRADE_LOCK_SECONDS = 180
CLUSTER_SIZE = 9

NAIROBI_TZ = pytz.timezone("Africa/Nairobi")
reverse_pair_map = {0: "BTC", 1: "ETH"}
regime_map = {0: "bull", 1: "bear", 2: "flat"}
WINDOWS = {
    "spread": deque(maxlen=200),
    "btc": deque(maxlen=200),
    "eth": deque(maxlen=200),
    "ethbtc": deque(maxlen=200),
    "zscore": deque(maxlen=200),
}
cluster_map = defaultdict(lambda: deque(maxlen=CLUSTER_SIZE))
active_trades = {}
locked_until = {}
reverse_cluster_map = defaultdict(lambda: deque(maxlen=4))
last_output_message = None
last_skip_reason = None
last_skip_ts = None

confidence_filter   = MLFilter(MODEL_PATHS.get("confidence_filter", "ml_model/triangular_rf_model.json"))
pair_selector       = MLFilter(MODEL_PATHS.get("pair_selector", "ml_model/pair_selector_model.json"))
cointegration_model = MLFilter(MODEL_PATHS.get("cointegration_model", "ml_model/cointegration_score_model.json"))
regime_classifier   = MLFilter(MODEL_PATHS.get("regime_classifier", "ml_model/regime_classifier.json"))

# === Startup Display ===
def color_text(text, color):
    colors = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m", "reset": "\033[0m"}
    return f"{colors.get(color,'')}{text}{colors['reset']}"

def print_msg(text: str, color: str = "yellow"):
    """Print a timestamped message with duplicate suppression."""
    global last_output_message
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {text}"
    if line != last_output_message:
        print(getattr(Fore, color.upper(), "") + line + Style.RESET_ALL)
        last_output_message = line

def print_skip_msg(reason: str, text: str, timestamp: datetime, color: str = "yellow"):
    """Record skip reason without printing to terminal."""
    global last_skip_reason, last_skip_ts
    if last_skip_reason == reason and last_skip_ts:
        elapsed = (timestamp - last_skip_ts).total_seconds()
        if elapsed < 30:
            return
    else:
        elapsed = None

    suffix = f" [{int(elapsed)}s]" if elapsed is not None else ""
    print_msg(f"{text}{suffix}", color)
    last_skip_reason = reason
    last_skip_ts = timestamp

def print_startup():
    print(
        color_text(
            "✅ XAlgo Signal Engine Started – Monitoring for High-Conviction Trades...\n",
            "green",
        )
    )

def print_shutdown():
    print(color_text("🛑 XAlgo Signal Engine Stopped Gracefully\n", "red"))

def graceful_exit(*args):
    print_shutdown()
    sys.exit(0)

atexit.register(print_shutdown)
signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

# === Helpers ===
def ensure_datetime(ts):
    if isinstance(ts, datetime):
        return ts.astimezone(NAIROBI_TZ) if ts.tzinfo else pytz.utc.localize(ts).astimezone(NAIROBI_TZ)
    if isinstance(ts, (int, float)):
        return datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc).astimezone(NAIROBI_TZ)
    return datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(NAIROBI_TZ)

def get_live_config(regime, direction):
    # Use "default" fallback if regime key is not found
    best = get_tuned_thresholds(MEMORY, regime, BEST_CONFIGS)

    sl = float(best["sl_percent"])
    tp = float(best["tp_percent"])
    threshold = float(best["base_thr_sell"] if direction == -1 else best["thr_buy"])

    if STRATEGY_MODE == "alpha":
        sl *= 0.7
        tp *= 0.65

    # Enforce max TP cap
    tp = min(tp, 1.00)

    return {
        "MASTER_CONVICTION_THRESHOLD": threshold,
        "SL_PERCENT": sl,
        "TP_PERCENT": tp,
    }

def check_trade_closed(price, direction, sl_level, tp_level):
    return (price >= tp_level or price <= sl_level) if direction == 1 else (price <= tp_level or price >= sl_level)

def compute_composite_exit_score(confidence, entry_conf, coint_score, spread_zscore):
    conf_decay = max(0, 1 - (confidence / entry_conf)) if entry_conf > 0 else 1
    coint_decay = max(0, 1 - coint_score)
    z_reversal = 1 if (spread_zscore < -0.25) or (spread_zscore > 0.25) else 0
    return 0.4 * conf_decay + 0.3 * coint_decay + 0.3 * z_reversal

def passes_entry_gates(confidence, coint_score, zscore, z_slope, threshold):
    """Return True if signal exceeds adaptive entry thresholds."""
    if confidence is None or coint_score is None or zscore is None:
        return False
    return (
        confidence > threshold
        and coint_score >= ENTRY_COINTEGRATION_MIN
        and abs(zscore) >= DYNAMIC_Z_MIN
        and (z_slope is None or z_slope <= -ENTRY_SLOPE_MIN)
    )

# === Main Tick Logic ===
def process_tick(timestamp, btc_price, eth_price, ethbtc_price):
    timestamp = ensure_datetime(timestamp)
    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price
    features = generate_live_features(btc_price, eth_price, ethbtc_price, WINDOWS)
    if not features:
        log_signal_event(timestamp, spread, 0.0, 0.0, None, 0, "veto_feature_fail")
        MISSED_OPPORTUNITIES.inc()
        return

    regime_code = regime_classifier.predict(pd.DataFrame([features]).reindex(columns=regime_classifier.model.feature_names_in_))[0]
    regime = regime_map.get(regime_code, "flat")
    features["regime"] = regime
    regime_gauge.set(regime_code)

    confidence, direction = confidence_filter.predict_with_confidence(pd.DataFrame([features]).reindex(columns=confidence_filter.model.feature_names_in_))
    direction = int(direction)
    confidence_gauge.set(confidence)
    if direction == 0:
        log_signal_event(timestamp, spread, confidence, features.get("spread_zscore", 0), 0, 0, "veto_no_trade")
        MISSED_OPPORTUNITIES.inc()
        display_signal_info(0, 0.0, 0.0, confidence, timestamp=timestamp)
        return

    coint_score, _ = cointegration_model.predict_with_confidence(pd.DataFrame([features]).reindex(columns=cointegration_model.model.feature_names_in_))
    cointegration_gauge.set(coint_score)
    pair_code = pair_selector.predict(pd.DataFrame([features]).reindex(columns=pair_selector.model.feature_names_in_))[0]
    selected_leg = reverse_pair_map.get(pair_code)

    if selected_leg is None:
        # Neutral leg prediction → suppress trade output
        log_signal_event(
            timestamp,
            spread,
            confidence,
            features.get("spread_zscore", 0),
            direction,
            0,
            "veto_neutral_leg",
            coint_score=coint_score,
            regime=regime,
        )
        MISSED_OPPORTUNITIES.inc()
        # Skip trade silently when no dominant leg is detected
        return

    entry_price = eth_price if selected_leg == "ETH" else btc_price
    pair = "ETHUSDT" if selected_leg == "ETH" else "BTCUSDT"
    live_price = entry_price

    config = get_live_config(regime, direction)

    # === Primary Entry Gates ===
    spread_zscore = features.get("spread_zscore", 0.0)
    zscore_slope = features.get("zscore_slope", 0.0)
    spread_slope = features.get("spread_slope", 0.0)
    zscore_gauge.set(spread_zscore)
    zscore_slope_gauge.set(zscore_slope)
    if not passes_entry_gates(confidence, coint_score, spread_zscore, zscore_slope, config["MASTER_CONVICTION_THRESHOLD"]):
        reason = []
        if confidence <= config["MASTER_CONVICTION_THRESHOLD"]:
            reason.append("conf")
        if coint_score < 0.8:
            reason.append("coint")
        if abs(spread_zscore) < DYNAMIC_Z_MIN:
            reason.append("z")
        if zscore_slope >= 0:
            reason.append("slope")
        log_signal_event(
            timestamp,
            spread,
            confidence,
            spread_zscore,
            direction,
            0,
            "veto_" + "_".join(reason),
            coint_score=coint_score,
            regime=regime,
        )
        MISSED_OPPORTUNITIES.inc()
        cluster_map[(direction, pair)].clear()
        # Criteria not met – log silently without terminal output
        return

    # === Cooldown Check ===
    if pair in locked_until and timestamp < locked_until[pair]:
        log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 0, "veto_cooldown_active",
                         coint_score=coint_score, regime=regime)
        MISSED_OPPORTUNITIES.inc()
        return

    spread_vol = features.get("spread_volatility", 0.0)

    cluster_key = (direction, pair)
    cluster = cluster_map[cluster_key]
    cluster.append({
        "direction": direction,
        "confidence": confidence,
        "coint": coint_score,
        "z": spread_zscore,
        "slope": spread_slope,
    })

    # === Existing Trade Check ===
    if pair in active_trades:
        manager = active_trades[pair]
        manager.feed.put_nowait((timestamp, live_price, confidence, coint_score, spread_zscore, spread_slope))
        if not manager._active:
            locked_until[pair] = timestamp + timedelta(seconds=TRADE_LOCK_SECONDS)
            del active_trades[pair]
        log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 0, "veto_trade_lock_active",
                         coint_score=coint_score, regime=regime)
        MISSED_OPPORTUNITIES.inc()
        return

    # === New Trade Entry ===
    if len(cluster) == cluster.maxlen and all(
        s["direction"] == direction
        and s["confidence"] > config["MASTER_CONVICTION_THRESHOLD"]
        and s["coint"] >= 0.8
        and abs(s["z"]) >= DYNAMIC_Z_MIN
        and s["slope"] < 0
        for s in cluster
    ):

        z_mag = abs(spread_zscore)
        regime_multipliers = {
            "flat": {"tp_mult": 0.8, "sl_mult": 0.6},
            "bull": {"tp_mult": 1.2, "sl_mult": 1.1},
            "bear": {"tp_mult": 1.2, "sl_mult": 1.1},
            "neutral": {"tp_mult": 1.0, "sl_mult": 1.0},
        }
        reg_mult = regime_multipliers.get(regime, {"tp_mult": 1.0, "sl_mult": 1.0})
        mod_label = "static"
        if USE_DYNAMIC_SL_TP:
            high = SL_TP_MODIFIERS.get("high", {})
            medium = SL_TP_MODIFIERS.get("medium", {})
            low = SL_TP_MODIFIERS.get("low", {})

            if coint_score >= high.get("threshold", 0.9):
                mod_cfg = high
                mod_label = "high"
            elif coint_score >= medium.get("threshold", 0.8):
                mod_cfg = medium
                mod_label = "medium"
            else:
                mod_cfg = low
                mod_label = "low"

            sl_mod = mod_cfg.get("sl", 1.0)
            tp_mod = mod_cfg.get("tp", 1.0)

            coint_mod_gauge.set({"low": 0, "medium": 1, "high": 2}.get(mod_label, -1))

            dynamic_sl_pct = config["SL_PERCENT"] * (1 / max(confidence, 1e-6)) * reg_mult["sl_mult"] * sl_mod
            dynamic_tp_pct = config["TP_PERCENT"] * z_mag * reg_mult["tp_mult"] * tp_mod
        else:
            dynamic_sl_pct = config["SL_PERCENT"]
            dynamic_tp_pct = config["TP_PERCENT"]
            coint_mod_gauge.set(-1)

        sl = entry_price * (1 - dynamic_sl_pct / 100) if direction == 1 else entry_price * (1 + dynamic_sl_pct / 100)
        tp = entry_price * (1 + dynamic_tp_pct / 100) if direction == 1 else entry_price * (1 - dynamic_tp_pct / 100)

        sl_gauge.set(sl)
        tp_gauge.set(tp)
        print_msg(
            f"\U0001F9E0 Dynamic SL/TP set: SL={sl:.4f}, TP={tp:.4f} (coint:{mod_label})",
            "green",
        )

        queue = asyncio.Queue()
        state = TradeState(
            asset=pair,
            direction=direction,
            entry_price=entry_price,
            entry_time=timestamp,
            confidence_entry=confidence,
            cointegration_entry=coint_score,
            entry_zscore=spread_zscore,
            sl_pct=dynamic_sl_pct,
            tp_pct=dynamic_tp_pct,
        )
        manager = TradeManager(
            state,
            queue,
            timeout_seconds=600,
            strategy_mode=STRATEGY_MODE,
            trailing_offset_pct=TRAILING_OFFSET_PCT,
            trailing_enabled=TRAILING_TP_ENABLED,
        )
        asyncio.get_running_loop().create_task(manager.start())
        queue.put_nowait((timestamp, entry_price, confidence, coint_score, spread_zscore, spread_slope))
        active_trades[pair] = manager

        log_execution_event(
            timestamp,
            pair,
            direction,
            entry_price,
            confidence,
            coint_score,
            regime,
            sl,
            tp,
            spread_zscore,
            features.get("spread_slope", 0.0),
            cointegration_modifier=mod_label,
        )

        log_signal_event(
            timestamp,
            spread,
            confidence,
            spread_zscore,
            direction,
            1,
            "signal_pass_cluster",
            coint_score=coint_score,
            regime=regime,
            selected_leg=selected_leg,
            entry_level=entry_price,
            stop_loss=sl,
            take_profit=tp,
            cointegration_modifier=mod_label,
        )

        display_signal_info(
            direction,
            dynamic_sl_pct,
            dynamic_tp_pct,
            confidence,
            pair=pair,
            entry_price=entry_price,
            sl_price=sl,
            tp_price=tp,
            regime=regime,
            timestamp=timestamp,
            zscore=spread_zscore,
            coint=coint_score,
        )
        cluster.clear()
        reverse_cluster_map[pair].clear()
        locked_until[pair] = timestamp + timedelta(seconds=TRADE_LOCK_SECONDS)
        return

    log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 0, "waiting_for_cluster",
                     coint_score=coint_score, regime=regime)

# === Main Entry ===
async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_metrics_server()
    asyncio.create_task(schedule_retrain())
    asyncio.create_task(retrain_on_drift(Path("triangular_rf_drift.flag"), "triangular_rf_model.pkl"))
    asyncio.create_task(weekly_retrain("cointegration_score_model.pkl"))
    print_startup()
    logging.info("Engine initialized. Awaiting ticks...")
    ws_url = CONFIG.get("websocket", {}).get("binance_url")
    ingestor = BinanceIngestor(ws_url=ws_url)
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
