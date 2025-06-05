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
import yaml
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.filters import MLFilter
from core.feature_pipeline import generate_live_features
from core.trade_logger import log_signal_event, log_execution_event
from core.trade_manager import TradeManager, TradeState
from core.execution_engine import display_signal_info
from data.binance_ingestor import BinanceIngestor
from core.prom_metrics import (
    CONFIDENCE_SCORE,
    COINTEGRATION_STABILITY,
    MISSED_OPPORTUNITIES,
    start_metrics_server,
)
from core.retrain_scheduler import (
    schedule_retrain,
    retrain_on_drift,
    weekly_retrain,
)

# === Load Configuration ===
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

STRATEGY_MODE = CONFIG.get("strategy_mode", "defensive").lower()

BEST_CONFIGS = CONFIG.get("regime_defaults", {})
MIN_Z_BUY = CONFIG.get("zscore_thresholds", {}).get("min_buy", 1.0)
MIN_Z_SELL = CONFIG.get("zscore_thresholds", {}).get("min_sell", -1.0)
MODEL_PATHS = CONFIG.get("model_paths", {})

# === Entry Gate Thresholds ===
ENTRY_CONFIDENCE_MIN = 0.65
ENTRY_COINTEGRATION_MIN = 0.75
ENTRY_ZSCORE_MIN = 2.0
ENTRY_ZSCORE_FLAT = 1.8
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
}
cluster_map = defaultdict(lambda: deque(maxlen=CLUSTER_SIZE))
active_trades = {}
locked_until = {}
reverse_cluster_map = defaultdict(lambda: deque(maxlen=4))

confidence_filter   = MLFilter(MODEL_PATHS.get("confidence_filter", "ml_model/triangular_rf_model.json"))
pair_selector       = MLFilter(MODEL_PATHS.get("pair_selector", "ml_model/pair_selector_model.json"))
cointegration_model = MLFilter(MODEL_PATHS.get("cointegration_model", "ml_model/cointegration_score_model.json"))
regime_classifier   = MLFilter(MODEL_PATHS.get("regime_classifier", "ml_model/regime_classifier.json"))

# === Startup Display ===
def color_text(text, color):
    colors = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m", "reset": "\033[0m"}
    return f"{colors.get(color,'')}{text}{colors['reset']}"

def print_startup():
    print(color_text("✅ XAlgo [Signal Engine Started]\n", "green"))
    print(color_text("📊 ACTIVE MODELS:", "yellow"))
    print(f"   • Confidence Filter       → {os.path.basename(MODEL_PATHS.get('confidence_filter', 'triangular_rf_model.json'))}")
    print(f"   • Pair Selector           → {os.path.basename(MODEL_PATHS.get('pair_selector', 'pair_selector_model.json'))}")
    print(f"   • Cointegration Scorer    → {os.path.basename(MODEL_PATHS.get('cointegration_model', 'cointegration_score_model.json'))}")
    print(f"   • Regime Classifier       → {os.path.basename(MODEL_PATHS.get('regime_classifier', 'regime_classifier.json'))}\n")

    print(color_text("⚙️  ENTRY FILTERS:", "yellow"))
    print(f"   • Confidence      ≥ {ENTRY_CONFIDENCE_MIN}")
    print(f"   • Cointegration   ≥ {ENTRY_COINTEGRATION_MIN}")
    print(
        f"   • Z-Score        ≥ {ENTRY_ZSCORE_MIN} (flat ≥ {ENTRY_ZSCORE_FLAT})\n"
    )

def print_shutdown():
    print(color_text("🛑 XAlgo [Signal Engine Stopped Gracefully]\n", "red"))

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
    best = BEST_CONFIGS.get(regime, BEST_CONFIGS["flat"])

    sl = float(best["sl_percent"])
    tp = float(best["tp_percent"])
    threshold = float(best["base_thr_sell"] if direction == -1 else best["thr_buy"])

    if STRATEGY_MODE == "alpha":
        sl *= 0.7
        tp *= 0.65

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

def passes_entry_gates(confidence, coint_score, zscore, slope, direction, regime):
    """Return True if signal exceeds all entry gate thresholds."""
    z_min = ENTRY_ZSCORE_FLAT if regime == "flat" else ENTRY_ZSCORE_MIN
    if direction == 1:
        slope_ok = slope > 0
    elif direction == -1:
        slope_ok = slope < 0
    else:
        slope_ok = False
    return (
        confidence >= ENTRY_CONFIDENCE_MIN
        and coint_score >= ENTRY_COINTEGRATION_MIN
        and abs(zscore) >= z_min
        and slope_ok
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

    confidence, direction = confidence_filter.predict_with_confidence(pd.DataFrame([features]).reindex(columns=confidence_filter.model.feature_names_in_))
    direction = int(direction)
    CONFIDENCE_SCORE.set(confidence)
    if direction == 0:
        log_signal_event(timestamp, spread, confidence, features.get("spread_zscore", 0), 0, 0, "veto_no_trade")
        MISSED_OPPORTUNITIES.inc()
        display_signal_info(0, 0.0, 0.0, confidence)
        return

    coint_score, _ = cointegration_model.predict_with_confidence(pd.DataFrame([features]).reindex(columns=cointegration_model.model.feature_names_in_))
    COINTEGRATION_STABILITY.set(coint_score)
    pair_code = pair_selector.predict(pd.DataFrame([features]).reindex(columns=pair_selector.model.feature_names_in_))[0]
    selected_leg = reverse_pair_map.get(pair_code)
    entry_price = eth_price if selected_leg == "ETH" else btc_price
    pair = "ETHUSDT" if selected_leg == "ETH" else "BTCUSDT"
    live_price = entry_price

    # === Primary Entry Gates ===
    spread_zscore = features.get("spread_zscore", 0.0)
    spread_slope = features.get("spread_slope", 0.0)
    if not passes_entry_gates(confidence, coint_score, spread_zscore, spread_slope, direction, regime):
        reason = []
        if confidence < ENTRY_CONFIDENCE_MIN:
            reason.append("conf")
        if coint_score < ENTRY_COINTEGRATION_MIN:
            reason.append("coint")
        z_min = ENTRY_ZSCORE_FLAT if regime == "flat" else ENTRY_ZSCORE_MIN
        if abs(spread_zscore) < z_min:
            reason.append("z")
        if not ((direction == 1 and spread_slope > 0) or (direction == -1 and spread_slope < 0)):
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
        display_signal_info(0, 0.0, 0.0, confidence)
        return

    # === Cooldown Check ===
    if pair in locked_until and timestamp < locked_until[pair]:
        log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 0, "veto_cooldown_active",
                         coint_score=coint_score, regime=regime)
        MISSED_OPPORTUNITIES.inc()
        display_signal_info(0, 0.0, 0.0, confidence)
        return

    config = get_live_config(regime, direction)
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
        display_signal_info(0, 0.0, 0.0, confidence)
        return

    # === New Trade Entry ===
    z_min = ENTRY_ZSCORE_FLAT if regime == "flat" else ENTRY_ZSCORE_MIN
    if len(cluster) == cluster.maxlen and all(
        s["direction"] == direction
        and s["confidence"] >= ENTRY_CONFIDENCE_MIN
        and s["coint"] >= ENTRY_COINTEGRATION_MIN
        and abs(s["z"]) >= z_min
        and (
            (direction == 1 and s["slope"] > 0)
            or (direction == -1 and s["slope"] < 0)
        )
        for s in cluster
    ):

        sl = entry_price * (1 - config["SL_PERCENT"] / 100) if direction == 1 else entry_price * (1 + config["SL_PERCENT"] / 100)
        tp = entry_price * (1 + config["TP_PERCENT"] / 100) if direction == 1 else entry_price * (1 - config["TP_PERCENT"] / 100)

        queue = asyncio.Queue()
        state = TradeState(
            asset=pair,
            direction=direction,
            entry_price=entry_price,
            entry_time=timestamp,
            confidence_entry=confidence,
            cointegration_entry=coint_score,
            entry_zscore=spread_zscore,
            sl_pct=config["SL_PERCENT"],
            tp_pct=config["TP_PERCENT"],
        )
        manager = TradeManager(state, queue, timeout_seconds=600, strategy_mode=STRATEGY_MODE)
        asyncio.get_running_loop().create_task(manager.start())
        queue.put_nowait((timestamp, entry_price, confidence, coint_score, spread_zscore, spread_slope))
        active_trades[pair] = manager

        log_execution_event(timestamp, pair, direction, entry_price, confidence, coint_score, regime,
                            sl, tp, spread_zscore, features.get("spread_slope", 0.0))

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
        )

        display_signal_info(direction, config["SL_PERCENT"], config["TP_PERCENT"], confidence)
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
