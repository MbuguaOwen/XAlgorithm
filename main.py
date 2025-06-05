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

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.filters import MLFilter
from core.feature_pipeline import generate_live_features
from core.trade_logger import log_signal_event, log_execution_event
from data.binance_ingestor import BinanceIngestor

# === Load Configuration ===
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

BEST_CONFIGS = CONFIG.get("regime_defaults", {})
MIN_Z_BUY = CONFIG.get("zscore_thresholds", {}).get("min_buy", 1.0)
MIN_Z_SELL = CONFIG.get("zscore_thresholds", {}).get("min_sell", -1.0)
MODEL_PATHS = CONFIG.get("model_paths", {})

NAIROBI_TZ = pytz.timezone("Africa/Nairobi")
reverse_pair_map = {0: "BTC", 1: "ETH"}
regime_map = {0: "bull", 1: "bear", 2: "flat"}
WINDOWS = {
    "spread": deque(maxlen=200),
    "btc": deque(maxlen=200),
    "eth": deque(maxlen=200),
    "ethbtc": deque(maxlen=200),
}
cluster_map = defaultdict(lambda: deque(maxlen=5))
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
    print(f"   • MIN_Z_BUY  ≥ {MIN_Z_BUY}")
    print(f"   • MIN_Z_SELL ≤ {MIN_Z_SELL}\n")

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
    return {
        "MASTER_CONVICTION_THRESHOLD": float(best["base_thr_sell"] if direction == -1 else best["thr_buy"]),
        "SL_PERCENT": float(best["sl_percent"]),
        "TP_PERCENT": float(best["tp_percent"])
    }

def check_trade_closed(price, direction, sl_level, tp_level):
    return (price >= tp_level or price <= sl_level) if direction == 1 else (price <= tp_level or price >= sl_level)

def compute_composite_exit_score(confidence, entry_conf, coint_score, spread_zscore):
    conf_decay = max(0, 1 - (confidence / entry_conf)) if entry_conf > 0 else 1
    coint_decay = max(0, 1 - coint_score)
    z_reversal = 1 if (spread_zscore < -0.25) or (spread_zscore > 0.25) else 0
    return 0.4 * conf_decay + 0.3 * coint_decay + 0.3 * z_reversal

# === Main Tick Logic ===
def process_tick(timestamp, btc_price, eth_price, ethbtc_price):
    timestamp = ensure_datetime(timestamp)
    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price
    features = generate_live_features(btc_price, eth_price, ethbtc_price, WINDOWS)
    if not features:
        log_signal_event(timestamp, spread, 0.0, 0.0, None, 0, "veto_feature_fail")
        return

    regime_code = regime_classifier.predict(pd.DataFrame([features]).reindex(columns=regime_classifier.model.feature_names_in_))[0]
    regime = regime_map.get(regime_code, "flat")
    features["regime"] = regime

    confidence, direction = confidence_filter.predict_with_confidence(pd.DataFrame([features]).reindex(columns=confidence_filter.model.feature_names_in_))
    direction = int(direction)
    if direction == 0:
        log_signal_event(timestamp, spread, confidence, features.get("spread_zscore", 0), 0, 0, "veto_no_trade")
        return

    coint_score, _ = cointegration_model.predict_with_confidence(pd.DataFrame([features]).reindex(columns=cointegration_model.model.feature_names_in_))
    pair_code = pair_selector.predict(pd.DataFrame([features]).reindex(columns=pair_selector.model.feature_names_in_))[0]
    selected_leg = reverse_pair_map.get(pair_code)
    entry_price = eth_price if selected_leg == "ETH" else btc_price
    pair = "ETHUSDT" if selected_leg == "ETH" else "BTCUSDT"
    live_price = entry_price

    # === Apply Hybrid Z-Score Entry Filter ===
    spread_zscore = features.get("spread_zscore", 0.0)
    if (direction == 1 and spread_zscore < MIN_Z_BUY) or (direction == -1 and spread_zscore > MIN_Z_SELL):
        log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 0, "veto_zscore_weak",
                         coint_score=coint_score, regime=regime)
        return

    # === Cooldown Check ===
    if pair in locked_until and timestamp < locked_until[pair]:
        log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 0, "veto_cooldown_active",
                         coint_score=coint_score, regime=regime)
        return

    config = get_live_config(regime, direction)
    spread_vol = features.get("spread_volatility", 0.0)

    # === Cluster Size Logic ===
    base_cluster = 5 if direction == 1 else 10
    z_adj = -1 if abs(spread_zscore) > 2.5 else (0 if abs(spread_zscore) > 1.5 else 1)
    vol_adj = 0 if spread_vol < 0.005 else (1 if spread_vol < 0.01 else 2)
    cluster_size = max(3, min(15, base_cluster + vol_adj + z_adj))

    cluster_key = (direction, pair)
    cluster = cluster_map[cluster_key]
    if cluster.maxlen != cluster_size:
        cluster_map[cluster_key] = deque(cluster, maxlen=cluster_size)
        cluster = cluster_map[cluster_key]
    cluster.append({"direction": direction, "confidence": confidence, "coint": coint_score})

    # === Existing Trade Check ===
    if pair in active_trades:
        trade = active_trades[pair]
        peak_price = max(trade["max_price"], live_price) if direction == 1 else min(trade["min_price"], live_price)
        trade["max_price"] = peak_price if direction == 1 else trade["max_price"]
        trade["min_price"] = peak_price if direction == -1 else trade["min_price"]

        entry_conf = trade.get("entry_confidence", confidence)
        reverse_cluster = reverse_cluster_map[pair]
        reverse_cluster.append({"confidence": confidence, "zscore": spread_zscore, "coint": coint_score})
        composite_score = compute_composite_exit_score(confidence, entry_conf, coint_score, spread_zscore)

        exit_reason = None
        if check_trade_closed(live_price, direction, trade["sl_level"], trade["tp_level"]):
            exit_reason = "tp_hit" if (direction == 1 and live_price >= trade["tp_level"]) or (direction == -1 and live_price <= trade["tp_level"]) else "sl_hit"
        elif composite_score > 0.65:
            exit_reason = "composite_exit"
        elif len(reverse_cluster) == reverse_cluster.maxlen and all(
            c["confidence"] < 0.6 or abs(c["zscore"]) < 0.25 or c["coint"] < 0.7 for c in reverse_cluster):
            exit_reason = "reverse_cluster_exit"

        if exit_reason:
            log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 0, exit_reason,
                             coint_score=coint_score, regime=regime)
            print(color_text(f"\n🚫 EXIT [{pair}] Reason: {exit_reason.upper()} | Price={live_price:.2f} | Score={composite_score:.2f} @ {timestamp.strftime('%H:%M:%S')}\n", "yellow"))
            del active_trades[pair]
            cluster.clear()
            reverse_cluster.clear()
            locked_until[pair] = timestamp + timedelta(seconds=15)
            return

        log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 0, "veto_trade_lock_active",
                         coint_score=coint_score, regime=regime)
        return

    # === New Trade Entry ===
    if len(cluster) == cluster.maxlen and all(
        s["direction"] == direction and s["confidence"] >= config["MASTER_CONVICTION_THRESHOLD"] and s["coint"] >= 0.8
        for s in cluster):

        sl = entry_price * (1 - config["SL_PERCENT"] / 100) if direction == 1 else entry_price * (1 + config["SL_PERCENT"] / 100)
        tp = entry_price * (1 + config["TP_PERCENT"] / 100) if direction == 1 else entry_price * (1 - config["TP_PERCENT"] / 100)

        active_trades[pair] = {
            "direction": direction,
            "entry_price": entry_price,
            "sl_level": sl,
            "tp_level": tp,
            "max_price": entry_price,
            "min_price": entry_price,
            "entry_confidence": confidence,
            "entry_z": spread_zscore,
        }

        log_execution_event(timestamp, pair, direction, entry_price, confidence, coint_score, regime,
                            sl, tp, spread_zscore, features.get("spread_slope", 0.0))

        log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 1, "signal_pass_cluster",
                         coint_score=coint_score, regime=regime, selected_leg=selected_leg,
                         entry_level=entry_price, stop_loss=sl, take_profit=tp)

        print(color_text(f"\n{['SELL','HOLD','BUY'][direction]} SIGNAL [{pair}] | Entry={entry_price:.2f} SL={sl:.2f} TP={tp:.2f} | Regime={regime} @ {timestamp.strftime('%H:%M:%S')}\n",
                         "green" if direction == 1 else "red"))
        cluster.clear()
        reverse_cluster_map[pair].clear()
        return

    log_signal_event(timestamp, spread, confidence, spread_zscore, direction, 0, "waiting_for_cluster",
                     coint_score=coint_score, regime=regime)

# === Main Entry ===
async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print_startup()
    logging.info("Engine initialized. Awaiting ticks...")
    ws_url = CONFIG.get("websocket", {}).get("binance_url")
    ingestor = BinanceIngestor(ws_url=ws_url)
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
