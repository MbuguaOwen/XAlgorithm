#!/usr/bin/env python
# main.py – XAlgo: Simple, Conviction-Driven ML Arbitrage Engine (No Double Locking, Pro Adaptive, Cluster-Guarded, Trade Logging)

import asyncio
import logging
import pandas as pd
from collections import deque
from datetime import datetime
import pytz
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.filters import MLFilter
from core.feature_pipeline import generate_live_features
from core.execution_engine import calculate_dynamic_sl_tp
from core.trade_logger import log_signal_event, log_execution_event
from data.binance_ingestor import BinanceIngestor

# === CONFIGURABLE THRESHOLDS ===
CLUSTER_SIZE = 1  # Number of consecutive high-conviction signals required
CONVICTION_WEIGHT = 0.0
COINT_WEIGHT = 0.0
ZSCORE_WEIGHT = 0.0
SLOPE_WEIGHT = 0.0
MASTER_CONVICTION_THRESHOLD = 0.00

MAX_HOLD_SECONDS = 1200
MIN_SPREAD_MAGNITUDE = 0.0000
USER_CONFIDENCE_THRESHOLD = None
USER_COINTEGRATION_THRESHOLD = None

GATE_MODEL_PATH = "ml_model/triangular_rf_model.json"
PAIR_MODEL_PATH = "ml_model/pair_selector_model.json"
COINT_MODEL_PATH = "ml_model/cointegration_score_model.json"
REGIME_MODEL_PATH = "ml_model/regime_classifier.json"

reverse_pair_map = {0: "BTC", 1: "ETH"}
regime_map = {0: "flat", 1: "volatile", 2: "trending"}

WINDOW = deque(maxlen=200)
NAIROBI_TZ = pytz.timezone("Africa/Nairobi")

def color_text(text, color):
    colors = {"green": "\033[92m", "red": "\033[91m", "reset": "\033[0m"}
    return f"{colors[color]}{text}{colors['reset']}"

def ensure_datetime(ts):
    if isinstance(ts, datetime):
        dt_utc = ts if ts.tzinfo else pytz.utc.localize(ts)
    elif isinstance(ts, (int, float)):
        dt_utc = datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc)
    else:
        dt_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    return dt_utc.astimezone(NAIROBI_TZ)

def get_adaptive_thresholds(regime: str, volatility: float):
    if regime == "trending":
        z = 1.2 + volatility * 0.3
        conf = 0.88
        coint = 0.7 - volatility * 0.15
        slope = 0.0002
    elif regime == "volatile":
        z = 1.0 + volatility * 0.2
        conf = 0.86
        coint = 0.65 - volatility * 0.10
        slope = 0.00015
    else:
        z = 0.8 + volatility * 0.1
        conf = 0.85
        coint = 0.7
        slope = 0.0001
    return round(z, 3), round(conf, 3), round(max(min(coint, 0.75), 0.5), 3), round(slope, 5)

def check_model_features(model, features_dict, model_name):
    if hasattr(model, 'feature_names_in_'):
        expected = set(model.feature_names_in_)
        actual = set(features_dict.keys())
        missing = expected - actual
        if missing:
            logging.critical(f"🚨 [FATAL] {model_name} missing required features: {missing}")
            raise RuntimeError(f"{model_name}: Model expects features missing in current pipeline: {missing}")

def check_all_models_loaded():
    dummy = {k: 1.0 for k in [
        "spread", "spread_zscore", "vol_spread", "spread_kalman", "spread_ewma",
        "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc", "btc_vol", "eth_vol",
        "ethbtc_vol", "momentum_btc", "momentum_eth", "rolling_corr", "vol_ratio", "spread_slope"
    ]}
    for model, name in [
        (confidence_filter.model, "Confidence Model"),
        (cointegration_model.model, "Cointegration Model"),
        (pair_selector.model, "Pair Selector Model"),
        (regime_classifier.model, "Regime Classifier Model"),
    ]:
        check_model_features(model, dummy, name)

confidence_filter = MLFilter(GATE_MODEL_PATH)
pair_selector = MLFilter(PAIR_MODEL_PATH)
cointegration_model = MLFilter(COINT_MODEL_PATH)
regime_classifier = MLFilter(REGIME_MODEL_PATH)
check_all_models_loaded()

if not hasattr(sys.modules[__name__], "signal_cluster"):
    signal_cluster = deque(maxlen=CLUSTER_SIZE)
else:
    signal_cluster = getattr(sys.modules[__name__], "signal_cluster")

def process_tick(timestamp, btc_price, eth_price, ethbtc_price):
    global signal_cluster

    timestamp = ensure_datetime(timestamp)
    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price

    if abs(spread) < MIN_SPREAD_MAGNITUDE:
        log_signal_event(timestamp, spread, 0.0, 0.0, None, 0, "veto_spread_too_small")
        return

    features = generate_live_features(btc_price, eth_price, ethbtc_price, WINDOW)
    if not features:
        log_signal_event(timestamp, spread, 0.0, 0.0, None, 0, "veto_feature_fail")
        return

    spread_z = features.get("spread_zscore", 0.0)
    volatility = features.get("vol_spread", 0.001)
    spread_slope = features.get("spread_slope", 0.0)

    regime_input = pd.DataFrame([features]).reindex(columns=regime_classifier.model.feature_names_in_)
    regime_code = regime_classifier.predict(regime_input)[0]
    regime = regime_map.get(regime_code, "flat")
    features["regime"] = regime

    zscore_threshold, confidence_threshold, cointegration_threshold, slope_threshold = get_adaptive_thresholds(regime, volatility)

    gate_input = pd.DataFrame([features]).reindex(columns=confidence_filter.model.feature_names_in_)
    check_model_features(confidence_filter.model, features, "Confidence Model")
    confidence, direction = confidence_filter.predict_with_confidence(gate_input)

    coint_input = pd.DataFrame([features]).reindex(columns=cointegration_model.model.feature_names_in_)
    check_model_features(cointegration_model.model, features, "Cointegration Model")
    coint_score, _ = cointegration_model.predict_with_confidence(coint_input)

    pass_conf = 1 if confidence >= confidence_threshold else 0
    pass_coint = 1 if coint_score >= cointegration_threshold else 0
    pass_z = 1 if abs(spread_z) >= zscore_threshold else 0
    pass_slope = 1 if abs(spread_slope) >= slope_threshold else 0

    master_conviction = (
        pass_conf * CONVICTION_WEIGHT +
        pass_coint * COINT_WEIGHT +
        pass_z * ZSCORE_WEIGHT +
        pass_slope * SLOPE_WEIGHT
    )

    logging.info(
        f"[THRESHOLDS] regime={regime} | zscore={spread_z:.3f}/{zscore_threshold} | "
        f"conf={confidence:.3f}/{confidence_threshold} | coint={coint_score:.3f}/{cointegration_threshold} | "
        f"slope={spread_slope:.5f}/{slope_threshold} | master_conviction={master_conviction:.2f}"
    )

    if direction == 0 or master_conviction < MASTER_CONVICTION_THRESHOLD:
        log_signal_event(timestamp, spread, confidence, spread_z, direction, 0, "veto_master_conviction",
            coint_score=coint_score, regime=regime, slope=spread_slope)
        return

    signal_cluster.append({
        "direction": direction,
        "confidence": confidence,
        "coint": coint_score,
        "zscore": abs(spread_z),
        "slope": abs(spread_slope),
        "master_conviction": master_conviction
    })

    if len(signal_cluster) == CLUSTER_SIZE and all(
        s["direction"] == direction and s["master_conviction"] >= MASTER_CONVICTION_THRESHOLD
        for s in signal_cluster
    ):
        pair_input = pd.DataFrame([features]).reindex(columns=pair_selector.model.feature_names_in_)
        check_model_features(pair_selector.model, features, "Pair Selector Model")
        pair_code = pair_selector.predict(pair_input)[0]
        selected_leg = reverse_pair_map.get(pair_code)

        if selected_leg == "ETH":
            entry_price = eth_price
            pair = "ETHUSDT"
        else:
            entry_price = btc_price
            pair = "BTCUSDT"

        stop_loss_pct, take_profit_pct = calculate_dynamic_sl_tp(
            spread_zscore=spread_z,
            vol_spread=volatility,
            confidence=confidence,
            regime=regime
        )
        sl_level = entry_price * (1 - stop_loss_pct / 100) if direction == 1 else entry_price * (1 + stop_loss_pct / 100)
        tp_level = entry_price * (1 + take_profit_pct / 100) if direction == 1 else entry_price * (1 - take_profit_pct / 100)

        # ---- LOG EXECUTION ----
        log_execution_event(
            timestamp=timestamp, pair=pair, direction=direction, entry_price=entry_price,
            confidence=confidence, cointegration_score=coint_score, spread_zscore=spread_z, spread_slope=spread_slope,
            regime=regime, stop_loss=sl_level, take_profit=tp_level
        )

        log_signal_event(
            timestamp, spread, confidence, spread_z, direction, 1, "signal_pass_cluster",
            coint_score=coint_score, cluster=list(signal_cluster), regime=regime, slope=spread_slope,
            selected_leg=selected_leg, entry_level=entry_price, stop_loss=sl_level, take_profit=tp_level
        )

        color = "green" if direction == 1 else "red"
        label = "BUY" if direction == 1 else "SELL"
        side = "LONG" if direction == 1 else "SHORT"
        local_time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = (f"\n{label} SIGNAL [{pair}]: "
               f"Entry={entry_price:.2f} | SL={sl_level:.2f} | TP={tp_level:.2f} | "
               f"Dir={side} | Leg={selected_leg} | Time={local_time_str}\n")
        print(color_text(msg, color))
    else:
        log_signal_event(
            timestamp, spread, confidence, spread_z, direction, 0, "waiting_for_cluster",
            coint_score=coint_score, regime=regime, slope=spread_slope
        )

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("🚀 XAlgo: Conviction-Driven, No Double-Locking, Cluster-Guarded Signal Engine Starting...")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
