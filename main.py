#!/usr/bin/env python
# main.py – XAlgo: Asymmetric Conviction ML Arbitrage Engine (Strict Sells, Adaptive Buys)

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

# === CONFIGURABLE ASYMMETRIC THRESHOLDS ===
CLUSTER_SIZE_SELL = 17      # STRICT: Only act on SELL if you get 17 in a row
CLUSTER_SIZE_BUY = 4        # ADAPTIVE: Only need 4 in a row to BUY

CONVICTION_WEIGHT = 0.15
COINT_WEIGHT = 0.5
ZSCORE_WEIGHT = 0.5
SLOPE_WEIGHT = 0.15

MASTER_CONVICTION_THRESHOLD_SELL = 0.80
MASTER_CONVICTION_THRESHOLD_BUY = 0.62

MAX_HOLD_SECONDS = 1200
MIN_SPREAD_MAGNITUDE = 0.00005
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

# --- Cluster buffers per direction ---
signal_cluster_buy = deque(maxlen=CLUSTER_SIZE_BUY)
signal_cluster_sell = deque(maxlen=CLUSTER_SIZE_SELL)

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

def get_adaptive_thresholds(regime: str, volatility: float, direction: int):
    # Sell = -1, Buy = +1
    if direction == -1:
        # STRICT SELL: Raise the bar
        if regime == "trending":
            z = 1.25 + volatility * 0.45
            conf = 0.91
            coint = 0.75 - volatility * 0.16
            slope = 0.00025
        elif regime == "volatile":
            z = 1.08 + volatility * 0.28
            conf = 0.89
            coint = 0.71 - volatility * 0.12
            slope = 0.00017
        else:
            z = 0.97 + volatility * 0.19
            conf = 0.88
            coint = 0.73
            slope = 0.00013
    else:
        # Adaptive BUY: More forgiving, let more longs through
        if regime == "trending":
            z = 0.75 + volatility * 0.12
            conf = 0.78
            coint = 0.63 - volatility * 0.10
            slope = 0.00009
        elif regime == "volatile":
            z = 0.65 + volatility * 0.09
            conf = 0.76
            coint = 0.60 - volatility * 0.07
            slope = 0.00007
        else:
            z = 0.55 + volatility * 0.05
            conf = 0.74
            coint = 0.62
            slope = 0.00006
    return round(z, 3), round(conf, 3), round(max(min(coint, 0.77), 0.5), 3), round(slope, 5)

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

def process_tick(timestamp, btc_price, eth_price, ethbtc_price):
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

    # --- Confidence, Cointegration, Direction ---
    gate_input = pd.DataFrame([features]).reindex(columns=confidence_filter.model.feature_names_in_)
    check_model_features(confidence_filter.model, features, "Confidence Model")
    confidence, direction = confidence_filter.predict_with_confidence(gate_input)

    coint_input = pd.DataFrame([features]).reindex(columns=cointegration_model.model.feature_names_in_)
    check_model_features(cointegration_model.model, features, "Cointegration Model")
    coint_score, _ = cointegration_model.predict_with_confidence(coint_input)

    # --- Asymmetric thresholds ---
    zscore_threshold, confidence_threshold, cointegration_threshold, slope_threshold = get_adaptive_thresholds(
        regime, volatility, direction
    )
    master_conviction_threshold = MASTER_CONVICTION_THRESHOLD_SELL if direction == -1 else MASTER_CONVICTION_THRESHOLD_BUY
    cluster_size = CLUSTER_SIZE_SELL if direction == -1 else CLUSTER_SIZE_BUY

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
        f"slope={spread_slope:.5f}/{slope_threshold} | master_conviction={master_conviction:.2f} | direction={direction}"
    )

    if direction == 0 or master_conviction < master_conviction_threshold:
        log_signal_event(timestamp, spread, confidence, spread_z, direction, 0, "veto_master_conviction",
            coint_score=coint_score, regime=regime, slope=spread_slope)
        return

    # --- Directional Cluster Guard ---
    cluster = signal_cluster_sell if direction == -1 else signal_cluster_buy
    cluster.append({
        "direction": direction,
        "confidence": confidence,
        "coint": coint_score,
        "zscore": abs(spread_z),
        "slope": abs(spread_slope),
        "master_conviction": master_conviction
    })

    if len(cluster) == cluster_size and all(
        s["direction"] == direction and s["master_conviction"] >= master_conviction_threshold
        for s in cluster
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
            coint_score=coint_score, cluster=list(cluster), regime=regime, slope=spread_slope,
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
        cluster.clear()  # Reset the cluster to avoid double-execution
    else:
        log_signal_event(
            timestamp, spread, confidence, spread_z, direction, 0, "waiting_for_cluster",
            coint_score=coint_score, regime=regime, slope=spread_slope
        )

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("🚀 XAlgo: Asymmetric, Strict-Sell, Adaptive-Buy, Cluster-Guarded Signal Engine Starting...")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
