#!/usr/bin/env python
# main.py – XAlgo: Adaptive Asymmetric Conviction ML Arbitrage Engine (Manual Thresholds Only)

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

# === MANUAL CONFIGS: Enter your preferred values here ===
BEST_CONFIGS = {
    "bull": {
        "base_cl_sell": 25, "cl_buy": 6, "conviction_weight": 0.20, "coint_weight": 0.55,
        "zscore_weight": 0.45, "slope_weight": 0.20, "base_thr_sell": 0.92, "thr_buy": 0.72,
        "min_spread": 0.00008, "sl_percent": 0.18, "tp_percent": 0.54
    },
    "bear": {
        "base_cl_sell": 15, "cl_buy": 5, "conviction_weight": 0.16, "coint_weight": 0.51,
        "zscore_weight": 0.41, "slope_weight": 0.17, "base_thr_sell": 0.88, "thr_buy": 0.68,
        "min_spread": 0.00007, "sl_percent": 0.17, "tp_percent": 0.51
    },
    "flat": {
        "base_cl_sell": 18, "cl_buy": 5, "conviction_weight": 0.18, "coint_weight": 0.53,
        "zscore_weight": 0.43, "slope_weight": 0.19, "base_thr_sell": 0.90, "thr_buy": 0.70,
        "min_spread": 0.000075, "sl_percent": 0.18, "tp_percent": 0.52
    },
    "neutral": {
        "base_cl_sell": 17, "cl_buy": 4, "conviction_weight": 0.15, "coint_weight": 0.50,
        "zscore_weight": 0.40, "slope_weight": 0.15, "base_thr_sell": 0.85, "thr_buy": 0.65,
        "min_spread": 0.00006, "sl_percent": 0.16, "tp_percent": 0.48
    }
}

reverse_pair_map = {0: "BTC", 1: "ETH"}
regime_map = {0: "bull", 1: "bear", 2: "flat"}

WINDOW = deque(maxlen=200)
NAIROBI_TZ = pytz.timezone("Africa/Nairobi")

# --- Cluster buffers (set dynamically, accessible for backtest.py)
signal_cluster_buy = deque(maxlen=4)
signal_cluster_sell = deque(maxlen=17)

veto_counters = {
    "veto_spread_too_small": 0,
    "veto_feature_fail": 0,
    "veto_master_conviction": 0,
    "waiting_for_cluster": 0,
}

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

GATE_MODEL_PATH = "ml_model/triangular_rf_model.json"
PAIR_MODEL_PATH = "ml_model/pair_selector_model.json"
COINT_MODEL_PATH = "ml_model/cointegration_score_model.json"
REGIME_MODEL_PATH = "ml_model/regime_classifier.json"

confidence_filter = MLFilter(GATE_MODEL_PATH)
pair_selector = MLFilter(PAIR_MODEL_PATH)
cointegration_model = MLFilter(COINT_MODEL_PATH)
regime_classifier = MLFilter(REGIME_MODEL_PATH)
check_all_models_loaded()

def get_live_config(regime, direction):
    # Default to 'flat' if regime not found
    best = BEST_CONFIGS.get(regime, BEST_CONFIGS["flat"])
    return {
        "CLUSTER_SIZE": int(best.get("base_cl_sell" if direction == -1 else "cl_buy", 17 if direction == -1 else 4)),
        "CONVICTION_WEIGHT": float(best.get("conviction_weight", 0.15)),
        "COINT_WEIGHT": float(best.get("coint_weight", 0.5)),
        "ZSCORE_WEIGHT": float(best.get("zscore_weight", 0.5)),
        "SLOPE_WEIGHT": float(best.get("slope_weight", 0.15)),
        "MASTER_CONVICTION_THRESHOLD": float(best.get("base_thr_sell" if direction == -1 else "thr_buy", 0.80 if direction == -1 else 0.62)),
        "MIN_SPREAD_MAGNITUDE": float(best.get("min_spread", 0.00005)),
        "SL_PERCENT": float(best.get("sl_percent", 0.15)),
        "TP_PERCENT": float(best.get("tp_percent", 0.55)),
    }

def process_tick(timestamp, btc_price, eth_price, ethbtc_price):
    global signal_cluster_buy, signal_cluster_sell, veto_counters

    timestamp = ensure_datetime(timestamp)
    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price

    features = generate_live_features(btc_price, eth_price, ethbtc_price, WINDOW)
    if not features:
        veto_counters["veto_feature_fail"] += 1
        log_signal_event(timestamp, spread, 0.0, 0.0, None, 0, "veto_feature_fail")
        return None

    regime_input = pd.DataFrame([features]).reindex(columns=regime_classifier.model.feature_names_in_)
    regime_code = regime_classifier.predict(regime_input)[0]
    regime = regime_map.get(regime_code, "flat")
    features["regime"] = regime

    gate_input = pd.DataFrame([features]).reindex(columns=confidence_filter.model.feature_names_in_)
    check_model_features(confidence_filter.model, features, "Confidence Model")
    confidence, direction = confidence_filter.predict_with_confidence(gate_input)
    direction = int(direction)

    config = get_live_config(regime, direction)
    CLUSTER_SIZE = config["CLUSTER_SIZE"]
    CONVICTION_WEIGHT = config["CONVICTION_WEIGHT"]
    COINT_WEIGHT = config["COINT_WEIGHT"]
    ZSCORE_WEIGHT = config["ZSCORE_WEIGHT"]
    SLOPE_WEIGHT = config["SLOPE_WEIGHT"]
    MASTER_CONVICTION_THRESHOLD = config["MASTER_CONVICTION_THRESHOLD"]
    MIN_SPREAD_MAGNITUDE = config["MIN_SPREAD_MAGNITUDE"]
    SL_PERCENT = config["SL_PERCENT"]
    TP_PERCENT = config["TP_PERCENT"]

    # Re-init clusters if config changes
    if direction == -1:
        if signal_cluster_sell.maxlen != CLUSTER_SIZE:
            signal_cluster_sell = deque(maxlen=CLUSTER_SIZE)
    else:
        if signal_cluster_buy.maxlen != CLUSTER_SIZE:
            signal_cluster_buy = deque(maxlen=CLUSTER_SIZE)

    if abs(spread) < MIN_SPREAD_MAGNITUDE:
        veto_counters["veto_spread_too_small"] += 1
        log_signal_event(timestamp, spread, 0.0, 0.0, None, 0, "veto_spread_too_small")
        return None

    spread_z = features.get("spread_zscore", 0.0)
    volatility = features.get("vol_spread", 0.001)
    spread_slope = features.get("spread_slope", 0.0)

    coint_input = pd.DataFrame([features]).reindex(columns=cointegration_model.model.feature_names_in_)
    check_model_features(cointegration_model.model, features, "Cointegration Model")
    coint_score, _ = cointegration_model.predict_with_confidence(coint_input)

    # Adaptive thresholds: stricter for shorts, adaptive for buys
    def get_adaptive_thresholds(regime, volatility, direction):
        if direction == -1:
            if regime == "bull":
                z = 1.25 + volatility * 0.45
                conf = 0.91
                coint = 0.75 - volatility * 0.16
                slope = 0.00025
            elif regime == "bear":
                z = 1.00 + volatility * 0.30
                conf = 0.86
                coint = 0.70 - volatility * 0.10
                slope = 0.00014
            else:
                z = 0.85 + volatility * 0.15
                conf = 0.83
                coint = 0.72
                slope = 0.00010
        else:
            if regime == "bull":
                z = 0.68 + volatility * 0.12
                conf = 0.76
                coint = 0.64 - volatility * 0.08
                slope = 0.00007
            elif regime == "bear":
                z = 0.58 + volatility * 0.07
                conf = 0.74
                coint = 0.62 - volatility * 0.05
                slope = 0.00005
            else:
                z = 0.52 + volatility * 0.04
                conf = 0.71
                coint = 0.61
                slope = 0.00004
        return round(z, 3), round(conf, 3), round(max(min(coint, 0.77), 0.5), 3), round(slope, 5)

    zscore_threshold, confidence_threshold, cointegration_threshold, slope_threshold = get_adaptive_thresholds(regime, volatility, direction)

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

    if direction == 0 or master_conviction < MASTER_CONVICTION_THRESHOLD:
        veto_counters["veto_master_conviction"] += 1
        log_signal_event(timestamp, spread, confidence, spread_z, direction, 0, "veto_master_conviction",
            coint_score=coint_score, regime=regime, slope=spread_slope)
        return None

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

    if len(cluster) == CLUSTER_SIZE and all(
        s["direction"] == direction and s["master_conviction"] >= MASTER_CONVICTION_THRESHOLD
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

        stop_loss_pct = SL_PERCENT
        take_profit_pct = TP_PERCENT
        sl_level = entry_price * (1 - stop_loss_pct / 100) if direction == 1 else entry_price * (1 + stop_loss_pct / 100)
        tp_level = entry_price * (1 + take_profit_pct / 100) if direction == 1 else entry_price * (1 - take_profit_pct / 100)

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

        # --- For backtest: return trade info ---
        return {
            "timestamp": timestamp,
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "confidence": confidence,
            "cointegration_score": coint_score,
            "spread_zscore": spread_z,
            "spread_slope": spread_slope,
            "regime": regime,
            "stop_loss": sl_level,
            "take_profit": tp_level,
            "pnl": None,
        }
    else:
        veto_counters["waiting_for_cluster"] += 1
        log_signal_event(
            timestamp, spread, confidence, spread_z, direction, 0, "waiting_for_cluster",
            coint_score=coint_score, regime=regime, slope=spread_slope
        )
        return None

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("🚀 XAlgo: Dynamic, Regime-Adaptive, Cluster-Guarded Signal Engine Starting...")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
