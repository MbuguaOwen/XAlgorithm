#!/usr/bin/env python
# main.py – XAlgo: Master Conviction, Reversal, & Trade Lock Engine (Production Grade)

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
from core.trade_logger import log_signal_event, log_execution_event
from data.binance_ingestor import BinanceIngestor

# === ONLY 4 MANUAL THRESHOLDS PER REGIME ===
BEST_CONFIGS = {
    "bull":    {"base_thr_sell": 0.98, "thr_buy": 0.77, "sl_percent": 0.19, "tp_percent": 0.61},
    "bear":    {"base_thr_sell": 0.98, "thr_buy": 0.77, "sl_percent": 0.19, "tp_percent": 0.61},
    "flat":    {"base_thr_sell": 0.90, "thr_buy": 0.70, "sl_percent": 0.20, "tp_percent": 0.81},
    "neutral": {"base_thr_sell": 0.90, "thr_buy": 0.70, "sl_percent": 0.20, "tp_percent": 0.81},
}

reverse_pair_map = {0: "BTC", 1: "ETH"}
regime_map = {0: "bull", 1: "bear", 2: "flat"}
NAIROBI_TZ = pytz.timezone("Africa/Nairobi")

WINDOW = deque(maxlen=200)
signal_cluster_buy = deque(maxlen=9)
signal_cluster_sell = deque(maxlen=19)

# --- Trade lock & reversal clusters ---
active_trades = {}  # {"BTCUSDT": {...}, "ETHUSDT": {...}}
reverse_signal_cluster_buy = deque(maxlen=24)   # 3 x 9
reverse_signal_cluster_sell = deque(maxlen= 39)  # 3 x 19
REVERSAL_CONVICTION_BONUS = 0.1                 # Must be +0.1 higher threshold

GATE_MODEL_PATH   = "ml_model/triangular_rf_model.json"
PAIR_MODEL_PATH   = "ml_model/pair_selector_model.json"
COINT_MODEL_PATH  = "ml_model/cointegration_score_model.json"
REGIME_MODEL_PATH = "ml_model/regime_classifier.json"

confidence_filter = MLFilter(GATE_MODEL_PATH)
pair_selector     = MLFilter(PAIR_MODEL_PATH)
cointegration_model = MLFilter(COINT_MODEL_PATH)
regime_classifier   = MLFilter(REGIME_MODEL_PATH)

def ensure_datetime(ts):
    if isinstance(ts, datetime):
        dt_utc = ts if ts.tzinfo else pytz.utc.localize(ts)
    elif isinstance(ts, (int, float)):
        dt_utc = datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc)
    else:
        dt_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    return dt_utc.astimezone(NAIROBI_TZ)

def get_live_config(regime, direction):
    best = BEST_CONFIGS.get(regime, BEST_CONFIGS["flat"])
    return {
        "MASTER_CONVICTION_THRESHOLD": float(best["base_thr_sell"] if direction == -1 else best["thr_buy"]),
        "SL_PERCENT": float(best["sl_percent"]),
        "TP_PERCENT": float(best["tp_percent"]),
        "CLUSTER_SIZE": 19 if direction == -1 else 9,
    }

def color_text(text, color):
    colors = {"green": "\033[92m", "red": "\033[91m", "reset": "\033[0m"}
    return f"{colors[color]}{text}{colors['reset']}"

def check_trade_closed(pair, price, direction, sl_level, tp_level):
    """Check if SL or TP hit for an active trade on this pair."""
    if direction == 1:  # Long
        if price <= sl_level or price >= tp_level:
            return True
    else:  # Short
        if price >= sl_level or price <= tp_level:
            return True
    return False

def process_tick(
    timestamp, btc_price, eth_price, ethbtc_price,
    base_thr_sell=None, thr_buy=None, sl_percent=None, tp_percent=None,
    cluster_buy=None, cluster_sell=None,
    signal_cluster_buy_override=None, signal_cluster_sell_override=None
):
    global signal_cluster_buy, signal_cluster_sell, active_trades
    global reverse_signal_cluster_buy, reverse_signal_cluster_sell

    timestamp = ensure_datetime(timestamp)
    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price

    features = generate_live_features(btc_price, eth_price, ethbtc_price, WINDOW)
    if not features:
        log_signal_event(timestamp, spread, 0.0, 0.0, None, 0, "veto_feature_fail")
        return None

    # --- 1. Predict regime with ML model
    regime_input = pd.DataFrame([features]).reindex(columns=regime_classifier.model.feature_names_in_)
    regime_code = regime_classifier.predict(regime_input)[0]
    regime = regime_map.get(regime_code, "flat")
    features["regime"] = regime

    # --- 2. ML Confidence filter: Should we trade, and which direction?
    gate_input = pd.DataFrame([features]).reindex(columns=confidence_filter.model.feature_names_in_)
    confidence, direction = confidence_filter.predict_with_confidence(gate_input)
    direction = int(direction)

    if direction == 0:
        log_signal_event(timestamp, spread, confidence, features.get("spread_zscore", 0), direction, 0, "veto_no_trade")
        return None

    # ---- Use passed-in thresholds if provided (for backtest/Optuna) ----
    config = get_live_config(regime, direction)
    if base_thr_sell is not None and direction == -1:
        config["MASTER_CONVICTION_THRESHOLD"] = base_thr_sell
    if thr_buy is not None and direction == 1:
        config["MASTER_CONVICTION_THRESHOLD"] = thr_buy
    if sl_percent is not None:
        config["SL_PERCENT"] = sl_percent
    if tp_percent is not None:
        config["TP_PERCENT"] = tp_percent
    if cluster_buy is not None and direction == 1:
        config["CLUSTER_SIZE"] = cluster_buy
    if cluster_sell is not None and direction == -1:
        config["CLUSTER_SIZE"] = cluster_sell

    CLUSTER_SIZE = config["CLUSTER_SIZE"]
    MASTER_CONVICTION_THRESHOLD = config["MASTER_CONVICTION_THRESHOLD"]
    SL_PERCENT = config["SL_PERCENT"]
    TP_PERCENT = config["TP_PERCENT"]

    # --- 3. Cointegration stability (final ML filter, model-driven)
    coint_input = pd.DataFrame([features]).reindex(columns=cointegration_model.model.feature_names_in_)
    coint_score, _ = cointegration_model.predict_with_confidence(coint_input)

    # --- 4. Directional cluster guard
    if direction == -1:
        cluster = signal_cluster_sell_override if signal_cluster_sell_override is not None else signal_cluster_sell
        if cluster.maxlen != CLUSTER_SIZE:
            cluster = deque(maxlen=CLUSTER_SIZE)
            if signal_cluster_sell_override is None:
                signal_cluster_sell = cluster
    else:
        cluster = signal_cluster_buy_override if signal_cluster_buy_override is not None else signal_cluster_buy
        if cluster.maxlen != CLUSTER_SIZE:
            cluster = deque(maxlen=CLUSTER_SIZE)
            if signal_cluster_buy_override is None:
                signal_cluster_buy = cluster

    cluster.append({
        "direction": direction,
        "confidence": confidence,
        "coint": coint_score,
    })

    spread_zscore = features.get("spread_zscore", 0.0)
    spread_slope  = features.get("spread_slope", 0.0)

    # --- 5. Pair selector: BTC or ETH trade
    pair_input = pd.DataFrame([features]).reindex(columns=pair_selector.model.feature_names_in_)
    pair_code = pair_selector.predict(pair_input)[0]
    selected_leg = reverse_pair_map.get(pair_code)

    if selected_leg == "ETH":
        entry_price = eth_price
        pair = "ETHUSDT"
        live_price = eth_price
    else:
        entry_price = btc_price
        pair = "BTCUSDT"
        live_price = btc_price

    # === MASTER TRADE LOCK & REVERSAL LOGIC ===
    if pair in active_trades:
        # Check if SL/TP hit
        trade = active_trades[pair]
        direction_lock = trade['direction']
        sl_level = trade['sl_level']
        tp_level = trade['tp_level']
        if check_trade_closed(pair, live_price, direction_lock, sl_level, tp_level):
            del active_trades[pair]
            # Clear all clusters so next entry requires full cluster
            signal_cluster_buy.clear()
            signal_cluster_sell.clear()
            reverse_signal_cluster_buy.clear()
            reverse_signal_cluster_sell.clear()
            return None  # Unlock, require new cluster next bar

        # --- Build reverse cluster for opposite direction ---
        reverse_direction = -direction_lock
        reverse_cluster_size = 24 if reverse_direction == 1 else 39  # 1.5x entry cluster size
        reverse_conviction_threshold = (get_live_config(regime, reverse_direction)["MASTER_CONVICTION_THRESHOLD"]
                                        + REVERSAL_CONVICTION_BONUS)
        if reverse_direction == 1:
            reverse_cluster = reverse_signal_cluster_buy
        else:
            reverse_cluster = reverse_signal_cluster_sell

        if direction == reverse_direction and \
           confidence >= reverse_conviction_threshold and coint_score >= 0.8:
            reverse_cluster.append({
                "direction": direction,
                "confidence": confidence,
                "coint": coint_score,
            })
        else:
            reverse_cluster.clear()  # Reset if no new strong reverse

        # Emergency exit only if reverse cluster is FULL and all strong
        if len(reverse_cluster) == reverse_cluster_size and all(
            s["direction"] == reverse_direction and
            s["confidence"] >= reverse_conviction_threshold and
            s["coint"] >= 0.8
            for s in reverse_cluster
        ):
            del active_trades[pair]
            signal_cluster_buy.clear()
            signal_cluster_sell.clear()
            reverse_signal_cluster_buy.clear()
            reverse_signal_cluster_sell.clear()
            log_signal_event(
                timestamp, spread, confidence, spread_zscore, reverse_direction, 0, "emergency_exit_reversal",
                coint_score=coint_score, regime=regime, selected_leg=selected_leg,
                entry_level=entry_price
            )
            print(color_text(f"\nEMERGENCY EXIT [{pair}]: "
                             f"Conviction {confidence:.3f} | Regime={regime} | Time={timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n", "red"))
            return None  # Exit, require full new cluster for re-entry

        # If trade still open and no reversal, block new entry
        log_signal_event(
            timestamp, spread, confidence, spread_zscore, direction, 0, "veto_trade_lock_active",
            coint_score=coint_score, regime=regime
        )
        return None

    # === NORMAL ENTRY LOGIC ===
    if len(cluster) == CLUSTER_SIZE and all(
        s["direction"] == direction and s["confidence"] >= MASTER_CONVICTION_THRESHOLD and s["coint"] >= 0.8
        for s in cluster
    ):
        stop_loss_pct = SL_PERCENT
        take_profit_pct = TP_PERCENT
        sl_level = entry_price * (1 - stop_loss_pct / 100) if direction == 1 else entry_price * (1 + stop_loss_pct / 100)
        tp_level = entry_price * (1 + take_profit_pct / 100) if direction == 1 else entry_price * (1 - take_profit_pct / 100)

        # Register active trade lock
        active_trades[pair] = {
            "direction": direction,
            "entry_price": entry_price,
            "sl_level": sl_level,
            "tp_level": tp_level
        }

        log_execution_event(
            timestamp=timestamp, pair=pair, direction=direction, entry_price=entry_price,
            confidence=confidence, cointegration_score=coint_score, regime=regime,
            stop_loss=sl_level, take_profit=tp_level,
            spread_zscore=spread_zscore, spread_slope=spread_slope
        )
        log_signal_event(
            timestamp, spread, confidence, spread_zscore, direction, 1, "signal_pass_cluster",
            coint_score=coint_score, regime=regime, selected_leg=selected_leg,
            entry_level=entry_price, stop_loss=sl_level, take_profit=tp_level
        )

        color = "green" if direction == 1 else "red"
        label = "BUY" if direction == 1 else "SELL"
        local_time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = (f"\n{label} SIGNAL [{pair}]: "
               f"Entry={entry_price:.2f} | SL={sl_level:.2f} | TP={tp_level:.2f} | "
               f"Regime={regime} | Time={local_time_str}\n")
        print(color_text(msg, color))
        cluster.clear()
        return {
            "timestamp": timestamp, "pair": pair, "direction": direction, "entry_price": entry_price,
            "confidence": confidence, "cointegration_score": coint_score, "regime": regime,
            "stop_loss": sl_level, "take_profit": tp_level, "pnl": None,
        }
    else:
        log_signal_event(
            timestamp, spread, confidence, spread_zscore, direction, 0, "waiting_for_cluster",
            coint_score=coint_score, regime=regime
        )
        return None

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("🚀 XAlgo: Master Conviction, Reversal & Trade Lock Engine Starting...")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
