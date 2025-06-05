import pandas as pd
import numpy as np
from collections import deque
from core.adaptive_filters import EWMA, KalmanFilter

Z_SCORE_WINDOW = 100
kalman = KalmanFilter()
ewma_mean = EWMA(alpha=0.05)

 codex/find-and-fix-a-bug-in-codebase
btc_price_window = deque(maxlen=Z_SCORE_WINDOW)
eth_price_window = deque(maxlen=Z_SCORE_WINDOW)
ethbtc_price_window = deque(maxlen=Z_SCORE_WINDOW)


def compute_triangle_features(
    btc_price,
    eth_price,
    ethbtc_price,
    spread_window,
    btc_window=btc_price_window,
    eth_window=eth_price_window,
    ethbtc_window=ethbtc_price_window,
):

def compute_triangle_features(btc_price, eth_price, ethbtc_price, windows):
    """Compute live trading features using rolling price windows."""
 main
    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price

    spread_window = windows["spread"]
    btc_window = windows["btc"]
    eth_window = windows["eth"]
    ethbtc_window = windows["ethbtc"]

    # Update all windows with latest prices
    spread_window.append(spread)
 codex/find-and-fix-a-bug-in-codebase
    if btc_window is not None:
        btc_window.append(btc_price)
    if eth_window is not None:
        eth_window.append(eth_price)
    if ethbtc_window is not None:
        ethbtc_window.append(ethbtc_price)

    btc_window.append(btc_price)
    eth_window.append(eth_price)
    ethbtc_window.append(ethbtc_price)
 main

    values = list(spread_window)

    # === Z-score and volatility ===
    if len(values) >= Z_SCORE_WINDOW:
        mean = np.mean(values)
        std = np.std(values)
        z_score = (spread - mean) / std if std > 1e-6 else 0.0
        vol_spread = std
    else:
        z_score = 0.0
        vol_spread = 0.0

    # === Filters ===
    kalman_filtered = kalman.update(spread)
    ewma_filtered = ewma_mean.update(spread)

    # === Momentum ===
 codex/find-and-fix-a-bug-in-codebase
    momentum_btc = (
        btc_price - btc_window[-2] if btc_window is not None and len(btc_window) > 1 else 0.0
    )
    momentum_eth = (
        eth_price - eth_window[-2] if eth_window is not None and len(eth_window) > 1 else 0.0
    )

    momentum_btc = btc_window[-1] - btc_window[-2] if len(btc_window) > 1 else 0.0
    momentum_eth = eth_window[-1] - eth_window[-2] if len(eth_window) > 1 else 0.0
 main

    # === Rolling correlation ===
    if len(btc_window) >= 20 and len(eth_window) >= 20:
        btc_returns = pd.Series(list(btc_window)[-20:]).pct_change().dropna()
        eth_returns = pd.Series(list(eth_window)[-20:]).pct_change().dropna()
        if len(btc_returns) > 2 and len(eth_returns) > 2 and np.std(btc_returns) > 1e-8 and np.std(eth_returns) > 1e-8:
            rolling_corr = btc_returns.corr(eth_returns)
        else:
            rolling_corr = 0.0
    else:
        rolling_corr = 0.0

    # === Volatility metrics ===
 codex/find-and-fix-a-bug-in-codebase
    btc_vol = np.std(btc_window) if btc_window is not None and len(btc_window) > 1 else 0.0
    eth_vol = np.std(eth_window) if eth_window is not None and len(eth_window) > 1 else 0.0
    ethbtc_vol = np.std(ethbtc_window) if ethbtc_window is not None and len(ethbtc_window) > 1 else 0.0

    btc_returns_full = pd.Series(list(btc_window)).pct_change().dropna()
    eth_returns_full = pd.Series(list(eth_window)).pct_change().dropna()
    ethbtc_returns_full = pd.Series(list(ethbtc_window)).pct_change().dropna()

    btc_vol = np.std(btc_returns_full) if len(btc_returns_full) > 1 else 0.0
    eth_vol = np.std(eth_returns_full) if len(eth_returns_full) > 1 else 0.0
    ethbtc_vol = np.std(ethbtc_returns_full) if len(ethbtc_returns_full) > 1 else 0.0
 main
    vol_ratio = eth_vol / btc_vol if btc_vol > 1e-8 else 1.0

    # === Spread Slope (required by ML model) ===
    if len(values) >= 5:
        spread_slope = pd.Series(values).diff().rolling(5).mean().iloc[-1]
    else:
        spread_slope = 0.0

    return {
        "btc_usd": btc_price,
        "eth_usd": eth_price,
        "eth_btc": ethbtc_price,
        "implied_ethbtc": implied_ethbtc,
        "spread": spread,
        "spread_zscore": z_score,
        "vol_spread": vol_spread,
        "spread_kalman": kalman_filtered,
        "spread_ewma": ewma_filtered,
        "btc_vol": btc_vol,
        "eth_vol": eth_vol,
        "ethbtc_vol": ethbtc_vol,
        "momentum_btc": momentum_btc,
        "momentum_eth": momentum_eth,
        "rolling_corr": rolling_corr,
        "vol_ratio": vol_ratio,
        "spread_slope": spread_slope  # ✅ Critical feature added
    }

def generate_features_from_csvs(csv_paths):
    btc_df = pd.read_csv(csv_paths["BTCUSDT"], parse_dates=["open_time"])
    eth_df = pd.read_csv(csv_paths["ETHUSDT"], parse_dates=["open_time"])
    ethbtc_df = pd.read_csv(csv_paths["ETHBTC"], parse_dates=["open_time"])

    btc_df = btc_df.rename(columns={"open_time": "timestamp", "close": "btc_usd", "volume": "btc_vol"})
    eth_df = eth_df.rename(columns={"open_time": "timestamp", "close": "eth_usd", "volume": "eth_vol"})
    ethbtc_df = ethbtc_df.rename(columns={"open_time": "timestamp", "close": "eth_btc", "volume": "ethbtc_vol"})

    df = btc_df[["timestamp", "btc_usd", "btc_vol"]] \
        .merge(eth_df[["timestamp", "eth_usd", "eth_vol"]], on="timestamp") \
        .merge(ethbtc_df[["timestamp", "eth_btc", "ethbtc_vol"]], on="timestamp")

 codex/find-and-fix-a-bug-in-codebase
    spread_window = deque(maxlen=Z_SCORE_WINDOW)
    btc_window = deque(maxlen=Z_SCORE_WINDOW)
    eth_window = deque(maxlen=Z_SCORE_WINDOW)
    ethbtc_window = deque(maxlen=Z_SCORE_WINDOW)
    features = []

    for _, row in df.iterrows():
        f = compute_triangle_features(
            row["btc_usd"],
            row["eth_usd"],
            row["eth_btc"],
            spread_window,
            btc_window,
            eth_window,
            ethbtc_window,
        )

    windows = {
        "spread": deque(maxlen=Z_SCORE_WINDOW),
        "btc": deque(maxlen=Z_SCORE_WINDOW),
        "eth": deque(maxlen=Z_SCORE_WINDOW),
        "ethbtc": deque(maxlen=Z_SCORE_WINDOW),
    }
    features = []

    for _, row in df.iterrows():
        f = compute_triangle_features(row["btc_usd"], row["eth_usd"], row["eth_btc"], windows)
 main
        f["timestamp"] = row["timestamp"]
        features.append(f)

    return pd.DataFrame(features)

 codex/find-and-fix-a-bug-in-codebase
def generate_live_features(btc_price, eth_price, ethbtc_price, spread_window):
    return compute_triangle_features(
        btc_price,
        eth_price,
        ethbtc_price,
        spread_window,
        btc_price_window,
        eth_price_window,
        ethbtc_price_window,
    )

def generate_live_features(btc_price, eth_price, ethbtc_price, windows):
    """Generate features from live prices using provided rolling windows."""
    if isinstance(windows, deque):
        windows = {
            "spread": windows,
            "btc": deque(maxlen=Z_SCORE_WINDOW),
            "eth": deque(maxlen=Z_SCORE_WINDOW),
            "ethbtc": deque(maxlen=Z_SCORE_WINDOW),
        }
    return compute_triangle_features(btc_price, eth_price, ethbtc_price, windows)
 main
