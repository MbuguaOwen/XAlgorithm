# label_all_targets.py
# 🧠 Unified labeling script for all XAlgoNexus models
# 🏗️ Output: 4 CSVs for ML training

import pandas as pd
import numpy as np
from pathlib import Path
from core.adaptive_filters import KalmanFilter

INPUT_PATH = Path("features_triangular_base.csv")

TRI_OUTPUT = Path("features_triangular_labeled.csv")
COINT_OUTPUT = Path("features_cointegration_labeled.csv")
PAIR_OUTPUT = Path("features_pair_selection.csv")
REGIME_OUTPUT = Path("features_regime.csv")

# === Load base features ===
df = pd.read_csv(INPUT_PATH, parse_dates=["open_time"])
df.reset_index(drop=True, inplace=True)

# === 1. Triple-Barrier Labeling ===
print("⚙️ Applying triple-barrier labeling...")
HORIZON = 30
TP_MULT = 1.5
SL_MULT = 1.0

labels, directions = [], []
for i in range(len(df) - HORIZON):
    entry = df.loc[i, "spread"]
    vol = df.loc[i, "vol_spread"]
    tp, sl = TP_MULT * vol, SL_MULT * vol
    label, direction = 0, 0

    for j in range(1, HORIZON + 1):
        future = df.loc[i + j, "spread"]
        diff = future - entry
        if diff >= tp:
            label, direction = 1, 1
            break
        elif diff <= -sl:
            label, direction = -1, -1
            break

    labels.append(label)
    directions.append(direction)

df_tri = df.iloc[:-HORIZON].copy()
df_tri["label"] = labels
df_tri["direction"] = directions
df_tri.to_csv(TRI_OUTPUT, index=False)
print(f"✅ Saved: {TRI_OUTPUT}")

# === 2. Cointegration Score (Kalman RMS) ===
print("⚙️ Calculating cointegration stability...")
kf = KalmanFilter()
residuals = []
kalman_vals = []

for val in df["spread"].fillna(0):
    filt = kf.update(val)
    kalman_vals.append(filt)
    residuals.append(val - filt)

rms_window = 30
resid_rms = pd.Series(residuals).rolling(rms_window).apply(lambda x: np.sqrt(np.mean(np.square(x))))
score = 1 - (resid_rms / resid_rms.max())  # normalize to [0, 1]

df_coint = df.copy()
df_coint["cointegration_score"] = score.fillna(0)
df_coint.to_csv(COINT_OUTPUT, index=False)
print(f"✅ Saved: {COINT_OUTPUT}")

# === 3. Best Leg (BTC or ETH) Selection ===
print("⚙️ Deriving best_leg from feature dominance...")
def select_leg(row):
    if abs(row["spread_zscore"]) < 0.1:
        return 2  # Neutral
    if abs(row["momentum_eth"]) > abs(row["momentum_btc"]):
        return 1  # ETH
    if abs(row["momentum_btc"]) > abs(row["momentum_eth"]):
        return 0  # BTC
    return 2  # Neutral fallback

df_pair = df.copy()
df_pair["momentum_btc"] = df_pair["btc_usd"].pct_change().rolling(3).mean()
df_pair["momentum_eth"] = df_pair["eth_usd"].pct_change().rolling(3).mean()
df_pair["best_leg"] = df_pair.apply(select_leg, axis=1)
df_pair.dropna(inplace=True)
df_pair.to_csv(PAIR_OUTPUT, index=False)
print(f"✅ Saved: {PAIR_OUTPUT}")

# === 4. Regime Classification (volatility buckets) ===
print("⚙️ Classifying regimes by volatility level...")
vol = df["vol_spread"].rolling(20).mean().fillna(method="bfill")
vol_bins = pd.qcut(vol, q=3, labels=[0, 1, 2])  # 0 = flat, 1 = volatile, 2 = trending

df_regime = df.copy()
df_regime["regime"] = vol_bins.astype(int)
df_regime.to_csv(REGIME_OUTPUT, index=False)
print(f"✅ Saved: {REGIME_OUTPUT}")

print("🏁 All label files generated successfully.")
