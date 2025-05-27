# label_with_triple_barrier.py
# 🧠 Applies triple-barrier labeling to features_triangular_base.csv
# 🚀 Outputs: features_triangular_labeled.csv

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("features_triangular_base.csv")
OUTPUT_PATH = Path("features_triangular_labeled.csv")

# Triple-barrier parameters
HORIZON = 30          # lookahead steps
TP_MULT = 1.5         # take-profit multiple of vol_spread
SL_MULT = 1.0         # stop-loss multiple of vol_spread

print("📂 Loading base features...")
df = pd.read_csv(INPUT_PATH, parse_dates=["open_time"])
df.reset_index(drop=True, inplace=True)

labels = []
directions = []

for i in range(len(df) - HORIZON):
    entry_price = df.loc[i, "spread"]
    vol = df.loc[i, "vol_spread"]
    tp = TP_MULT * vol
    sl = SL_MULT * vol

    label = 0
    direction = 0

    for j in range(1, HORIZON + 1):
        future_price = df.loc[i + j, "spread"]
        diff = future_price - entry_price

        if diff >= tp:
            label = 1
            direction = 1
            break
        elif diff <= -sl:
            label = -1
            direction = -1
            break

    labels.append(label)
    directions.append(direction)

# Align labels with the dataframe (trim the end)
df = df.iloc[:-HORIZON].copy()
df["label"] = labels
df["direction"] = directions

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved labeled file to: {OUTPUT_PATH}")
