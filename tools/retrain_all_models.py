# retrain_all_models.py
# 🚀 Unified retraining script for all 4 ML models in XAlgoNexus

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_DIR = Path("ml_model")
MODEL_DIR.mkdir(exist_ok=True)

# === Common training function ===
def train_and_save_model(X, y, model_name):
    print(f"\n🎯 Training {model_name}...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"📊 {model_name} Classification Report:\n{classification_report(y_test, y_pred)}")

    # Save .pkl
    pkl_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump(model, pkl_path)
    print(f"💾 Saved: {pkl_path}")

    # Save .json
    json_path = MODEL_DIR / f"{model_name}.json"
    model.save_model(json_path)
    print(f"💾 Saved: {json_path}")

# === 1. Triangular RF Model ===
# Class mapping: –1 = SHORT → 0, 0 = HOLD → 1, 1 = LONG → 2
df_tri = pd.read_csv("features_triangular_labeled.csv")
X_tri = df_tri.drop(columns=["label", "direction", "open_time"])
y_tri = df_tri["label"].map({-1: 0, 0: 1, 1: 2})
train_and_save_model(X_tri, y_tri, "triangular_rf_model")

# === 2. Cointegration Score Model ===
# Binarized: cointegration_score ≥ 0.8 → 1 (stable), else 0
df_coint = pd.read_csv("features_cointegration_labeled.csv")
X_coint = df_coint.drop(columns=["cointegration_score", "open_time"])
y_coint = (df_coint["cointegration_score"] > 0.8).astype(int)
train_and_save_model(X_coint, y_coint, "cointegration_score_model")

# === 3. Pair Selector Model ===
# best_leg: 0 = BTC, 1 = ETH, 2 = Neutral
df_pair = pd.read_csv("features_pair_selection.csv")
X_pair = df_pair.drop(columns=["best_leg", "momentum_btc", "momentum_eth", "open_time"])
y_pair = df_pair["best_leg"]
train_and_save_model(X_pair, y_pair, "pair_selector_model")

# === 4. Regime Classifier Model ===
# regime: 0 = flat, 1 = volatile, 2 = trending
df_regime = pd.read_csv("features_regime.csv")
X_reg = df_regime.drop(columns=["regime", "open_time"])
y_reg = df_regime["regime"]
train_and_save_model(X_reg, y_reg, "regime_classifier")

print("\n✅ All models trained and saved successfully.")
