import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
from collections import deque
from core.feature_pipeline import generate_live_features, Z_SCORE_WINDOW


def test_generate_live_features_structure():
    window = deque(maxlen=Z_SCORE_WINDOW)
    features = generate_live_features(30000.0, 2000.0, 0.066, window)
    expected_keys = {
        'btc_usd', 'eth_usd', 'eth_btc', 'implied_ethbtc', 'spread',
        'spread_zscore', 'vol_spread', 'spread_kalman', 'spread_ewma',
        'btc_vol', 'eth_vol', 'ethbtc_vol', 'momentum_btc', 'momentum_eth',
        'rolling_corr', 'vol_ratio', 'spread_slope', 'zscore_slope'
    }
    assert set(features.keys()) == expected_keys


