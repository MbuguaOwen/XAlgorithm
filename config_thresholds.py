import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'default.yaml'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

ENTRY_THRESHOLDS = CONFIG.get('entry_thresholds', {})
REGIME_DEFAULTS = CONFIG.get('regime_defaults', {})
MODELS = CONFIG.get('models', {})
BACKTEST_CFG = CONFIG.get('backtest', {})
USE_DYNAMIC_SL_TP = CONFIG.get('use_dynamic_sl_tp', True)
TRAILING_TP_ENABLED = CONFIG.get('trailing_tp_enabled', False)
TRAILING_TP_OFFSET_PCT = CONFIG.get('trailing_tp_offset_pct', 0.002)

SL_TP_MODIFIERS = CONFIG.get('sl_tp_modifiers', {}).get('cointegration', {})


def get_dynamic_zscore_min() -> float:
    return float(ENTRY_THRESHOLDS.get('dynamic_zscore_min', 1.5))
