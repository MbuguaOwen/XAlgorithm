from prometheus_client import Gauge, Counter, start_http_server

# --- Gauges ---
CONFIDENCE_SCORE = Gauge('confidence_score', 'Latest confidence score from model')
COINTEGRATION_STABILITY = Gauge('cointegration_stability', 'Latest cointegration stability score')
TRADE_PNL = Gauge('trade_pnl', 'Realized PnL percentage from last trade')
TUNED_CONFIDENCE_THRESHOLD = Gauge(
    'tuned_confidence_threshold',
    'Current tuned confidence threshold'
)
REGIME_TRADE_SUCCESS_RATE = Gauge(
    'regime_trade_success_rate',
    'Current win rate per regime',
    ['regime']
)

# --- Counters ---
EXIT_REASON_COUNTS = Counter('exit_reason_counts', 'Number of exits by reason', ['reason'])
MISSED_OPPORTUNITIES = Counter('missed_opportunities', 'Signals vetoed by entry filters')
ENTRY_REASON_COUNTS = Counter('entry_reason_counts', 'Number of entry decisions by reason', ['reason'])
META_FALLBACK_USED_TOTAL = Counter('meta_fallback_used_total', 'Fallback config usage count')
CONFIG_OVERFIT_BLOCK_COUNT = Counter(
    'config_overfit_block_count',
    'Tuning updates blocked due to overfitting risk'
)

def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server."""
    start_http_server(port)
