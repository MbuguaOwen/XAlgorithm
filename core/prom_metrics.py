from prometheus_client import Gauge, Counter, start_http_server

# --- Gauges ---
CONFIDENCE_SCORE = Gauge('confidence_score', 'Latest confidence score from model')
COINTEGRATION_STABILITY = Gauge('cointegration_stability', 'Latest cointegration stability score')
TRADE_PNL = Gauge('trade_pnl', 'Realized PnL percentage from last trade')

# --- Counters ---
EXIT_REASON_COUNTS = Counter('exit_reason_counts', 'Number of exits by reason', ['reason'])
MISSED_OPPORTUNITIES = Counter('missed_opportunities', 'Signals vetoed by entry filters')
ENTRY_REASON_COUNTS = Counter('entry_reason_counts', 'Number of entry decisions by reason', ['reason'])

def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server."""
    start_http_server(port)
