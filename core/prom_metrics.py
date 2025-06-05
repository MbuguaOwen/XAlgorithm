from prometheus_client import Gauge, Counter, start_http_server

CONFIDENCE_SCORE = Gauge('confidence_score', 'Latest confidence score from model')
TRADE_PNL = Gauge('trade_pnl', 'Realized PnL percentage from last trade')
EXIT_REASON_COUNTS = Counter('exit_reason_counts', 'Number of exits by reason', ['reason'])

def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server."""
    start_http_server(port)
