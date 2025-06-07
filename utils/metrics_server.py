from prometheus_client import Gauge, Counter, start_http_server
import logging

# Define Prometheus gauges
confidence_gauge = Gauge(
    "xalgo_confidence_score",
    "Latest confidence score from model",
)
cointegration_gauge = Gauge(
    "xalgo_cointegration_score",
    "Latest cointegration stability score",
)
regime_gauge = Gauge(
    "xalgo_regime_label",
    "Current regime label as integer",
)

# New gauges for additional metrics
zscore_gauge = Gauge(
    "xalgo_spread_zscore",
    "Current spread z-score",
)
zscore_slope_gauge = Gauge(
    "xalgo_zscore_slope",
    "Slope of the spread z-score",
)
sl_gauge = Gauge(
    "xalgo_dynamic_sl",
    "Dynamic stop-loss price",
)
tp_gauge = Gauge(
    "xalgo_dynamic_tp",
    "Dynamic take-profit price",
)

# Gauge for applied cointegration modifier label (0=low,1=medium,2=high)
coint_mod_gauge = Gauge(
    "xalgo_cointegration_modifier",
    "Applied cointegration modifier label code",
)

# Additional metrics for adaptive tuning
tuned_confidence_threshold = Gauge(
    "tuned_confidence_threshold",
    "Current tuned confidence buy threshold",
)
regime_trade_success_rate = Gauge(
    "regime_trade_success_rate",
    "Historical trade win rate for regime",
    ["regime"],
)
meta_fallback_used_total = Counter(
    "meta_fallback_used_total",
    "Number of times default config used due to poor performance",
)
config_overfit_block_count = Counter(
    "config_overfit_block_count",
    "Tuning updates blocked by overfitting guard",
)


def start_metrics_server(port: int = 8001) -> None:
    """Start Prometheus metrics server on the specified port."""
    try:
        start_http_server(port)
        logging.info(
            "\u2705 Prometheus metrics running at http://localhost:%d/metrics", port
        )
    except OSError as exc:  # Port already in use or other socket errors
        logging.error("Failed to start metrics server on port %d: %s", port, exc)
