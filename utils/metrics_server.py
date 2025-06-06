from prometheus_client import Gauge, start_http_server
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


def start_metrics_server(port: int = 8001) -> None:
    """Start Prometheus metrics server on the specified port."""
    try:
        start_http_server(port)
        logging.info(
            "\u2705 Prometheus metrics running at http://localhost:%d/metrics", port
        )
    except OSError as exc:  # Port already in use or other socket errors
        logging.error("Failed to start metrics server on port %d: %s", port, exc)
