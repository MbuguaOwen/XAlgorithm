import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import csv
import os
from datetime import datetime
from core import trade_logger


def test_logging_functions(tmp_path, monkeypatch):
    signal_path = tmp_path / "signal.csv"
    exec_path = tmp_path / "exec.csv"
    exit_path = tmp_path / "exit.csv"
    monkeypatch.setattr(trade_logger, "LOG_FILE", str(signal_path))
    monkeypatch.setattr(trade_logger, "EXECUTION_LOG_FILE", str(exec_path))

    ts = datetime.utcnow()
    trade_logger.log_signal_event(ts, 1.0, 0.9, 1.2, 1, 1, "test")
    trade_logger.log_execution_event(ts, "BTCUSDT", 1, 50000, 0.9, 0.8, "bull", 0.1, 0.2, 1.2, 0.0)
    monkeypatch.setattr(trade_logger, "EXECUTION_LOG_FILE", str(exit_path))
    trade_logger.log_trade_exit(ts, "BTCUSDT", 1, 50000, 50100, 0.2, 0.9, 0.95, 0.8, 0.85, "TP", 10.5)

    with open(signal_path) as f:
        rows = list(csv.reader(f))
    assert rows and len(rows[0]) >= 10

    with open(exit_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows and rows[-1]["exit_reason"] == "TP"
    assert "duration" in rows[-1]


