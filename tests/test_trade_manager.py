import asyncio
import csv
from datetime import datetime

import pytest

from core.trade_manager import TradeManager, TradeState
from core import trade_logger

@pytest.mark.asyncio
async def test_tp_hit(tmp_path, monkeypatch):
    log_path = tmp_path / "exec.csv"
    monkeypatch.setattr(trade_logger, "EXECUTION_LOG_FILE", str(log_path))

    q = asyncio.Queue()
    state = TradeState("BTCUSDT", 1, 100.0, datetime.utcnow(), 0.8, 0.9, 3.0)
    manager = TradeManager(state, q, timeout_seconds=5)
    task = asyncio.create_task(manager.start())
    q.put_nowait((datetime.utcnow(), 100.0, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 101.0, 0.8, 0.9, -0.1, -0.1))
    await asyncio.sleep(0.2)
    task.cancel()
    assert state.exit_reason == "TP_REVERSION"
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["exit_reason"] == "TP_REVERSION"
    assert float(rows[0]["duration"]) > 0

@pytest.mark.asyncio
async def test_sl_hit(tmp_path, monkeypatch):
    log_path = tmp_path / "exec.csv"
    monkeypatch.setattr(trade_logger, "EXECUTION_LOG_FILE", str(log_path))

    q = asyncio.Queue()
    state = TradeState("BTCUSDT", 1, 100.0, datetime.utcnow(), 0.8, 0.9, 3.0)
    manager = TradeManager(state, q, timeout_seconds=5)
    task = asyncio.create_task(manager.start())
    q.put_nowait((datetime.utcnow(), 100.0, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 99.0, 0.8, 0.9, 3.0, -0.2))
    await asyncio.sleep(0.2)
    task.cancel()
    assert state.exit_reason == "ADAPTIVE_SL_TREND"

@pytest.mark.asyncio
async def test_confidence_drop(tmp_path, monkeypatch):
    log_path = tmp_path / "exec.csv"
    monkeypatch.setattr(trade_logger, "EXECUTION_LOG_FILE", str(log_path))

    q = asyncio.Queue()
    state = TradeState("BTCUSDT", 1, 100.0, datetime.utcnow(), 0.8, 0.9, 3.0)
    manager = TradeManager(state, q, timeout_seconds=5)
    task = asyncio.create_task(manager.start())
    q.put_nowait((datetime.utcnow(), 100.0, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 100.0, 0.4, 0.9, 3.0, 0.1))
    await asyncio.sleep(0.2)
    task.cancel()
    assert state.exit_reason == "EMERGENCY_CONFIDENCE"

@pytest.mark.asyncio
async def test_cointegration_fail(tmp_path, monkeypatch):
    log_path = tmp_path / "exec.csv"
    monkeypatch.setattr(trade_logger, "EXECUTION_LOG_FILE", str(log_path))

    q = asyncio.Queue()
    state = TradeState("BTCUSDT", 1, 100.0, datetime.utcnow(), 0.8, 0.9, 3.0)
    manager = TradeManager(state, q, timeout_seconds=5)
    task = asyncio.create_task(manager.start())
    q.put_nowait((datetime.utcnow(), 100.0, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 100.0, 0.8, 0.6, 3.0, 0.1))
    await asyncio.sleep(0.2)
    task.cancel()
    assert state.exit_reason == "EMERGENCY_COINTEGRATION"

@pytest.mark.asyncio
async def test_timeout_exit(tmp_path, monkeypatch):
    log_path = tmp_path / "exec.csv"
    monkeypatch.setattr(trade_logger, "EXECUTION_LOG_FILE", str(log_path))

    q = asyncio.Queue()
    state = TradeState("BTCUSDT", 1, 100.0, datetime.utcnow(), 0.8, 0.9, 3.0)
    manager = TradeManager(state, q, timeout_seconds=1)
    task = asyncio.create_task(manager.start())
    q.put_nowait((datetime.utcnow(), 100.0, 0.8, 0.9, 3.0, 0.1))
    await asyncio.sleep(1.2)
    task.cancel()
    assert state.exit_reason == "TIMEOUT"


@pytest.mark.asyncio
async def test_trailing_tp_exit(tmp_path, monkeypatch):
    log_path = tmp_path / "exec.csv"
    monkeypatch.setattr(trade_logger, "EXECUTION_LOG_FILE", str(log_path))

    q = asyncio.Queue()
    state = TradeState(
        "BTCUSDT", 1, 100.0, datetime.utcnow(), 0.8, 0.9, 3.0, tp_pct=2.0
    )
    manager = TradeManager(state, q, timeout_seconds=5)
    task = asyncio.create_task(manager.start())
    q.put_nowait((datetime.utcnow(), 100.0, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 101.0, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 102.0, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 100.9, 0.8, 0.9, 3.0, 0.1))
    await asyncio.sleep(0.2)
    task.cancel()
    assert state.exit_reason == "TRAILING_TP"


@pytest.mark.asyncio
async def test_sl_ratchet_exit(tmp_path, monkeypatch):
    log_path = tmp_path / "exec.csv"
    monkeypatch.setattr(trade_logger, "EXECUTION_LOG_FILE", str(log_path))

    q = asyncio.Queue()
    state = TradeState(
        "BTCUSDT", 1, 100.0, datetime.utcnow(), 0.8, 0.9, 3.0, tp_pct=2.0
    )
    manager = TradeManager(state, q, timeout_seconds=5)
    task = asyncio.create_task(manager.start())
    q.put_nowait((datetime.utcnow(), 100.0, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 101.0, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 101.5, 0.8, 0.9, 3.0, 0.1))
    q.put_nowait((datetime.utcnow(), 100.04, 0.8, 0.9, 3.0, 0.1))
    await asyncio.sleep(0.2)
    task.cancel()
    assert state.exit_reason == "SL_RATCHETED_EXIT"

