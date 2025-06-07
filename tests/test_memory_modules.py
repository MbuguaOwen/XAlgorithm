from datetime import datetime

from memory import MemoryCore, SignalRecord, SignalMemory, RegimeRecall, RegimeOutcome


def test_signal_memory_roundtrip(tmp_path):
    mem = MemoryCore(signal_path=tmp_path / "sig.pkl")
    sm = SignalMemory(mem, maxlen=5)

    rec = SignalRecord(timestamp=datetime.utcnow(), features={"a": 1}, confidence=0.9)
    sm.log_signal(rec)

    assert len(mem.signal_memory) == 1
    mem.save_all()

    mem2 = MemoryCore(signal_path=tmp_path / "sig.pkl")
    assert len(mem2.signal_memory) == 1


def test_regime_recall(tmp_path):
    mem = MemoryCore(regime_path=tmp_path / "reg.json")
    rr = RegimeRecall(mem)
    out = RegimeOutcome(timestamp=datetime.utcnow(), regime="bull", win=True, pnl_pct=1.0)
    rr.log_outcome(out)
    profile = rr.get_profile()
    assert profile["bull"]["wins"] == 1
