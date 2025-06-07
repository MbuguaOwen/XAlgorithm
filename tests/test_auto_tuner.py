from memory import MemoryCore
from memory.auto_tuner import run_tuning_cycle, get_tuned_thresholds
from memory.replay_engine import ReplayEngine


def test_auto_tuner_empty(tmp_path):
    mem = MemoryCore(
        signal_path=tmp_path / "sig.pkl",
        regime_path=tmp_path / "reg.json",
        ghost_path=tmp_path / "ghost.json",
        tuned_path=tmp_path / "tuned.json",
    )
    run_tuning_cycle(mem)
    cfg = get_tuned_thresholds(mem, "flat", {"default": {"thr_buy": 0.75}})
    assert cfg["thr_buy"] >= 0.7


def test_replay_engine(tmp_path):
    mem = MemoryCore(
        signal_path=tmp_path / "sig.pkl",
        regime_path=tmp_path / "reg.json",
        ghost_path=tmp_path / "ghost.json",
        tuned_path=tmp_path / "tuned.json",
    )
    mem.signal_memory.append(
        {
            "confidence": 0.9,
            "features": {"spread_zscore": 2.5},
            "outcome": {"win": True},
        }
    )
    engine = ReplayEngine(mem)
    score = engine.replay({"confidence": 0.8, "zscore": 2.0})
    assert 0.0 <= score <= 1.0
