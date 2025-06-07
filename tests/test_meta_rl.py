from memory import MemoryCore
from memory.meta_rl import MetaReinforcer


def test_meta_reinforcer_fallback(tmp_path):
    mem = MemoryCore(
        signal_path=tmp_path / "sig.pkl",
        regime_path=tmp_path / "reg.json",
        ghost_path=tmp_path / "ghost.json",
        tuned_path=tmp_path / "tuned.json",
    )
    meta = MetaReinforcer(mem, fallback_threshold=0.6)
    # poor performance lowers weight below 0.5
    for _ in range(6):
        meta.update_performance("flat", 0.1, -0.1, 100)
    mem.tuned_configs["flat"] = {"thr_buy": 0.9}
    cfg = meta.choose_config("flat", mem.tuned_configs["flat"], {"thr_buy": 0.8})
    assert cfg["thr_buy"] == 0.8

