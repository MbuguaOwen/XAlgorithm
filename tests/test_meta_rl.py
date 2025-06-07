from memory.meta_rl import MetaReinforcer


def test_meta_reinforcer_decay():
    mr = MetaReinforcer(fallback_threshold=0.5, min_trades=10)
    mr.update_performance("bull", 0.6, 0.1, 20)
    assert mr.get_weight("bull") > 0.9
    mr.update_performance("bull", 0.4, -0.1, 30)
    assert mr.get_weight("bull") < 1.0
    for _ in range(3):
        mr.update_performance("bull", 0.3, -0.2, 30)
    assert mr.get_weight("bull") <= 0.1
