from permpy.staircase import gen_interval_divisions


def test_gen_interval_divisions():
    m = 4
    k = 2
    result = list(gen_interval_divisions(m, k))
    expected = [
        [(), (0, 1, 2, 3)],
        [(0,), (1, 2, 3)],
        [(0, 1), (2, 3)],
        [(0, 1, 2), (3,)],
        [(0, 1, 2, 3), ()],
    ]
    assert (
        result == expected
    ), f"gen_interval_divisions({m},{k}) did not generate correctly!"
