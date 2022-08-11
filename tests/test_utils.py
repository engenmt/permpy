from math import comb
from permpy.utils import (
    gen_compositions,
    gen_interval_divisions,
)


def test_gen_compositions():
    for k in range(1, 9):
        for n in range(k, 17):
            result = len(set(gen_compositions(n, k)))
            expected = comb(n - 1, k - 1)
            assert result == expected, (
                f"gen_compositions({n}, {k}) yielded {result} compositions,"
                f" but it should have yeilded {expected}."
            )


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
