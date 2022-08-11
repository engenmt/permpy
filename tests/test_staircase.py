from math import comb
from permpy.utils import (
    gen_compositions,
    gen_interval_divisions,
)
from permpy.staircase import (
    first_two_cells,
    add_two_cells,
    MonotoneStaircase,
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


def test_monotone_class_two_cells():
    N = 10
    cases = [
        ([+1, +1], lambda n: 2 ** n - n),
        ([-1, -1], lambda n: 2 ** n - n),
        ([+1, -1], lambda n: 2 ** (n - 1)),
        ([-1, +1], lambda n: 2 ** (n - 1)),
    ]
    for cells, count in cases:
        M = MonotoneStaircase(cells, n=N)
        assert (
            len(M.perm_class) == N + 1
        ), "MonotoneStaircase has the incorrect length after generation!"
        for n, perm_set in enumerate(M.perm_class[1:], start=1):
            result = len(perm_set)
            expected = count(n)
            assert result == expected, (
                f"MonotoneStaircase({cells})[{n}]) has {result} permutations,"
                f" but it should have {expected}!"
            )


def test_monotone_class_three_cells():
    N = 8
    cases = {
        (+1, -1, +1): [1, 1, 2, 5, 13, 34, 89, 233, 610],
        (-1, +1, -1): [1, 1, 2, 6, 18, 51, 139, 371, 980],
        (+1, +1, +1): [1, 1, 2, 5, 14, 42, 128, 384, 1123],
        (-1, -1, +1): [1, 1, 2, 6, 19, 58, 170, 483, 1342],
        (-1, +1, +1): [1, 1, 2, 6, 19, 58, 170, 483, 1342],
        (+1, -1, -1): [1, 1, 2, 6, 19, 58, 170, 483, 1342],
        (+1, +1, -1): [1, 1, 2, 6, 19, 58, 170, 483, 1342],
        (-1, -1, -1): [1, 1, 2, 6, 21, 72, 231, 698, 2018],
    }
    for cells, count in cases.items():
        M = MonotoneStaircase(cells, n=N)
        for n, perm_set in enumerate(M.perm_class[1:], start=1):
            result = len(perm_set)
            expected = count[n]
            assert result == expected, (
                f"MonotoneStaircase({cells})[{n}]) has {result} permutations,"
                f" but it should have {expected}!"
            )


def test_first_two_cells():
    for n in range(1, 10):
        S = first_two_cells(n)
        result = len(S)
        expected = 2 ** (n + 1) - 1
        assert result == expected, (
            f"first_two_cells({n}) has {result} partial griddings,"
            f" but it should have {expected}!"
        )


def test_first_two_cells_distinct():
    for n in range(1, 10):
        S = set(p for p, _ in first_two_cells(n))
        result = len(S)
        expected = 2 ** n
        assert result == expected, (
            f"first_two_cells({n}) has {result} distinct permutations,"
            f" but it should have {expected}!"
        )


def test_add_two_cells():
    expected_vals = [1, 3, 8, 22, 63, 185, 550, 1644, 4925, 14767]
    for n, expected in enumerate(expected_vals):
        two_cells = first_two_cells(n)
        four_cells = add_two_cells(two_cells, n)
        result = len(four_cells)
        assert result == expected, (
            f"four_cells({n}) has {result} partial griddings,"
            f" but it should have {expected}!"
        )


def test_add_two_cell_distinct():
    expected_vals = [1, 2, 4, 10, 30, 96, 308, 974, 3034, 9340]
    for n, expected in enumerate(expected_vals):
        two_cells = first_two_cells(n)
        four_cells = add_two_cells(two_cells, n)
        four_cells_distinct = set(p for p, _ in four_cells)
        result = len(four_cells_distinct)
        assert result == expected, (
            f"four_cells({n}) has {result} distinct permutations,"
            f" but it should have {expected}!"
        )
