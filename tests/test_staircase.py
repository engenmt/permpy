from permpy.staircase import (
    MonotoneStaircase,
)


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
        (-1, -1, -1): [1, 1, 2, 6, 21, 72, 231, 698, 2018],
        (-1, -1, +1): [1, 1, 2, 6, 19, 58, 170, 483, 1342],
        (-1, +1, -1): [1, 1, 2, 6, 18, 51, 139, 371, 980],
        (-1, +1, +1): [1, 1, 2, 6, 19, 58, 170, 483, 1342],
        (+1, -1, -1): [1, 1, 2, 6, 19, 58, 170, 483, 1342],
        (+1, -1, +1): [1, 1, 2, 5, 13, 34, 89, 233, 610],
        (+1, +1, -1): [1, 1, 2, 6, 19, 58, 170, 483, 1342],
        (+1, +1, +1): [1, 1, 2, 5, 14, 42, 128, 384, 1123],
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
