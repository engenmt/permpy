from permpy.staircase import gen_interval_divisions, first_two_cells, add_two_cells


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
