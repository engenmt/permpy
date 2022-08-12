from permpy.permutation import Permutation
from permpy.permset import PermSet
from permpy.staircase import MonotoneStaircase


def test_one_cell_enumeration():
    N = 10
    cases = {
        (-1,): lambda _: 1,
        (+1,): lambda _: 1,
    }
    for cells, count in cases.items():
        M = MonotoneStaircase(cells, n=N)
        assert (
            len(M) == N + 1
        ), "MonotoneStaircase has the incorrect length after generation!"
        for n, perm_set in enumerate(M[1:], start=1):
            result = len(perm_set)
            expected = count(n)
            assert result == expected, (
                f"MonotoneStaircase({cells})[{n}] has {result} permutations,"
                f" but it should have {expected}!"
            )


def test_one_cell_basis():
    N = 10
    cases = {
        (-1,): [12],
        (+1,): [21],
    }
    for cells, expected_basis in cases.items():
        M = MonotoneStaircase(cells, n=N)
        result = M.guess_basis(max_length=N)
        expected_basis = PermSet(
            [Permutation(basis_elt) for basis_elt in expected_basis]
        )
        assert result == expected_basis, (
            f"MonotoneStaircase({cells}).guess_basis() returns {result},"
            f" but it should return {expected_basis}!"
        )


def test_two_cells_enumeration():
    N = 10
    cases = {
        (-1, -1): lambda n: 2 ** n - n,
        (-1, +1): lambda n: 2 ** (n - 1),
        (+1, -1): lambda n: 2 ** (n - 1),
        (+1, +1): lambda n: 2 ** n - n,
    }
    for cells, count in cases.items():
        M = MonotoneStaircase(cells, n=N)
        assert (
            len(M) == N + 1
        ), "MonotoneStaircase has the incorrect length after generation!"
        for n, perm_set in enumerate(M[1:], start=1):
            result = len(perm_set)
            expected = count(n)
            assert result == expected, (
                f"MonotoneStaircase({cells})[{n}] has {result} permutations,"
                f" but it should have {expected}!"
            )


def test_two_cells_basis():
    N = 10
    cases = {
        (-1, -1): [123, 2413, 3412],
        (-1, +1): [132, 231],
        (+1, -1): [312, 213],
        (+1, +1): [321, 2143, 3142],
    }
    for cells, expected_basis in cases.items():
        M = MonotoneStaircase(cells, n=N)
        result = M.guess_basis(max_length=N)
        expected_basis = PermSet(
            [Permutation(basis_elt) for basis_elt in expected_basis]
        )
        assert result == expected_basis, (
            f"MonotoneStaircase({cells}).guess_basis() returns {result},"
            f" but it should return {expected_basis}!"
        )


def test_three_cells_enumeration():
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
        for n, perm_set in enumerate(M[1:], start=1):
            result = len(perm_set)
            expected = count[n]
            assert result == expected, (
                f"MonotoneStaircase({cells})[{n}] has {result} permutations,"
                f" but it should have {expected}!"
            )


def test_three_cells_basis():
    N = 8
    cases = {
        (-1, -1, -1): [24135, 3412, 1234, 24153, 14253, 4123],
        (-1, -1, +1): [3412, 1243, 2413, 1423, 4123],
        (-1, +1, -1): [1324, 52143, 2314, 2341, 4231, 1342, 3421],
        (-1, +1, +1): [4132, 4231, 2431, 13524, 1432, 3421, 23154, 13254, 23514],
        (+1, -1, -1): [3412, 2134, 3142, 4123, 3124],
        (+1, -1, +1): [312, 2143],
        (+1, +1, -1): [21435, 31425, 31452, 3214, 3241, 4213, 3421, 4231, 21453],
        (+1, +1, +1): [321, 314265, 214635, 314625, 214365],
    }
    for cells, expected_basis in cases.items():
        M = MonotoneStaircase(cells, n=N)
        result = M.guess_basis(max_length=N)
        expected_basis = PermSet(
            [Permutation(basis_elt) for basis_elt in expected_basis]
        )
        assert result == expected_basis, (
            f"MonotoneStaircase({cells}).guess_basis() returns {result},"
            f" but it should return {expected_basis}!"
        )
