from permpy import Permutation as Perm


def test_is_monotone_increasing():
    cases_true = [Perm.monotone_increasing(n) for n in range(10)]
    for p in cases_true:
        assert (
            p.is_increasing()
        ), f"{p} is increasing, but Perm.is_increasing returns False."

    cases_false = [Perm(12354), Perm(43215), Perm(51234)]
    for p in cases_false:
        assert (
            not p.is_increasing()
        ), f"{p} is not increasing, but Perm.is_increasing returns True."


def test_is_monotone_decreasing():
    cases_true = [Perm.monotone_decreasing(n) for n in range(10)]
    for p in cases_true:
        assert (
            p.is_decreasing()
        ), f"{p} is decreasing, but Perm.is_decreasing returns False."

    cases_false = [Perm(54312), Perm(15432), Perm(25314)]
    for p in cases_false:
        assert (
            not p.is_decreasing()
        ), f"{p} is not decreasing, but Perm.is_decreasing returns True."
