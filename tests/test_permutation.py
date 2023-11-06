import random
from collections import Counter
from permpy import Permutation as Perm


def test_new_increasing():
    p = Perm(12345)
    candidates = [
        Perm("12345"),
        Perm([1, 2, 3, 4, 5]),
        Perm([-100, -3, 0, 1.2, 10e6]),
        Perm(p),
        Perm.monotone_increasing(5),
        sum((Perm(1) for _ in range(4)), Perm(1)),
    ]
    for q in candidates:
        assert p == q, f"Expected the increasing perm of length 5, got {q} instead."


def test_new_perm():
    p = Perm(51423)
    candidates = [
        Perm("51423"),
        Perm([5, 1, 4, 2, 3]),
        Perm([10e5, 10e1, 10e4, 10e2, 10e3]),
        Perm(p),
    ]
    for q in candidates:
        assert p == q, f"Expected the perm {p}, got {q} instead."


def test_direct_sum():
    p = Perm(4321)
    q = Perm(3124)
    r = Perm(1)
    cases = {
        (p, p): Perm(43218765),
        (p, q): Perm(43217568),
        (q, p): Perm(31248765),
        (q, q): Perm(31247568),
        (q, r): Perm(31245),
    }
    for (r, s), intended in cases.items():
        direct_sum = r + s
        assert direct_sum == intended, (
            f"Expected the direct sum of {r} and {s} to be {intended},"
            f" got {direct_sum} instead."
        )


def test_skew_sum():
    p = Perm(4321)
    q = Perm(3124)
    r = Perm(1)
    cases = {
        (p, p): Perm(87654321),
        (p, q): Perm(87653124),
        (q, p): Perm(75684321),
        (q, q): Perm(75683124),
        (q, r): Perm(42351),
    }
    for (r, s), intended in cases.items():
        skew_sum = r - s
        assert skew_sum == intended, (
            f"Expected the skew sum of {r} and {s} to be {intended},"
            f" got {skew_sum} instead."
        )


def test_monotone_increasing():
    p = Perm.monotone_increasing(5)
    assert p == Perm(12345), f"Perm.monotone_increasing is not increasing: {p}"


def test_monotone_decreasing():
    p = Perm.monotone_decreasing(5)
    assert p == Perm(54321), f"Perm.monotone_decreasing is not decreasing: {p}"


def test_random_length():
    for n in range(10):
        p = Perm.random(n)
        assert len(p) == n, f"Length of Perm.random({n}) is {len(p)}!"


def test_random_avoider():
    random.seed(123)
    q = 123
    n = 8
    for _ in range(10):
        p = Perm.random_avoider(n, [q])
        assert not p.involves(
            q
        ), f"A random perm of length {n} avoiding {q} doesn't avoid {q}!"


def test_call():
    p = Perm(4132)
    idx = 2
    val = p(idx)
    expected = 2
    assert val == expected, f"Called Perm({p})({idx}), got {val} instead of {expected}!"


def test_containment():
    containments = [
        (Perm(1), Perm(21)),
        (Perm(132), Perm(4132)),
    ]
    for p, q in containments:
        assert q.__contains__(p), f"The containment of {p} in {q} was not affirmed!"

    noncontainments = [(Perm(231), Perm(1234))]
    for p, q in noncontainments:
        assert not q.__contains__(
            p
        ), f"The noncontainment of {p} in {q} was not affirmed!"


def test_pow():
    p = Perm(12345)
    for n in range(-5, 0, 5):
        result = p**n
        assert p == result, (
            f"The identity permutation raised to the power {n}"
            f" resulted in {result}, not the identity permutation."
        )

    q = Perm(41352)
    powers = [Perm(12345), q, Perm(54321), Perm(25314), Perm(12345)]
    for exp, val in enumerate(powers):
        assert q**exp == val, f"Perm({q})^{exp} should be {val}, but it's {q**exp}!"


def test_perm_to_ind():
    p = Perm(24513)
    expected = 42
    result = p.perm_to_ind()
    assert (
        result == expected
    ), f"Perm({p}).perm_to_ind() returned {result} instead of {expected}!"


def test_ind_to_perm():
    k, n = 42, 5
    expected = Perm(24513)
    result = Perm(k, n)
    assert (
        result == expected
    ), f"Perm({k}, {n}).ind_to_perm() returned {result} instead of {expected}!"


def test_delete():
    p = Perm(35214)
    cases = [
        (dict(indices=2), Perm(2413)),
        (dict(indices=[2, 4]), Perm(231)),
        (dict(values=1), Perm(2413)),
        (dict(values=[4]), Perm(3214)),
    ]
    for kwargs, expected in cases:
        result = p.delete(**kwargs)
        assert (
            result == expected
        ), f"Perm({p}).delete({kwargs}) returned incorrect value!"

    for idx, val in enumerate(p):
        assert p.delete(indices=idx) == p.delete(values=val), (
            f"Perm({p}).delete(indices={idx}) did not equal"
            f" Perm({p}).delete(values={val})!"
        )


def test_insert():
    p = Perm(2413)
    inserted = p.insert(2, 1)
    expected = Perm(35214)
    assert (
        inserted == expected
    ), f"Perm({p}).insert(2,1) should equal Perm({expected}), not {inserted}."


def test_complement():
    cases = [
        (Perm(12345), Perm(54321)),
        (Perm(25314), Perm(41352)),
    ]
    for p, expected in cases:
        result = p.complement()
        assert (
            result == expected
        ), f"Perm({p}).complement() should equal {expected}, not {result}."


def test_reverse():
    cases = [
        (Perm(12345), Perm(54321)),
        (Perm(25314), Perm(41352)),
    ]
    for p, expected in cases:
        result = p.reverse()
        assert (
            result == expected
        ), f"Perm({p}).reverse() should equal {expected}, not {result}."


def test_inverse():
    cases = [
        (Perm(12345), Perm(12345)),
        (Perm(25314), Perm(41352)),
    ]
    for p, expected in cases:
        result = p.inverse()
        assert (
            result == expected
        ), f"Perm({p}).inverse() should equal {expected}, not {result}."


def test_pretty_out():
    p = Perm("1 9 3 7 5 6 4 8 2 10")
    expected = "\n".join(
        [
            "                  10",
            "   9",
            "               8",
            "       7",
            "           6",
            "         5",
            "             4",
            "     3",
            "                 2",
            " 1",
        ]
    )
    assert p.pretty_out() == expected, f"Perm({p}).pretty_out() is incorrect!"


def test_fixed_points():
    p = Perm(521436)
    result = p.fixed_points()
    expected = [1, 3, 5]
    assert (
        result == expected
    ), f"Perm({p}).fixed_points() should equal {expected}, not {result}."


def test_sum_decomposable():
    cases_true = [
        Perm(1234),
        Perm(1324),
        Perm(2134),
        Perm(2314),
        Perm(3124),
        Perm(3214),
    ]
    for p in cases_true:
        assert (
            p.sum_decomposable()
        ), f"Perm({p}).sum_decomposable() returns False, which is incorrect."

    cases_false = [
        Perm(3142),
        Perm(4123),
        Perm(4132),
        Perm(4213),
        Perm(4231),
        Perm(4312),
        Perm(4321),
    ]
    for p in cases_false:
        assert (
            not p.sum_decomposable()
        ), f"Perm({p}).sum_decomposable() returns True, which is incorrect."


def test_sum_decomposition():
    summands = [Perm(1), Perm(312), Perm(21)]
    direct_sum = summands[0] + summands[1] + summands[2]
    decomposition = direct_sum.sum_decomposition()
    assert decomposition == summands, (
        f"Perm({direct_sum}).sum_decomposition() returned {decomposition},"
        f" but it should return {summands}.",
    )


def test_descents():
    p = Perm(42561873)
    expected = [0, 3, 5, 6]
    result = p.descents()
    assert (
        result == expected
    ), f"Perm({p}).descents() returned {result}, but it should return {expected}."


def test_ascents():
    p = Perm(42561873)
    expected = [1, 2, 4]
    result = p.ascents()
    assert (
        result == expected
    ), f"Perm({p}).ascents() returned {result}, but it should return {expected}."


def test_peaks():
    p = Perm(2341765)
    expected = [2, 4]
    result = p.peaks()
    assert (
        result == expected
    ), f"Perm({p}).peaks() returned {result}, but it should return {expected}."


def test_valleys():
    p = Perm(3241756)
    expected = [1, 3, 5]
    result = p.valleys()
    assert (
        result == expected
    ), f"Perm({p}).valleys() returned {result}, but it should return {expected}."


def test_ltr_min():
    p = Perm(35412)
    expected = [0, 3]
    result = p.ltr_min()
    assert (
        result == expected
    ), f"Perm({p}).ltr_min() returned {result}, but it should return {expected}."


def test_rtl_min():
    p = Perm(315264)
    expected = [5, 3, 1]
    result = p.rtl_min()
    assert (
        result == expected
    ), f"Perm({p}).rtl_min() returned {result}, but it should return {expected}."


def test_ltr_max():
    p = Perm(35412)
    expected = [0, 1]
    result = p.ltr_max()
    assert (
        result == expected
    ), f"Perm({p}).ltr_max() returned {result}, but it should return {expected}."


def test_rtl_max():
    p = Perm(35412)
    expected = [4, 2, 1]
    result = p.rtl_max()
    assert (
        result == expected
    ), f"Perm({p}).rtl_max() returned {result}, but it should return {expected}."


def test_inversions():
    cases = [
        (Perm(4132), [(0, 1), (0, 2), (0, 3), (2, 3)]),
        (Perm.monotone_increasing(7), []),
    ]
    for p, expected in cases:
        result = p.inversions()
        assert (
            result == expected
        ), f"Perm({p}).inversions() returned {result}, but it should return {expected}."


def test_noninversions():
    cases = [
        (Perm(4132), [(1, 2), (1, 3)]),
        (Perm.monotone_decreasing(7), []),
    ]
    for p, expected in cases:
        result = p.noninversions()
        assert result == expected, (
            f"Perm({p}).noninversions() returned {result},"
            f" but it should return {expected}."
        )


def test_breadth():
    p = Perm(3142)
    expected = 3
    result = p.breadth()
    assert (
        result == expected
    ), f"Perm({p}).breadth() returned {result}, but it should return {expected}."


def test_pattern_counts():
    p = Perm(1324)
    expected = Counter({Perm(123): 2, Perm(132): 1, Perm(213): 1})
    result = p.pattern_counts(3)
    assert result == expected, (
        f"Perm({p}).pattern_counts(3) returned {result},"
        f" but it should return {expected}."
    )


def test_avoids():
    cases = [
        (Perm(123456), 231, True),
        (Perm(123456), 123, False),
    ]
    for p, q, expected in cases:
        result = p.avoids(q)
        assert (
            result == expected
        ), f"Perm({p}).avoids({q}) is {result}, but it should be {expected}."


def test_involves():
    cases = [
        (Perm(123456), 231, False),
        (Perm(123456), 123, True),
    ]
    for p, q, expected in cases:
        result = p.involves(q)
        assert (
            result == expected
        ), f"Perm({p}).involves({q}) is {result}, but it should be {expected}."


def test_involved_in():
    cases = [
        (Perm(123), 31542, False),
        (Perm(213), 54213, True),
    ]
    for p, q, expected in cases:
        result = p.involved_in(q)
        assert (
            result == expected
        ), f"Perm({p}).involved_in({q}) is {result}, but it should be {expected}."


def test_involved_in():
    cases = [
        (Perm(123), [set([Perm()]), set([Perm(1)]), set([Perm(12)]), set([Perm(123)])]),
    ]
    for perm, expected in cases:
        result = perm.downset()
        assert (
            result == expected
        ), f"Perm({perm}).downset() is {result}, but it should be {expected}."
