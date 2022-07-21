from permpy import PermClass, PermSet, Permutation


def test_all():
    C = PermClass.all(6)
    expected = [1, 1, 2, 6, 24, 120, 720]
    result = [len(S) for S in C]
    assert (
        result == expected
    ), "PermClass.all(6) does not contain all permutations up to length 6!"


def test_maximally_extend():
    C = PermClass.all(4)
    C[4].remove(Permutation(1234))
    C.maximally_extend(1)
    expected = 103
    result = len(C[5])  # All but the 17 permutations covering 1234
    assert result == expected, (
        f"Maximally extending the class without 1234 gives {result} permutations"
        f" of length 5 instead of the correct number, {expected}."
    )


def test_filter_by():
    C = PermClass.all(6)
    p = Permutation(21)
    C.filter_by(lambda q: p not in q)  # C now only contains permutations avoiding 21
    result = all(
        len(S) == 1 for S in C
    )  # This should only be one permutation of each length
    expected = True
    assert (
        result == expected
    ), "Filtering by 21-avoidance didn't yield one permutation of each length!"


def test_guess_basis():
    p = Permutation(12)

    C = PermClass.all(8)
    C.filter_by(lambda q: p not in q)  # Class of decreasing permutations
    assert C.guess_basis() == PermSet(p), (
        "PermClass.guess_basis guessed incorrectly"
        " on the class of decreasing permutations."
    )

    D = C.sum_closure()  # Class of layered permutations
    assert D.guess_basis() == PermSet([Permutation(312), Permutation(231)]), (
        "PermClass.guess_basis guessed incorrectly"
        " on the class of layered permutations."
    )


def test_skew_closure():
    p = Permutation(21)
    C = PermClass.all(8)
    C.filter_by(lambda q: p not in q)  # Class of increasing permutations
    D = C.skew_closure(max_len=7)  # Class of co-layered permutations

    expected = 64
    result = len(D[7])

    assert result == expected, (
        "PermClass.skew_closure incorrectly generated"
        " the class of co-layered permtuations!"
    )


def test_sum_closure():
    p = Permutation(12)
    C = PermClass.all(8)
    C.filter_by(lambda q: p not in q)  # Class of decreasing permutations
    D = C.sum_closure(max_len=7)  # Class of layered permutations

    expected = 64
    result = len(D[7])

    assert (
        result == expected
    ), "PermClass.sum_closure incorrectly generated the class of layered permtuations!"
