from permpy import PropertyClass, Permutation


def test_union():
    inc = Permutation(12)
    Dec = PropertyClass(lambda p: inc not in p)  # Class of decreasing permutations
    dec = Permutation(21)
    Inc = PropertyClass(lambda p: dec not in p)  # Class of increasing permutations

    U = Dec.union(Inc)
    result = len(U[8])  # Length of ultimate PermSet
    expected = 2
    assert (
        result == expected
    ), "The union of the increasing and decreasing PropertyClasses is incorrect!"

    U.extend(1)
    result_extension = len(U[9])
    expected_extension = 2
    assert result_extension == expected_extension, (
        "The extension of the union of the increasing and decreasing PropertyClasses"
        " is incorrect!"
    )


def test_skew_closure():
    p = Permutation(21)
    C = PropertyClass(lambda q: p not in q)  # Class of increasing permutations
    D = C.skew_closure(max_len=7)  # Class of co-layered permutations

    result = len(D[7])
    expected = 64
    assert result == expected, (
        "PropertyClass.skew_closure incorrectly generated"
        " the class of co-layered permutations!"
    )


def test_sum_closure():
    p = Permutation(12)
    C = PropertyClass(lambda q: p not in q)  # Class of decreasing permutations
    D = C.sum_closure(max_len=7)  # Class of layered permutations

    result = len(D[7])
    expected = 64
    assert result == expected, (
        "PropertyClass.sum_closure incorrectly generated"
        " the class of layered permutations!"
    )
