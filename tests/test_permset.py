import random
from permpy import Permutation, PermSet


def test_union():
    S = PermSet.all(3) + PermSet.all(4)
    expected = 30
    result = len(S)
    assert result == expected, (
        "The union of PermSet.all(3) and PermSet.all(4) should have"
        f" {expected} permutations, but it has {result} instead."
    )


def test_difference():
    S = PermSet.all(3) - PermSet(Permutation(123))
    expected = 5
    result = len(S)
    assert result == expected, (
        "The result of removing Permutation(123) from PermSet.all(3)"
        f" should have {expected} permutations, but it has {result} instead."
    )


def test_all():
    expected = PermSet([Permutation(12), Permutation(21)])
    result = PermSet.all(2)
    assert result == expected, "PermSet.all(2) is incorrect!"


def test_get_random():
    random.seed(1324)
    S = PermSet.all(4)
    p = S.get_random()
    assert p in S, "PermSet.get_random() returned a permutation not in the PermSet!"


def test_get_length():
    S = PermSet.all(3) + PermSet.all(4)
    expected = PermSet.all(3)
    result = S.get_length(3)
    assert result == expected, "PermSet.get_length incorrectly filtered by length!"
