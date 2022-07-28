import permpy
import doctest


def test_permutation():
    doctest.testmod(permpy.permutation)


def test_permset():
    doctest.testmod(permpy.permset)


def test_permclass():
    doctest.testmod(permpy.permclass)


def test_propertyclass():
    doctest.testmod(permpy.propertyclass)


def test_avclass():
    doctest.testmod(permpy.avclass)
