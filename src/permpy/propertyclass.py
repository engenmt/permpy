import copy
import logging

from .permutation import Permutation
from .permset import PermSet
from .permclass import PermClass
from .utils import copy_func

logging.basicConfig(level=logging.INFO)


class PropertyClass(PermClass):
    def __init__(self, property, max_len=8):
        """Return the smallest PermClass of all permutations that satisfy the test.

        Args:
            property (func): function that accepts a permutation and returns a
                Boolean. Should be a hereditary property.
            max_len (int): maximum length to be included in class

        Returns:
            PropertyClass: class of permutations that satisfy the property.

        """

        C = [
            PermSet(Permutation())
        ]  # List consisting of just the PermSet containing the empty Permutation

        for _ in range(max_len):
            C.append(C[-1].right_extensions(test=property, trust=True))

        super().__init__(C)
        self.property = property

    def __contains__(self, p):
        p_length = len(p)
        if p_length > self.max_len:
            return self.property(p)
        return p in self[p_length]

    def add_property(self, property):
        """Modify self by removing those permutations that fail the test."""
        for length in range(len(self)):
            for p in list(self[length]):
                if not property(p):
                    self[length].remove(p)
        self.property = lambda p: self.property(p) and property(p)

    def union(self, other):
        """
        Examples:
            >>> inc = Permutation(12)
            >>> D = PropertyClass(lambda p: inc not in p)
            >>> dec = Permutation(21)
            >>> I = PropertyClass(lambda p: dec not in p)
            >>> U = D.union(I)
            >>> len(U[8])
            2
            >>> U.extend(1)
            >>> len(U[9])
            2

        """
        property_self = copy_func(self.property)
        property_other = copy_func(other.property)

        C = PermClass.union(self, other)
        C.__class__ = PropertyClass
        C.property = lambda p: property_self(p) or property_other(p)

        return C

    def extend(self, t):
        for _ in range(t):
            self.data.append(self[-1].right_extensions(test=self.property))

    def extended(self, t):
        C = copy.deepcopy(self)
        C.extend(t)
        return C

    def skew_closure(self, max_len=8):
        """
        Examples:
            >>> p = Permutation(21)
            >>> C = PropertyClass(lambda q: p not in q) # Class of increasing permutations
            >>> D = C.skew_closure(max_len=7) # Class of co-layered permutations
            >>> len(D[7]) == 64
            True
        """
        property = copy_func(self.property)

        def is_skew(p):
            return all(property(q) for q in p.skew_decomposition())

        return PropertyClass(is_skew, max_len=max_len)

    def sum_closure(self, max_len=8):
        """
        Examples:
            >>> p = Permutation(12)
            >>> C = PropertyClass(lambda q: p not in q) # Class of decreasing permutations
            >>> D = C.sum_closure(max_len=7) # Class of layered permutations
            >>> len(D[7]) == 64
            True
        """
        property = copy_func(self.property)

        def is_skew(p):
            return all(property(q) for q in p.sum_decomposition())

        return PropertyClass(is_skew, max_len=max_len)
