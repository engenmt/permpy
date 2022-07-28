import copy
import logging

from math import factorial

from .permutation import Permutation
from .permset import PermSet
from .deprecated.permclassdeprecated import PermClassDeprecatedMixin

logging.basicConfig(level=logging.INFO)


class ClassTooShortError(Exception):
    pass


class PermClass(PermClassDeprecatedMixin):
    """A minimal Python class representing a Permutation class.

    Notes:
        This class assumes the Permutation class is closed downwards but
            does not assert this fact.

    """

    def __init__(self, C):
        self.data = C
        self.max_len = len(C) - 1

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        try:
            return self.data[idx]
        except IndexError:
            raise ClassTooShortError

    def __add__(self, other):
        return self.union(other)

    def __contains__(self, p):
        p_length = len(p)
        if p_length > self.max_len:
            return False
        return p in self[p_length]

    @classmethod
    def all(cls, max_length):
        """Return the PermClass containing all permutations up to the given length."""
        return PermClass([PermSet.all(length) for length in range(max_length + 1)])

    def append(self, S):
        self.max_len += 1
        self.data.append(S)

    def maximally_extend(self, additional_length=1):
        """Extend `self` maximally.

        Notes:
            Includes only those permutations whose downsets lie entirely in `self`.

        """
        for _ in range(additional_length):
            self.data.append(
                PermSet(
                    p
                    for p in Permutation.gen_all(self.max_len + 1)
                    if p.covers().issubset(self[-1])
                )
            )
            self.max_len += 1

    def filter_by(self, property):
        """Modify `self` by removing permutations that do not satisfy the `property`."""
        for length in range(len(self)):
            for p in list(self[length]):
                if not property(p):
                    self[length].remove(p)

    def filtered_by(self, property):
        """Return a copy of `self` that has been filtered using the `property`."""
        C = copy.deepcopy(self)
        C.filter_by(property)
        return C

    def guess_basis(self, max_length=6):
        """Guess a basis for the class up to "max_length" by iteratively
        generating the class with basis elements known so far (initially the
        empty set) and adding elements that should be avoided to the basis.

        Search mode goes up to the max length in the class and prints out the
        number of basis elements of each length on the way.

        """
        assert (
            max_length <= self.max_len
        ), "The class is not big enough to check that far!"

        # Find the first length at which perms are missing.
        for length, S in enumerate(self):
            if len(S) < factorial(length):
                start_length = length
                break
        else:
            # If we're here, then `self` is the class of all permutations.
            return PermSet()

        # Add missing perms of minimum length to basis.
        missing = PermSet.all(start_length) - self[start_length]
        basis = missing

        length = start_length
        current = PermSet.all(length - 1)
        current = current.right_extensions(basis=basis)

        # Go up in length, adding missing perms at each step.
        while length < max_length:
            length += 1
            current = current.right_extensions(basis=basis)

            for perm in list(current):
                if perm not in self[length]:
                    basis.add(perm)
                    current.remove(perm)

        return basis

    def union(self, other):
        """Return the union of the two permutation classes."""
        return PermClass([S_1 + S_2 for S_1, S_2 in zip(self, other)])

    def heatmap(self, **kwargs):
        permset = PermSet(
            set().union(*self)
        )  # Collect all perms in self into one PermSet
        permset.heatmap(**kwargs)

    def skew_closure(self, max_len=8):
        """Return the skew closure of `self`.

        Todo:
            Implement constructively.

        """
        assert max_len <= self.max_len, "Can't make a skew-closure of that size!"
        L = []
        for length in range(max_len + 1):
            new_set = PermSet()
            for p in Permutation.gen_all(length):
                if all(q in self for q in set(p.skew_decomposition())):
                    new_set.add(p)
            L.append(new_set)

        return PermClass(L)

    def sum_closure(self, max_len=8):
        """Return the sum closure of `self`.

        Todo:
            Implement constructively.

        """
        assert max_len <= self.max_len, "Can't make a sum-closure of that size!"
        L = []
        for length in range(max_len + 1):
            new_set = PermSet()
            for p in Permutation.gen_all(length):
                if all(q in self for q in set(p.sum_decomposition())):
                    new_set.add(p)
            L.append(new_set)

        return PermClass(L)
