import copy
import logging
from math import factorial

from .deprecated.permclassdeprecated import PermClassDeprecatedMixin
from .permset import PermSet
from .permutation import Permutation
from .utils import copy_func

logging.basicConfig(level=logging.INFO)


class PermClass(list, PermClassDeprecatedMixin):
    @staticmethod
    def class_from_test(test, max_len=8, has_all_syms=False):
        """Return the smallest PermClass of all permutations that satisfy the test.

        Args:
            test (func): function that accepts a permutation and returns a Boolean.
                Should be closed downward.
            max_len (int): maximum length to be included in class
            has_all_syms (bool): whether the class is known to be closed under
                all symmetries.

        Returns:
            PermClass: class of permutations that satisfy the test.
        """

        C = [
            PermSet(Permutation())
        ]  # List consisting of just the PermSet containing the empty Permutation
        for length in range(1, max_len + 1):
            if len(C[length - 1]) == 0:
                return PermClass(C)

            new_set = PermSet()
            to_check = PermSet(set.union(*[p.covered_by() for p in C[length - 1]]))
            to_check = PermSet(
                p for p in to_check if PermSet(p.covers()).issubset(C[length - 1])
            )

            while to_check:
                p = to_check.pop()

                if test(p):
                    if has_all_syms:
                        syms = PermSet(p.symmetries())
                        new_set += syms
                        to_check -= syms
                    else:
                        new_set.add(p)
                else:
                    if has_all_syms:
                        to_check -= PermSet(p.symmetries())

            C.append(new_set)

        return PermClass(C, test)

    def __init__(cls, C, test=None):
        super(PermClass, cls).__init__(C)
        cls.length = len(C) + 1
        cls.test = test

    def __contains__(self, p):
        if len(p) > len(self):
            return self.test(p)

        return p in self[len(p)]

    def filter_by(self, test):
        """Modify self by removing those permutations that fail the test.

        Note:
            Does not actually ensure the result is a class.
        """
        for i in range(len(self)):
            for p in list(self[i]):
                if not test(p):
                    self[i].remove(p)

    def guess_basis(self, max_length=6, search_mode=False):
        """Guess a basis for the class up to "max_length" by iteratively
        generating the class with basis elements known so far (initially {})
        and adding elements that should be avoided to the basis.

        Search mode goes up to the max length in the class and prints out the
        number of basis elements of each length on the way.
        """
        assert max_length < len(self), "Class not big enough to check that far!"

        # Find the first length at which perms are missing.
        for idx, S in enumerate(self):
            if len(S) < factorial(idx):
                start_length = idx
                break
        else:
            # If we're here, then self is the class of all permutations.
            return PermSet()

        # Add missing perms of minimum length to basis.
        missing = PermSet.all(start_length) - self[start_length]
        basis = missing

        length = start_length
        current = PermSet(Permutation.gen_all(length - 1))
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
        """
        Notes:
            The resulting class has a test built from those of `self` and `other`.
        """
        self_test = copy_func(self.test)
        other_test = copy_func(other.test)
        return PermClass(
            [S_1 + S_2 for S_1, S_2 in zip(self, other)],
            test=lambda p: self_test(p) or other_test(p),
        )

    def extend(self, t):
        for i in range(t):
            self.append(self[-1].right_extensions(test=self.test))

    def extended(self, t):
        C = copy.deepcopy(self)
        C.extend(t)
        return C

    def heatmap(self, **kwargs):
        permset = PermSet(
            set.union(*self)
        )  # Collect all perms in self into one PermSet
        permset.heatmap(**kwargs)

    def skew_closure(self, max_len=8, has_all_syms=False):
        """
        Notes:
            This will raise an IndexError if the resulting class is extended.
        Todos:
            Check that the `test` works properly, even if `self` is modified.
        Examples:
            >>> p = Permutation(21)
            >>> C = PermClass.class_from_test(lambda q: p not in q)
            >>> D = C.skew_closure()
            >>> len(D[8]) == 128
            True
        """
        if self.test:
            test = copy_func(self.test)

            def is_skew(p):
                return all(test(q) for q in p.skew_decomposition())

        else:
            C = copy.deepcopy(self)

            def is_skew(p):
                return all(q in C for q in p.skew_decomposition())

        return PermClass.class_from_test(
            is_skew, max_len=max_len, has_all_syms=has_all_syms
        )

    def sum_closure(self, max_len=8, has_all_syms=False):
        """
        Notes:
            This will raise an IndexError if the resulting class is extended.
        Todos:
            Check that the `test` works properly, even if `self` is modified.
        Examples:
            >>> p = Permutation(12)
            >>> C = PermClass.class_from_test(lambda q: p not in q)
            >>> D = C.sum_closure()
            >>> len(D[8]) == 128
            True
        """
        if self.test:
            test = copy.deepcopy(self.test)

            def is_sum(p):
                return all(test(q) for q in p.sum_decomposition())

        else:
            C = copy.deepcopy(self)

            def is_sum(p):
                return all(q in C for q in p.sum_decomposition())

        return PermClass.class_from_test(
            is_sum, max_len=max_len, has_all_syms=has_all_syms
        )


if __name__ == "__main__":
    pass
