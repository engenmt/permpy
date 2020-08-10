import logging
from math import factorial

from .permclass import PermClass
from .permset import PermSet
from .permutation import Permutation


class AvClass(PermClass):
    """An object representing an avoidance class.

    Notes:
        Does not contain the empty permutation.

    Examples:
        >>> B = [123]
        >>> A = AvClass(B, length=4)
        >>> for S in A:
        ...    print(S)
        ...
        Set of 0 permutations
        Set of 1 permutations
        Set of 2 permutations
        Set of 5 permutations
        Set of 14 permutations
    """

    def __init__(self, basis, length=8, verbose=0):

        list.__init__(self, [PermSet()])
        if isinstance(basis, Permutation):
            self.basis = [basis]
        else:
            self.basis = [Permutation(b) for b in basis]

        self.test = lambda p: all(b not in p for b in basis)

        p = Permutation([0], clean=True)
        if length >= 1:
            if p not in self.basis:
                self.append(PermSet(p))
                self.length = 1
                self.extend_to_length(length)
            else:
                for _ in range(length):
                    self.append(PermSet())

    def extend_by_one(self, trust=True):
        """Extend `self` by right-extending its ultimate PermSet.

        Args:
            trust (bool): Whether of not we can trust the insertion values of
                the ultimate PermSet. In this context, we generally can.
        """
        logging.debug(f"Calling extend_by_one({self}, trust={trust})")
        self.length += 1
        self.append(self[-1].right_extensions(basis=self.basis, trust=trust))

    def extend_to_length(self, length, trust=True):
        if length <= self.length:
            return

        for n in range(self.length + 1, length + 1):
            self.extend_by_one(trust=trust)

    def extend_by_length(self, length, trust=True):
        for n in range(length):
            self.extend_by_one(trust=trust)

    def right_juxtaposition(self, C, generate_perms=True):
        A = PermSet()
        max_length = max([len(P) for P in self.basis]) + max([len(P) for P in C.basis])
        for n in range(2, max_length + 1):
            for i in range(0, factorial(n)):
                P = Permutation(i, n)
                for Q in self.basis:
                    for R in C.basis:
                        if len(Q) + len(R) == n:
                            if Q == Permutation(P[0 : len(Q)]) and R == Permutation(
                                P[len(Q) : n]
                            ):
                                A.add(P)
                        elif len(Q) + len(R) - 1 == n:
                            if Q == Permutation(P[0 : len(Q)]) and Permutation(
                                R
                            ) == Permutation(P[len(Q) - 1 : n]):
                                A.add(P)
        return AvClass(list(A.minimal_elements()), length=(8 if generate_perms else 0))

    def above_juxtaposition(self, C, generate_perms=True):
        inverse_class = AvClass([P.inverse() for P in C.basis], 0)
        horizontal_juxtaposition = self.right_juxtaposition(
            inverse_class, generate_perms=False
        )
        return AvClass(
            [B.inverse() for B in horizontal_juxtaposition.basis],
            length=(8 if generate_perms else 0),
        )

    def contains(self, other):
        """Check if `self` contains `other` as a permutation class using their bases.
        """
        for p in self.basis:
            for q in other.basis:
                if p in q:
                    break
            else:
                # If we're here, then `p` is not involved in any of the basis elements
                # of `q`, so the permutation `p` lies in `other` but not `self`.
                return False
        return True


# TODO: What is this for?
if __name__ == "__main__":
    print()

    B = [123]
    A = AvClass(B, 12)
    for idx, S in enumerate(A):
        print(S)
