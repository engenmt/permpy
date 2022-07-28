from math import factorial
import logging

from .permutation import Permutation
from .permset import PermSet
from .permclass import PermClass


class AvClass(PermClass):
    """An object representing an avoidance class."""

    def __init__(self, basis, max_len=8):

        if isinstance(basis, Permutation):
            basis = [basis]
        else:
            basis = [Permutation(b) for b in basis]

        C = [
            PermSet(Permutation())
        ]  # List consisting of just the PermSet containing the empty Permutation

        if max_len > 0:
            if Permutation(1) not in basis:
                C.append(PermSet(Permutation(1)))
            else:
                C.append(PermSet())

            for length in range(max_len - 1):
                C.append(C[-1].right_extensions(basis=basis, trust=True))

        super().__init__(C)
        self.basis = basis

    def __repr__(self):
        basis_str = ", ".join(f"{p}" for p in self.basis)
        return f"Av({basis_str})"

    def extend_by_one(self, trust=True):
        """Extend `self` by right-extending its ultimate PermSet.

        Args:
            trust (bool): Whether of not we can trust the insertion values of
                the ultimate PermSet. In this context, we generally can.

        """
        logging.debug(f"Calling extend_by_one({self}, trust={trust})")
        self.append(self[-1].right_extensions(basis=self.basis, trust=trust))

    def extend_to_length(self, length, trust=True):
        for _ in range(self.max_len + 1, length + 1):
            self.extend_by_one(trust=trust)

    def extend_by_length(self, length, trust=True):
        for _ in range(length):
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
        """Check if `self` contains `other` as a permutation class using their bases."""
        for p in self.basis:
            for q in other.basis:
                if p in q:
                    break
            else:
                # If we're here, then `p` is not involved in any of the basis elements of `q`, so
                # the permutation `p` lies in `other` but not `self`.
                return False
        return True
