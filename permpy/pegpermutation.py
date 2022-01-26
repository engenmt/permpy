from math import fabs
from sympy import *
from itertools import chain, combinations

from .permutation import Permutation
from .permset import *
from .permclass import *
from .vectorset import *


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))


class PegPermutation(Permutation):
    def __new__(cls, p, signs):
        if isinstance(p, int):
            p = list(str(p))
        assert len(p) == len(signs), "improper sign string"
        for i in range(0, len(signs)):
            assert (
                signs[i] == "+" or signs[i] == "-" or signs[i] == "."
            ), "improper sign string"

        return Permutation.__new__(cls, p)

    def __init__(self, p, signs):
        self.signs = list(signs)

    def __repr__(self):
        s = " "
        for i in range(0, len(self)):
            s += str(self[i] + 1) + self.signs[i] + " "
        return s

    def __hash__(self):
        return hash((tuple(self[:]), tuple(self.signs)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def num_signs(self):
        return sum([1 for s in self.signs if s != "."])

    def sign_subset(self, P):
        for i in range(0, len(self)):
            t = (self.signs[i], P.signs[i])
            if t in [("+", "."), ("+", "-"), ("-", "+"), ("-", ".")]:
                return False
        return True

    def filling_vector(self):
        L = []
        for s in self.signs:
            if s == "+" or s == "-":
                L.append(2)
            else:
                L.append(1)
        return Vector(L)

    def all_dotted_monotone_intervals(self):
        mi = []
        difference = 0
        c_start = 0
        c_length = 0
        for i in range(0, len(self) - 1):
            if (
                self.signs[i] == "."
                and self.signs[i + 1] == "."
                and math.fabs(self[i] - self[i + 1]) == 1
                and (c_length == 0 or self[i] - self[i + 1] == difference)
            ):
                if c_length == 0:
                    c_start = i
                c_length += 1
                difference = self[i] - self[i + 1]
            else:
                if c_length != 0:
                    mi.append((c_start, c_start + c_length))
                c_start = 0
                c_length = 0
                difference = 0
        if c_length != 0:
            mi.append((c_start, c_start + c_length))
        return mi

    def is_compact(self):
        for i in range(0, len(self) - 1):
            if self[i] + 1 == self[i + 1]:
                s = "".join(self.signs[i : i + 2])
                if s == "++" or s == "+." or s == ".+":
                    return False
            elif self[i] - 1 == self[i + 1]:
                s = "".join(self.signs[i : i + 2])
                if s == "--" or s == "-." or s == ".-":
                    return False
        return True

    def is_compact_and_clean(self):
        if not self.is_compact():
            return False
        for i in range(0, len(self) - 1):
            if (
                self.signs[i] == "."
                and self.signs[i + 1] == "."
                and fabs(self[i] - self[i + 1]) == 1
            ):
                return False
        return True

    def clean_basis(self):
        dotted_intervals = self.all_dotted_monotone_intervals()
        in_intervals = []
        for (start, end) in dotted_intervals:
            in_intervals.extend(range(start, end + 1))
        for i in range(0, len(self)):
            if i not in in_intervals:
                dotted_intervals.append((i, i))
        dotted_intervals.sort()
        S = VectorSet()
        for (i, (start, end)) in enumerate(dotted_intervals):
            L = [0] * len(dotted_intervals)
            if start == end:
                if self.signs[start] == ".":
                    L[i] = 0
            else:
                length_to_avoid = end - start + 2
                L[i] = length_to_avoid
            for j in L:
                if j != 0:
                    if Vector(L).norm() > 0:
                        S.append(Vector(L))
                    break
        return S

    def split(self, indices):
        entries = list(self[:])
        signs = list(self.signs)
        indices.sort()
        offset = 0
        for index in indices:
            new_entries = entries[: index + offset]
            new_signs = signs[: index + 1 + offset]
            new_signs.append(signs[index + offset])

            if signs[index + offset] == "+":
                new_entries.append(entries[index + offset] - 0.5)
                new_entries.append(entries[index + offset] + 0.5)
            else:
                new_entries.append(entries[index + offset] + 0.5)
                new_entries.append(entries[index + offset] - 0.5)
            new_entries.extend(entries[index + offset + 1 :])
            new_signs.extend(signs[index + offset + 1 :])

            entries = Permutation.standardize(new_entries)
            signs = new_signs
            offset += 1

        return PegPermutation(entries, signs)

    def clean(self):
        if not self.is_compact():
            return PegPermutation(self, self.signs)
        dotted_intervals = self.all_dotted_monotone_intervals()
        copy = list(self[:])
        copysigns = list(self.signs[:])
        for (start, end) in dotted_intervals:
            type_of_interval = self[start + 1] - self[start]
            copysigns[start] = "+" if type_of_interval == 1 else "-"
            for i in range(start + 1, end + 1):
                copy[i] = -1
                copysigns[i] = -1
        while -1 in copy:
            copy.remove(-1)
            copysigns.remove(-1)
        return PegPermutation(copy, copysigns)

    def downset(self):
        return PermSet([self]).downset()

    def shrink_by_one(self):
        S = PermSet()

        for i in range(0, len(self)):
            if self.signs[i] != ".":
                new_signs = list(self.signs)
                new_signs[i] = "."

                S.add(PegPermutation(list(self), new_signs))

            else:
                if len(self) == 1:
                    continue
                new_signs = list(self.signs)
                new_signs.pop(i)

                new_entries = list(self)
                new_entries.pop(i)

                S.add(PegPermutation(new_entries, new_signs))

        return S

    def csgf(self, basis):
        x = Symbol("x")
        subsets = powerset(basis)
        f = 0
        for subset in subsets:
            V = VectorSet(subset)
            negpow = pow(-1, len(subset))
            fv = self.filling_vector()
            meet_B = V.meet_all() if len(V) != 0 else [1] * len(fv)
            xpow = fv.meet(meet_B).norm()
            f += negpow * pow(x, xpow) / pow(1 - x, self.num_signs())
        return f

    def reverse(self):
        entries = self[::-1]
        signs = self.signs[::-1]
        for i in range(0, len(signs)):
            if signs[i] == "-":
                signs[i] = "+"
            elif signs[i] == "+":
                signs[i] = "-"
        return PegPermutation(entries, signs)

    def involved_in(self, P):
        if not self.bounds_set:
            (self.lower_bound, self.upper_bound) = self.set_up_bounds()
            self.bounds_set = True
        L = list(self)
        n = len(L)
        p = len(P)
        # if n <= 1 and n <= p:
        # return True

        indices = [0] * n

        while indices[0] < p:
            if self.involvement_check(
                self.upper_bound, self.lower_bound, indices, P, 1
            ):
                if self.sign_subset(P):
                    return True
            indices[0] += 1

        return False

    def involvement_check(self, upper_bound, lower_bound, indices, q, next):
        if next == len(indices):
            return True
        lq = len(q)
        indices[next] = indices[next - 1] + 1
        while indices[next] < lq:
            if self.involvement_fits(
                upper_bound, lower_bound, indices, q, next
            ) and self.involvement_check(
                upper_bound, lower_bound, indices, q, next + 1
            ):
                if self.sign_subset(q):
                    return True
            indices[next] += 1
        return False

    def involvement_fits(self, upper_bound, lower_bound, indices, q, next):
        return (
            lower_bound[next] == -1 or q[indices[next]] > q[indices[lower_bound[next]]]
        ) and (
            upper_bound[next] == -1 or q[indices[next]] < q[indices[upper_bound[next]]]
        )
