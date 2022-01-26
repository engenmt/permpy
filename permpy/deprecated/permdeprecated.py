import warnings
import functools


def deprecated(func):
    """This is a decorator that can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            f"Call to deprecated function {func.__name__}.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


class PermutationDeprecatedMixin:
    """A mixin for deprecated methods kept for backward compatability."""

    @deprecated
    def all_syms(self):
        return self.symmetries()

    @deprecated
    def ascent_set(self):
        return self.ascents()

    @deprecated
    def descent_set(self):
        return self.descents()

    @deprecated
    def avoids_set(self, B):
        for p in B:
            if not isinstance(p, Permutation):
                p = Permutation(p)
            if p in self:
                return False
        return True

    @deprecated
    def buildupset(self, height):
        return self.upset(height, stratified=True)

    @deprecated
    def _ascii_plot(self):
        """Prints a simple plot of the given Permutation."""
        n = len(self)
        array = [[" " for _ in range(n)] for _ in range(n)]
        for i in range(n):
            array[self[i]][i] = "*"
        array.reverse()
        s = "\n".join((" ".join(l) for l in array))
        print(s)

    @deprecated
    def greedy_sum(self):
        """This provides a sum-decomposition of `self` in which consecutive increasing sum-components are merged."""
        parts = []
        sofar = 0
        while sofar < len(p):
            if len(p) - sofar == 1:
                parts.append(Permutation(1))
                return parts
            i = 1
            while sofar + i <= len(p) and list(p[sofar : sofar + i]) == range(
                sofar, sofar + i
            ):
                i += 1
            i -= 1
            if i > 0:
                parts.append(Permutation(range(i)))
            sofar += i
            i = 2
            while sofar + i <= len(p) and not (
                max(p[sofar : sofar + i]) - min(p[sofar : sofar + i]) + 1 == i
                and min(p[sofar : sofar + i]) == sofar
            ):
                i += 1
            if sofar + i <= len(p):
                parts.append(Permutation(p[sofar : sofar + i]))
            sofar += i
        return parts

    @deprecated
    def chom_sum(p):
        L = []
        p = p.greedy_sum()
        for i in p:
            if i.inversions() == 0:
                L.extend([Permutation(1)] * len(i))
            else:
                L.append(i)
        return L

    @deprecated
    def chom_skew(p):
        return [r.reverse() for r in p.reverse().chom_sum()]

    @deprecated
    def christiecycles(self):
        # builds a permutation induced by the black and gray edges separately, and
        # counts the number of cycles in their product. used for transpositions
        p = list(self)
        n = self.__len__()
        q = [0] + [p[i] + 1 for i in range(n)]
        grayperm = range(1, n + 1) + [0]
        blackperm = [0 for _ in range(n + 1)]
        for i in range(n + 1):
            ind = q.index(i)
            blackperm[i] = q[ind - 1]
        newperm = []
        for i in range(n + 1):
            k = blackperm[i]
            j = grayperm[k]
            newperm.append(j)
        return Permutation(newperm).numcycles()

    @deprecated
    def coveredby(self):
        return self.covered_by()

    @deprecated
    def children(self):
        return self.covers()

    @deprecated
    def fixedptsplusbonds(self):
        return self.fixed_points() + self.bonds()

    @deprecated
    def num_immediate_copies_of(self, other):
        return self.num_contiguous_copies(other)

    @deprecated
    def threepats(self):
        return {str(p): count for p, count in self.pattern_counts(3).items()}

    @deprecated
    def fourpats(self):
        return {str(p): count for p, count in self.pattern_counts(4).items()}

    @deprecated
    @classmethod
    def ind2perm(cls, k, n):
        return cls.ind_to_perm(k, n)

    @deprecated
    def perm2ind(self):
        return self.perm_to_ind()

    @deprecated
    def ind_to_perm(self):
        return self.perm_to_ind()

    @deprecated
    @classmethod
    def listall(cls, n):
        return cls.list_all(n)

    @deprecated
    def longestrun(self):
        return self.len_max_run()

    @deprecated
    def longestrunA(self):
        return self.max_ascending_run()[1]

    @deprecated
    def longestrunD(self):
        return self.max_descending_run()[1]

    @deprecated
    def ltrmax(self):
        return self.ltr_max()

    @deprecated
    def ltrmin(self):
        return self.ltr_min()

    @deprecated
    def rtlmax(self):
        return self.rtl_max()

    @deprecated
    def rtlmin(self):
        return self.rtl_min()

    @deprecated
    def num_ltrmin(self):
        return self.num_ltr_min()

    @deprecated
    def majorindex(self):
        return self.major_index()

    @deprecated
    def min_gapsize(self):
        return self.breadth()

    @deprecated
    def occurrences(self, other):
        return self.copies(other)

    @deprecated
    def num_cycles(self):
        return len(self.cycle_decomp())

    @deprecated
    def othercycles(self):
        """Builds a permutation induced by the black and gray edges separately,
        and counts the number of cycles in their product.
        """
        p = list(self)
        n = self.__len__()
        q = [0] + [p[i] + 1 for i in range(n)]
        grayperm = [n] + range(n)
        blackperm = [0 for _ in range(n + 1)]
        for i in range(n + 1):
            ind = q.index(i)
            blackperm[i] = q[ind - 1]
        newperm = []
        for i in range(n + 1):
            k = blackperm[i]
            j = grayperm[k]
            newperm.append(j)
        return Permutation(newperm).numcycles()

    @deprecated
    def sumcycles(self):
        return self.othercycles() + self.christiecycles()

    @deprecated
    def maxcycles(self):
        return max(self.othercycles() - 1, self.christiecycles())

    @deprecated
    def peak_list(self):
        return self.peaks()

    @deprecated
    def valley_list(self):
        return self.peaks()

    @deprecated
    @classmethod
    def plentiful(cls, gap):
        """Return the gap-plentiful permutation of minimal(?) length."""
        # if gap == 6:
        #     return Permutation([5,10,15,2,7,12,17,4,9,14,1,6,11,16,3,8,13])
        d = gap - 1
        if d % 2:
            firsts = list(range(2, d + 1, 2)) + list(range(1, d + 2, 2))
        else:
            firsts = list(range(1, d + 1, 2)) + list(range(2, d + 2, 2))

        def segment(first):
            return list(range(first, first + (d - 2) * (d + 1) + 1, d + 1))

        segments = [segment(f) for f in firsts]
        entries = [val for segment in segments for val in segment]
        try:
            entries.remove(1)
            entries.remove(2)
            n = max(entries)
            entries.remove(n)
        except ValueError:
            pass
        return Permutation(entries)

    @deprecated
    def all_extensions(self):
        return self.covered_by()

    @deprecated
    def all_extensions_track_index(self, track_index):
        L = []
        for i in range(0, len(self) + 1):
            for j in range(0, len(self) + 1):
                # insert (i-0.5) after entry j (i.e., first when j=0)
                l = list(self)
                l.insert(j, i - 0.5)
                if j < track_index:
                    L.append((Permutation(l), track_index + 1))
                else:
                    L.append((Permutation(l), track_index))
        return L
