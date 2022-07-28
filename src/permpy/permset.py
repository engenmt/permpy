import random

from collections import Counter, defaultdict

from .permutation import Permutation
from .permmisc import lcm

from .deprecated.permsetdeprecated import PermSetDeprecatedMixin

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl_imported = True
except ImportError:
    mpl_imported = False


class PermSet(set, PermSetDeprecatedMixin):
    """Represents a set of permutations, and allows statistics to be computed
    across the set."""

    def __repr__(self):
        return f"Set of {len(self)} permutations"

    def __init__(cls, s=[]):
        """Return the PermSet from the iterable provided, or just with a single permutation."""
        if isinstance(s, Permutation):
            super().__init__([s])
        else:
            super().__init__(s)

    def __add__(self, other):
        """Return the union of the two permutation sets."""
        return PermSet(super().__or__(other))

    def __or__(self, other):
        """Wrapper for _add__."""
        return self + other

    def __sub__(self, other):
        """Return the union of the two permutation sets."""
        return PermSet(super().__sub__(other))

    @classmethod
    def all(cls, length):
        """Return the set of all permutations of a given length.

        Args:
            length (int): the length of the permutations

        """
        return PermSet(Permutation.gen_all(length))

    def union(self, other):
        """Wrapper for __add__"""
        return self + other

    def get_random(self):
        """Return a random element from the set."""
        return random.sample(self, 1)[0]

    def by_length(self):
        """Return a dictionary stratifying the permutations in `self`."""
        D = defaultdict(PermSet)
        for p in self:
            D[len(p)].add(p)
        return D

    def get_length(self, length):
        """Return the subset of permutations that have the specified length.

        Args:
            length (int): length of permutations to be returned

        """
        return PermSet(p for p in self if len(p) == length)

    def show_all(self):
        """The default representation doesn't print the entire set, this function does."""
        return set.__repr__(self)

    def minimal_elements(self):
        """Return the elements of `self` that are minimal with respect to the
        permutation pattern order.

        """

        shortest_len = min(len(p) for p in self)
        shortest_perms = PermSet(p for p in self if len(p) == shortest_len)
        S = PermSet(p for p in self if p.avoids(B=shortest_perms))

        return shortest_perms + S.minimal_elements()

    def symmetries(self):
        """Return the PermSet of all symmetries of all permutations in `self`."""
        S = set(self)
        S.update([p.reverse() for p in S])
        S.update([p.complement() for p in S])
        S.update([p.inverse() for p in S])
        return PermSet(S)

    def covers(self):
        """Return those permutations that `self` covers."""
        return PermSet(set().union(*[p.covers() for p in self]))

    def covered_by(self):
        """Return those permutations that `self` is covered by."""
        return PermSet(set().union(*[p.covered_by() for p in self]))

    def right_extensions(self, basis=None, test=None, trust=False):
        """Return the 'one layer' upset of `self`.

        Notes:
            Requires each permutation in `self` to be the same size.
            Requires either basis or test.
            Implicit assumption is that the test is hereditary.

        Args:
            basis (iter:optional): permutations to avoid. Useful for building classes.
            test (optional): Function that accepts a permutation and returns a boolean.
                Only returns those permutations that pass the test.
            trust (boolean:optional): Whether or not to trust the `insertion_values`
                existing in the Permutations in `self`.

        """
        if len(self) == 0:
            return PermSet()

        if test is None and basis is not None:
            if trust:
                lr = 2
                # If we trust the previous insertion_values, then right-extending
                # a permutation with an insertion value only makes us fail the test
                # when the rightmost two entries are used.
            else:
                lr = 1

            def test(p):
                return p.avoids(B=basis, lr=lr)

        S = set().union(*[set(p.right_extensions(test=test)) for p in self])
        return PermSet(S)

    def upset(self, up_to_length):
        """Return the upset of `self`, stratified by length.

        Args:
            basis (iter:optional): permutations to avoid. Useful for building classes.

        """
        if not self:
            return []

        by_length = self.by_length
        min_length = min(by_length.keys())
        max_length = max(by_length.keys())
        upset = [PermSet() for _ in range(min_length)]
        upset.append(PermSet(by_length[min_length]))

        if max_length > up_to_length:
            raise ValueError(
                f"PermSet.upset called with up_to_length = {up_to_length} on a PermSet with a longer permutation."
            )

        for length in range(min_length + 1, max_length + 1):
            prev_length = upset[length - 1] + by_length[-1]
            upset.append(PermSet(set().union(p.covered_by() for p in prev_length)))

        return upset

    def downset(self):
        """Return the downset of `self` as a list."""
        bottom_edge = PermSet()
        bottom_edge.update(self)

        max_len = max(len(p) for p in self)
        levels = [PermSet() for _ in range(max_len + 1)]
        for p in self:
            levels[len(p)].add(p)

        downset = [None for _ in range(max_len + 1)]

        for n in range(max_len, -1, -1):
            upper = downset[n] + levels[n]
            lower = upper.covers()
            downset[n - 1] = lower

        return downset

    def pattern_counts(self, k):
        """Return a dictionary counting the copies of all `k`-perms in each permutation in `self`."""
        C = Counter()
        for pi in self:
            C += pi.pattern_counts(k)
        return C

    def total_statistic(self, statistic, default=0):
        """Return the sum of the given statistic over all perms in `self`.

        Notes:
            Works as long as the statistic is a number. If the statistic is a
                Counter or something, this will fail as written.

        """
        return sum((statistic(p) for p in self), default)

    def heatmap(self, only_length=None, ax=None, blur=False, gray=False, **kwargs):
        """Visalization of a set of permutations, which, for each length, shows
        the relative frequency of each value in each position.

        Args:
            only_length (int:optional):  If given, restrict to the permutations of this length.

        """
        if not mpl_imported:
            raise NotImplementedError(
                "PermSet.heatmap requires matplotlib to be imported."
            )

        try:
            import numpy as np
        except ImportError as exc:
            raise exc("PermSet.heatmap requires numpy to be imported!")
        # first group permutations by length
        perms_by_length = self.by_length()

        # if given a length, ignore all other lengths
        if only_length:
            perms_by_length = {only_length: perms_by_length[only_length]}
        lengths = list(perms_by_length.keys())
        grid_size = lcm(lengths)
        grid = np.zeros((grid_size, grid_size))

        def inflate(a, n):
            """Inflates a k x k array A by into a nk x nk array by inflating
            each entry from A into a n x n matrix."""
            ones = np.ones((n, n))
            c = np.multiply.outer(a, ones)
            c = np.concatenate(np.concatenate(c, axis=1), axis=1)
            return c

        for length, permset in perms_by_length.items():
            small_grid = np.zeros((length, length))
            for p in permset:
                for idx, val in enumerate(p):
                    small_grid[length - val - 1, idx] += 1
            mul = grid_size // length
            inflated = inflate(small_grid, mul)
            inflated /= inflated.max()
            grid += inflated

        if not ax:
            ax = plt.gca()
        if blur:
            interpolation = "bicubic"
        else:
            interpolation = "nearest"

        def get_cubehelix(gamma=1, start=1, rot=-1, hue=1, light=1, dark=0):
            """Get a cubehelix palette."""
            cdict = mpl._cm.cubehelix(gamma, start, rot, hue)
            cmap = mpl.colors.LinearSegmentedColormap("cubehelix", cdict)
            x = np.linspace(light, dark, 256)
            pal = cmap(x)
            cmap = mpl.colors.ListedColormap(pal)
            return cmap

        if gray:
            cmap = get_cubehelix(start=0.5, rot=1, light=1, dark=0.2, hue=0)
        else:
            cmap = get_cubehelix(start=0.5, rot=-0.5, light=1, dark=0.2)

        ax.imshow(grid, cmap=cmap, interpolation=interpolation)
        ax.set_aspect("equal")
        ax.set(**kwargs)
        ax.axis("off")
        return ax

    def stack_inverse(self):
        """Return the PermSet of stack-inverses of elements of self.

        Notes:
            Uses dynamic programming!

        """
        A = [tuple([val + 1 for val in pi]) for pi in self]
        n = len(A[0])
        assert all(
            len(pi) == n for pi in self
        ), "Not designed to handle this, unfortunately!"
        L = [set() for _ in range(n + 1)]
        L[n].update((pi, tuple(), tuple()) for pi in A)
        for k in range(n)[::-1]:
            unpop_temp = set(unpop(state) for state in L[k + 1])
            L[k].update(state for state in unpop_temp if state is not None)
            old = L[k]

            unpush_temp = set(unpush(state) for state in old)
            new = set(state for state in unpush_temp if state is not None)
            while new:
                L[k].update(new)
                old = new
                unpush_temp = set(unpush(state) for state in old)
                new = set(state for state in unpush_temp if state is not None)
        return PermSet(
            Permutation(state[2])
            for state in L[0]
            if state is not None and not state[1]
        )


def unpop(state):
    """Given the before, stack, and after tuples, returns the (one-step) preimage."""
    after, stack, before = state
    if after and after[-1] and (not stack or after[-1] < stack[-1]):
        return (after[:-1], stack + (after[-1],), before)
    else:
        return


def unpush(state):
    """Given the before, stack, and after tuples, returns the (one-step) preimage."""
    after, stack, before = state
    if stack:
        return (after, stack[:-1], (stack[-1],) + before)
    else:
        return
