from __future__ import print_function

from .permutation import Permutation
from .permset import PermSet
from .utils import gen_interval_divisions


class MonotoneStaircase:
    def __init__(self, cells, n=0, begin_east=True):
        self.cells = cells
        self.n = n
        self.next_cell_east = begin_east  # Otherwise, North
        self.generate()

    def generate(self):
        self.perms = {Permutation(): 0}
        for cell_orientation in self.cells:
            if self.next_cell_east:
                self.add_cell_east(cell_orientation)
            else:
                self.add_cell_north(cell_orientation)
            self.next_cell_east = not self.next_cell_east

        self.perm_class = [PermSet() for _ in range(self.n + 1)]
        for perm in self.perms:
            self.perm_class[len(perm)].add(perm)

    def add_cell_east(self, cell_orientation):
        decreasing = cell_orientation == -1
        perms_new = {}
        for perm, k in self.perms.items():
            for m in range(self.n - len(perm) + 1):
                for perm_new in all_horizontal_extensions(
                    perm, m, k, decreasing=decreasing
                ):
                    # Keep the partial gridding that places the maximum number of points
                    # in the final cell, as it has the most descendents.
                    perms_new[perm_new] = max(perms_new.get(perm_new, 0), m)
        self.perms = perms_new

    def add_cell_north(self, cell_orientation):
        decreasing = cell_orientation == -1
        perms_new = {}
        for perm, k in self.perms.items():
            for m in range(self.n - len(perm) + 1):
                for perm_new in all_vertical_extensions(
                    perm, m, k, decreasing=decreasing
                ):
                    # Keep the partial gridding that places the maximum number of points
                    # in the final cell, as it has the most descendents.
                    perms_new[perm_new] = max(perms_new.get(perm_new, 0), m)
        self.perms = perms_new


def pretty_out(pi, k=0, width=2, vert_line=False, horiz_line=False):
    """Return a string to visualize `pi`."""
    lines = []
    n = len(pi)

    min_width = len(str(n + 1))
    width = max(min_width, width)

    for val in range(n)[::-1]:
        idx = pi.index(val) + 1
        prefix = f"{val+1:>{width*idx}}"
        line = f"{prefix:<{width*n}}"
        lines.append(line)

    if vert_line:
        if k == 0:
            for idx in range(n):
                lines[idx] += " |"
        else:
            for idx, line in enumerate(lines):
                prefix = line[: -width * k]
                suffix = line[-width * k :]
                lines[idx] = f"{prefix}{'|':>{width}}{suffix}"
    elif horiz_line:
        lines.insert(k, "-" * (width * n))

    return "\n".join(lines)


def all_vertical_extensions(pi, m, k, verbose=False, decreasing=False):
    """Given a permutation `pi`, generate all ways to add an increasing sequence
    of length `m` above its right `k` points.

    """
    n = len(pi)

    # Split pi on its last k elements.
    if k == 0:
        prefix = pi
        suffix = ()
    else:
        prefix = pi[:-k]
        suffix = pi[-k:]

    if verbose:
        print(f"Vertically extending (pi, m, k) = {(pi, m, k)}")
        print(f"prefix = {prefix}")
        print(f"suffix = {suffix}")

    for uppers in gen_interval_divisions(m, k + 1, shift=n, reverse=decreasing):
        new_suffix = sum([uppers[i] + (suffix[i],) for i in range(k)], ()) + uppers[-1]

        if verbose:
            print(f"uppers = {uppers:20}, new_suffix = {new_suffix:20}")
            print(f"Yielding {prefix + new_suffix}.")

        yield prefix + new_suffix


def all_horizontal_extensions(pi, m, k, verbose=False, decreasing=True):
    """Given a permutation `pi`, generate all ways to add an decreasing sequence
    of length `m` to the right of its upper `k` points.

    """

    tau = inverse(pi)
    n = len(tau)

    if k == 0:
        prefix = tau
        suffix = ()
    else:
        prefix = tau[:-k]
        suffix = tau[-k:]

    if verbose:
        print(f"Horizontally extending (pi, m, k) = {(pi,m,k)}")
        print(f"prefix = {prefix}")
        print(f"suffix = {suffix}")

    for uppers in gen_interval_divisions(m, k + 1, shift=n, reverse=decreasing):
        new_suffix = sum([uppers[i] + (suffix[i],) for i in range(k)], ()) + uppers[-1]

        if verbose:
            print(f"uppers = {uppers:20}, new_suffix = {new_suffix:20}")
            print(f"Yielding the inverse of {prefix + new_suffix}.")

        yield inverse(prefix + new_suffix)


def inverse(pi):
    q = tuple(pi.index(val) for val in range(len(pi)))
    return q


def first_two_cells(n):
    """Return the set of initial configurations of points in the first two cells."""

    initial = ((), 0)
    R = set([initial])  # The set containing the empty tuple.

    S = set()

    for pi, k in R:
        for m in range(0, n + 1):
            S.update((tau, m) for tau in all_vertical_extensions(pi, m, k))

    T = set()

    for pi, k in S:
        if k == 0 and len(pi) != 0:
            T.add((pi, 0))
        else:
            for m in range(0, n - len(pi) + 1):
                T.update((tau, m) for tau in all_horizontal_extensions(pi, m, k))

    return T


def add_two_cells(R, n):
    S = set()
    for pi, k in R:
        S.add((pi, 0))
        for m in range(1, n - len(pi) + 1):
            S.update((tau, m) for tau in all_vertical_extensions(pi, m, k))

    T = set()
    for pi, k in S:
        T.add((pi, 0))
        for m in range(1, n - len(pi) + 1):
            T.update((tau, m) for tau in all_horizontal_extensions(pi, m, k))

    return T
