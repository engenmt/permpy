from __future__ import print_function

from .permutation import Permutation
from .permset import PermSet
from .permclass import PermClass
from .utils import gen_interval_divisions


class MonotoneStaircase(PermClass):
    def __init__(self, cells, n=0, begin_east=True):
        self.cells = cells
        self.n = n
        self.next_cell_east = begin_east  # Otherwise, North
        super().__init__(self._generate())

    def _generate(self):
        self.perms = {Permutation(): 0}
        for cell_orientation in self.cells:
            self.add_cell(
                is_east=self.next_cell_east, cell_orientation=cell_orientation
            )
            self.next_cell_east = not self.next_cell_east

        perm_class = [PermSet() for _ in range(self.n + 1)]
        for perm in self.perms:
            perm_class[len(perm)].add(perm)
        return perm_class

    def add_cell(self, is_east, cell_orientation):
        all_relevant_extensions = (
            all_vertical_extensions if is_east else all_horizontal_extensions
        )
        is_decreasing = cell_orientation == -1
        perms_new = {}
        for perm, k in self.perms.items():
            for m in range(self.n - len(perm) + 1):
                for perm_new in all_relevant_extensions(
                    perm, m, k, decreasing=is_decreasing
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
                lines[idx] += f"{'|':>{width}}"
        else:
            for idx, line in enumerate(lines):
                prefix = line[: -width * k]
                suffix = line[-width * k :]
                lines[idx] = f"{prefix}{'|':>{width}}{suffix}"
    elif horiz_line:
        lines.insert(k, "-" * (width * n))

    return "\n".join(lines)


def all_vertical_extensions(pi, m, k, decreasing):
    """Generate all ways to add a monotone sequence of length m above
    the rightmost k points of pi.

    """
    n = len(pi)
    if m == 0:
        yield pi
        return

    if k == 0:
        addition = (
            Permutation.monotone_decreasing(m)
            if decreasing
            else Permutation.monotone_increasing(m)
        )
        yield pi + addition
        return

    prefix = list(pi[:-k])
    suffix = pi[-k:]

    for uppers in gen_interval_divisions(m, k + 1, shift=n, reverse=decreasing):
        new_suffix = list(uppers[0])
        for suffix_val, upper in zip(suffix, uppers[1:]):
            new_suffix.append(suffix_val)
            new_suffix.extend(upper)
        yield Permutation(prefix + new_suffix)


def all_horizontal_extensions(pi, m, k, decreasing):
    """Generate all ways to add a monotone sequence of length m to the right of
    the uppermost k points of pi.

    """

    for tau in all_vertical_extensions(pi.inverse(), m, k, decreasing=decreasing):
        yield tau.inverse()
