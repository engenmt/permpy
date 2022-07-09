from __future__ import print_function

from itertools import combinations_with_replacement as cwr


def pretty_out(pi, k, vert_line=True, by_lines=False, width=2):
    """Return a nice string to visualize `pi`.
    If `by_lines == True`, then will return the list of strings by line,
    in case you want to append some stuff to each line.

    """
    print(pi, k)
    lines = []
    n = len(pi)

    max_width = len(str(n + 1))  # This is the width of each value.
    if max_width > width:
        width = max_width

    blank = " " * width
    for val in range(n)[::-1]:
        idx = pi.index(val)
        line = blank * (idx) + str(val + 1).rjust(width) + blank * (n - idx - 1)
        lines.append(line)

    if vert_line:
        if k == 0:
            for idx in range(n):
                lines[idx] += " |"
        else:
            for idx in range(n):
                lines[idx] = lines[idx][: -width * k] + " |" + lines[idx][-width * k :]
    else:
        lines.insert(k, "-" * (width * n))

    if by_lines:
        return lines
    else:
        return "\n".join(lines)


def gen_compositions(n, k=0):
    """Generate all compositions (as lists) of `n` into `k` parts.
    If `k == 0`, then generate all compositions of `n`.

    """
    assert n >= k, f"Need weight to be at least length: {n} ≥ {k}"

    if k == 0:
        for i in range(1, n + 1):
            for c in gen_compositions(n, i):
                yield c
    else:
        if k == 1:
            yield [n]
        elif n == k:
            yield [1] * n
        else:
            for i in range(1, n - k + 2):
                for c in gen_compositions(n - i, k - 1):
                    yield c + [i]


def gen_weak_compositions(n, k):
    """Generate all weak compositions (as lists) of `n` into `k` parts."""
    for c in gen_compositions(n + k, k):
        yield [part - 1 for part in c]


def gen_interval_divisions(m, k, shift=0, reverse=False):
    """Generate all ways of splitting the interval `[1, m]` shifted up by `shift` into `k` pieces.

    Example:
        >>> list(gen_interval_divisions(4, 2))
        [
            [ ()          , (0, 1, 2, 3) ],
            [ (0,)        ,    (1, 2, 3) ],
            [ (0, 1)      ,       (2, 3) ],
            [ (0, 1, 2)   ,          (3,)],
            [ (0, 1, 2, 3),            ()]
        ]
    """
    if reverse:
        direction = -1
    else:
        direction = +1

    L = range(shift, shift + m)[::direction]
    for c in cwr(range(m + 1), k - 1):
        # For each choice of divisions...

        c = (0,) + c + (m,)
        yield [tuple(val for val in L[c[i] : c[i + 1]]) for i in range(k)]


def all_vertical_extensions(pi, m, k, verbose=False):
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

    for uppers in gen_interval_divisions(m, k + 1, shift=n):
        new_suffix = sum([uppers[i] + (suffix[i],) for i in range(k)], ()) + uppers[-1]

        if verbose:
            print(f"uppers = {uppers:20}, new_suffix = {new_suffix:20}")
            print(f"Yielding {prefix + new_suffix}.")

        yield prefix + new_suffix


def all_horizontal_extensions(pi, m, k, verbose=False):
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

    for uppers in gen_interval_divisions(m, k + 1, shift=n, reverse=True):
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
