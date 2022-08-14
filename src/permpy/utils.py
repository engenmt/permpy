import types

from itertools import combinations_with_replacement


def copy_func(f, name=None):
    """Return a function with same code, globals, defaults, closure, and
    name (or provide a new name)
    """
    fn = types.FunctionType(
        f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__
    )
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__)
    return fn


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
            yield (n,)
        elif n == k:
            yield tuple(1 for _ in range(n))
        else:
            for i in range(1, n - k + 2):
                for c in gen_compositions(n - i, k - 1):
                    yield c + (i,)


def gen_weak_compositions(n, k):
    """Generate all weak compositions (as lists) of `n` into `k` parts."""
    for c in gen_compositions(n + k, k):
        yield tuple(part - 1 for part in c)


def gen_interval_divisions(m, k, shift=0, reverse=False):
    """Generate all ways of splitting the interval `[1, m]` shifted up by `shift` into `k` pieces."""
    # L is the complete list of values to return
    if reverse:
        L = range(shift + m - 1, shift - 1, -1)
    else:
        L = range(shift, shift + m)
    for c in combinations_with_replacement(range(m + 1), k - 1):
        # For each choice of divisions...

        c = (0,) + c + (m,)
        yield [tuple(L[c[i] : c[i + 1]]) for i in range(k)]
