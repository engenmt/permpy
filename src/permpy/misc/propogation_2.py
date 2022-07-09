from collections import Counter, namedtuple

# X = namedtuple('X', ['a', 'b', 'c'])
X = namedtuple("X", ["v", "s"])


def next_states(S, mul):
    """Compute the states that would arise from the given state."""
    to_return = Counter()

    def include(new_v, new_s):
        to_return[X(new_v, new_s)] += mul

    if len(S.v) == 4:
        a, b, c, d = S.v

        if S.s == 1:
            include((b, c, d, 1), 2)

            for k in range(1, c + d):
                include((k, -1, 0), 1)

            include((a, b, c + 1, d), 1)

            for k in range(1, b):
                include((k, -1, 0), 1)

            for k in range(0, a + 1):
                include((k, -1, c + d), 1)

        elif S.s == 2:

            include((b + c, d, 1), 0)

            for k in range(1, d):
                include((k, -1, 0), 1)

            include((a, b, c, d + 1), 2)

            for k in range(1, b + c):
                include((k, -1, d), 2)

            for k in range(0, a + 1):
                include((k, -1, d), 1)
    elif len(S.v) == 3:

        a, b, c = S.v

        if S.s == 0:
            if c > 0:
                include((b, c, 1), 0)

            if b > 0:
                for k in range(1, c):
                    include((b, k, 1, c - k), 1)
            else:
                for k in range(1, c):
                    include((k, 1, c - k), 3)

            include((a, b, c + 1), 0)

            for k in range(1, b):
                include((k, -1, c), 2)

            if b > 0:
                if a > 0:
                    for k in range(0, a + 1):
                        include((k, -1, c), 1)
                else:
                    include((0, -1, c), 2)

        elif S.s == 1:
            include((c, -1, 0), 1)

            for k in range(1, c):
                include((k, -1, 0), 1)

            include((a, -1, c + 1), 1)

            for k in range(0, a + 1):
                include((k, -1, c), 1)

        elif S.s == 2:

            include((a, -1, 1), 1)

            for k in range(1, c):
                include((k, -1, 0), 1)

            include((a, -1, c + 1), 1)

            for k in range(0, a + 1):
                include((k, -1, c), 2)
        elif S.s == 3:
            assert a > 0 and b > 0 and c > 0

            include((a, b, c, 1), 2)

            for k in range(1, b + c):
                include((k, -1, 0), 1)

            include((a, b + 1, c), 3)

            for k in range(1, a):
                include((k, -1, 0), 1)

            include((0, -1, b + c), 1)
        else:
            raise Exception

    else:
        raise Exception

    return to_return


def iterate(states):
    """Iterate the current Counter of states once."""
    ns = sum([next_states(S, mul) for S, mul in states.items()], Counter())
    return ns


def states(n):
    """Return the states corresponding to permutations of length n in NS."""
    states = Counter()
    states[X(0, 0, 0)] = 1

    for _ in range(n):
        states = iterate(states)

    return states


def pstring(state, num):
    """Return a nice string give a state and its count.

    Example:
    >>> pstring(X(0,-1,1),4)
    ( 0, *, 1) :  4

    """
    if len(state.v) == 4:
        a, b, c, d = state.v

        return f"({a:2},{b:2},{c:2},{d:2})_{state.s} : {num:2}"
    elif len(state.v) == 3:
        a, b, c = state.v
        if b == -1:
            b = " *"
        return f"({a:2},{b:2},{c:2})_{state.s}    : {num:2}"


if __name__ == "__main__":
    S = Counter([X((0, 0, 0), 0)])

    N = 6

    for n in range(1, N + 1):
        print(f"\nn = {n-1} -> {n}\n")

        for state, num in sorted(S.items()):
            NS = next_states(state, num)
            if 0 < sum(NS.values()) // num < 100:
                print(f"{pstring(state, num)} -> ")
                for nstate, nnum in sorted(NS.items()):
                    print("\t" + pstring(nstate, nnum // num))
            # print("\t" + pstring(state, num))

        S = iterate(S)

        print(f"{n:2}, {sum(S.values()):10}")
