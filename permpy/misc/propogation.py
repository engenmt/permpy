from collections import Counter, namedtuple

X = namedtuple("X", ["a", "b", "c"])


def next_states(S, mul):
    """Compute the states that would arise from the given state."""
    to_return = Counter()

    def include(a, b, c):
        to_return[X(a, b, c)] += mul

    if S.b != -1:

        if S.c > 0:
            include(S.a, S.b, S.c + 1)

        include(S.b, S.c, 1)

        # for k in range(0,S.a):
        # 	include(k,-1,S.c)

        # for k in range(0,S.b):
        # 	include(k,-1,S.c)
        if S.b > 0:
            for k in range(0, S.a + 1):
                include(k, -1, S.c)

        for k in range(1, S.b):
            include(k, -1, S.c)

        # for k in range(1,S.c):
        # 	include(k, -1, 0)

        for k in range(1, S.c):
            include(0, -1, S.c - k + 1)

    else:

        if S.c > 0:
            include(S.a, -1, S.c + 1)
            include(0, S.c, 1)
        else:
            include(S.a, -1, 1)

        for k in range(0, S.a + 1):
            include(k, -1, S.c)

        for k in range(1, S.c):
            include(k, -1, 0)

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
    a, b, c = state
    if b == -1:
        b = " *"
    return f"({a:2},{b:2},{c:2}) : {num:2}"


if __name__ == "__main__":
    S = Counter([X(0, 0, 0)])

    for n in range(1, 6):
        print(f"\nn = {n}\n")

        for state, num in sorted(S.items()):
            NS = next_states(state, num)
            if 3 < sum(NS.values()) // num < 5:
                print(f"{pstring(state, num)} -> ")
                for nstate, nnum in sorted(NS.items()):
                    print("\t" + pstring(nstate, nnum // num))
            # print("\t" + pstring(state, num))

        S = iterate(S)

        # if n == 4:
        # 	S[X(0,-1,0)] -= 1

        print(f"{n:2}, {sum(S.values()):10}")
