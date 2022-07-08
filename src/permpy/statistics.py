import fractions

from functools import reduce

from permpy.permutation import Permutation


def bend_list(perm):
    """Returns the list of indices at that the permutation changes
    direction. That is, the number of non-monotone consecutive triples of
    the permutation. A permutation p can be expressed as the concatenation
    of len(p.bend_list()) + 1 monotone segments."""
    raise NotImplementedError

    # this isn't quite correct....
    # return len(
    #     [
    #         i
    #         for i in range(1, len(perm) - 1)
    #         if (perm[i - 1] > perm[i] and perm[i + 1] > perm[i])
    #         or (perm[i - 1] < perm[i] and perm[i + 1] < perm[i])
    #     ]
    # )


def order(perm):
    L = map(len, perm.cycle_decomp())
    return reduce(lambda x, y: x * y // fractions.gcd(x, y), L)


def bonds(perm):
    numbonds = 0
    p = list(perm)
    for i in range(1, len(p)):
        if p[i] - p[i - 1] == 1 or p[i] - p[i - 1] == -1:
            numbonds += 1
    return numbonds


def majorindex(perm):
    sum = 0
    p = list(perm)
    n = perm.__len__()
    for i in range(0, n - 1):
        if p[i] > p[i + 1]:
            sum += i + 1
    return sum


def fixedptsplusbonds(perm):
    return perm.fixed_points() + perm.bonds()


def christiecycles(perm):
    # builds a permutation induced by the black and gray edges separately, and
    # counts the number of cycles in their product. used for transpositions
    p = list(perm)
    n = perm.__len__()
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


def othercycles(perm):
    # builds a permutation induced by the black and gray edges separately, and
    # counts the number of cycles in their product
    p = list(perm)
    n = perm.__len__()
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


def sumcycles(perm):
    return perm.othercycles() + perm.christiecycles()


def maxcycles(perm):
    return max(perm.othercycles() - 1, perm.christiecycles())
