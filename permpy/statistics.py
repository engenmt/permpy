from permpy.permutation import Permutation

def fixed_points(perm):
    """Returns the number of fixed points of the permutation.

    >>> Permutation(521436).fixed_points()
    3
    """
    sum = 0
    for i in range(perm.__len__()):
        if perm(i) == i:
            sum += 1
    return sum


def skew_decomposable(perm):
    """Determines whether the permutation is expressible as the skew sum of
    two permutations.

    >>> p = Permutation.random(8).direct_sum(Permutation.random(12))
    >>> p.skew_decomposable()
    False
    >>> p.complement().skew_decomposable()
    True
    """

    p = list(perm)
    n = perm.__len__()
    for i in range(1,n):
        if set(range(n-i,n)) == set(p[0:i]):
            return True
    return False

def sum_decomposable(perm):
    """Determines whether the permutation is expressible as the direct sum of
    two permutations.

    >>> p = Permutation.random(4).direct_sum(Permutation.random(15))
    >>> p.sum_decomposable()
    True
    >>> p.reverse().sum_decomposable()
    False
    """

    p = list(perm)
    n = perm.__len__()
    for i in range(1,n):
        if set(range(0,i)) == set(p[0:i]):
            return True
    return False

def num_cycles(perm):
    """Returns the number of cycles in the permutation.

    >>> Permutation(53814276).num_cycles()
    3
    """

    return len(perm.cycle_decomp())



def descent_set(perm):
    """Returns descent set of the permutation

    >>> Permutation(42561873).descent_set()
    [1, 4, 6, 7]
    """

    p = list(perm)
    n = perm.__len__()
    descents = []
    for i in range(1,n):
        if p[i-1] > p[i]:
            descents.append(i)
    return descents

def num_descents(perm):
    """Returns the number of descents of the permutation

    >>> Permutation(42561873).num_descents()
    4
    """
    return len(perm.descent_set())

def ascent_set(perm):
    """Returns the ascent set of the permutation

    >>> Permutation(42561873).ascent_set()
    [2, 3, 5]
    """
    descents = perm.descent_set()
    return [i for i in range(1, len(perm)) if i not in descents]

def num_ascents(perm):
    """Returns the number of ascents of the permutation

    >>> Permutation(42561873).num_ascents()
    3
    """
    return len(perm.ascent_set())


def peak_list(perm):
    """Returns the list of peaks of the permutation.

    >>> Permutation(2341765).peak_list()
    [2, 4]
    """

    def check(i):
        return perm[i-1] < perm[i] > perm[i+1]
    return [i for i in range(1, len(perm)-1) if check(i)]


def num_peaks(perm):
    """Returns the number of peaks of the permutation

    >>> Permutation(2341765).num_peaks()
    2
    """

    return len(perm.peak_list())

def valley_list(perm):
    """Returns the list of valleys of the permutation.

    >>> Permutation(3241756).valley_list()
    [1, 3, 5]
    """

    return perm.complement().peak_list()


def num_valleys(perm):
    """Returns the number of peaks of the permutation

    >>> Permutation(3241756).num_valleys()
    3
    """

    return len(perm.valley_list())

def bend_list(perm):
    """Returns the list of indices at which the permutation changes
    direction. That is, the number of non-monotone consecutive triples of
    the permutation. A permutation p can be expressed as the concatenation
    of len(p.bend_list()) + 1 monotone segments."""
    raise NotImplementedError

    # this isn't quite correct....
    return len([i for i in range(1, len(perm)-1) if (perm[i-1] > perm[i] and perm[i+1] > perm[i]) or (perm[i-1] < perm[i] and perm[i+1] < perm[i])])


def trivial(perm):
    """The trivial permutation statistic, for convenience

    >>> Permutation.random(10).trivial()
    0
    """
    return 0

def order(perm):
    L = map(len, perm.cycle_decomp())
    return reduce(lambda x,y: x*y // fractions.gcd(x,y), L)

def ltrmin(perm):
    """Returns the positions of the left-to-right minima.

    >>> Permutation(35412).ltrmin()
    [0, 3]
    """

    n = perm.__len__()
    L = []
    minval = len(perm) + 1
    for idx, val in enumerate(perm):
        if val < minval:
            L.append(idx)
            minval = val
    return L

def rtlmin(perm):
    """Returns the positions of the left-to-right minima.

    >>> Permutation(315264).rtlmin()
    [5, 3, 1]
    """
    rev_perm = perm.reverse()
    return [len(perm) - val - 1 for val in rev_perm.ltrmin()]

def ltrmax(perm):
    return [len(perm)-i-1 for i in Permutation(perm[::-1]).rtlmax()][::-1]

def rtlmax(perm):
    return [len(perm)-i-1 for i in perm.complement().reverse().ltrmin()][::-1]

def num_ltrmin(perm):
    return len(perm.ltrmin())

def inversions(perm):
    """Returns the number of inversions of the permutation, i.e., the
    number of pairs i,j such that i < j and perm(i) > perm(j).

    >>> Permutation(4132).inversions()
    4
    >>> Permutation.monotone_decreasing(6).inversions() == 5*6 / 2
    True
    >>> Permutation.monotone_increasing(7).inversions()
    0
    """

    p = list(perm)
    n = perm.__len__()
    inv = 0
    for i in range(n):
        for j in range(i+1,n):
            if p[i]>p[j]:
                inv+=1
    return inv

def noninversions(perm):
    p = list(perm)
    n = perm.__len__()
    inv = 0
    for i in range(n):
        for j in range(i+1,n):
            if p[i]<p[j]:
                inv+=1
    return inv

def bonds(perm):
    numbonds = 0
    p = list(perm)
    for i in range(1,len(p)):
        if p[i] - p[i-1] == 1 or p[i] - p[i-1] == -1:
            numbonds+=1
    return numbonds

def majorindex(perm):
    sum = 0
    p = list(perm)
    n = perm.__len__()
    for i in range(0,n-1):
        if p[i] > p[i+1]:
            sum += i + 1
    return sum

def fixedptsplusbonds(perm):
    return perm.fixed_points() + perm.bonds()

def longestrunA(perm):
    p = list(perm)
    n = perm.__len__()
    maxi = 0
    length = 1
    for i in range(1,n):
        if p[i-1] < p[i]:
            length += 1
            if length > maxi: maxi = length
        else:
            length = 1
    return max(maxi,length)

def longestrunD(perm):
    return perm.complement().longestrunA()

def longestrun(perm):
    return max(perm.longestrunA(), perm.longestrunD())

def christiecycles(perm):
    # builds a permutation induced by the black and gray edges separately, and
    # counts the number of cycles in their product. used for transpositions
    p = list(perm)
    n = perm.__len__()
    q = [0] + [p[i] + 1 for i in range(n)]
    grayperm = range(1,n+1) + [0]
    blackperm = [0 for i in range(n+1)]
    for i in range(n+1):
        ind = q.index(i)
        blackperm[i] = q[ind-1]
    newperm = []
    for i in range(n+1):
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
    blackperm = [0 for i in range(n+1)]
    for i in range(n+1):
        ind = q.index(i)
        blackperm[i] = q[ind-1]
    newperm = []
    for i in range(n+1):
        k = blackperm[i]
        j = grayperm[k]
        newperm.append(j)
    return Permutation(newperm).numcycles()

def sumcycles(perm):
    return perm.othercycles() + perm.christiecycles()

def maxcycles(perm):
    return max(perm.othercycles() - 1,perm.christiecycles())

def is_involution(perm):
    """Checks if the permutation is an involution, i.e., is equal to it's
    own inverse. """

    return perm == perm.inverse()

def is_identity(perm):
    """Checks if the permutation is the identity.

    >>> p = Permutation.random(10)
    >>> (p * p.inverse()).is_identity()
    True
    """

    return perm == Permutation.identity(len(perm))

def threepats(perm):
    p = list(perm)
    n = perm.__len__()
    patnums = {'123' : 0, '132' : 0, '213' : 0,
                         '231' : 0, '312' : 0, '321' : 0}
    for i in range(n-2):
        for j in range(i+1,n-1):
            for k in range(j+1,n):
                patnums[''.join(map(lambda x:
                                                        str(x+1),Permutation([p[i], p[j], p[k]])))] += 1
    return patnums

def fourpats(perm):
    p = list(perm)
    n = perm.__len__()
    patnums = {'1234' : 0, '1243' : 0, '1324' : 0,
                         '1342' : 0, '1423' : 0, '1432' : 0,
                         '2134' : 0, '2143' : 0, '2314' : 0,
                         '2341' : 0, '2413' : 0, '2431' : 0,
                         '3124' : 0, '3142' : 0, '3214' : 0,
                         '3241' : 0, '3412' : 0, '3421' : 0,
                         '4123' : 0, '4132' : 0, '4213' : 0,
                         '4231' : 0, '4312' : 0, '4321' : 0 }

    for i in range(n-3):
        for j in range(i+1,n-2):
            for k in range(j+1,n-1):
                for m in range(k+1,n):
                    patnums[''.join(map(lambda x:
                                        str(x+1),list(Permutation([p[i], p[j], p[k], p[m]]))))] += 1
    return patnums

def num_consecutive_3214(perm):
    number = 0
    n = len(perm)
    for i in range(n-3):
        if perm[i+2] < perm[i+1] < perm[i] < perm[i+3]:
            number += 1
    return number
