import sys
import os
import subprocess
import time
import math
import random
import fractions
import itertools


# python 2/3 compatibility
from functools import reduce

try:
    import matplotlib.pyplot as plt
    mpl_imported = True
except ImportError:
    mpl_imported = False


import permpy.permset

__author__ = 'Cheyne Homberger, Jay Pantone'

def _is_iter(obj):
    try:
        iter(obj)
        result = True
    except TypeError:
        result = False
    return result


# a class for creating permutation objects
class Permutation(tuple):
    """Class for Permutation objects, representing permutations of an ordered
    n-element set. Permutation objects are immutable, and represented internally
    as a n-tuple of the integers 0 through n-1."""

    # static class variable, controls permutation representation
    _REPR = 'oneline'

    # default to displaying permutations as 1-based
    _BASE = 1

    lower_bound = []
    upper_bound = []
    bounds_set = False;
    insertion_locations = []

    # some useful functions for playing with permutations
    @classmethod
    def monotone_increasing(cls, n):
        """Returns a monotone increasing permutation of length n.

        >>> Permutation.monotone_increasing(5)
        1 2 3 4 5
        """
        return cls(range(n))

    @classmethod
    def monotone_decreasing(cls, n):
        """Returns a monotone decreasing permutation of length n.

        >>> Permutation.monotone_decreasing(5)
        5 4 3 2 1
        """
        return cls(range(n)[::-1])

    @classmethod
    def identity(cls, n):
        """Returns the identity permutation of length n. Same as
        monotone_increasing."""
        return cls.monotone_increasing(n)

    @classmethod
    def random(cls, n):
        """Outputs a random permutation of length n.

        >>> len( Permutation.random(10) ) == 10
        True
        """
        L = list(range(n))
        random.shuffle(L)
        return cls(L)

    @classmethod
    def random_avoider(cls, n, B, simple=False, involution=False, verbose=-1):
        """Generates a (uniformly) random permutation which avoids the patterns
        contained in `B`.

        Parameters
        ----------
        B : PermSet or list of objects which can be coerced to Permutations
            Basis of permutations to be avoided
        simple : Boolean
            Restrict to simple permutations
        involution : Boolean
            Restrict to involutions
        verbose : int
            Level of verbosity (-1 for none). Doubling the integer doubles the
            number of messages printed

        Returns
        -------
        p : Permutation instance
            A permutation avoiding all the patterns in `B`

        >>> p = Permutation.random_avoider(8, [123])
        >>> p.involves(123)
        False
        """

        i = 1
        p = cls.random(n)
        while (involution and not p.is_involution()) \
            or (simple and not p.is_simple()) or not p.avoids_set(B):
            i += 1
            p = cls.random(n)
            if verbose != -1 and i % verbose == 0:
                print("Tested: "+str(i)+" permutations.");
        return p


    @classmethod
    def listall(cls, n):
        """Returns a list of all permutations of length `n`"""
        if n == 0:
            return []
        else:
            L = []
            for k in range(math.factorial(n)):
                L.append(cls(k,n))
            return L

    @classmethod
    def standardize(cls, L):
        """Standardizes a list `L` of unique elements by mapping them to the set
        {0,1, ..., len(L)} by an order-preserving bijection"""
        assert len(set(L)) == len(L), 'make sure elements are distinct!'
        ordered = L[:]
        ordered.sort()
        return [ordered.index(x) for x in L]

    @classmethod
    def change_repr(cls, representation=None):
        """Toggles globally between cycle notation or one-line notation. Note
        that internal representation is still one-line."""
        L = ['oneline', 'cycle', 'both']
        if representation in L:
            cls._REPR = representation
        else:
            k = int(input('1 for oneline, 2 for cycle, 3 for both\n '))
            k -= 1
            cls._REPR = L[k]

    @classmethod
    def ind2perm(cls, k, n):
        """De-indexes a permutation by a bijection from the set S_n to [n!].
        See also the `Permutation.perm2ind` method.

        Parameters
        ----------
        k : int
            An integer between 0 and n! - 1, to be mapped to S_n.
        n : int
            Length of the permutation.

        Returns
        -------
        p : Permutation instance

        >>> Permutation.ind2perm(12,8).perm2ind()
        12
        """

        result = list(range(n))
        def swap(i,j):
            t = result[i]
            result[i] = result[j]
            result[j] = t
        for i in range(n, 0, -1):
            j = k % i
            swap(i-1,j)
            k //= i
        p = cls(result)
        return p

    @classmethod
    def plentiful(cls, gap):
        """Returns the gap-plentiful permutation of minimal(?) length."""
        # if gap == 6:
        #     return Permutation([5,10,15,2,7,12,17,4,9,14,1,6,11,16,3,8,13])
        d = gap-1
        if d % 2:
            firsts = list(range(2, d+1, 2)) + list(range(1, d+2, 2))
        else:
            firsts = list(range(1, d+1, 2)) + list(range(2, d+2, 2))
        def segment(first):
            return list(range(first, first+(d-2)*(d+1)+1, d+1))
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


    # overloaded built in functions:
    def __new__(cls, p, n = None):
        """Creates a new permutation object. Supports a variety of creation
        methods.

        Parameters
        ----------
        p : Permutation, tuple, list, or int
        n : int (optional)
            If `p` is an iterable containing distinct elements, they will be
            standardized to produce a permutation of length `len(p)`.
            If `n` is given, and `p` is an integer, use `ind2perm` to create a
            permutation.
            If `p` is an integer with fewer than 10 digits, try to create a
            permutation from the digits.

        Returns
        -------
        Permutation instance

        >>> Permutation(35124) == Permutation([3, 5, 1, 2, 4])
        True
        >>> Permutation(5, 12) == Permutation.ind2perm(5, 12)
        True
        >>> Permutation([215, -99, 30, 12.1351, 0]) == Permutation(51432)
        True
        """
        def _is_iterable(obj):
            """Quick utility to check if object is iterable."""
            res = True
            try: iter(obj)
            except: res = False
            return res
        entries = []
        if n:
            return Permutation.ind2perm(p, n)
        else:
            if isinstance(p, Permutation):
                return tuple.__new__(cls, p)
            elif _is_iterable(p):
                entries = list(p)[:]
            elif isinstance(p, int):
                entries = [int(digit) for digit in str(p)]

            if len(entries) == 0:
                if len(p) > 0:
                    err = 'Invalid inputs'
                    raise ValueError(err)

            standardization = Permutation.standardize(entries)
            return tuple.__new__(cls, standardization)

    # Not sure what this function does... Jay?
    def __init__(self,p,n=None):
        self.insertion_locations = [1]*(len(self)+1)

    def __call__(self,i):
        """Allows permutations to be called as functions. Used extensively for
        internal methods (e.g., counting cycles). Note that permutations are
        zero-based internally.

        >>> Permutation(4132)(2)
        2
        """

        return self[i]

    def oneline(self):
        """Returns the one-line notation representation of the permutation (as a
        sequence of integers 1 through n)."""
        base = Permutation._BASE
        s = ' '.join( str(entry + base) for entry in self )
        return s

    def cycles(self):
        """Returns the cycle notation representation of the permutation."""
        base = Permutation._BASE
        stringlist = ['( ' + ' '.join([str(x + base) for x in cyc]) + ' )'
                            for cyc in self.cycle_decomp()]
        return ' '.join(stringlist)

    def __repr__(self):
        """Tells python how to display a permutation object."""
        if Permutation._REPR == 'oneline':
            return self.oneline()
        if Permutation._REPR == 'cycle':
            return self.cycles()
        else:
            return '\n'.join([self.oneline(), self.cycles()])


    # __hash__, __eq__, __ne__ inherited from tuple class

    def __mul__(self, other):
        """Returns the composition of two permutations."""
        assert len(self) == len(other)
        L = list(other)
        for i in range(len(L)):
            L[i] = self.__call__(L[i])
        return Permutation(L)

    def __add__(self, other):
        """Returns the direct sum of two permutations.
        >>> p = Permutation.monotone_increasing(10)
        >>> p + p == Permutation.monotone_increasing(20)
        True
        """
        return self.direct_sum(other)

    def __sub__(self, other):
        """Returns the skew sum of two permutations.
        >>> p = Permutation.monotone_decreasing(10)
        >>> p - p == Permutation.monotone_decreasing(20)
        True
        """
        return self.skew_sum(other)

    def __pow__(self, power):
        """Returns the permutation raised to a (positive integer) power.

        >>> p = Permutation.random(10)
        >>> p**p.order() == Permutation.monotone_increasing(10)
        True
        """

        try:
            assert power >= 0 and (isinstance(power, int) or power.is_integer())
        except ValueError:
            err = 'Power must be a positive integer'
            raise ValueError(err)
        power = int(power)
        if power == 0:
            return Permutation(range(len(self)))
        else:
            ans = self
            for i in range(power - 1):
                ans *= self
            return ans

    def perm2ind(self):
        """De-indexes a permutation, by mapping it to an integer between 0 and
        len(self)! - 1. See also `Permutation.ind2perm`.

        >>> p = Permutation(41523)
        >>> Permutation.ind2perm(p.perm2ind(), len(p)) == p
        True
        """
        q = list(self)
        n = self.__len__()
        def swap(i,j):
            t = q[i]
            q[i] = q[j]
            q[j] = t
        result = 0
        multiplier = 1
        for i in range(n-1,0,-1):
            result += q[i]*multiplier
            multiplier *= i+1
            swap(i, q.index(i))
        return result

    def delete(self, idx):
        """Returns the permutation which results from deleting the entry at
        position `idx` from `self`. Recall that indices are zero-indexed.

        >>> Permutation(35214).delete(2)
        2 4 1 3
        """
        p = list(self)
        if _is_iter(idx):
            sorted_idx = sorted(idx, reverse=True)
            for ix in sorted_idx:
                del p[ix]
        else:
            del p[idx]
        return Permutation(p)

    def insert(self,idx,val):
        """Returns the permutation resulting from inserting an entry with value
        just below `val` into the position just before the entry at position
        `idx`. Both the values and index are zero-indexed.

        >>> Permutation(2413).insert(2, 1)
        3 5 2 1 4

        >>> p = Permutation.random(10)
        >>> p == p.insert(4, 7).delete(4)
        True
        """
        p = list(self)
        for k in range(len(p)):
            if p[k] >= val:
                p[k] += 1
        p = p[:idx] + [val] + p[idx:]
        return Permutation(p)

    def complement(self):
        """Returns the complement of the permutation. That is, the permutation
        obtained by subtracting each of the entries from `len(self)`.

        >>> Permutation(2314).complement() == Permutation(3241)
        True
        """
        n = self.__len__()
        L = [n-1-i for i in self]
        return Permutation(L)

    def reverse(self):
        """Returns the reverse of the permutation.

        >>> Permutation(2314).reverse() == Permutation(4132)
        True
        """
        q = list(self)
        q.reverse()
        return Permutation(q)

    def inverse(self):
        """Returns the group-theoretic inverse of the permutation.

        >>> p = Permutation.random(10)
        >>> p * p.inverse() == Permutation.monotone_increasing(10)
        True
        """

        p = list(self)
        n = self.__len__()
        q = [0 for j in range(n)]
        for i in range(n):
            q[p[i]] = i
        return Permutation(q)

    def _ascii_plot(self):
        """Prints a simple plot of the given Permutation."""
        n = self.__len__()
        array = [[' ' for i in range(n)] for j in range(n)]
        for i in range(n):
            array[self[i]][i] = '*'
        array.reverse()
        s = '\n'.join( (' '.join(l) for l in array))
        # return s
        print(s)

    def cycle_decomp(self):
        """Calculates the cycle decomposition of the permutation. Returns a list
        of cycles, each of which is represented as a list.

        >>> Permutation(53814276).cycle_decomp()
        [[4, 3, 0], [6], [7, 5, 1, 2]]
        """
        n = self.__len__()
        seen = set()
        cyclelist = []
        while len(seen) < n:
            a = max(set(range(n)) - seen)
            cyc = [a]
            b = self(a)
            seen.add(b)
            while b != a:
                cyc.append(b)
                b = self(b)
                seen.add(b)
            cyclelist.append(cyc)
        cyclelist.reverse()
        return cyclelist


    def direct_sum(self, Q):
        """Calculates and returns the direct sum of two permutations.

        >>> Permutation(312).direct_sum(Permutation(1234))
        3 1 2 4 5 6 7
        """
        return Permutation(list(self)+[i+len(self) for i in Q])

    def skew_sum(self, Q):
        """Calculates and returns the skew sum of two permutations.

        >>> Permutation(312).skew_sum(Permutation(1234))
        7 5 6 1 2 3 4
        """
        return Permutation([i+len(Q) for i in self]+list(Q))


    # Permutation Statistics - somewhat self-explanatory

    def fixed_points(self):
        """Returns the number of fixed points of the permutation.

        >>> Permutation(521436).fixed_points()
        3
        """
        sum = 0
        for i in range(self.__len__()):
            if self(i) == i:
                sum += 1
        return sum


    def skew_decomposable(self):
        """Determines whether the permutation is expressible as the skew sum of
        two permutations.

        >>> p = Permutation.random(8).direct_sum(Permutation.random(12))
        >>> p.skew_decomposable()
        False
        >>> p.complement().skew_decomposable()
        True
        """

        p = list(self)
        n = self.__len__()
        for i in range(1,n):
            if set(range(n-i,n)) == set(p[0:i]):
                return True
        return False

    def sum_decomposable(self):
        """Determines whether the permutation is expressible as the direct sum of
        two permutations.

        >>> p = Permutation.random(4).direct_sum(Permutation.random(15))
        >>> p.sum_decomposable()
        True
        >>> p.reverse().sum_decomposable()
        False
        """

        p = list(self)
        n = self.__len__()
        for i in range(1,n):
            if set(range(0,i)) == set(p[0:i]):
                return True
        return False

    def num_cycles(self):
        """Returns the number of cycles in the permutation.

        >>> Permutation(53814276).num_cycles()
        3
        """

        return len(self.cycle_decomp())



    def descent_set(self):
        """Returns descent set of the permutation

        >>> Permutation(42561873).descent_set()
        [1, 4, 6, 7]
        """

        p = list(self)
        n = self.__len__()
        descents = []
        for i in range(1,n):
            if p[i-1] > p[i]:
                descents.append(i)
        return descents

    def num_descents(self):
        """Returns the number of descents of the permutation

        >>> Permutation(42561873).num_descents()
        4
        """
        return len(self.descent_set())

    def ascent_set(self):
        """Returns the ascent set of the permutation

        >>> Permutation(42561873).ascent_set()
        [2, 3, 5]
        """
        descents = self.descent_set()
        return [i for i in range(1, len(self)) if i not in descents]

    def num_ascents(self):
        """Returns the number of ascents of the permutation

        >>> Permutation(42561873).num_ascents()
        3
        """
        return len(self.ascent_set())


    def peak_list(self):
        """Returns the list of peaks of the permutation.

        >>> Permutation(2341765).peak_list()
        [2, 4]
        """

        def check(i):
            return self[i-1] < self[i] > self[i+1]
        return [i for i in range(1, len(self)-1) if check(i)]


    def num_peaks(self):
        """Returns the number of peaks of the permutation

        >>> Permutation(2341765).num_peaks()
        2
        """

        return len(self.peak_list())

    def valley_list(self):
        """Returns the list of valleys of the permutation.

        >>> Permutation(3241756).valley_list()
        [1, 3, 5]
        """

        return self.complement().peak_list()


    def num_valleys(self):
        """Returns the number of peaks of the permutation

        >>> Permutation(3241756).num_valleys()
        3
        """

        return len(self.valley_list())

    def bend_list(self):
        """Returns the list of indices at which the permutation changes
        direction. That is, the number of non-monotone consecutive triples of
        the permutation. A permutation p can be expressed as the concatenation
        of len(p.bend_list()) + 1 monotone segments."""
        raise NotImplementedError

        # this isn't quite correct....
        return len([i for i in range(1, len(self)-1) if (self[i-1] > self[i] and self[i+1] > self[i]) or (self[i-1] < self[i] and self[i+1] < self[i])])


    def trivial(self):
        """The trivial permutation statistic, for convenience

        >>> Permutation.random(10).trivial()
        0
        """
        return 0

    def order(self):
        L = map(len, self.cycle_decomp())
        return reduce(lambda x,y: x*y // fractions.gcd(x,y), L)

    def ltrmin(self):
        """Returns the positions of the left-to-right minima.

        >>> Permutation(35412).ltrmin()
        [0, 3]
        """

        n = self.__len__()
        L = []
        minval = len(self) + 1
        for idx, val in enumerate(self):
            if val < minval:
                L.append(idx)
                minval = val
        return L

    def rtlmin(self):
        """Returns the positions of the left-to-right minima.

        >>> Permutation(315264).rtlmin()
        [5, 3, 1]
        """
        rev_perm = self.reverse()
        return [len(self) - val - 1 for val in rev_perm.ltrmin()]

    def ltrmax(self):
        return [len(self)-i-1 for i in Permutation(self[::-1]).rtlmax()][::-1]

    def rtlmax(self):
        return [len(self)-i-1 for i in self.complement().reverse().ltrmin()][::-1]

    def num_ltrmin(self):
        return len(self.ltrmin())

    def inversions(self):
        """Returns the number of inversions of the permutation, i.e., the
        number of pairs i,j such that i < j and self(i) > self(j).

        >>> Permutation(4132).inversions()
        4
        >>> Permutation.monotone_decreasing(6).inversions() == 5*6 / 2
        True
        >>> Permutation.monotone_increasing(7).inversions()
        0
        """

        p = list(self)
        n = self.__len__()
        inv = 0
        for i in range(n):
            for j in range(i+1,n):
                if p[i]>p[j]:
                    inv+=1
        return inv

    def min_gapsize(self):
        """Returns the minimum gap between any two entries in the permutation 
        (computed with the taxicab metric).

        >>> Permutation(3142).min_gapsize()
        3
        """
        # currently uses the naive algorithm --- can be improved 
        min_dist = len(self)
        for i, j in itertools.combinations(range(len(self)), 2):
            h_dist = abs(i - j)
            v_dist = abs(self[i] - self[j])
            dist = h_dist + v_dist
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def noninversions(self):
        p = list(self)
        n = self.__len__()
        inv = 0
        for i in range(n):
            for j in range(i+1,n):
                if p[i]<p[j]:
                    inv+=1
        return inv

    def bonds(self):
        numbonds = 0
        p = list(self)
        for i in range(1,len(p)):
            if p[i] - p[i-1] == 1 or p[i] - p[i-1] == -1:
                numbonds+=1
        return numbonds

    def majorindex(self):
        sum = 0
        p = list(self)
        n = self.__len__()
        for i in range(0,n-1):
            if p[i] > p[i+1]:
                sum += i + 1
        return sum

    def fixedptsplusbonds(self):
        return self.fixed_points() + self.bonds()

    def longestrunA(self):
        p = list(self)
        n = self.__len__()
        maxi = 0
        length = 1
        for i in range(1,n):
            if p[i-1] < p[i]:
                length += 1
                if length > maxi: maxi = length
            else:
                length = 1
        return max(maxi,length)

    def longestrunD(self):
        return self.complement().longestrunA()

    def longestrun(self):
        return max(self.longestrunA(), self.longestrunD())

    def christiecycles(self):
        # builds a permutation induced by the black and gray edges separately, and
        # counts the number of cycles in their product. used for transpositions
        p = list(self)
        n = self.__len__()
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

    def othercycles(self):
        # builds a permutation induced by the black and gray edges separately, and
        # counts the number of cycles in their product
        p = list(self)
        n = self.__len__()
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

    def sumcycles(self):
        return self.othercycles() + self.christiecycles()

    def maxcycles(self):
        return max(self.othercycles() - 1,self.christiecycles())

    def is_involution(self):
        """Checks if the permutation is an involution, i.e., is equal to it's
        own inverse. """

        return self == self.inverse()

    def is_identity(self):
        """Checks if the permutation is the identity.

        >>> p = Permutation.random(10)
        >>> (p * p.inverse()).is_identity()
        True
        """

        return self == Permutation.identity(len(self))

    def threepats(self):
        p = list(self)
        n = self.__len__()
        patnums = {'123' : 0, '132' : 0, '213' : 0,
                             '231' : 0, '312' : 0, '321' : 0}
        for i in range(n-2):
            for j in range(i+1,n-1):
                for k in range(j+1,n):
                    patnums[''.join(map(lambda x:
                                                            str(x+1),Permutation([p[i], p[j], p[k]])))] += 1
        return patnums

    def fourpats(self):
        p = list(self)
        n = self.__len__()
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

    def num_consecutive_3214(self):
        number = 0
        n = len(self)
        for i in range(n-3):
            if self[i+2] < self[i+1] < self[i] < self[i+3]:
                number += 1
        return number

    def coveredby(self):
        S = set()
        n = len(self)
        for i in range(n+1):
            for j in range(n+1):
                S.add(self.ins(i,j))
        return S

    def buildupset(self, height):
        n = len(self)
        L = [set() for i in range(n)]
        L.append( set([self]) )
        for i in range(n + 1, height):
            oldS = list(L[i-1])
            newS    = set()
            for perm in oldS:
                newS = newS.union(perm.coveredby())
            L.append(newS)
        return L

    def set_up_bounds(self):
        L = list(self)
        n = len(L)
        upper_bound = [-1]*n
        lower_bound = [-1]*n
        for i in range(0,n):
            min_above = -1
            max_below = -1
            for j in range(i+1,len(self)):
                if L[j] < L[i]:
                    if L[j] > max_below:
                        max_below = L[j]
                        lower_bound[i] = j
                else:
                    if L[j] < min_above or min_above == -1:
                        min_above = L[j]
                        upper_bound[i] = j
        return (lower_bound, upper_bound)

    def avoids(self, p, lr=0):
        #TODO Am I correct on the lr?
        """Check if the permutation avoids the pattern `p`.

        Parameters
        ----------
        p : Permutation-like object
        lr : int
            Require the last entry to be equal to this

        >>> Permutation(123456).avoids(231)
        True
        >>> Permutation(123456).avoids(123)
        False
        """
        if not isinstance(p, Permutation):
            p = Permutation(p)
        return not p.involved_in(self, last_require=lr)

    def avoids_set(self, B):
        """Check if the permutation avoids the set of patterns.

        Parameters
        ----------
        B : iterable of Permutation-like objects
            Can be a PermSet or an iterable of objects which can be coerced to
            permutations.

        >>> Permutation(123456).avoids_set([321, 213])
        True
        >>> Permutation(123456).avoids_set([321, 123])
        False
        """
        for p in B:
            if not isinstance(p, Permutation):
                p = Permutation(p)
            if p.involved_in(self):
                return False
        return True

    def involves(self, p, lr=0):
        """Check if the permutation avoids the pattern `p`.

        Parameters
        ----------
        p : Permutation-like object
        lr : int
            Require the last entry to be equal to this

        >>> Permutation(123456).involves(231)
        False
        >>> Permutation(123456).involves(123)
        True
        """

        if not isinstance(p, Permutation):
            p = Permutation(p)
        return p.involved_in(self,last_require=lr)

    def involved_in(self, P, last_require=0):
        """ Checks if the permutation is contained as a pattern in `P`.

        >>> Permutation(123).involved_in(31542)
        False
        >>> Permutation(213).involved_in(54213)
        True
        """
        if not isinstance(P, Permutation):
            P = Permutation(P)

        if not self.bounds_set:
            (self.lower_bound, self.upper_bound) = self.set_up_bounds()
            self.bounds_set = True
        L = list(self)
        n = len(L)
        p = len(P)
        if n <= 1 and n <= p:
            return True

        indices = [0]*n

        if last_require == 0:
            indices[len(self)-1] = p - 1
            while indices[len(self)-1] >= 0:
                if self.involvement_check(self.upper_bound, self.lower_bound, indices, P, len(self)-2):
                    return True
                indices[len(self)-1] -= 1
            return False
        else:
            for i in range(1, last_require+1):
                indices[n-i] = p-i
            if not self.involvement_check_final(self.upper_bound, self.lower_bound, indices, P, last_require):
                return False

            return self.involvement_check(self.upper_bound, self.lower_bound, indices, P, len(self) - last_require - 1)

    def involvement_check_final(self, upper_bound, lower_bound, indices, q, last_require):
        for i in range(1,last_require):
            if not self.involvement_fits(upper_bound, lower_bound, indices, q, len(self)-i-1):
                return False
        return True

    def involvement_check(self, upper_bound, lower_bound, indices, q, next):
        if next < 0:
            return True
        # print indices,next
        indices[next] = indices[next+1]-1

        while indices[next] >= 0:
            if self.involvement_fits(upper_bound, lower_bound, indices, q, next) and self.involvement_check(upper_bound, lower_bound, indices, q, next-1):
                return True
            indices[next] -= 1
        return False

    def involvement_fits(self, upper_bound, lower_bound, indices, q, next):
        return (lower_bound[next] == -1 or q[indices[next]] > q[indices[lower_bound[next]]]) and (upper_bound[next] == -1 or q[indices[next]] < q[indices[upper_bound[next]]])


    def occurrences(self, pattern):
        total = 0
        for subseq in itertools.combinations(self, len(pattern)):
            if Permutation(subseq) == pattern:
                total += 1
        return total

    def all_intervals(self, return_patterns=False):
        blocks = [[],[]]
        for i in range(2, len(self)):
            blocks.append([])
            for j in range (0,len(self)-i+1):
                if max(self[j:j+i]) - min(self[j:j+i]) == i-1:
                    blocks[i].append(j)
        if return_patterns:
            patterns = []
            for length in range(0, len(blocks)):
                for start_index in blocks[length]:
                    patterns.append(Permutation(self[start_index:start_index+length]))
            return patterns
        else:
            return blocks

    def all_monotone_intervals(self, with_ones=False):
        mi = []
        difference = 0
        c_start = 0
        c_length = 0
        for i in range(0,len(self)-1):
            if math.fabs(self[i] - self[i+1]) == 1 and (c_length == 0 or self[i] - self[i+1] == difference):
                if c_length == 0:
                    c_start = i
                c_length += 1
                difference = self[i] - self[i+1]
            else:
                if c_length != 0:
                    mi.append((c_start, c_start+c_length))
                c_start = 0
                c_length = 0
                difference = 0
        if c_length != 0:
            mi.append((c_start, c_start+c_length))

        if with_ones:
            in_int = []
            for (start,end) in mi:
                in_int.extend(range(start, end+1))
            for i in range(len(self)):
                if i not in in_int:
                    mi.append((i,i))
            mi.sort(key=lambda x : x[0])
        return mi

    def monotone_quotient(self):
        return Permutation([self[k[0]] for k in self.all_monotone_intervals(with_ones=True)])



    def maximal_interval(self):
        ''' finds the biggest interval, and returns (i,j) is one is found,
            where i is the size of the interval, and j is the index
            of the first entry in the interval

        returns (0,0) if no interval is found, i.e., if the permutation
            is simple'''
        for i in range(2, len(self))[::-1]:
            for j in range (0,len(self)-i+1):
                if max(self[j:j+i]) - min(self[j:j+i]) == i-1:
                    return (i,j)
        return (0,0)

    def simple_location(self):
        ''' searches for an interval, and returns (i,j) if one is found,
            where i is the size of the interval, and j is the
            first index of the interval

        returns (0,0) if no interval is found, i.e., if the permutation
            is simple'''
        mins = list(self)
        maxs = list(self)
        for i in range(1,len(self)-1):
            for j in reversed(range(i,len(self))):
                mins[j] = min(mins[j-1], self[j])
                maxs[j] = max(maxs[j-1], self[j])
                if maxs[j] - mins[j] == i:
                    return (i,j)
        return (0,0)

    def is_simple(self):
        ''' returns True is this permutation is simple, False otherwise'''
        (i,j) = self.simple_location()
        return i == 0

    def is_strongly_simple(self):
        return self.is_simple() and all([p.is_simple() for p in self.children()])

    def decomposition(self):
        base = Permutation(self)
        components = [Permutation([1])for i in range(0,len(base))]
        while not base.is_simple():
            assert len(base) == len(components)
            (i,j) = base.maximal_interval()
            assert i != 0
            interval = list(base[j:i+j])
            new_base = list(base[0:j])
            new_base.append(base[j])
            new_base.extend(base[i+j:len(base)])
            new_components = components[0:j]
            new_components.append(Permutation(interval))
            new_components.extend(components[i+j:len(base)])
            base = Permutation(new_base)
            components = new_components
        return (base, components)

    def inflate(self, components):
        assert len(self) == len(components), 'number of components must equal length of base'
        L = list(self)
        NL = list(self)
        current_entry = 0
        for entry in range(0, len(self)):
            index = L.index(entry)
            NL[index] = [components[index][i]+current_entry for i in range(0, len(components[index]))]
            current_entry += len(components[index])
        NL_flat = [a for sl in NL for a in sl]
        return Permutation(NL_flat)

    def right_extensions(self):
        L = []
        if len(self.insertion_locations) > 0:
            indices = self.insertion_locations
        else:
            indices = [1]*(len(self)+1)

        R = [j for j in range(len(indices)) if indices[j] == 1]
        for i in R:
            A = [self[j] + (1 if self[j] > i-1 else 0) for j in range(0,len(self))]
            A.append(i)
            L.append(Permutation(A))
        return L

    # def all_right_extensions(self, max_length, l, S):
    #       if l == max_length:
    #           return S
    #       else:
    #           re = self.right_extensions()
    #           for p in re:
    #               S.add(p)
    #               S = p.all_right_extensions(max_length, l+1, S)
    #       return S

    def all_extensions(self):
        S = set()
        for i in range(0, len(self)+1):
            for j in range(0, len(self)+1):
                # insert (i-0.5) after entry j (i.e., first when j=0)
                l = list(self[:])
                l.insert(j, i-0.5)
                S.add(Permutation(l))
        return permpy.permset.PermSet(S)

    def all_extensions_track_index(self, ti):
        L = []
        for i in range(0, len(self)+1):
            for j in range(0, len(self)+1):
                # insert (i-0.5) after entry j (i.e., first when j=0)
                l = list(self[:])
                l.insert(j, i-0.5)
                if j < ti:
                    L.append((Permutation(l), ti+1))
                else:
                    L.append((Permutation(l), ti))
        return L

    def plot(self, show=True, ax=None, use_mpl=True, fname=None, **kwargs):
        """Draws a matplotlib plot of the permutation. Can be used for both
        quick visualization, or to build a larger figure. Unrecognized arguments
        are passed as options to the axes object to allow for customization
        (i.e., setting a figure title, or setting labels on the axes). Falls
        back to an ascii_plot if matplotlib isn't found, or if use_mpl is set to
        False.
        """
        if not mpl_imported or not use_mpl:
            return self._ascii_plot()
        xs = [val + Permutation._BASE for val in range(len(self))]
        ys = [val + Permutation._BASE for val in self]
        if not ax:
            ax = plt.gca()
        scat = ax.scatter(xs, ys, s=40, c='k')
        ax_settings = {'xticks': xs, 'yticks': ys,
                    'xticklabels': '', 'yticklabels': '',
                    'xlim': (min(xs) - 1, max(xs) + 1),
                    'ylim': (min(ys) - 1, max(ys) + 1)}
        ax.set(**ax_settings)
        ax.set(**kwargs)
        ax.set_aspect('equal')
        if fname:
            fig = plt.gcf()
            fig.savefig(fname, dpi=300)
        if show:
            plt.show()
        return ax


    def _show(self):
        if sys.platform == 'linux2':
            opencmd = 'gnome-open'
        else:
            opencmd = 'open'
        s = r"\documentclass{standalone}\n\usepackage{tikz}\n\n\\begin{document}\n\n"
        s += self.to_tikz()
        s += "\n\n\end{document}"
        dname = random.randint(1000,9999)
        os.system('mkdir t_'+str(dname))
        with open('t_'+str(dname)+'/t.tex', 'w') as f:
            f.write(s)
        subprocess.call(['pdflatex', '-output-directory=t_'+str(dname), 't_'+str(dname)+'/t.tex'],
            stderr = subprocess.PIPE, stdout = subprocess.PIPE)
        # os.system('pdflatex -output-directory=t_'+str(dname)+' t_'+str(dname)+'/t.tex')
        subprocess.call([opencmd, 't_'+str(dname)+'/t.pdf'],
            stderr = subprocess.PIPE, stdout = subprocess.PIPE)
        time.sleep(1)
        if sys.platform != 'linux2':
            subprocess.call(['rm', '-r', 't_'+str(dname)+'/'])

    def to_tikz(self):
        s = r'\begin{tikzpicture}[scale=.3,baseline=(current bounding box.center)]';
        s += '\n\t'
        s += r'\draw[ultra thick] (1,0) -- ('+str(len(self))+',0);'
        s += '\n\t'
        s += r'\draw[ultra thick] (0,1) -- (0,'+str(len(self))+');'
        s += '\n\t'
        s += r'\foreach \x in {1,...,'+str(len(self))+'} {'
        s += '\n\t\t'
        s += r'\draw[thick] (\x,.09)--(\x,-.5);'
        s += '\n\t\t'
        s += r'\draw[thick] (.09,\x)--(-.5,\x);'
        s += '\n\t'
        s += r'}'
        for (i,e) in enumerate(self):
            s += '\n\t'
            s += r'\draw[fill=black] ('+str(i+1)+','+str(e+1)+') circle (5pt);'
        s += '\n'
        s += r'\end{tikzpicture}'
        return s

    def shrink_by_one(self):
        return permpy.permset.PermSet([Permutation(p) for p in [self[:i]+self[i+1:] for i in range(0,len(self))]])

    def children(self):
        """Returns all patterns of length one less than the permutation."""
        return self.shrink_by_one()

    def downset(self):
        return permpy.permset.PermSet([self]).downset()

    def sum_indecomposable_sequence(self):
        S = self.downset()
        return [len([p for p in S if len(p)==i and not p.sum_decomposable()]) for i in range(1,max([len(p) for p in S])+1)]

    def sum_indec_bdd_by(self, n):
        l = [1]
        S = list(self.children())
        while len(S) > 0 and len(S[0]) > 0:
            l = [len([s for s in S if not s.sum_decomposable()])]+l
            if l[0] > n:
                return False
            S = list(permpy.permset.PermSet(S).layer_down())
        return True

    def contains_locations(self, Q):
        locs = []
        sublocs = itertools.combinations(range(len(self)), len(Q))
        for subloc in sublocs:
            if Permutation([self[i] for i in subloc]) == Q:
                locs.append(subloc)

        return locs

    def rank_val(self, i):
        return len([j for j in range(i+1,len(self)) if self[j] < self[i]])

    def rank_encoding(self):
        return [self.rank_val(i) for i in range(len(self))]

    def num_rtlmax_ltrmin_layers(self):
        return len(self.rtlmax_ltrmin_decomposition())

    def rtlmax_ltrmin_decomposition(self):
        P = Permutation(self)
        num_layers = 0
        layers = []
        while len(P) > 0:
            num_layers += 1
            positions = sorted(list(set(P.rtlmax()+P.ltrmin())))
            layers.append(positions)
            P = Permutation([P[i] for i in range(len(P)) if i not in positions])
        return layers

    def num_inc_bonds(self):
        return len([i for i in range(len(self)-1) if self[i+1] == self[i]+1])

    def num_dec_bonds(self):
        return len([i for i in range(len(self)-1) if self[i+1] == self[i]-1])

    def num_bonds(self):
        return len([i for i in range(len(self)-1) if self[i+1] == self[i]+1 or self[i+1] == self[i]-1])

    def contract_inc_bonds(self):
        P = Permutation(self)
        while P.num_inc_bonds() > 0:
            for i in range(0,len(P)-1):
                if P[i+1] == P[i]+1:
                    P = Permutation(P[:i]+P[i+1:])
                    break
        return P

    def contract_dec_bonds(self):
        P = Permutation(self)
        while P.num_dec_bonds() > 0:
            for i in range(0,len(P)-1):
                if P[i+1] == P[i]-1:
                    P = Permutation(P[:i]+P[i+1:])
                    break
        return P

    def contract_bonds(self):
        P = Permutation(self)
        while P.num_bonds() > 0:
            for i in range(0,len(P)-1):
                if P[i+1] == P[i]+1 or P[i+1] == P[i]-1:
                    P = Permutation(P[:i]+P[i+1:])
                    break
        return P

    def all_syms(self):
        S = permpy.permset.PermSet([self])
        S = S.union(permpy.permset.PermSet([P.reverse() for P in S]))
        S = S.union(permpy.permset.PermSet([P.complement() for P in S]))
        S = S.union(permpy.permset.PermSet([P.inverse() for P in S]))
        return S

    def is_representative(self):
        return self == sorted(self.all_syms())[0]

    def greedy_sum(p):
        parts = []
        sofar = 0
        while sofar < len(p):
            if len(p)-sofar == 1:
                parts.append(Permutation(1))
                return parts
            i = 1
            while sofar+i <= len(p) and list(p[sofar:sofar+i]) == range(sofar,sofar+i):
                i += 1
            i -= 1
            if i > 0:
                parts.append(Permutation(range(i)))
            sofar += i
            i = 2
            while sofar+i <= len(p) and not (max(p[sofar:sofar+i]) - min(p[sofar:sofar+i])+1 == i and min(p[sofar:sofar+i]) == sofar):
                i += 1
            if sofar+i <= len(p):
                parts.append(Permutation(p[sofar:sofar+i]))
            sofar += i
        return parts

    def chom_sum(p):
        L = []
        p = p.greedy_sum()
        for i in p:
            if i.inversions() == 0:
                L.extend([Permutation(1)]*len(i))
            else:
                L.append(i)
        return L

    def chom_skew(p):
        return [r.reverse() for r in p.reverse().chom_sum()]

if __name__ == '__main__':
    import doctest
    doctest.testmod()


