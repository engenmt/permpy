import random
import fractions
from functools import reduce

import permpy.permutation
from permpy.permutation import Permutation
# import permpy.permclass

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl_imported = True
except ImportError:
    mpl_imported = False


class PermSet(set):
    """Represents a set of permutations, and allows statistics to be computed
    across the set."""

    def __repr__(self):
        return 'Set of {} permutations'.format(len(self))

    def __add__(self, other):
        """Returns the union of the two permutation sets. Does not modify in
        place.

        Example
        -------
        >>> S = PermSet.all(3) + PermSet.all(4)
        >>> len(S)
        30
        """
        result = PermSet()
        result.update(self); result.update(other)
        return result


    @classmethod
    def all(cls, length):
        """Builds the set of all permutations of a given length.

        Parameters:
        -----------
        length : int
            the length of the permutations

        Examples
        --------
        >>> p = Permutation(12); q = Permutation(21)
        >>> PermSet.all(2) == PermSet([p, q])
        True
        """
        return PermSet(Permutation.listall(length))

    def get_random(self):
        """Returns a random element from the set.

        Example
        -------
        >>> p = PermSet.all(4).get_random()
        >>> p in PermSet.all(4) and len(p) == 4
        True
        """

        return random.sample(self, 1)[0]


    def get_length(self, length=None):
        """Returns the subset of permutations which have the specified length.

        Parameters
        ----------
        length : int
            lenght of permutations to be returned

        Example
        -------
        >>> S = PermSet.all(4) + PermSet.all(3)
        >>> S.get_length(3) == PermSet.all(3)
        True
        """
        return PermSet(p for p in self if len(p) == length)

    def heatmap(self, only_length=None, ax=None, blur=False, gray=False, **kwargs):
        """Visalization of a set of permutations, which, for each length, shows
        the relative frequency of each value in each position.

        Paramters
        ---------
        only_length : int or None
            if given, restrict to the permutations of this length
        """
        if not mpl_imported:
            err = 'heatmap requires matplotlib to be imported'
            raise NotImplementedError(err)
        try:
            import numpy as np
        except ImportError as e:
            err = 'heatmap function requires numpy'
            raise e(err)
        # first group permutations by length
        total_size = len(self)
        perms_by_length = {}
        for perm in self:
            n = len(perm)
            if n in perms_by_length:
                perms_by_length[n].add(perm)
            else:
                perms_by_length[n] = PermSet([perm])
        # if given a length, ignore all other lengths
        if only_length:
            perms_by_length = {only_length: perms_by_length[only_length]}
        lengths = list(perms_by_length.keys())
        def lcm(l):
            """Returns the least common multiple of the list l."""
            lcm = reduce(lambda x,y: x*y // fractions.gcd(x,y), l)
            return lcm
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
                    small_grid[length-val-1, idx] += 1
            mul = grid_size // length
            inflated = inflate(small_grid, mul)
            num_perms = len(permset)
            inflated /= inflated.max()
            grid += inflated

        if not ax:
            ax = plt.gca()
        if blur:
            interpolation = 'bicubic'
        else:
            interpolation = 'nearest'
        def get_cubehelix(gamma=1, start=1, rot=-1, hue=1, light=1, dark=0):
            """Get a cubehelix palette."""
            cdict = mpl._cm.cubehelix(gamma, start, rot, hue)
            cmap = mpl.colors.LinearSegmentedColormap("cubehelix", cdict)
            x = np.linspace(light, dark, 256)
            pal = cmap(x)
            cmap = mpl.colors.ListedColormap(pal)
            return cmap
        if gray:
            cmap = get_cubehelix(start=.5, rot=1, light=1, dark=.2, hue=0)
        else:
            cmap = get_cubehelix(start=.5, rot=-.5, light=1, dark=.2)

        ax.imshow(grid, cmap=cmap, interpolation=interpolation)
        ax.set_aspect('equal')
        ax.set(**kwargs)
        ax.axis('off')
        return ax











    def show_all(self):
        """The default representation doesn't print the entire set, this
        function does."""
        return set.__repr__(self)

    def __add__(self, other):
        result = PermSet()
        result.update(self)
        result.update(other)
        return result

    def minimal_elements(self):
        """Returns the elements of the set which are minimal with respect to
        the permutation pattern order.

        Examples
        --------
        """



        B = list(self)
        B = sorted(B, key=len)
        C = B[:]
        n = len(B)
        for (i,b) in enumerate(B):
            # if i % 1 == 0:
                # print i,'/',n
            if b not in C:
                continue
            for j in range(i+1,n):
                if B[j] not in C:
                    continue
                if b.involved_in(B[j]):
                    C.remove(B[j])
        return PermSet(C)

    def all_syms(self):
        sym_set = [frozenset(self)]
        sym_set.append(frozenset([i.reverse() for i in self]))
        sym_set.append(frozenset([i.complement() for i in self]))
        sym_set.append(frozenset([i.reverse().complement() for i in self]))
        sym_set.extend([frozenset([k.inverse() for k in L]) for L in sym_set])
        return frozenset(sym_set)

    def layer_down(self):
        S = PermSet()
        i = 1
        n = len(self)
        for P in self:
            # if i % 10000 == 0:
                # print('\t',i,'of',n,'. Now with',len(S),'.')
            S.update(P.shrink_by_one())
            i += 1
        return S

    def downset(self, return_class=False):
        bottom_edge = PermSet()
        bottom_edge.update(self)

        done = PermSet(bottom_edge)
        while len(bottom_edge) > 0:
            oldsize = len(done)
            next_layer = bottom_edge.layer_down()
            done.update(next_layer)
            del bottom_edge
            bottom_edge = next_layer
            del next_layer
            newsize = len(done)
            # print '\t\tDownset currently has',newsize,'permutations, added',(newsize-oldsize),'in the last run.'
        if not return_class:
            return done
        cl = [PermSet([])]
        max_length = max([len(P) for P in done])
        for i in range(1,max_length+1):
            cl.append(PermSet([P for P in done if len(P) == i]))
        return permclass.PermClass(cl)


    def total_statistic(self, statistic):
        return sum([statistic(p) for p in self])

    def threepats(self):
        patnums = {'123' : 0, '132' : 0, '213' : 0,
                             '231' : 0, '312' : 0, '321' : 0}
        L = list(self)
        for p in L:
            n = len(p)
            for i in range(n-2):
                for j in range(i+1,n-1):
                    for k in range(j+1,n):
                        std = permutation.Permutation.standardize([p[i], p[j], p[k]])
                        patnums[''.join([str(x + 1) for x in std])] += 1
        return patnums

    def fourpats(self):
        patnums = {'1234' : 0, '1243' : 0, '1324' : 0,
                             '1342' : 0, '1423' : 0, '1432' : 0,
                             '2134' : 0, '2143' : 0, '2314' : 0,
                             '2341' : 0, '2413' : 0, '2431' : 0,
                             '3124' : 0, '3142' : 0, '3214' : 0,
                             '3241' : 0, '3412' : 0, '3421' : 0,
                             '4123' : 0, '4132' : 0, '4213' : 0,
                             '4231' : 0, '4312' : 0, '4321' : 0 }
        L = list(self)
        for p in L:
            n = len(p)
            for i in range(n-3):
                for j in range(i+1,n-2):
                    for k in range(j+1,n-1):
                        for m in range(k+1,n):
                            std = permutation.Permutation.standardize([p[i], p[j], p[k], p[m]])
                            patnums[''.join([str(x + 1) for x in std])] += 1
        return patnums
