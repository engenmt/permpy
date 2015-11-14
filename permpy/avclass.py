from math import factorial
import types

import permpy.permset
import permpy.permclass
from permpy.permutation import Permutation
from permpy.permset import PermSet


class AvClass(permpy.permclass.PermClass):
    """Object representing an avoidance class
    >>> p = Permutation(123)
    >>> av = AvClass(p)
    """


    def __init__(self, basis, length=8, verbose=0):
        list.__init__(self, [PermSet() for i in range(0, length+1)])
        self.length = length

        temp_basis = []
        for P in basis:
            temp_basis.append(Permutation(P))
        basis = temp_basis
        self.basis = basis

        if length >= 1:
                self[1].add(Permutation([1]));
        for n in range(2,length+1):
            k = 0
            outof = len(self[n-1])
            for P in self[n-1]:
                k += 1
                if verbose > 0 and k % verbose == 0:
                    # print '\t\t\t\tRight Extensions:',k,'/',outof,'\t( length',n,')'
                    print('\t\t\t\tRight Extenstions: {}/{}\t( length {}'.format(
                                k, outof, n))
                insertion_locations = P.insertion_locations
                add_this_time = []
                for Q in P.right_extensions():
                    is_good = True
                    for B in basis:
                        if B.involved_in(Q,last_require=2):
                            is_good = False
                            insertion_locations[Q[-1]] = 0
                            # break
                    if is_good:

                        add_this_time.append(Q)
                for Q in add_this_time:
                    # print Q,'is good'
                    # print '\tchanging IL from ',Q.insertion_locations,'to',(insertion_locations[:Q[-1]+1]+    insertion_locations[Q[-1]:])
                    Q.insertion_locations = insertion_locations[:Q[-1]+1] + insertion_locations[Q[-1]:]
                    self[n].add(Q)

    def extend_to_length(self, l):
        for i in range(self.length+1, l+1):
            self.append(permset.PermSet())
        if (l <= self.length):
            return
        old = self.length
        self.length = l
        for n in range(old+1,l+1):
            for P in self[n-1]:
                insertion_locations = P.insertion_locations
                add_this_time = []
                for Q in P.right_extensions():
                    is_good = True
                    for B in self.basis:
                        if B.involved_in(Q,last_require=2):
                            is_good = False
                            insertion_locations[Q[-1]] = 0
                            # break
                    if is_good:

                        add_this_time.append(Q)
                for Q in add_this_time:
                    # print Q,'is good'
                    # print '\tchanging IL from ',Q.insertion_locations,'to',(insertion_locations[:Q[-1]+1]+    insertion_locations[Q[-1]:])
                    Q.insertion_locations = insertion_locations[:Q[-1]+1] + insertion_locations[Q[-1]:]
                    self[n].add(Q)

    def right_juxtaposition(self, C, generate_perms=True):
        A = permset.PermSet()
        max_length = max([len(P) for P in self.basis]) + max([len(P) for P in C.basis])
        for n in range(2, max_length+1):
            for i in range(0, factorial(n)):
                P = Permutation(i,n)
                for Q in self.basis:
                    for R in C.basis:
                        if len(Q) + len(R) == n:
                            if (Q == Permutation(P[0:len(Q)]) and R == Permutation(P[len(Q):n])):
                                A.add(P)
                        elif len(Q) + len(R) - 1 == n:
                            if (Q == Permutation(P[0:len(Q)]) and Permutation(R) == Permutation(P[len(Q)-1:n])):
                                A.add(P)
        return AvClass(list(A.minimal_elements()), length=(8 if generate_perms else 0))

    def above_juxtaposition(self, C, generate_perms=True):
        inverse_class = AvClass([P.inverse() for P in C.basis], 0)
        horizontal_juxtaposition = self.right_juxtaposition(inverse_class, generate_perms=False)
        return AvClass([B.inverse() for B in horizontal_juxtaposition.basis], length=(8 if generate_perms else 0))

    def contains(self, C):
        for P in self.basis:
            good = False
            for Q in C.basis:
                if P.involved_in(Q):
                    good = True
                    break
            if not good:
                return False
        return True

