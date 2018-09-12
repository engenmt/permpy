import copy
import time
from math import factorial

import permutation
import permset

class PermClass(list):

    # def __init__(self, n = 8):
        # list.__init__(self, [permset.PermSet(Permutation.listall(i)) for i in range(n + 1)])
        # self.avoids = []
        # self.length = n

    # def __len__(self):
    #       return self.length

    @staticmethod
    def class_from_test(test, l=8, has_all_syms=False):
        """Return the smallest PermClass of all permutations which satisfy the test.

        Args:
            test (func): function which accepts a permutation and returns a Boolean.
            l (int): maximum length to be included in class
            has_all_syms (Boolean): whether the class should be closed under all symmetries.

        Returns:
            PermClass: smallest PermClass of permutations which satisfy the test.

        """

        C = [permset.PermSet([permutation.Permutation([])])] # List consisting of just the PermSet containing the empty Permutation
        for cur_length in range(1,l+1):
            this_len = permset.PermSet([])
            if len(C[cur_length-1]) == 0:
                return PermClass(C)
            to_check = permset.PermSet(set.union(*[P.all_extensions() for P in C[cur_length-1]]))
            to_check = [P for P in to_check if permset.PermSet(P.children()).issubset(C[cur_length-1])]
            while len(to_check) > 0:
                P = to_check.pop() # one permutation
                print(str(P))
                if has_all_syms:
                    syms = permset.PermSet([P.all_syms()])
                if test(P):
                    if has_all_syms:
                        for Q in syms:
                            this_len.add(Q)
                    else:
                        this_len.add(P)
                if has_all_syms:
                    for Q in syms:
                        if Q in to_check:
                            to_check.remove(Q)

            C.append(this_len)
        return PermClass(C)

    def filter_by(self, test):
        """Modify self by removing those permutations which fail the test.

        Note:
            Does not actually ensure the result is a class.
        """
        for i in range(0, len(self)):
            D = list(self[i])
            for P in D:
                if not test(P):
                    self[i].remove(P)

    def guess_basis(self, max_length=6, search_mode=False):
        """
            Guess a basis for the class up to "max_length" by iteratively generating
            the class with basis elements known so far ({}, to start with) and adding
            elements which should be avoided to the basis.

            Search mode goes up to the max length in the class and prints out the number
            of basis elements of each length on the way.
        """

        t = time.time()

        assert max_length < len(self), 'Class not big enough to check that far!'

        if search_mode:
            max_length = len(self)-1

        # Find the first length at which perms are missing.\
        for idx, S in self:
            if idx == 0:
                continue
            
            if len(S) < factorial(idx):
                start_length = idx
                break
        else:
            # If we're here, then self is the class of all permutations.
            return permset.PermSet([])
        
        # Add missing perms of minimum length to basis.
        start_length = min(not_all_perms)
        basis = permset.PermSet(permutation.Permutation.listall(start_length)).difference(self[start_length])

        if search_mode:
            print('\t'+str(len(basis))+' basis elements of length '+str(start_length)+'\t\t'+("{0:.2f}".format(time.time() - t)) + ' seconds')
            t = time.time()

        basis_elements_so_far = len(basis)

        current_length = start_length + 1

        # Go up in length, adding missing perms at each step.
        while current_length <= max_length:
            C = avclass.AvClass(basis, current_length)
            basis = basis.union(C[-1].difference(self[current_length]))

            if search_mode:
                print('\t'+str(len(basis)-basis_elements_so_far)+' basis elements of length ' + str(current_length) + '\t\t' + ("{0:.2f}".format(time.time() - t)) + ' seconds')
                t = time.time()

            basis_elements_so_far = len(basis)

            current_length += 1

        return basis


    # def guess_basis(self, max_length=8):
    #       max_length = min(max_length, len(self)-1)
    #       B = permset.PermSet()
    #       B.update(self.check_tree_basis(max_length, permutation.Permutation([1,2]), permset.PermSet()))
    #       B.update(self.check_tree_basis(max_length, permutation.Permutation([2,1]), permset.PermSet()))
    #       return B.minimal_elements()

    # def check_tree_basis(self, max_length, R, S):
    #       if R not in self[len(R)]:
    #           for s in S:
    #               if s.involved_in(R):
    #                   return S
    #           S.add(R)
    #           return S
    #       else:
    #           if len(R) == max_length:
    #               return S
    #           re = R.right_extensions()
    #           for p in re:
    #               S = self.check_tree_basis(max_length, p, S)
    #           return S

    def plus_class(self,t):
        C = copy.deepcopy(self)
        for i in range(0,t):
            C = C.plus_one_class()
        return C

    def plus_one_class(self):
        D = copy.deepcopy(self)
        D.append(permset.PermSet())
        for l in range(0,len(self)):
            for P in self[l]:
                D[l+1] = D[l+1].union(P.all_extensions())
        return D

    def heatmap(self, **kwargs):
        permset = permutation.Permutation()
        for item in self:
            permutation.Permutation.update(item)
        permutation.Permutation.heatmap(**kwargs)

    def sum_closure(self,length=8, has_syms=False):
        return PermClass.class_from_test(lambda P : ((len(P) < len(self) and P in self[len(P)]) or P.sum_decomposable()) and all([Q in self[len(Q)] for Q in P.chom_sum()]), l=length, has_all_syms=has_syms)


