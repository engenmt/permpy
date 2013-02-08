from permutation import *
from permset import *
from permtools import *
from math import factorial

class AvClassWithBasis(list):
  
  def __init__(self, basis, max_length=8): 
    list.__init__(self, [PermSet() for i in range(0, max_length+1)])
    self.max_length = max_length
    self.basis = basis


    basis_perms = [Permutation(basis[i]) for i in range(0,len(basis))]

    self[1].add(Permutation([1]));
    for n in range(2,max_length+1):
      for P in self[n-1]:
        for Q in right_extensions(P):
          is_good = True
          for B in basis_perms:
            if not is_good:
              continue
            if B.involved_in(Q):
              is_good = False
          if is_good:
            self[n].add(Q)


  def __len__(self):
    return self.max_length
  