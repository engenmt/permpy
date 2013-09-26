from .permutation import *
from .permset import *
from .permclass import *
from math import factorial

class AvClass(PermClass):
  
  def __init__(self, basis, length=8): 
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
      for P in self[n-1]:
        for Q in P.right_extensions():
          is_good = True
          for B in basis:
            if not is_good:
              continue
            if B.involved_in(Q):
              is_good = False
          if is_good:
            self[n].add(Q)

  def right_juxtaposition(self, C, generate_perms=True):
    A = PermSet()
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