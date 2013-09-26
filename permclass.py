from .permutation import *
from .permset import *
import copy

class PermClass(list):
  
  # def __init__(self, n = 8): 
    # list.__init__(self, [permset.PermSet(permutation.Permutation.listall(i)) for i in range(n + 1)])
    # self.avoids = []
    # self.length = n

  # def __len__(self):
  #   return self.length

  def filter_by(self, test):
    for i in range(0, len(self)):
      D = list(self[i])
      for P in D:
        if not test(P):
          self[i].remove(P)

  def guess_basis(self, max_length=8):
    B = PermSet()
    B.update(self.check_tree_basis(max_length, Permutation([1,2]), PermSet()))
    B.update(self.check_tree_basis(max_length, Permutation([2,1]), PermSet()))
    return B.minimal_elements()

  def check_tree_basis(self, max_length, R, S):
    if R not in self[len(R)]:
      for s in S:
        if s.involved_in(R):
          return S
      S.add(R)
      return S
    else:
      if len(R) == max_length:
        return S
      re = R.right_extensions()
      for p in re:
        S = self.check_tree_basis(max_length, p, S)
      return S
    
  def plus_class(self,t):
    C = copy.deepcopy(self)
    for i in range(0,t):
      C = C.plus_one_class()
    return C

  def plus_one_class(self):
    D = copy.deepcopy(self)
    D.append(PermSet())
    for l in range(0,len(self)):
      for P in self[l]:
        D[l+1] = D[l+1].union(P.all_extensions())
    return D