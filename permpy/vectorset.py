from .vector import *

class VectorSet(list):

  def basis_union(self, B):
    if B and B[0] == -1:
      return self
    if self and self[0] == -1:
      return B
    S = set()
    for V in self:
      for W in B:
        S.add(V.meet(W))
    return VectorSet(S).minimal_elements()

  def minimal_elements(self):
    C = self[:]
    for V in self:
      for W in C:
  	    if V != W and W in V:
  	      C.remove(V)
  	      break
    return C

  def meet_all(self):
    if len(self) == 0:
      return Vector([])

    l = len(self[0])
    assert all(len(v) == l for v in self), "Can't meet a set of differently lengthed vectors!"
    
    return Vector([max(*vals, 1) for vals in zip(self)])
    