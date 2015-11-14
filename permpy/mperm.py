
import itertools as it


# Generic Permutation class, others build from this?



class _GeneralPermutation(tuple):
  
  # determines whether permutations are indexed from 0 or 1 
  _startIndex = 1

  def __new__(cls, p, distinct = True):
    ''' Initializes a mpermutation object, internal indexing starts at zero. '''
    if isinstance(p, cls):
      return tuple.__new__(cls, p)
    elif isinstance(p, tuple):
      entries = list(p)[:]
    elif isinstance(p, list):
      entries = list(set(p))
    # standardizes, starting at zero
    if distinct:
      assert len(entries) == len(p)
    entries.sort()
    standardization =  map(lambda e: entries.index(e), p)
    return tuple.__new__(cls, standardization)

  def oneline_repr(self):
    s = ' '
    for i in self:
      s += str(i +  self._startIndex) + ' '
    return s

  def __repr__(self):
    return self.oneline_repr()

  def __call__(self,i):
    '''allows permutations to be used as functions 
    (useful for counting cycles)'''
    return self[i]

  def delete(self, idx):
    L = list(self)
    del L[idx]
    return self.__class__(L)




class MPerm(genperm._GeneralPermutation):
  ''' Class representing multiset permutations '''
  
  # static class variable, controls permutation representation
  _REPR = 'oneline'  

  def __new__(cls, p):
    return super(MPerm, cls).__new__(cls, p, distinct = False)





