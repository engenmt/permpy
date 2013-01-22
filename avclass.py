import permutation
import permset

class AvClass(list):
  
  def __init__(self, n = 8): 
    list.__init__(self, [permset.PermSet(permutation.Perm.listall(i)) for i in range(n + 1)])
    self.avoids = []
    self.length = n

  def __len__(self):
    return self.length
    

  def avoid(self, perm):
    ''' builds the avoidance set in the most naive way possible.
        very slow for length 9 or higher'''
    # TODO: use a better algorithm!
    n = self.__len__()
    k = len(perm)
    upset = perm.buildupset(n + 1)
    for i in range(k, n+1):
      self[i] = self[i].difference(upset[i])
    self.avoids.append(perm)
      
