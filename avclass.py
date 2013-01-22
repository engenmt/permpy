from .permutation import Perm

class AvClass(list):
  
  def __init__(self, n = 8): 
    list.__init__(self, [PermSet(Perm.listall(i)) for i in range(n + 1)])
    self.avoids = []
    self.length = n

  def __len__(self):
    return self.length
    

  def avoid(self, perm):
    n = self.__len__()
    k = len(perm)
    upset = perm.buildupset(n + 1)
    for i in range(k, n+1):
      self[i] = self[i].difference(upset[i])
    self.avoids.append(perm)
      
