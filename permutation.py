import math
import random
import fractions

__author__ = 'Cheyne Homberger, Jay Pantone'



# a class for creating permutation objects
class Permutation(tuple):
  'can be initialized with either (index,length) or a list of entries'

  # static class variable, controls permutation representation
  _REPR = 'oneline'
  lower_bound = [] 
  upper_bound = []
  bounds_set = False;

  #===============================================================%
  # some useful functions for playing with permutations
  
  @staticmethod
  def random(n):
    '''outputs a random permutation of length n'''
    L = range(n)
    random.shuffle(L)
    return Permutation(L)

  @staticmethod
  def listall(n):
    '''returns a list of all permutations of length n'''
    if n == 0:
      return []
    else:
      L = []
      for k in range(math.factorial(n)):
        L.append(Permutation(k,n))
      return L
  
  @staticmethod
  def standardize(L):
    # copy by value
    assert len(set(L)) == len(L), 'make sure elements are distinct!'
    ordered = L[:] 
    ordered.sort()
    return [ordered.index(x) for x in L]

  @staticmethod
  def change_repr():
    '''changes between cycle notation or one-line notation. note that
    internal representation is still one-line'''
    L = ['oneline', 'cycles', 'both']
    k = input('1 for oneline, 2 for cycles, 3 for both\n ')
    k -= 1
    Permutation._REPR = L[k]

  @staticmethod
  def ind2perm(k, n):   
    '''de-indexes a permutation'''
    result = range(n)
    def swap(i,j):
      t = result[i]
      result[i] = result[j]
      result[j] = t
    for i in range(n, 0, -1):
      j = k % i
      swap(i-1,j)
      k /= i
    return Permutation(result)

  #================================================================#
  # overloaded built in functions:

  def __new__(cls, p, n = None):
    ''' Initializes a permutation object, internal indexing starts at zero. '''
    if n:
      return Permutation.ind2perm(p, n) 
    else: 
      if isinstance(p, Permutation):
        return tuple.__new__(cls, p)
      elif isinstance(p, tuple):
        entries = list(p)[:]
      elif isinstance(p, list):
        entries = p[:]
      # standardizes, starting at zero
      assert len(set(entries)) == len(entries), 'make sure elements are distinct!'
      entries.sort()
      standardization =  map(lambda e: entries.index(e), p)
      return tuple.__new__(cls, standardization)

  # def __init__(self,p,n=None):
  #   '''initializes a permutation object'''
  #   if isinstance(p, Permutation):
  #     list.__init__(self, p)
  #   else:
  #     if isinstance(p, tuple):
  #       p = list(p)
  #     if not n:
  #       std = Permutation.standardize(p)
  #       list.__init__(self, std)
  #     else:
  #       list.__init__(self,Permutation.ind2perm(p,n))
    
  def __call__(self,i):
    '''allows permutations to be used as functions 
    (useful for counting cycles)'''
    return self[i]

  def oneline(self):
    s = ' '
    for i in self:
      s += str( i+1 ) + ' '
    return s

  def cycles(self):
    stringlist = ['( ' + ' '.join([str(x+1) for x in cyc]) + ' )' 
                                    for cyc in self.cycle_decomp()]
    return ' '.join(stringlist)

  def __repr__(self):
    '''tells python how to display a permutation object'''
    if Permutation._REPR == 'oneline':
      return self.oneline()
    if Permutation._REPR == 'cycles':
      return self.cycles()
    else: 
      return '\n '.join([self.oneline(), self.cycles()])

  # def __hash__(self):
  #   '''allows fast comparisons of permutations, and allows sets of
  #   permutations'''
  #   return (self.perm2ind(), self.__len__()).__hash__()

  # def __eq__(self,other):
  #   ''' checks if two permutations are equal '''
  #   if len(self) != len(other):
  #     return False
  #   for i in range(len(self)):
  #     if self[i] != other[i]:
  #       return False
  #   return True

  # def __ne__(self,other):
  #   return not self == other

  def __mul__(self,other):
    ''' multiplies two permutations '''
    assert len(self) == len(other)
    L = list(other)
    for i in range(len(L)):
      L[i] = self.__call__(L[i])
    return Permutation(L)

  def __pow__(self, power):
    assert isinstance(power, int) and power >= 0
    if power == 0:
      return Permutation(range(len(self)))
    else:
      ans = self 
      for i in range(power - 1):
        ans *= self
      return ans

  def perm2ind(self):      
    ''' De-indexes a permutation. '''
    q = list(self)
    n = self.__len__()
    def swap(i,j):
      t = q[i]
      q[i] = q[j]
      q[j] = t
    result = 0
    multiplier = 1
    for i in range(n-1,0,-1):
      result += q[i]*multiplier
      multiplier *= i+1
      swap(i, q.index(i))
    return result

  def delete(self,i):
    p = list(self)
    del p[i]
    return Permutation(p)

  def ins(self,i,j):
    p = list(self)
    for k in range(len(p)):
      if p[k] >= j:
        p[k] += 1
    p = p[:i] + [j] + p[i:]
    return Permutation(p)

  # returns the complement of the permutation
  def complement(self):
    n = self.__len__()
    L = [n-1-i for i in self]
    return Permutation(L)
    
  # returns the reverse of the permutation  
  def reverse(self):
    q = list(self)
    q.reverse()
    return Permutation(q)

  def inverse(self):
    p = list(self)
    n = self.__len__()
    q = [0 for j in range(n)]
    for i in range(n):
      q[p[i]] = i
    return Permutation(q)

  def plot(self):
    ''' Draws a plot of the given Permutation. '''
    n = self.__len__()
    array = [[' ' for i in range(n)] for j in range(n)]
    for i in range(n):
      array[self[i]][i] = '*'
    array.reverse()
    s = '\n'.join( (''.join(l) for l in array)) 
    # return s
    print(s)

  def cycle_decomp(self):
    n = self.__len__()
    seen = set()
    cyclelist = []
    while len(seen) < n:
      a = max(set(range(n)) - seen)
      cyc = [a]
      b = self(a)
      seen.add(b)
      while b != a:
        cyc.append(b)
        b = self(b)
        seen.add(b)
      cyclelist.append(cyc)
    cyclelist.reverse()
    return cyclelist


    

# Permutation Statistics - somewhat self-explanatory
  
  def fixedpoints(self):
    sum = 0
    for i in range(self.__len__()):
      if self(i) == i:
        sum+=1
    return sum
    

  def decomposable(self):
    p = list(self)
    n = self.__len__()
    for i in range(1,n):
      if set(range(n-i,n)) == set(p[0:i]):
        return True
    return False


  def numcycles(self):
    num = 1
    n = self.__len__()
    list = range(n)
    k = 0
    while list:
      if k in list:
        del list[list.index(k)]
        k = self(k)
      else:
        k = list[0]
        num += 1
    return num
      
  def descents(self):
    p = list(self)
    n = self.__len__()
    numd = 0
    for i in range(1,n):
      if p[i-1] > p[i]:
        numd+=1
    return numd
   
  def ascents(self):
    return self.__len__()-1-self.descents()
    
  def bends(self):
        # Bends measures the number of times the permutation p
        # "changes direction".  Bends is also the number of
        # non-monotone consecutive triples in p.
        # The permutation p can be expressed as the concatenation of
        # bend(p)+1 monotone segments.
    b = 0
    curr_seg = 0
    p = list(self)
        # curr_seq is +1 if the current segment is increasing, -1 for
        # decreasing, and 0 if the current seqment has a single entry
        # in it (and thus could go either way)
    for i in range(1, len(p)):
      if curr_seg == 0:
        if p[i] > p[i-1]:
          curr_seg = 1
        else:
          curr_seg = -1
      elif curr_seg == 1:
        if p[i] < p[i-1]:
          b += 1
          curr_seg = 0
      elif curr_seg == -1:
        if p[i] > p[i-1]:
          b += 1
          curr_seg = 0
    return b
  
  def trivial(self):
    return 0

  def order(self):
    L = map(len, self.cycle_decomp())
    return reduce(lambda x,y: x*y / fractions.gcd(x,y), L) 
  
  def ltrmin(self):
    p = list(self)
    n = self.__len__()
    L = []
    for i in range(n):
      flag = True
      for j in range(i):
        if p[i] > p[j]:
          flag = False
      if flag: L.append(i)
    return L

  def numltrmin(self):
    p = list(self)
    n = self.__len__()
    num = 1
    m = p[0]
    for i in range(1,n):
      if p[i] < m:
        num += 1
        m = p[i]
    return num

  def inversions(self):
    p = list(self)
    n = self.__len__()
    inv = 0
    for i in range(n):
      for j in range(i+1,n):
        if p[i]>p[j]:
          inv+=1
    return inv

  def noninversions(self):
    p = list(self)
    n = self.__len__()
    inv = 0
    for i in range(n):
      for j in range(i+1,n):
        if p[i]<p[j]:
          inv+=1
    return inv
    
  def bonds(self):
    numbonds = 0
    p = list(self)
    for i in range(1,len(p)):
      if p[i] - p[i-1] == 1 or p[i] - p[i-1] == -1:
        numbonds+=1
    return numbonds
    
  def majorindex(self):
    sum = 0
    p = list(self)
    n = self.__len__()
    for i in range(0,n-1):
      if p[i] > p[i+1]:
        sum += i + 1
    return sum
       
  def fixedptsplusbonds(self):
    return self.fixedpoints() + self.bonds()
 
  def longestrunA(self):
    p = list(self)
    n = self.__len__()
    maxi = 0
    length = 1
    for i in range(1,n):
      if p[i-1] < p[i]:
        length += 1
        if length > maxi: maxi = length
      else:
        length = 1
    return max(maxi,length)
  
  def longestrunD(self):
    return self.complement().longestrunA()
  
  def longestrun(self):
    return max(self.longestrunA(),self.longestrunD())
  
  def christiecycles(self): 
    # builds a permutation induced by the black and gray edges separately, and
    # counts the number of cycles in their product. used for transpositions
    p = list(self)
    n = self.__len__()
    q = [0] + [p[i] + 1 for i in range(n)]
    grayperm = range(1,n+1) + [0]
    blackperm = [0 for i in range(n+1)]
    for i in range(n+1):
      ind = q.index(i)
      blackperm[i] = q[ind-1]
    newperm = []
    for i in range(n+1):
      k = blackperm[i]
      j = grayperm[k]
      newperm.append(j)
    return Permutation(newperm).numcycles()
  
  def othercycles(self): 
    # builds a permutation induced by the black and gray edges separately, and
    # counts the number of cycles in their product
    p = list(self)
    n = self.__len__()
    q = [0] + [p[i] + 1 for i in range(n)]
    grayperm = [n] + range(n)
    blackperm = [0 for i in range(n+1)]
    for i in range(n+1):
      ind = q.index(i)
      blackperm[i] = q[ind-1]
    newperm = []
    for i in range(n+1):
      k = blackperm[i]
      j = grayperm[k]
      newperm.append(j)
    return Permutation(newperm).numcycles()
    
  def sumcycles(self):
    return self.othercycles() + self.christiecycles()
   
  def maxcycles(self):
    return max(self.othercycles() - 1,self.christiecycles())

  def threepats(self):
    p = list(self)
    n = self.__len__()
    patnums = {'123' : 0, '132' : 0, '213' : 0, 
               '231' : 0, '312' : 0, '321' : 0}
    for i in range(n-2):
      for j in range(i+1,n-1):
        for k in range(j+1,n):
          patnums[''.join(map(lambda x: 
                              str(x+1),Permutation([p[i], p[j], p[k]])))] += 1
    return patnums

  def fourpats(self):
    p = list(self)
    n = self.__len__()
    patnums = {'1234' : 0, '1243' : 0, '1324' : 0, 
               '1342' : 0, '1423' : 0, '1432' : 0,
               '2134' : 0, '2143' : 0, '2314' : 0,
               '2341' : 0, '2413' : 0, '2431' : 0,
               '3124' : 0, '3142' : 0, '3214' : 0,
               '3241' : 0, '3412' : 0, '3421' : 0,
               '4123' : 0, '4132' : 0, '4213' : 0,
               '4231' : 0, '4312' : 0, '4321' : 0 }

    for i in range(n-3):
      for j in range(i+1,n-2):
        for k in range(j+1,n-1):
          for m in range(k+1,n):
            patnums[''.join(map(lambda x: 
                      str(x+1),Permutation([p[i], p[j], p[k], p[m]]).p))] += 1
    return patnums

  def num_consecutive_3214(self):
    number = 0
    n = len(self)
    for i in range(n-3):
      if self[i+2] < self[i+1] < self[i] < self[i+3]:
        number += 1
    return number

  def coveredby(self):
    S = set()
    n = len(self)
    for i in range(n+1):
      for j in range(n+1):
        S.add(self.ins(i,j))
    return S

  def buildupset(self, height):
    n = len(self)
    L = [set() for i in range(n)]
    L.append( set([self]) )
    for i in range(n + 1, height):
      oldS = list(L[i-1])
      newS  = set()
      for perm in oldS:
        newS = newS.union(perm.coveredby())
      L.append(newS)
    return L

  def set_up_bounds(self):
    L = list(self)
    n = len(L)
    upper_bound = [-1]*n
    lower_bound = [-1]*n
    for i in range(0,n):
      min_above = -1
      max_below = -1
      for j in range(0,i):
        if L[j] < L[i]:
          if L[j] > max_below:
            max_below = L[j]
            lower_bound[i] = j
        else:
          if L[j] < min_above or min_above == -1:
            min_above = L[j]
            upper_bound[i] = j
    return (lower_bound, upper_bound)

  def involved_in(self, P):
    if not self.bounds_set:
      (self.lower_bound, self.upper_bound) = self.set_up_bounds()
      self.bounds_set = True
    L = list(self)
    n = len(L)
    p = len(P)
    if n <= 1 and n <= p:
      return True

    indices = [0]*n
    
    while indices[0] < p:
      if self.involvement_check(self.upper_bound, self.lower_bound, indices, P, 1):
        return True
      indices[0] += 1

    return False

  def involvement_check(self, upper_bound, lower_bound, indices, q, next):
    if next == len(indices):
      return True
    lq = len(q)
    indices[next] = indices[next-1]+1
    while indices[next] < lq:
      if self.involvement_fits(upper_bound, lower_bound, indices, q, next) and self.involvement_check(upper_bound, lower_bound, indices, q, next+1):
        return True
      indices[next] += 1
    return False

  def involvement_fits(self, upper_bound, lower_bound, indices, q, next):
    return (lower_bound[next] == -1 or q[indices[next]] > q[indices[lower_bound[next]]]) and (upper_bound[next] == -1 or q[indices[next]] < q[indices[upper_bound[next]]])





      
    






