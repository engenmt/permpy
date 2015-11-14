from __future__ import print_function
import sys, os, subprocess, time
import math
import random
import fractions
import itertools
from math import factorial
import types

import copy
import time
from math import factorial


__author__ = 'Cheyne Homberger, Jay Pantone'



# a class for creating permutation objects
class Permutation(tuple):
  'can be initialized with either (index,length) or a list of entries'

  # static class variable, controls permutation representation
  _REPR = 'oneline'
  lower_bound = []
  upper_bound = []
  bounds_set = False;
  insertion_locations = []

  #===============================================================%
  # some useful functions for playing with permutations

  @staticmethod
  def monotone_increasing(n):
    return Permutation(range(n))

  @staticmethod
  def monotone_decreasing(n):
    return Permutation(range(n)[::-1])

  @staticmethod
  def random(n):
    '''outputs a random permutation of length n'''
    L = range(n)
    random.shuffle(L)
    return Permutation(L)

  @staticmethod
  def random_avoider(n, B, simple=False, involution=False, verbose=-1):
    i = 1
    p = Permutation.random(n)
    while (involution and not p.is_involution()) or (simple and not p.is_simple()) or not p.avoids_set(B):
      i += 1
      p = Permutation.random(n)
      if verbose != -1 and i % verbose == 0:
        print("Tested: "+str(i)+" permutations.");
    return p


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
    result = list(range(n))
    def swap(i,j):
      t = result[i]
      result[i] = result[j]
      result[j] = t
    for i in range(n, 0, -1):
      j = k % i
      swap(i-1,j)
      k //= i
    return Permutation(result)

  #================================================================#
  # overloaded built in functions:

  def __new__(cls, p, n = None):
    ''' Initializes a permutation object, internal indexing starts at zero. '''
    entries = []
    if n:
      return Permutation.ind2perm(p, n)
    else:
      if isinstance(p, Permutation):
        return tuple.__new__(cls, p)
      elif isinstance(p, tuple):
        entries = list(p)[:]
      elif isinstance(p, list):
        entries = p[:]
      elif isinstance(p, int):
        p = str(p)
        entries = list(p)
      # standardizes, starting at zero
      assert len(set(entries)) == len(entries), 'elements not distinct'
      assert len(entries) > 0 or p==list() or p==set() or p==tuple(), 'invalid permutation'
      entries.sort()
      standardization =  map(lambda e: entries.index(e), p)
      return tuple.__new__(cls, standardization)

  def __init__(self,p,n=None):
    self.insertion_locations = [1]*(len(self)+1)

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

  def num_disjoint_cycles(self):
    return len(self.cycle_decomp())

  def sum(self, Q):
    return Permutation(list(self)+[i+len(self) for i in Q])

  def skew_sum(self, Q):
    return Permutation([i+len(Q) for i in self]+list(Q))


# Permutation Statistics - somewhat self-explanatory

  def fixed_points(self):
    sum = 0
    for i in range(self.__len__()):
      if self(i) == i:
        sum += 1
    return sum


  def skew_decomposable(self):
    p = list(self)
    n = self.__len__()
    for i in range(1,n):
      if set(range(n-i,n)) == set(p[0:i]):
        return True
    return False

  def sum_decomposable(self):
    p = list(self)
    n = self.__len__()
    for i in range(1,n):
      if set(range(0,i)) == set(p[0:i]):
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
    return len([i for i in range(1, len(self)-1) if (self[i-1] > self[i] and self[i+1] > self[i]) or (self[i-1] < self[i] and self[i+1] < self[i])])

  def peaks(self):
    return len([i for i in range(1, len(self)-1) if self[i-1] < self[i] and self[i+1] < self[i]])

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

  def rtlmin(self):
    return [len(self)-i-1 for i in Permutation(self[::-1]).ltrmin()]

  def ltrmax(self):
    return [len(self)-i-1 for i in Permutation(self[::-1]).rtlmax()][::-1]

  def rtlmax(self):
    return [len(self)-i-1 for i in self.complement().reverse().ltrmin()][::-1]

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
    return self.fixed_points() + self.bonds()

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

  def is_involution(self):
    return self == self.inverse()

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
                      str(x+1),list(Permutation([p[i], p[j], p[k], p[m]]))))] += 1
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
      for j in range(i+1,len(self)):
        if L[j] < L[i]:
          if L[j] > max_below:
            max_below = L[j]
            lower_bound[i] = j
        else:
          if L[j] < min_above or min_above == -1:
            min_above = L[j]
            upper_bound[i] = j
    return (lower_bound, upper_bound)

  def avoids(self, P,lr=0):
    return not P.involved_in(self, last_require=lr)

  def avoids_set(self, B):
    for b in B:
      if b.involved_in(self):
        return False
    return True

  def involves(self, P, lr=0):
    return P.involved_in(self,last_require=lr)

  def involved_in(self, P, last_require=0):
    if not self.bounds_set:
      (self.lower_bound, self.upper_bound) = self.set_up_bounds()
      self.bounds_set = True
    L = list(self)
    n = len(L)
    p = len(P)
    if n <= 1 and n <= p:
      return True

    indices = [0]*n

    if last_require == 0:
      indices[len(self)-1] = p - 1
      while indices[len(self)-1] >= 0:
        if self.involvement_check(self.upper_bound, self.lower_bound, indices, P, len(self)-2):
          return True
        indices[len(self)-1] -= 1
      return False
    else:
      for i in range(1, last_require+1):
        indices[n-i] = p-i
      if not self.involvement_check_final(self.upper_bound, self.lower_bound, indices, P, last_require):
        return False

      return self.involvement_check(self.upper_bound, self.lower_bound, indices, P, len(self) - last_require - 1)

  def involvement_check_final(self, upper_bound, lower_bound, indices, q, last_require):
    for i in range(1,last_require):
      if not self.involvement_fits(upper_bound, lower_bound, indices, q, len(self)-i-1):
        return False
    return True

  def involvement_check(self, upper_bound, lower_bound, indices, q, next):
    if next < 0:
      return True
    # print indices,next
    indices[next] = indices[next+1]-1

    while indices[next] >= 0:
      if self.involvement_fits(upper_bound, lower_bound, indices, q, next) and self.involvement_check(upper_bound, lower_bound, indices, q, next-1):
        return True
      indices[next] -= 1
    return False

  def involvement_fits(self, upper_bound, lower_bound, indices, q, next):
    return (lower_bound[next] == -1 or q[indices[next]] > q[indices[lower_bound[next]]]) and (upper_bound[next] == -1 or q[indices[next]] < q[indices[upper_bound[next]]])


  def occurrences(self, pattern):
    total = 0
    for subseq in itertools.combinations(self, len(pattern)):
      if Permutation(subseq) == pattern:
        total += 1
    return total



  def all_intervals(self, return_patterns=False):
    blocks = [[],[]]
    for i in range(2, len(self)):
      blocks.append([])
      for j in range (0,len(self)-i+1):
        if max(self[j:j+i]) - min(self[j:j+i]) == i-1:
          blocks[i].append(j)
    if return_patterns:
      patterns = []
      for length in range(0, len(blocks)):
        for start_index in blocks[length]:
          patterns.append(Permutation(self[start_index:start_index+length]))
      return patterns
    else:
      return blocks



  def all_monotone_intervals(self):
    mi = []
    difference = 0
    c_start = 0
    c_length = 0
    for i in range(0,len(self)-1):
      if math.fabs(self[i] - self[i+1]) == 1 and (c_length == 0 or self[i] - self[i+1] == difference):
        if c_length == 0:
          c_start = i
        c_length += 1
        difference = self[i] - self[i+1]
      else:
        if c_length != 0:
          mi.append((c_start, c_start+c_length))
        c_start = 0
        c_length = 0
        difference = 0
    if c_length != 0:
      mi.append((c_start, c_start+c_length))
    return mi

  def maximal_interval(self):
    ''' finds the biggest interval, and returns (i,j) is one is found,
			where i is the size of the interval, and j is the index
			of the first entry in the interval

		returns (0,0) if no interval is found, i.e., if the permutation
			is simple'''
    for i in range(2, len(self))[::-1]:
      for j in range (0,len(self)-i+1):
        if max(self[j:j+i]) - min(self[j:j+i]) == i-1:
          return (i,j)
    return (0,0)

  def simple_location(self):
    ''' searches for an interval, and returns (i,j) if one is found,
			where i is the size of the interval, and j is the
			first index of the interval

		returns (0,0) if no interval is found, i.e., if the permutation
			is simple'''
    mins = list(self)
    maxs = list(self)
    for i in range(1,len(self)-1):
      for j in reversed(range(i,len(self))):
        mins[j] = min(mins[j-1], self[j])
        maxs[j] = max(maxs[j-1], self[j])
        if maxs[j] - mins[j] == i:
          return (i,j)
    return (0,0)

  def is_simple(self):
    ''' returns True is this permutation is simple, False otherwise'''
    (i,j) = self.simple_location()
    return i == 0

  def is_strongly_simple(self):
    return self.is_simple() and all([p.is_simple() for p in self.children()])

  def decomposition(self):
    base = Permutation(self)
    components = [Permutation([1])for i in range(0,len(base))]
    while not base.is_simple():
      assert len(base) == len(components)
      (i,j) = base.maximal_interval()
      assert i != 0
      interval = list(base[j:i+j])
      new_base = list(base[0:j])
      new_base.append(base[j])
      new_base.extend(base[i+j:len(base)])
      new_components = components[0:j]
      new_components.append(Permutation(interval))
      new_components.extend(components[i+j:len(base)])
      base = Permutation(new_base)
      components = new_components
    return (base, components)

  def inflate(self, components):
    assert len(self) == len(components), 'number of components must equal length of base'
    L = list(self)
    NL = list(self)
    current_entry = 0
    for entry in range(0, len(self)):
      index = L.index(entry)
      NL[index] = [components[index][i]+current_entry for i in range(0, len(components[index]))]
      current_entry += len(components[index])
    NL_flat = [a for sl in NL for a in sl]
    return Permutation(NL_flat)

  def right_extensions(self):
    L = []
    if len(self.insertion_locations) > 0:
      indices = self.insertion_locations
    else:
      indices = [1]*(len(self)+1)

    R = [j for j in range(len(indices)) if indices[j] == 1]
    for i in R:
      j = 0
      A = [self[j] + (1 if self[j] > i-1 else 0) for j in range(0,len(self))]
      A.append(i)
      L.append(Permutation(A))
    return L

  # def all_right_extensions(self, max_length, l, S):
  #   if l == max_length:
  #     return S
  #   else:
  #     re = self.right_extensions()
  #     for p in re:
  #       S.add(p)
  #       S = p.all_right_extensions(max_length, l+1, S)
  #   return S

  def all_extensions(self):
    S = set()
    for i in range(0, len(self)+1):
      for j in range(0, len(self)+1):
        # insert (i-0.5) after entry j (i.e., first when j=0)
        l = list(self[:])
        l.insert(j, i-0.5)
        S.add(Permutation(l))
    return PermSet(S)

  def show(self):
    if sys.platform == 'linux2':
      opencmd = 'gnome-open'
    else:
      opencmd = 'open'
    s = "\\documentclass{standalone}\n\\usepackage{tikz}\n\n\\begin{document}\n\n"
    s += self.to_tikz()
    s += "\n\n\end{document}"
    dname = random.randint(1000,9999)
    os.system('mkdir t_'+str(dname))
    with open('t_'+str(dname)+'/t.tex', 'w') as f:
      f.write(s)
    subprocess.call(['pdflatex', '-output-directory=t_'+str(dname), 't_'+str(dname)+'/t.tex'],
      stderr = subprocess.PIPE, stdout = subprocess.PIPE)
    # os.system('pdflatex -output-directory=t_'+str(dname)+' t_'+str(dname)+'/t.tex')
    subprocess.call([opencmd, 't_'+str(dname)+'/t.pdf'],
      stderr = subprocess.PIPE, stdout = subprocess.PIPE)
    time.sleep(1)
    if sys.platform != 'linux2':
      subprocess.call(['rm', '-r', 't_'+str(dname)+'/'])

  def to_tikz(self):
    s = r'\begin{tikzpicture}[scale=.3,baseline=(current bounding box.center)]';
    s += '\n\t'
    s += r'\draw[ultra thick] (1,0) -- ('+str(len(self))+',0);'
    s += '\n\t'
    s += r'\draw[ultra thick] (0,1) -- (0,'+str(len(self))+');'
    s += '\n\t'
    s += r'\foreach \x in {1,...,'+str(len(self))+'} {'
    s += '\n\t\t'
    s += r'\draw[thick] (\x,.09)--(\x,-.5);'
    s += '\n\t\t'
    s += r'\draw[thick] (.09,\x)--(-.5,\x);'
    s += '\n\t'
    s += r'}'
    for (i,e) in enumerate(self):
      s += '\n\t'
      s += r'\draw[fill=black] ('+str(i+1)+','+str(e+1)+') circle (5pt);'
    s += '\n'
    s += r'\end{tikzpicture}'
    return s

  def shrink_by_one(self):
    return PermSet([Permutation(p) for p in [self[:i]+self[i+1:] for i in range(0,len(self))]])

  def children(self):
    return self.shrink_by_one()

  def downset(self):
    return PermSet([self]).downset()

  def sum_indecomposable_sequence(self):
    S = self.downset()
    return [len([p for p in S if len(p)==i and not p.sum_decomposable()]) for i in range(1,max([len(p) for p in S])+1)]

  def sum_indec_bdd_by(self, n):
    l = [1]
    S = list(self.children())
    while len(S) > 0 and len(S[0]) > 0:
      l = [len([s for s in S if not s.sum_decomposable()])]+l
      if l[0] > n:
        return False
      S = list(PermSet(S).layer_down())
    return True

  def contains_locations(self, Q):
    locs = []
    sublocs = itertools.combinations(range(len(self)), len(Q))
    for subloc in sublocs:
      if Permutation([self[i] for i in subloc]) == Q:
        locs.append(subloc)

    return locs

  def rank_val(self, i):
    return len([j for j in range(i+1,len(self)) if self[j] < self[i]])

  def rank_encoding(self):
    return [self.rank_val(i) for i in range(len(self))]

  def num_rtlmax_ltrmin_layers(self):
    return len(self.rtlmax_ltrmin_decomposition())

  def rtlmax_ltrmin_decomposition(self):
    P = Permutation(self)
    num_layers = 0
    layers = []
    while len(P) > 0:
      num_layers += 1
      positions = sorted(list(set(P.rtlmax()+P.ltrmin())))
      layers.append(positions)
      P = Permutation([P[i] for i in range(len(P)) if i not in positions])
    return layers

  def num_inc_bonds(self):
    return len([i for i in range(len(self)-1) if self[i+1] == self[i]+1])

  def num_dec_bonds(self):
    return len([i for i in range(len(self)-1) if self[i+1] == self[i]-1])

  def num_bonds(self):
    return len([i for i in range(len(self)-1) if self[i+1] == self[i]+1 or self[i+1] == self[i]-1])

  def contract_inc_bonds(self):
    P = Permutation(self)
    while P.num_inc_bonds() > 0:
      for i in range(0,len(P)-1):
        if P[i+1] == P[i]+1:
          P = Permutation(P[:i]+P[i+1:])
          break
    return P

  def contract_dec_bonds(self):
    P = Permutation(self)
    while P.num_dec_bonds() > 0:
      for i in range(0,len(P)-1):
        if P[i+1] == P[i]-1:
          P = Permutation(P[:i]+P[i+1:])
          break
    return P

  def contract_bonds(self):
    P = Permutation(self)
    while P.num_bonds() > 0:
      for i in range(0,len(P)-1):
        if P[i+1] == P[i]+1 or P[i+1] == P[i]-1:
          P = Permutation(P[:i]+P[i+1:])
          break
    return P

  def all_syms(self):
    S = PermSet([self])
    S = S.union(PermSet([P.reverse() for P in S]))
    S = S.union(PermSet([P.complement() for P in S]))
    S = S.union(PermSet([P.inverse() for P in S]))
    return S

  def is_representative(self):
    return self == sorted(self.all_syms())[0]



class PermClass(list):

  # def __init__(self, n = 8):
    # list.__init__(self, [PermSet(permutation.Permutation.listall(i)) for i in range(n + 1)])
    # self.avoids = []
    # self.length = n

  # def __len__(self):
  #   return self.length

  @staticmethod
  def class_from_test(test, l=8, has_all_syms=False):
    C = [PermSet([Permutation([])])]
    for cur_length in range(1,l+1):
      this_len = PermSet([])
      if len(C[cur_length-1]) == 0:
        return PermClass(C)
      to_check = PermSet(set.union(*[P.all_extensions() for P in C[cur_length-1]]))
      to_check = [P for P in to_check if PermSet(P.children()).issubset(C[cur_length-1])]
      while len(to_check) > 0:
        P = to_check.pop()
        if has_all_syms:
          syms = PermSet([
              P,
              P.reverse(),
              P.complement(),
              P.reverse().complement(),
              P.inverse(),
              P.reverse().inverse(),
              P.complement().inverse(),
              P.reverse().complement().inverse()
            ])
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

    assert max_length < len(self), 'class not big enough to check that far'

    if search_mode:
      max_length = len(self)-1

    # Find the first length at which perms are missing.
    not_all_perms = [i for i in range(len(self)) if i >= 1 and len(self[i]) != factorial(i)]

    # If no perms are missing, we have all perms, so return empty basis.
    if len(not_all_perms) == 0:
      return PermSet([])

    # Add missing perms of minimum length to basis.
    start_length = min(not_all_perms)
    basis = PermSet(Permutation.listall(start_length)).difference(self[start_length])

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
  #   max_length = min(max_length, len(self)-1)
  #   B = PermSet()
  #   B.update(self.check_tree_basis(max_length, Permutation([1,2]), PermSet()))
  #   B.update(self.check_tree_basis(max_length, Permutation([2,1]), PermSet()))
  #   return B.minimal_elements()

  # def check_tree_basis(self, max_length, R, S):
  #   if R not in self[len(R)]:
  #     for s in S:
  #       if s.involved_in(R):
  #         return S
  #     S.add(R)
  #     return S
  #   else:
  #     if len(R) == max_length:
  #       return S
  #     re = R.right_extensions()
  #     for p in re:
  #       S = self.check_tree_basis(max_length, p, S)
  #     return S

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

class PermSet(set):
  ''' Provides functions for dealing with sets of Permutation objects '''

  def __repr__(self):
    # if len(self) > 10:
    return 'Set of %d permutations' % len(self)
    # else:
      # return set.__repr__(self)

  @staticmethod
  def all(n):
    ''' builds the set of all permutations of length n'''
    return PermSet(Permutation.listall(n))

  def show_all(self):
    return set.__repr__(self)

  def minimal_elements(self):
    B = list(self)
    B = sorted(B, key=len)
    C = B[:]
    n = len(B)
    for (i,b) in enumerate(B):
      # if i % 1 == 0:
        # print i,'/',n
      if b not in C:
        continue
      for j in range(i+1,n):
        if B[j] not in C:
          continue
        if b.involved_in(B[j]):
          C.remove(B[j])
    return PermSet(C)

  def all_syms(self):
    sym_set = [frozenset(self)]
    sym_set.append(frozenset([i.reverse() for i in self]))
    sym_set.append(frozenset([i.complement() for i in self]))
    sym_set.append(frozenset([i.reverse().complement() for i in self]))
    sym_set.extend([frozenset([k.inverse() for k in L]) for L in sym_set])
    return frozenset(sym_set)

  def layer_down(self):
    S = PermSet()
    i = 1
    n = len(self)
    for P in self:
      # if i % 10000 == 0:
        # print('\t',i,'of',n,'. Now with',len(S),'.')
      S.update(P.shrink_by_one())
      i += 1
    return S

  def downset(self, return_class=False):
    bottom_edge = PermSet()
    bottom_edge.update(self)

    done = PermSet(bottom_edge)
    while len(bottom_edge) > 0:
      oldsize = len(done)
      next_layer = bottom_edge.layer_down()
      done.update(next_layer)
      del bottom_edge
      bottom_edge = next_layer
      del next_layer
      newsize = len(done)
      # print '\t\tDownset currently has',newsize,'permutations, added',(newsize-oldsize),'in the last run.'
    if not return_class:
      return done
    cl = [PermSet([])]
    max_length = max([len(P) for P in done])
    for i in range(1,max_length+1):
      cl.append(PermSet([P for P in done if len(P) == i]))
    return permclass.PermClass(cl)


  def total_statistic(self, statistic):
    return sum([statistic(p) for p in self])

  def threepats(self):
    patnums = {'123' : 0, '132' : 0, '213' : 0,
               '231' : 0, '312' : 0, '321' : 0}
    L = list(self)
    for p in L:
      n = len(p)
      for i in range(n-2):
        for j in range(i+1,n-1):
          for k in range(j+1,n):
            std = Permutation.standardize([p[i], p[j], p[k]])
            patnums[''.join([str(x + 1) for x in std])] += 1
    return patnums

  def fourpats(self):
    patnums = {'1234' : 0, '1243' : 0, '1324' : 0,
               '1342' : 0, '1423' : 0, '1432' : 0,
               '2134' : 0, '2143' : 0, '2314' : 0,
               '2341' : 0, '2413' : 0, '2431' : 0,
               '3124' : 0, '3142' : 0, '3214' : 0,
               '3241' : 0, '3412' : 0, '3421' : 0,
               '4123' : 0, '4132' : 0, '4213' : 0,
               '4231' : 0, '4312' : 0, '4321' : 0 }
    L = list(self)
    for p in L:
      n = len(p)
      for i in range(n-3):
        for j in range(i+1,n-2):
          for k in range(j+1,n-1):
            for m in range(k+1,n):
              std = Permutation.standardize([p[i], p[j], p[k], p[m]])
              patnums[''.join([str(x + 1) for x in std])] += 1
    return patnums



class AvClass(PermClass):

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
          print('\t\t\t\tRight Extensions:',k,'/',outof,'\t( length',n,')')
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
          # print '\tchanging IL from ',Q.insertion_locations,'to',(insertion_locations[:Q[-1]+1]+  insertion_locations[Q[-1]:])
          Q.insertion_locations = insertion_locations[:Q[-1]+1] + insertion_locations[Q[-1]:]
          self[n].add(Q)

  def extend_to_length(self, l):
    for i in range(self.length+1, l+1):
      self.append(PermSet())
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
          # print '\tchanging IL from ',Q.insertion_locations,'to',(insertion_locations[:Q[-1]+1]+  insertion_locations[Q[-1]:])
          Q.insertion_locations = insertion_locations[:Q[-1]+1] + insertion_locations[Q[-1]:]
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
