from .permutation import *
from .permset import *
from .permclass import *
import collections, itertools, operator
from math import factorial

class GeometricGridClass(PermClass):
  
  ## Todo: Automatic row/col signs if possible, otherwise use the x2 trick

  def __init__(self, M, col, row, max_length=8, generate=True): 
    M = M[::-1]
    M = map(list, zip(*M))
    self.M = M
    self.col = col
    self.row = row
    list.__init__(self, [PermSet() for i in range(0, max_length+1)])

    self.alphabet_size = sum([(1 if self.M[i][j] != 0 else 0) for i in range(0,len(self.M)) for j in range(0,len(self.M[i]))])
    self.alphabet = [(i,j) for i in range(0, len(self.M)) for j in range(0,len(self.M[i])) if self.M[i][j] != 0]

    self.commuting_pairs = []

    self.dots = []
    for (index, coords) in enumerate(self.alphabet):
      if self.M[coords[0]][coords[1]] == 2:
        self.dots.append(index)
    
    for (i,l) in enumerate(self.alphabet):
      for j in range(i,len(self.alphabet)):
        if self.alphabet[i][0] != self.alphabet[j][0] and self.alphabet[i][1] != self.alphabet[j][1]:
          self.commuting_pairs.append((i,j))

    self.alphabet_indices = [i for i in range(0,len(self.alphabet))]     

    if generate:
      self.generate_perms(max_length)

    self.length = max_length

  def find_word_for_perm(self, P):
    l = len(P)

    M = self.M
    column_signs = self.col
    row_signs = self.row
    
    all_words = itertools.product(self.alphabet_indices,repeat=l)
      
    for word in all_words:
      perm = self.dig_word_to_perm(word)
      if perm == P:
        return dig_to_num(word)
    
  def generate_perms(self, max_length):
    M = self.M
    column_signs = self.col
    row_signs = self.row

    list.__init__(self, [PermSet() for i in range(0, max_length+1)])
    self[1].add(Permutation([1])) 
    
    for length in range(2,max_length+1):
      ''' Try all words of length 'length' with alphabet 
      equal to the cell alphabet of M.'''
      all_words = itertools.product(self.alphabet_indices,repeat=length)
      
      for word in all_words:
        P = self.dig_word_to_perm(word)
        if P:
          self[length].add(P)

  def dig_word_to_perm(self, word, ignore_bad=False):
    if not ignore_bad:
      bad_word = False
      for index in self.dots:
        if word.count(index) > 1:
          return False
      if not self.is_valid_word(word):
        return False
    points = []
    height = len(word)+2
    for (position, letter) in enumerate(word):
      if self.col[self.alphabet[letter][0]] == 1:
        x_point = (self.alphabet[letter][0]+1)*height - position
      else:
        x_point = (self.alphabet[letter][0])*height + position
      if self.row[self.alphabet[letter][1]] == 1:
        y_point = (self.alphabet[letter][1]+1)*height - position
      else:
        y_point = (self.alphabet[letter][1])*height + position
      points.append((x_point,y_point))
    points.sort(key=operator.itemgetter(0))  
    widths = [p[0] for p in points] 
    heights = [p[1] for p in points] 
    new_points = Permutation.standardize(heights)
    return Permutation(new_points)

  def alpha_word_to_perm(self, word):
    w = []
    for i in range(0, len(word)):
      n = ord(word[i])-97
      if n < 0 or n >= self.alphabet_size:
        return False
      w.append(n)
    return self.dig_word_to_perm(w, ignore_bad=True)

  def is_valid_word(self, word):
    for i in range(0,len(word)-1):
      if (word[i+1], word[i]) in self.commuting_pairs:
        return False
    return True

  def find_inflations_avoiders(self, basis):
    max_basis_length = max([len(B) for B in basis])
    allowed_inflations = [Permutation([1])]
    allowed_inflations.extend(list(set([P for sublist in [B.all_intervals(return_patterns=True) for B in basis] for P in sublist])))
    avoidence_inflations = []
    print allowed_inflations
    for length in range(2,max_basis_length):
      all_words = itertools.product(self.alphabet_indices,repeat=length)
      for word in all_words:
        P = self.dig_word_to_perm(word)
        if not P:
          continue
        combos = itertools.product(range(0,len(allowed_inflations)),repeat=len(P))
        for combo in combos:
          components = [allowed_inflations[i] for i in combo]
          Q = P.inflate(components)
          if Q in basis:
            what_to_avoid = [chr(e+97)+'_'+str(allowed_inflations.index(components[i])) for (i,e) in enumerate(word)]
            print what_to_avoid,'=',Q
            if what_to_avoid not in avoidence_inflations:
              avoidence_inflations.append(what_to_avoid)
    return avoidence_inflations

  def inflation_rules(self, basis):
    avoidence_inflations = self.find_inflations_avoiders(basis)
    rules = []
    letters = set()
    for ai in avoidence_inflations:
      rule = 'SS, '
      rule += (', SS, '.join(ai))
      rule += ', SS'
      rules.append('C(P('+rule+'))')
      letters = letters.union(ai)
    big_rule = 'I(' + (','.join(rules)) + ')'
    letters = list(letters)
    letters.sort()
    return (letters, big_rule)

def dig_to_num(w):
  return ''.join([chr(a+97) for a in w])




