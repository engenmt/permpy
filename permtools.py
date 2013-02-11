from permutation import *
from avclass import *
from permset import *
from math import factorial
import itertools
import operator

def juxtaposition_basis(S):
  if len(S) == 0 or len(S) == 1:
    return S

  while len(S) >= 2:
    A = []
    max_length = max([len(p) for p in S[0]]) + max([len(p) for p in S[1]])
    for n in range(2, max_length+1):
      for i in range(0, factorial(n)):
        P = Permutation(i,n)
        for a in S[0]:
          for b in S[1]:
            if len(a) + len(b) == n:
              if (Permutation(a) == Permutation(P[0:len(a)]) and 
                          Permutation(b) == Permutation(P[len(a):n])):
                A.append(list(P))
            elif len(a) + len(b) - 1 == n:
              if (Permutation(a) == Permutation(P[0:len(a)]) and 
                          Permutation(b) == Permutation(P[len(a)-1:n])):
                A.append(list(P))

      
    if len(S) > 2:
      B = [A]
      B.extend(S[2:])
      return juxtaposition_basis(B)
    else:
      return reduce_basis(A)
      
def reduce_basis(B):
  ''' takes a list of lists '''
  B = sorted(B, key=len)
  C = B[:]
  n = len(B)
  for (i,b) in enumerate(B):
    if b not in C:
      continue
    for j in range(i+1,n):
      if B[j] not in C:
        continue
      if Permutation(b).involved_in(Permutation(B[j])):
        C.remove(B[j])
  return C

def right_extensions(P):
  L = []
  for i in range(0,len(P)+1):
    A = list(P[:])
    A = [A[j] + (1 if A[j] > i-1 else 0) for j in range(0,len(P))]
    A.append(i)
    L.append(Permutation(A))
  return L

def generate_permutations_in_geometric_grid_class(M, max_length, column_signs, row_signs):
  ''' Need to check that the cell graph of M is a forest. '''
  M = M[::-1]
  M = map(list, zip(*M))
  
  L = [PermSet() for i in range(0, max_length+1)]
  L[1].add(Permutation([1]))

  alphabet_size = sum([(1 if M[i][j] != 0 else 0) for i in range(0,len(M)) for j in range(0,len(M[i]))])
  alphabet = [(i,j) for i in range(0, len(M)) for j in range(0,len(M[i])) if M[i][j] != 0]

  commuting_pairs = []
  
  for (i,l) in enumerate(alphabet):
	  for j in range(i,len(alphabet)):
		  if alphabet[i][0] != alphabet[j][0] and alphabet[i][1] != alphabet[j][1]:
			  commuting_pairs.append((i,j))

  alphabet_indices = [i for i in range(0,len(alphabet))]		  
  
  for length in range(2,max_length+1):
    ''' Try all words of length 'length' with alphabet 
    equal to the cell alphabet of M.'''
    height = length+2
    distance = 1
    all_words = itertools.product(alphabet_indices,repeat=length)
    
    for word in all_words:
      if not is_valid_word(word, commuting_pairs):
		    continue
      points = []
      for (position, letter) in enumerate(word):
        if column_signs[alphabet[letter][0]] == 1:
          x_point = (alphabet[letter][0]+1)*height - (position*distance)
        else:
          x_point = (alphabet[letter][0])*height + (position*distance)
        if row_signs[alphabet[letter][1]] == 1:
          y_point = (alphabet[letter][1]+1)*height - (position*distance)
        else:
          y_point = (alphabet[letter][1])*height + (position*distance)
        points.append((x_point,y_point))
      points.sort(key=operator.itemgetter(0))  
      widths = [p[0] for p in points] 
      heights = [p[1] for p in points] 
      new_points = Permutation.standardize(heights)
      P = Permutation(new_points)
      L[length].add(P)
  
  return L

def is_valid_word(word, commuting_pairs):
  for i in range(0,len(word)-1):
    if (word[i+1], word[i]) in commuting_pairs:
      return False
  return True

def find_basis_given_class(C, max_length):
  B = []
  B.extend(check_tree_basis(C, max_length, Permutation([1,2]), set()))
  B.extend(check_tree_basis(C, max_length, Permutation([2,1]), set()))
  return reduce_basis(B)

def check_tree_basis(C, max_length, R, S):
  if R not in C[len(R)]:
    for s in S:
      if s.involved_in(R):
        return S
    S.add(R)
    return S
  else:
    if len(R) == max_length:
      return S
    re = right_extensions(R)
    for p in re:
      S = check_tree_basis(C, max_length, p, S)
    return S

def all_right_extensions(P, max_length, l, S):
  if l == max_length:
    return S
  else:
    re = right_extensions(P)
    for p in re:
      S.add(p)
      S = all_right_extensions(p, max_length, l+1, S)
  return S

