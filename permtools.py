from permutation import *
from avclass import *
from math import factorial
import itertools

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
              if Permutation(a) == Permutation(P[0:len(a)]) and Permutation(b) == Permutation(P[len(a):n]):
                A.append(list(P))
            elif len(a) + len(b) - 1 == n:
              if Permutation(a) == Permutation(P[0:len(a)]) and Permutation(b) == Permutation(P[len(a)-1:n]):
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

# def generate_permutations_in_geometric_grid_class(M, max_length):
#   ''' Need to check that the cell graph of M is a forest. '''
#   L = [PermSet() for i in range(0, max_length+1)]
#   L[1].add(Permutation([1]))

#   alphabet_size = sum([(1 if M[i][j] != 0 else 0) for i in range(0,len(M)) for j in range(0,len(M[i]))])
#   alphabet = [(i,j) for i in range(0, len(M)) for j in range(0,len(M[i])) if M[i][j] != 0]
#   commuting_pairs = []
  
#   for (i,l) in enumerate(alphabet):
# 	  for j in range(i,len(alphabet)):
# 		  if alphabet[i][0] != alphabet[j][0] and alphabet[i][1] != alphabet[j][1]:
# 			  commuting_pairs.append((i,j))
	
#   alphabet_indices = [i for i in range(0,len(alphabet))]		  
  
#   for length in (2,max_length+1):
#     ''' Try all words of length 'length' with alphabet 
#     equal to the cell alphabet of M.'''
#     distance = 1/(length+1)
#     all_words = itertools.product(alphabet_indices,repeat=length)
    
#     ''' todo: MAKE BASE POINTS '''
    
#     for word in all_words:
#       if is_valid_word(w, commuting_pairs):
# 		  points = []
		
