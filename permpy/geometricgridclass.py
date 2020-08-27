import collections
import itertools
import logging
import operator
from math import factorial

from .permutation import Permutation
from .permset import PermSet
from .permclass import PermClass

logging.basicConfig(level=10)

class BadMatrixException(Exception):
	pass

class GeometricGridClass(PermClass):
	
	## Todo: Automatic row/col signs if possible, otherwise use the x2 trick

	def __init__(self, M, col=None, row=None, max_length=8, generate=True): 
		"""M goes from top to bottom, then left to right.
		"""
		
		M = M[::-1]
		M = list(map(list, zip(*M)))
		self.M = M
		# self.M goes left to right, bottom to top.
		
		self.col = col
		self.row = row
		PermClass.__init__(self, [PermSet() for _ in range(max_length+1)])

		if self.col is None or self.row is None:
			self.compute_signs()
		
		self.alphabet = [
			(col_idx, row_idx)
			for col_idx, col in enumerate(self.M) 
			for row_idx, val in enumerate(col) 
			if val
		]
		self.alphabet_size = len(self.alphabet)

		self.commuting_pairs = []

		self.dots = [
			idx 
			for idx, (x,y) in enumerate(self.alphabet)
			if self.M[x][y] == 2
		]
		
		for i, l_i in enumerate(self.alphabet):
			for j, l_j in enumerate(self.alphabet[i+1:], i+1):
				if l_i[0] != l_j[0] and l_i[1] != l_j[1]:
					self.commuting_pairs.append((i,j))

		self.alphabet_indices = list(range(self.alphabet_size))

		if generate:
			self.generate_perms(max_length)

	def find_word_for_perm(self, p):
		
		all_words = itertools.product(self.alphabet_indices, repeat=len(p))
			
		for word in all_words:
			perm = self.dig_word_to_perm(word)
			if perm == p:
				return dig_to_num(word)

	def compute_signs(self):
		"""
		Examples:
			>>> M = [[0,1,-1],[1,0,1]]
			>>> G = GeometricGridClass(M, generate=False)
			>>> (G.col, G.row) == ([1,-1,1], [1,-1])
			True
		"""
		col_signs = [0 for _ in range(len(self.M))]
		row_signs = [0 for _ in range(len(self.M[0]))]
	
		unsigned_vals = {0,2} # These represent empty cells and point-cells respectively
	
		for col_idx, col in enumerate(self.M):
			if all(val in unsigned_vals for val in col):
				# This column has no entries that need a sign, so we set it arbitrarily.
				col_signs[col_idx] = 1
	
		for row_idx in range(len(row_signs)):
			if all(col[row_idx] in unsigned_vals for col in self.M):
				# This row has no entries that need a sign, so we set it arbitrarily.
				row_signs[row_idx] = 1
	
		while not (all(col_signs) and all(row_signs)):
			# This loop will continue until all col_signs and row_signs are non-zero
			# It will make at most one "arbitrary" column assignment per loop.
			logging.debug(f"Starting loop again.")
			logging.debug(f"\tself.M = {self.M}")
			logging.debug(f"\tcol_signs = {col_signs}")
			logging.debug(f"\trow_signs = {row_signs}")
			choice_made = False
		
			for col_idx, col in enumerate(self.M):
				if col_signs[col_idx]:
					# This column has a sign already.
					continue
			
				for row_idx, (row_sign, entry) in enumerate(zip(row_signs, col)):
					if entry in unsigned_vals:
						continue
				
					if not row_sign:
						continue
				
					# If we're here, then:
					# - there's a signed entry in entry = self.M[col_idx][row_idx]
					# - row_sign = row_signs[row_idx] is defined.
					col_signs[col_idx] = entry * row_sign
					break
				else:
					# If we're here, then col_signs[col_idx] is undefined.
					if not choice_made:
						# Make our arbitrary choice.
						col_signs[col_idx] = 1
						choice_made = True
			
				if col_signs[col_idx]:
					for row_idx, entry in enumerate(col):
						if entry in unsigned_vals:
							continue
						if row_signs[row_idx]:
							assert row_signs[row_idx] == entry * col_signs[col_idx], \
								f"The signs are all messed up now: {self.M}, {col_signs}, {row_signs} ({col_idx}, {row_idx})"
						else:
							row_signs[row_idx] = entry * col_signs[col_idx]

		# CHECKING
		for col_idx, (col, col_sign) in enumerate(zip(self.M, col_signs)):
			for row_idx, (entry, row_sign) in enumerate(zip(col, row_signs)):
				if entry not in unsigned_vals: 
					if entry != col_sign * row_sign:
						raise BadMatrixException(f"Signs can't be computed for this matrix: {self.M}")
	
		self.col = col_signs
		self.row = row_signs

		
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

#   def alpha_word_to_perm(self, word):
#     w = []
#     for i in range(0, len(word)):
#       n = ord(word[i])-97
#       if n < 0 or n >= self.alphabet_size:
#         return False
#       w.append(n)
#     return self.dig_word_to_perm(w, ignore_bad=True)

	def is_valid_word(self, word):
		for i in range(0,len(word)-1):
			if (word[i+1], word[i]) in self.commuting_pairs:
				return False
		return True

#   def find_inflations_avoiders(self, basis):
#     max_basis_length = max([len(B) for B in basis])
#     allowed_inflations = [Permutation([1])]
#     allowed_inflations.extend(list(set([P for sublist in [B.all_intervals(return_patterns=True) for B in basis] for P in sublist])))
#     avoidence_inflations = []
#     print(allowed_inflations)
#     for length in range(2,max_basis_length):
#       all_words = itertools.product(self.alphabet_indices,repeat=length)
#       for word in all_words:
#         P = self.dig_word_to_perm(word)
#         if not P:
#           continue
#         combos = itertools.product(range(0,len(allowed_inflations)),repeat=len(P))
#         for combo in combos:
#           components = [allowed_inflations[i] for i in combo]
#           Q = P.inflate(components)
#           if Q in basis:
#             what_to_avoid = [chr(e+97)+'_'+str(allowed_inflations.index(components[i])) for (i,e) in enumerate(word)]
#             print(what_to_avoid,'=',Q)
#             if what_to_avoid not in avoidence_inflations:
#               avoidence_inflations.append(what_to_avoid)
#     return avoidence_inflations

#   def inflation_rules(self, basis):
#     avoidence_inflations = self.find_inflations_avoiders(basis)
#     rules = []
#     letters = set()
#     for ai in avoidence_inflations:
#       rule = 'SS, '
#       rule += (', SS, '.join(ai))
#       rule += ', SS'
#       rules.append('C(P('+rule+'))')
#       letters = letters.union(ai)
#     big_rule = 'I(' + (','.join(rules)) + ')'
#     letters = list(letters)
#     letters.sort()
#     return (letters, big_rule)

# def dig_to_num(w):
#   return ''.join([chr(a+97) for a in w])

if __name__ == "__main__":
	G = GeometricGridClass([[1,1],[1,-1]], generate=False)




