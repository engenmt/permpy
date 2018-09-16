from __future__ import division, print_function

from collections import Counter, defaultdict

import sys
import os
import subprocess
import time
import math
import random
import fractions
import itertools

from permpy.permstats import PermutationStatsMixin
from permpy.permmisc import PermutationMiscMixin

from scipy.misc import comb

try:
	import matplotlib.pyplot as plt
	mpl_imported = True
except ImportError:
	mpl_imported = False

__author__ = 'Cheyne Homberger, Jay Pantone'

"""
Todo:
	* Permutation.random_avoider
"""

def _is_iter(obj):
	try:
		iter(obj)
		return True
	except TypeError:
		return False

# a class for creating permutation objects
class Permutation(tuple, 
				  PermutationStatsMixin, 
				  PermutationMiscMixin):
	"""Class for representing permutations as immutable 0-indexed tuples.
	"""

	# static class variable, controls permutation representation
	_REPR = 'oneline'

	# default to displaying permutations as 1-based
	_BASE = 1

	lower_bound = []
	upper_bound = []
	bounds_set = False
	insertion_values = [] # When creating a class, this keeps track of what new values are allowed.

	@classmethod
	def monotone_increasing(cls, n):
		"""Return the monotone increasing permutation of length n.
		
		Examples:
			>>> Permutation.monotone_increasing(5)
			1 2 3 4 5

		ME: Done!
		"""
		return cls(range(n), clean=True)

	@classmethod
	def monotone_decreasing(cls, n):
		"""Return the monotone decreasing permutation of length n.

		Examples:
			>>> Permutation.monotone_decreasing(5)
			5 4 3 2 1
		
		ME: Done!
		"""
		return cls(range(n)[::-1], clean=True)

	@classmethod
	def identity(cls, n):
		"""Return the identity permutation of length n. 
		Same as monotone_increasing.

		ME: Done!
		"""
		return cls.monotone_increasing(n)

	@classmethod
	def random(cls, n):
		"""Return a random permutation of length n.
	
		Examples:
			>>> len( Permutation.random(10) ) == 10
			True

		ME: Done!
		"""
		L = list(range(n))
		random.shuffle(L)
		return cls(L, clean=True)

	@classmethod
	def random_avoider(cls, n, B, simple=False, involution=False, verbose=-1):
		"""Generate a (uniformly) random permutation which avoids the patterns
		contained in `B`.

		Args:
			n (int): length of permutation to generate
			B (iterable): Iterable of permutation-like objects to avoid.
			simple (Boolean, optional): Whether the returned Permutation should be simple.
				Defaults to False.
			involution (Boolean, optional): Whether the returned Permutation should be an involution.
				Defaults to False.
			verbose (int, optional): Level of verbosity (-1 for no verbosity)
				Doubling the integer doubles the number of messages printed.
				Defaults to -1.

		Returns:
			p (Permutation): A permutation avoiding all the patterns in `B`

		>>> p = Permutation.random_avoider(8, [123])
		>>> p.involves(123)
		False

		ME: TODO! Ideally, we should use MCMC for this.
		"""

		i = 1
		p = cls.random(n)
		while (involution and not p.is_involution()) \
				or (simple and not p.is_simple()) \
				or not p.avoids(B = B):
			i += 1
			p = cls.random(n)
			if verbose != -1 and i % verbose == 0:
				print(f"Tested: {str(i)} permutations.")
		return p

	@classmethod
	def genall(cls, n):
		"""Generate all permutations of length n.

		ME: Done!
		"""
		for pi in itertools.permutations(range(n)):
			yield Permutation(pi,clean=True)

	@classmethod
	def listall(cls, n):
		"""Return a list of all permutations of length `n`.

		ME: Done!
		"""
		return list(cls.genall(n))

	@classmethod
	def standardize(cls, L):
		"""Standardize the list `L` of distinct elements by mapping them to the 
		set {0,1, ..., len(L)} by an order-preserving bijection.
		
		See the following for some interesting discussion on this:
		https://stackoverflow.com/questions/17767646/relative-order-of-elements-in-list
		
		ME: Done!
		"""
		assert len(set(L)) == len(L), "Ensure elements are distinct!"
		ordered = sorted(L)
		return tuple(ordered.index(x) for x in L)

	@classmethod
	def change_repr(cls, representation=None):
		"""Toggle globally between cycle notation or one-line notation. 
		Note that internal representation is still one-line.

		ME: Done!
		"""
		L = ['oneline', 'cycle', 'both']
		if representation in L:
			cls._REPR = representation
		else:
			k = int(input('1 for oneline, 2 for cycle, 3 for both\n'))
			cls._REPR = L[k-1]

	@classmethod
	def ind2perm(cls, k, n):
		"""De-index the permutation by a bijection from the set S_n to [n!].
		See also the `Permutation.perm2ind` method.

		Args:
			k (int): An integer between 0 and n! - 1, to be mapped to S_n.
			n (int): Length of the permutation.

		Returns:
			Permutation of index k of length n.
		
		Examples:
			>>> Permutation.ind2perm(12,8).perm2ind()
			12

		ME: Todo!
		"""
		assert isinstance(k, int), f"Got confused: Permutation.ind2perm(k = {k}, n = {n}) was called."
		result = list(range(n))
		for i in range(n, 0, -1):
			j = k % i
			result[i-1], result[j] = result[j], result[i-1]
			k //= i
		p = cls(result, clean=True)
		return p

	# overloaded built in functions:
	def __new__(cls, p, n = None, clean=False):
		"""Create a new permutation object. Supports a variety of creation
		methods.

		Notes:
			If `p` is an iterable containing distinct elements, they will be
				standardized to produce a permutation of length `len(p)`.
			If `n` is given, and `p` is an integer, use `ind2perm` to create a
			permutation.
			If `p` is an integer with fewer than 10 digits, try to create a
			permutation from the digits.

		Args:
			p (Permutation-like object): object to be coerced into a Permutation.
				Accepts Permutation, tuple, str, int, or iterable.
			n (int, optional): If this is provided, the method appeals to Permutation.ind2perm(p, n).
			clean (Boolean, optional): Whether the input is known to be an 
				iterable containing each element from range(len(p)) exactly once.

		Returns
		-------
		Permutation instance

		>>> Permutation('3 5 1 2 4') == Permutation([3, 5, 1, 2, 4])
		True
		>>> Permutation(5, 12) == Permutation.ind2perm(5, 12)
		True
		>>> Permutation([215, -99, 30, 12.1351, 0]) == Permutation(51432)
		True

		ME: Done!
		"""
		if clean:
			return tuple.__new__(cls, p)
		elif isinstance(p, Permutation):
			return p
		elif n is not None:
			return Permutation.ind2perm(p, n)
		else:
			if isinstance(p, str):
				if ' ' in p:
					p = p.split()
				entries = [int(digit) for digit in p]
			elif isinstance(p, int):
				entries = [int(digit) for digit in str(p)]
			else:
				entries = p
			entries = Permutation.standardize(entries)
			return tuple.__new__(cls, entries)

	def __init__(self,p,n=None,clean=False):
		"""Initialize the Permutation. 

		Notes:
			self.insertion_values is used for creating classes later.

			If only the "bad spots" are noted, then initializing perms is faster!

		ME: Done!
		"""
		self.insertion_values = list(range(len(self)+1))

	def __call__(self,i):
		"""Allow the permutation to be called as a function. 

		Notes:
			Recall that permutations are zero-based internally.
		
		Examples:
			>>> Permutation(4132)(2)
			2

		ME: Done!
		"""
		return self[i]

	def __contains__(self, other):
		"""Return True if `self` contains `other`.

		Examples:
			>>> Permutation(21).__contains__(Permutation(1))
			True
			>>> Permutation(132) in Permutation(4132)
			True
			>>> Permutation(231) in Permutation(1234)
			False
		"""
		return other.involved_in(self)

	def oneline(self):
		"""Return the one-line notation representation of the permutation (as a
		sequence of integers 1 through n).

		ME: Done!
		"""
		base = Permutation._BASE
		s = " ".join( str(entry + base) for entry in self )
		return s

	def __repr__(self):
		"""Return a string representation of the permutation depending on the 
		chosen representation (`Permutation._REPR`).

		ME: Done!
		"""
		if Permutation._REPR == 'oneline':
			return self.oneline()
		elif Permutation._REPR == 'cycle':
			return self.cycles()
		else:
			return '\n'.join([self.oneline(), self.cycles()])

	# __hash__, __eq__, __ne__ inherited from tuple class

	def __mul__(self, other):
		"""Return the functional composition of the two permutations.

		ME: Done!
		"""
		assert len(other) == len(self)
		L = (self[val] for val in other)
		return Permutation(L,clean=True)

	def __add__(self, other):
		"""Return the direct sum of the two permutations.

		Examples:
			>>> p = Permutation.monotone_increasing(10)
			>>> p + p == Permutation.monotone_increasing(20)
			True

		ME: Done!
		"""
		n = len(self)
		return Permutation(list(self) + [i+n for i in other],clean=True)

	def direct_sum(self, other):
		"""Return the direct sum of the two permutations.
		
		Examples:
			>>> Permutation(312).direct_sum(Permutation(1234))
			3 1 2 4 5 6 7

		ME: Done!
		"""
		return self + other

	def __sub__(self, other):
		"""Return the skew sum of the two permutations.

		Examples:
			>>> p = Permutation.monotone_decreasing(10)
			>>> p - p == Permutation.monotone_decreasing(20)
			True

		ME: Done!
		"""
		m = len(other)
		return Permutation([i+m for i in self] + list(other),clean=True)

	def skew_sum(self, other):
		"""Return the skew sum of the two permutations.
		
		Examples:
			>>> Permutation(312).skew_sum(Permutation(1234))
			7 5 6 1 2 3 4

		ME: Done!
		"""
		return self - other

	def __pow__(self, power):
		"""Return the permutation raised to a power.
		
		Examples:
			>>> p = Permutation.random(10)
			>>> p**p.order() == Permutation.monotone_increasing(10)
			True

		ME: Done!
		"""
		assert isinstance(power,int), 'Power must be an integer!'
		if power < 0:
			p = self.inverse()
			return p.__pow__(-power)
		elif power == 0:
			return Permutation.identity(len(self))
		else:
			ans = self
			for _ in range(power - 1):
				ans *= self
			return ans

	def perm2ind(self):
		"""De-index the permutation, by mapping it to an integer between 0 and
		len(self)! - 1. See also `Permutation.ind2perm`.
		
		Examples:
			>>> p = Permutation(41523)
			>>> Permutation.ind2perm(p.perm2ind(), len(p)) == p
			True

		ME: TODO!
		"""
		q = list(self)
		n = len(self)
		result = 0
		multiplier = 1
		for i in range(0,n)[::-1]:
			result += q[i]*multiplier
			multiplier *= i+1
			j = q.index(i)
			q[i], q[j] = q[j], q[i]
		return result

	def delete(self, indices=None, values=None):
		"""Return the permutation which results from deleting the indices or 
		values given.

		Notes:
			Recall that both indices and values are zero-indexed.

		Examples:
			>>> Permutation(35214).delete(indices=2)
			2 4 1 3
			>>> Permutation(35214).delete(indices=[2,4])
			2 3 1
			>>> Permutation(35214).delete(values=[4])
			3 2 1 4
			>>> Permutation(35214).delete(indices=[2]) == Permutation(35214).delete(values=1)
			True

		ME: Done!
		"""
		if indices is not None:
			try:
				indices = list(indices)
				p = [val for idx, val in enumerate(self) if idx not in indices]
				return Permutation(p)
			except TypeError: 
				val = self[indices]
				p = [old_val if old_val < val else old_val-1 for old_val in self if old_val != val]
				return Permutation(p,clean=True)
		elif values is not None:
			try:
				values = list(values)
				p = [val for idx, val in enumerate(self) if val not in values]
				return Permutation(p)
			except TypeError: 
				val = values
				p = [old_val if old_val < val else old_val-1 for old_val in self if old_val != val]
				return Permutation(p,clean=True)
		else:
			raise Exception(f"Permutation({self}).delete() was called, which doesn't make sense.")

	def insert(self,idx,val):
		"""Return the permutation resulting from inserting an entry with value
		just below `val` into the position just before the entry at position
		`idx`. 

		Notes:
			Recall that both indices and values are zero-indexed.
		
		Examples:
			>>> Permutation(2413).insert(2, 1)
			3 5 2 1 4
			>>> p = Permutation.random(10)
			>>> p == p.insert(4, 7).delete(indices = 4)
			True

		ME: Done!
		"""
		p = [old_val if old_val < val else old_val+1 for old_val in self]
		p.insert(idx, int(math.ceil(val)))
		return Permutation(p,clean=True)

	def complement(self):
		"""Return the complement of the permutation. That is, the permutation
		obtained by subtracting each of the entries from `len(self)`.
		
		Examples:
			>>> Permutation(2314).complement() == Permutation(3241)
			True
			>>> p = Permutation.random(10)
			>>> p == p.complement().complement()
			True

		ME: Done!
		"""
		n = len(self) - 1
		return Permutation([n-i for i in self],clean=True)

	def reverse(self):
		"""Return the reverse of the permutation.
		
		Examples:
			>>> Permutation(2314).reverse() == Permutation(4132)
			True

		ME: Done!
		"""
		return Permutation(self[::-1],clean=True)

	def inverse(self):
		"""Return the group-theoretic or functional inverse of self.

		ME: Done!
		"""
		q = list(self[:])
		for idx, val in enumerate(self):
			q[val] = idx
		return Permutation(q,clean=True)

	def pretty_out(self, by_lines=False, width = 2):
		"""Return a nice string to visualize `self`.
		
		Notes:
			If `by_lines == True`, then will return the list of strings by line,
			in case you want to append some stuff to each line.

		Examples:
			>>> print(Permutation([1,9,3,7,5,6,4,8,2,10]).pretty_out())
			                  10
			   9                
			               8    
			       7            
			           6        
			         5          
			             4      
			     3              
			                 2  
			 1                  

			>>> for line in Permutation([1,9,3,7,5,6,4,8,2,10]).pretty_out(by_lines = True):
			...     print(repr(line))
			... 
			'                  10'
			'   9                '
			'               8    '
			'       7            '
			'           6        '
			'         5          '
			'             4      '
			'     3              '
			'                 2  '
			' 1                  '

		ME: Done!
		"""
		lines = []
		n = len(self)
		
		max_width = len(str(n+1)) # This is the width of each value.
		if max_width > width:
			width = max_width

		blank = " " * width
		for val in range(n)[::-1]:
			idx = self.index(val)
			line = (blank * (idx) \
					+ str(val+1).rjust(width)
					+ blank * (n-idx-1))
			lines.append(line)

		if by_lines:
			return lines
		else:
			return "\n".join(lines)

	def fixed_points(self):
		"""Return the fixed points of the permutation as a list. Recall that 
		both indices and values are zero-indexed.
		
		>>> Permutation(521436).fixed_points()
		[1, 3, 5]

		ME: Done!
		"""
		L = [idx for idx, val in enumerate(self) if idx == val]
		return L

	def sum_decomposable(self):
		"""Determine whether the permutation is expressible as the direct sum of
		two smaller permutations.
		
		Examples:
			>>> p = Permutation.random(4).direct_sum(Permutation.random(15))
			>>> p.sum_decomposable()
			True
			>>> p.reverse().sum_decomposable()
			False

		ME: Done!
		"""
		indices = set()
		vals    = set()
		for idx, val in enumerate(self[:-1]):
			# Iterates through the permutation up until the penultimate entry.
			indices.add(idx)
			vals.add(val)
			if indices == vals:
				return True
		return False

	def sum_decomposition(self):
		"""Return the list of sum-indecomposable permutations which sum to self.
		
		Examples:
			>>> p = Permutation(1) + Permutation(312) + Permutation(21)
			>>> p.sum_decomposition()
			[1, 3 1 2, 2 1]
			>>> p == sum(p.sum_decomposition(), Permutation([]))
			True

		ME: Done!
		"""
		if len(self) == 0:
			return []

		indices = set()
		vals    = set()
		for idx, val in enumerate(self[:-1]):
			# Iterates through the permutation up until the penultimate entry.
			indices.add(idx)
			vals.add(val)
			if indices == vals:
				component = [Permutation(self[:idx+1],clean=True)]
				rest = Permutation((val-idx-1 for val in self[idx+1:]), clean=True)
				return component + rest.sum_decomposition()
		
		# If we didn't return already, then self is sum-indecomposable.
		return [self]

	def skew_decomposable(self):
		"""Determine whether the permutation is expressible as the skew sum of
		two smaller permutations.
		
		Examples:
			>>> p = Permutation.random(8).direct_sum(Permutation.random(12))
			>>> p.skew_decomposable()
			False
			>>> p.complement().skew_decomposable()
			True

		ME: Done!
		"""
		indices = set()
		vals    = set()
		n = len(self)
		for idx, val in enumerate(self[:-1]):
			indices.add(idx)
			vals.add(n-val-1)
			if indices == vals:
				return True
		return False

	def skew_decomposition(self):
		"""Return the list of skew-indecomposable permutations which skew sum to self.
		
		Examples:
			>>> p = Permutation.random(4) + Permutation.random(15)
			>>> p.sum_decomposable()
			True
			>>> p.reverse().sum_decomposable()
			False

		ME: Done!
		"""
		if not self:
			return []

		indices = set()
		vals    = set()
		n = len(self)
		for idx, val in enumerate(self[:-1]):
			# Iterates through the permutation up until the penultimate entry.
			indices.add(idx)
			vals.add(n-val-1)
			if indices == vals:
				component = [Permutation((value-(n-idx)+1 for value in self[:idx+1]), clean=True)]
				rest = Permutation(self[idx+1:],clean=True)
				return component + rest.skew_decomposition()

		# If we didn't return already, then self is skew-indecomposable.
		return [self]

	def descents(self):
		"""Return the list of (positions of) descents of the permutation.
		
		Examples:
			>>> Permutation(42561873).descents()
			[0, 3, 5, 6]

		ME: Done!
		"""
		return [i for i in range(len(self)-1) if self[i] >= self[i+1]] # >= is a bit faster than > for some reason.

	def ascents(self):
		"""Return the list of (positions of) ascents of the permutation.

		Examples:
			>>> Permutation(42561873).ascents()
			[1, 2, 4]

		ME: Done!
		"""
		return [i for i in range(len(self)-1) if self[i] <= self[i+1]] # <= is a bit faster than < for some reason.

	def peaks(self):
		"""Return the list of (positions of) peaks of the permutation.
		
		Examples:
			>>> Permutation(2341765).peaks()
			[2, 4]

		ME: Done!
		"""
		return [i for i in range(1, len(self)-1) if self[i-1] < self[i] > self[i+1]]

	def valleys(self):
		"""Return the list of (positions of) valleys of the permutation.
		
		Examples:
			>>> Permutation(3241756).valleys()
			[1, 3, 5]

		ME: Done!
		"""
		return [i for i in range(1, len(self)-1) if self[i-1] > self[i] < self[i+1]]

	def ltr_min(self):
		"""Return the positions of the left-to-right minima.
		
		Examples:
			>>> Permutation(35412).ltr_min()
			[0, 3]

		ME: Done!
		"""
		L = []
		minval = len(self)
		for idx, val in enumerate(self):
			if val < minval:
				L.append(idx)
				minval = val
		return L

	def rtl_min(self):
		"""Return the positions of the right-to-left minima.
		
		Examples:
			>>> Permutation(315264).rtl_min()
			[5, 3, 1]

		ME: Done!
		"""
		L = []
		n = len(self)
		minval = n
		for idx, val in enumerate(self[::-1]):
			if val < minval:
				L.append(n-idx-1)
				minval = val
		return L

	def ltr_max(self):
		"""Return the positions of the left-to-right maxima.
		
		Examples:
			>>> Permutation(35412).ltr_max()
			[0, 1]

		ME: Done!
		"""
		L = []
		maxval = -1
		for idx, val in enumerate(self):
			if val > maxval:
				L.append(idx)
				maxval = val
		return L

	def rtl_max(self):
		"""Return the positions of the right-to-left maxima.
		
		Examples:
			>>> Permutation(35412).rtl_max()
			[4, 2, 1]

		ME: Done!
		"""
		L = []
		n = len(self)
		maxval = -1
		for idx, val in enumerate(self[::-1]):
			if val > maxval:
				L.append(n-idx-1)
				maxval = val
		return L

	def inversions(self):
		"""Return the list of inversions of the permutation, i.e., the
		pairs (i,j) such that i < j and self(i) > self(j).
		
		Examples:
			>>> Permutation(4132).inversions()
			[(0, 1), (0, 2), (0, 3), (2, 3)]
			>>> Permutation.monotone_increasing(7).inversions()
			[]

		ME: Done!
		"""
		L = [(i,j+i+1) for i, val_i in enumerate(self)\
					   for j, val_j in enumerate(self[i+1:]) if val_i >= val_j]
		return L

	def noninversions(self):
		"""Return the list of noninversions of the permutation, i.e., the
		pairs (i,j) such that i < j and self(i) < self(j).

		ME: Done!
		"""
		n = len(self)
		L = [(i,j+i+1) for i, val_i in enumerate(self)\
					   for j, val_j in enumerate(self[i+1:]) if val_i <= val_j]
		return L

	def breadth(self):
		"""Return the minimum taxicab distance between any two entries in the 
		permutation.
		
		Examples:
			>>> Permutation(3142).breadth()
			3

		ME: TODO! Currently uses the naive algorithm--can be improved.
		"""

		min_dist = len(self)
		for i, j in itertools.combinations(range(len(self)), 2):
			h_dist = abs(i - j)
			v_dist = abs(self[i] - self[j])
			dist = h_dist + v_dist
			if dist < min_dist:
				min_dist = dist
		return min_dist

	def bonds(self):
		"""Return the list of (initial) indices of the bonds of `self`.

		Notes:
			A bond is an interval of size 2.

		ME: Done!
		"""
		L = [idx for idx,val in enumerate(self[:-1]) if val - self[idx+1] in [-1,1]]
		return L

	# def fixedptsplusbonds(self):
	#     return len(self.fixed_points() + self.bonds())

	def pattern_counts(self, k):
		"""Return a Counter (dictionary) counting the occurrences of each perm of length `k` in `self`.

		Examples:
			>>> a = Permutation(1324)
			>>> a.pattern_counts(3)
			Counter({1 2 3: 2, 1 3 2: 1, 2 1 3: 1})

		ME: Done!
		"""
		C = Counter()
		for vals in itertools.combinations(self,k):
			C[ Permutation(vals) ] += 1
		return C

	def max_ascending_run(self):
		"""Return the (inital) index and length of a longest ascending run of `self`.
		
		Notes:
			An ascending run is a consecutive sequence of increasing entries.

		ME: Done!
		"""
		max_idx = 0
		max_len = 0
		current_run_max = -1
		current_run_idx =  0
		current_run_len =  0
		for idx, val in enumerate(self):
			if val > current_run_max:
				current_run_len += 1
				current_run_max = val
			else:
				if current_run_len > max_len:
					max_idx = current_run_idx
					max_len = current_run_len
					current_run_max = val
					current_run_idx = idx
					current_run_len = 1
		return (max_idx, max_len)

	def max_descending_run(self):
		"""Return the (inital) index and length of a longest descending run of `self`.
		A descending run is a consecutive sequence of decreasing entries.

		ME: Done!
		"""
		max_idx = 0
		max_len = 0
		current_run_min = n
		current_run_idx = 0
		current_run_len = 0
		for idx, val in enumerate(self):
			if val < current_run_min:
				current_run_len += 1
				current_run_min = val
			else:
				if current_run_len > max_len:
					max_idx = current_run_idx
					max_len = current_run_len
					current_run_min = val
					current_run_idx = idx
					current_run_len = 1
		return (max_idx, max_len)

	def covered_by(self):
		"""Return the set of permutations which cover `self`.

		ME: Done!
		"""
		S = set()
		n = len(self)
		for idx, val in enumerate(self):
			for j in range(val):
				S.add(self.insert(idx,j))
			for j in range(val+2,n+1):
				S.add(self.insert(idx,j))
		for j in range(n+1):
			S.add(self.insert(n,j))
		return S

	def covers(self):
		"""Return the set of permutations which are covered by `self`.

		ME: Done!
		"""
		S = set(self.delete(values=val) for val in self)
		return S

	def upset(self, height, stratified=False):
		"""Return the upset of `self` using repeated applications of `covered_by`.
		
		Notes:
			If `stratified` == True, return the upset as a list `L` such that 
			`L[i]` is the set of permutations of length `i` which contain `self`.

		ME: Done!
		"""
		n = len(self)
		L = [set()]*n
		L.append( set([self]) )
		for i in range(n+1, height+1):
			new_set = set()
			for perm in L[i-1]:
				new_set.update(perm.covered_by())
			L.append(new_set)

		if stratified:
			return L
		else:
			return set().union(*L)

	def set_up_bounds(self):
		"""Use for something?
		"""
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

	def avoids(self, p=None, lr=0, B=None):
		""" Check if the permutation avoids the pattern `p`.

		Args:
			p (Permutation-like object): permutation to be avoided
			lr (int): Require the last entry to be equal to this
			B (iterable of permutation-like objects:optional): A collection of permutations to be avoided.

		>>> Permutation(123456).avoids(231)
		True
		>>> Permutation(123456).avoids(123)
		False

		ME: "#TODO Am I correct on the lr?"
		"""
		if p is not None:
			p = Permutation(p)
			if p.involved_in(self, last_require=lr):
				return False
			else:
				return True
		elif B:
			B = [Permutation(b) for b in B]
			for p in B:
				if p.involved_in(self, last_require=lr):
					return False
			return True
		else:
			# If we're here, neither a permutation `p` nor a set `B` was provided.
			return True

	def involves(self, P, lr=0):
		"""Check if the permutation contains the pattern `P`.

		Args:
			P (Permutation-like object): Pattern to be contained.
			lr (int, optional): Require the last entry to be equal to this.
		
		Examples:
			>>> Permutation(123456).involves(231)
			False
			>>> Permutation(123456).involves(123)
			True
		"""
		return Permutation(P).involved_in(self,last_require=lr)

	def involved_in(self, P, last_require=0):
		"""Check if `self` is contained as a pattern in `P`.

		Args:
			P (Permutation-like object): Pattern to be contained.
			lr (int, optional): Require the last entry to be equal to this.
		
		Examples:
			>>> Permutation(123).involved_in(31542)
			False
			>>> Permutation(213).involved_in(54213)
			True
		"""
		P = Permutation(P)

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
			if self.involvement_fits(upper_bound, lower_bound, indices, q, next) and \
					self.involvement_check(upper_bound, lower_bound, indices, q, next-1):
				return True
			indices[next] -= 1
		return False

	def involvement_fits(self, upper_bound, lower_bound, indices, q, next):
		return (lower_bound[next] == -1 or q[indices[next]] > q[indices[lower_bound[next]]]) \
		   and (upper_bound[next] == -1 or q[indices[next]] < q[indices[upper_bound[next]]])

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

	def all_monotone_intervals(self, with_ones=False):
		"""Return all monotone intervals of size at least 2.

		If `with_ones == True`, then return all monotone intervals of size at least 1.

		ME: TODO!
		"""

		mi = []
		difference = 0
		c_start = 0
		c_length = 0
		for i in range(0,len(self)-1):
			if (self[i] - self[i+1]) in [-1,1] and \
				(c_length == 0 or (self[i] - self[i+1]) == difference):
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

		if with_ones:
			in_int = []
			for (start,end) in mi:
				in_int.extend(range(start, end+1))
			for i in range(len(self)):
				if i not in in_int:
					mi.append((i,i))
			mi.sort(key=lambda x : x[0])
		return mi

	def monotone_quotient(self):
		"""Quotient `self` by its monotone intervals.

		ME: TODO!
		"""

		return Permutation([self[k[0]] for k in self.all_monotone_intervals(with_ones=True)])

	def maximal_interval(self):
		"""Find the biggest interval, and return (i,j) is one is found,
		where i is the size of the interval, and j is the index
		of the first entry in the interval

		Returns (0,0) if no interval is found, i.e., if the permutation
			is simple.

		ME: TODO!
		"""
		for i in range(2, len(self))[::-1]:
			for j in range (0,len(self)-i+1):
				if max(self[j:j+i]) - min(self[j:j+i]) == i-1:
					return (i,j)
		return (0,0)

	def simple_location(self):
		"""Searche for an interval, and return (i,j) if one is found,
		where i is the size of the interval, and j is the
		first index of the interval.

		Returns (0,0) if no interval is found, i.e., if the permutation
		is simple.

		ME: TODO!
		"""
		mins = list(self)
		maxs = list(self)
		for i in range(1,len(self)-1):
			for j in reversed(range(i,len(self))):
				mins[j] = min(mins[j-1], self[j])
				maxs[j] = max(maxs[j-1], self[j])
				if maxs[j] - mins[j] == i:
					return (i,j)
		return (0,0)

	def decomposition(self):
		"""

		ME: TODO!
		"""

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
		""" Returns the list of right extensions of `self`, only including those in which the new
			value comes from `self.insertion_values`.
		"""
		L = []
		for new_val in self.insertion_values:
			p = [val if val < new_val else val+1 for val in self]
			p.append(new_val)
			L.append(Permutation(p,clean=True))
		return L

	# def all_right_extensions(self, max_length, l, S):
	#       if l == max_length:
	#           return S
	#       else:
	#           re = self.right_extensions()
	#           for p in re:
	#               S.add(p)
	#               S = p.all_right_extensions(max_length, l+1, S)
	#       return S

	def all_extensions(self):
		S = set()
		for i in range(0, len(self)+1):
			for j in range(0, len(self)+1):
				# insert (i-0.5) after entry j (i.e., first when j=0)
				l = list(self[:])
				l.insert(j, i-0.5)
				S.add(Permutation(l))
		return permset.PermSet(S)

	def all_extensions_track_index(self, ti):
		L = []
		for i in range(0, len(self)+1):
			for j in range(0, len(self)+1):
				# insert (i-0.5) after entry j (i.e., first when j=0)
				l = list(self[:])
				l.insert(j, i-0.5)
				if j < ti:
					L.append((Permutation(l), ti+1))
				else:
					L.append((Permutation(l), ti))
		return L

	def plot(self, show=True, ax=None, use_mpl=True, fname=None, **kwargs):
		""" Draws a matplotlib plot of the permutation. Can be used for both
		quick visualization, or to build a larger figure. Unrecognized arguments
		are passed as options to the axes object to allow for customization
		(i.e., setting a figure title, or setting labels on the axes). Falls
		back to an ascii_plot if matplotlib isn't found, or if use_mpl is set to
		False.
		"""
		if not mpl_imported or not use_mpl:
			return self._ascii_plot()
		xs = [val + Permutation._BASE for val in range(len(self))]
		ys = [val + Permutation._BASE for val in self]
		if not ax:
			ax = plt.gca()
		scat = ax.scatter(xs, ys, s=40, c='k')
		ax_settings = {'xticks': xs, 'yticks': ys,
					'xticklabels': '', 'yticklabels': '',
					'xlim': (min(xs) - 1, max(xs) + 1),
					'ylim': (min(ys) - 1, max(ys) + 1)}
		ax.set(**ax_settings)
		ax.set(**kwargs)
		ax.set_aspect('equal')
		if fname:
			fig = plt.gcf()
			fig.savefig(fname, dpi=300)
		if show:
			plt.show()
		return ax


	def _show(self):
		if sys.platform == 'linux2':
			opencmd = 'gnome-open'
		else:
			opencmd = 'open'
		s = r"\documentclass{standalone}\n\usepackage{tikz}\n\n\\begin{document}\n\n"
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
		return permpy.permset.PermSet([Permutation(p) for p in [self[:i]+self[i+1:] for i in range(0,len(self))]])

	def children(self):
		"""Returns all patterns of length one less than the permutation."""
		return self.shrink_by_one()

	def downset(self):
		return permset.PermSet([self]).downset()

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
			S = list(permset.PermSet(S).layer_down())
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
		"""Return the set (not PermSet) of all symmetries of `self`.

		ME: Done!
		"""
		S = set([self])
		S.update([P.reverse() for P in S])
		S.update([P.complement() for P in S])
		S.update([P.inverse() for P in S])
		return S

	def is_representative(self):
		"""Check if self is the (lexicographically) least element of its symmetry class.

		ME: Done!
		"""
		return self == sorted(self.all_syms())[0]

	def greedy_sum(p):
		parts = []
		sofar = 0
		while sofar < len(p):
			if len(p)-sofar == 1:
				parts.append(Permutation(1))
				return parts
			i = 1
			while sofar+i <= len(p) and list(p[sofar:sofar+i]) == range(sofar,sofar+i):
				i += 1
			i -= 1
			if i > 0:
				parts.append(Permutation(range(i)))
			sofar += i
			i = 2
			while sofar+i <= len(p) and not (max(p[sofar:sofar+i]) - min(p[sofar:sofar+i])+1 == i and min(p[sofar:sofar+i]) == sofar):
				i += 1
			if sofar+i <= len(p):
				parts.append(Permutation(p[sofar:sofar+i]))
			sofar += i
		return parts

	@classmethod
	def one_cycles(cls, n):
		"""Generate those permutations of length n which consist of one cycle.
		"""
		for pi in itertools.permutations(range(n-1)):
			# print(f"pi={pi}")
			cycle = [n-1] + list(pi)
			tau = [None for _ in range(n)]
			for idx, val in enumerate(cycle[:-1]):
				# print(val, tau)
				tau[val] = cycle[idx+1]
			tau[cycle[-1]] = cycle[0]
			yield (Permutation(tau), cycle)

	def copies_of(self, other):
		"""Return the list of (values corresponding to) copies of `other` in `self`.
		"""
		copies = []
		for subseq in itertools.combinations(self,len(other)):
			if Permutation(subseq) == other:
				copies.append(subseq)
		return copies

	def immediate_copies_of(self, other):
		"""Return the list of (indices corresponding to) immediate copies of `other` in `self`.
		"""
		immediate_copies = []
		m = len(other)
		for initial_idx in range(len(self)-m):
			if Permutation(self[initial_idx:initial_idx+m]) == other:
				immediate_copies.append(initial_idx)
		return immediate_copies

	def density_of(self, pi):
		"""Return the density of copies of pi in self.
		"""
		copies = self.copies_of(pi)
		num_copies = len(copies)

		return float(num_copies)/comb(len(self),len(pi))

	def optimizers(self, n):
		# print(self)
		max_copies = 0
		best_perms = []
		for tau in Permutation.genall(n):
			# print("tau = {}".format(tau))
			num_copies = len(tau.copies_of(self))
			# print("num_copies = {}".format(num_copies))
			if num_copies > max_copies:
				max_copies = num_copies
				best_perms = [tau]
				# print("\tnum_copies = {}".format(num_copies))
			elif num_copies == max_copies:
				best_perms.append(tau)
				# print("\tnum_copies = {}".format(num_copies))

		return best_perms


if __name__ == '__main__':
	pass
	# B = [Permutation([1]) - Permutation(b) for b in [312,231,123]]
	# for b in B:
	# 	print(b)

	# n = 5
	# for pi in Permutation.genall(n):
	# 	if all(pi.avoids(b) for b in B):
	# 		if not pi.sum_decomposable():
	# 			print(pi.pretty_out())
	# 			print("-"*(2*n))





			








