import copy
import logging
import time

from math import factorial

from .permutation import Permutation
from .permset import PermSet
from .deprecated.permclassdeprecated import PermClassDeprecatedMixin
from .utils import copy_func

logging.basicConfig(level=logging.INFO)


class PermClass(PermClassDeprecatedMixin):
	"""A minimal Python class representing a Permutation class.
	
	Notes:
		Relies on the Permutation class being closed downwards, but does not assert this.
	"""
	def __init__(self, C):
		self.data = C
		self.max_len = len(C)-1
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.data[idx]
	
	def __add__(self, other):
		return self.union(other)
	
	def __contains__(self, p):
		p_length = len(p)
		if p_length > self.max_len:
			return False
		return p in self[p_length]
	
	@classmethod
	def all(cls, max_length):
		"""Return the PermClass that contains all permutations up to the given length.
		
		Examples:
			>>> C = PermClass.all(6)
			>>> print([len(S) for S in C])
			[1, 1, 2, 6, 24, 120, 720]
		"""
		return PermClass([PermSet.all(length) for length in range(max_length+1)])
	
	def maximally_extend(self, additional_length=1):
		"""Extend `self` maximally.
		
		Notes: Includes only those permutations whose downsets lie entirely in `self`.
		Examples:
			>>> C = PermClass.all(4)
			>>> C[4].remove(Permutation(1234))
			>>> C.maximally_extend(1)
			>>> len(C[5]) # All but the 17 permutations covering 1234
			103
		"""
		for _ in range(additional_length):
			self.data.append(PermSet(
				p for p in Permutation.gen_all(self.max_len+1) if p.covers().issubset(self[-1])
			))
			self.max_len += 1
		

	def filter_by(self, property):
		"""Modify `self` by removing those permutations that do not satisfy the `property``.
		
		Examples:
			>>> C = PermClass.all(6)
			>>> p = Permutation(21)
			>>> C.filter_by(lambda q: p not in q)
			>>> all(len(S) == 1 for S in C)
			True
		"""
		for length in range(len(self)):
			for p in list(self[length]):
				if not property(p):
					self[length].remove(p)

	def filtered_by(self, property):
		"""Return a copy of `self` that has been filtered using the `property`."""
		C = copy.deepcopy(self)
		C.filter_by(property)
		return C

	def guess_basis(self, max_length=6, search_mode=False):
		"""Guess a basis for the class up to "max_length" by iteratively
		generating the class with basis elements known so far (initially the 
		empty set) and adding elements that should be avoided to the basis.

		Search mode goes up to the max length in the class and prints out the 
		number of basis elements of each length on the way.
		
		Examples:
			>>> p = Permutation(12)
			>>> C = PermClass.all(8)
			>>> C.filter_by(lambda q: p not in q) # Class of decreasing permutations
			>>> C.guess_basis() == PermSet(p)
			True
			>>> D = C.sum_closure() # Class of layered permutations
			>>> D.guess_basis() == PermSet([Permutation(312), Permutation(231)])
			True
		"""
		assert max_length <= self.max_len, 'The class is not big enough to check that far!'

		# Find the first length at which perms are missing.
		for length, S in enumerate(self):
			if len(S) < factorial(length):
				start_length = length
				break
		else:
			# If we're here, then `self` is the class of all permutations.
			return PermSet()

		# Add missing perms of minimum length to basis.
		missing = PermSet.all(start_length) - self[start_length]
		basis = missing

		length = start_length
		current = PermSet.all(length-1)
		current = current.right_extensions(basis=basis)

		# Go up in length, adding missing perms at each step.
		while length < max_length:
			length += 1
			current = current.right_extensions(basis=basis)

			for perm in list(current):
				if perm not in self[length]:
					basis.add(perm)
					current.remove(perm)

		return basis

	def union(self, other):
		"""Return the union of the two permutation classes.
		"""
		return PermClass([S_1 + S_2 for S_1, S_2 in zip(self, other)])

	def heatmap(self, **kwargs):
		permset = PermSet(set.union(*self)) # Collect all perms in self into one PermSet
		permset.heatmap(**kwargs)
	
	def skew_closure(self, max_len=8):
		"""
		Notes:
			This could be done constructively.
		Examples:
			>>> p = Permutation(21)
			>>> C = PermClass.all(8)
			>>> C.filter_by(lambda q: p not in q) # Class of increasing permutations
			>>> D = C.skew_closure(max_len=7)
			>>> len(D[7]) == 64
			True
		"""
		assert max_len <= self.max_len, "Can't make a skew-closure of that size!"
		L = []
		for length in range(max_len+1):
			new_set = PermSet()
			for p in Permutation.gen_all(length):
				if all(q in self for q in set(p.skew_decomposition())):
					new_set.add(p)
			L.append(new_set)
					
		return PermClass(L)

	def sum_closure(self, max_len=8):
		"""
		Notes:
			This could be done constructively.
		Examples:
			>>> p = Permutation(12)
			>>> C = PermClass.all(8)
			>>> C.filter_by(lambda q: p not in q) # Class of decreasing permutations
			>>> D = C.sum_closure(max_len=7)
			>>> len(D[7]) == 64
			True
		"""
		assert max_len <= self.max_len, "Can't make a sum-closure of that size!"
		L = []
		for length in range(max_len+1):
			new_set = PermSet()
			for p in Permutation.gen_all(length):
				if all(q in self for q in set(p.sum_decomposition())):
					new_set.add(p)
			L.append(new_set)
					
		return PermClass(L)

if __name__ == "__main__":
	pass
