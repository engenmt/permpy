import copy
import logging
import time
from math import factorial

from .permutation import Permutation
from .permset import PermSet
from .deprecated.permclassdeprecated import PermClassDeprecatedMixin
from .utils import copy_func

logging.basicConfig(level=logging.INFO)


class PermClass(list, PermClassDeprecatedMixin):
	"""A minimal Python class representing a Permutation class.
	
	Notes:
		Relies on the Permutation class being closed downwards, but does not assert this.
	"""

	def __init__(cls, C):
		super(PermClass, cls).__init__(C)
		cls.length = len(C)-1

	def __contains__(self, p):
		p_length = len(p)
		if p_length > self.length:
			return False
		return p in self[p_length]

	def filter_by(self, test):
		"""Modify `self` by removing those permutations that fail the test."""
		for i, S in range(len(self)):
			for p in list(self[i]):
				if not test(p):
					self[i].remove(p)

	def guess_basis(self, max_length=6, search_mode=False):
		"""Guess a basis for the class up to "max_length" by iteratively
		generating the class with basis elements known so far (initially the 
		empty set) and adding elements that should be avoided to the basis.

		Search mode goes up to the max length in the class and prints out the 
		number of basis elements of each length on the way.
		"""
		assert max_length < self.length, 'Class not big enough to check that far!'

		# Find the first length at which perms are missing.
		for idx, S in enumerate(self):
			if len(S) < factorial(idx):
				start_length = idx
				break
		else:
			# If we're here, then self is the class of all permutations.
			return PermSet()

		# Add missing perms of minimum length to basis.
		missing = PermSet.all(start_length) - self[start_length]
		basis = missing

		length = start_length
		current = PermSet(Permutation.gen_all(length-1))
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
	
	def skew_closure(self, max_len=8, has_all_syms=False):
		"""
		Notes:
			This could be done constructively.
		Examples:
			>>> p = Permutation(21)
			>>> C = PermClass.all(8)
			>>> C.filter_by(lambda q: p not in q) # Class of increasing permutations
			>>> D = C.skew_closure()
			>>> len(D[8]) == 128
			True
		"""
		L = []
		for length in range(len(self)):
			new_set = PermSet()
			for p in Permutation.gen_all(length):
				if all(q in self for q in set(p.skew_decomposition())):
					new_set.add(p)
			L.append(new_set)
					
		return PermClass(L)

	def sum_closure(self, max_len=8, has_all_syms=False):
		"""
		Notes:
			This could be done constructively.
		Examples:
			>>> p = Permutation(12)
			>>> C = PermClass.all(8)
			>>> C.filter_by(lambda q: p not in q) # Class of decreasing permutations
			>>> D = C.skew_closure()
			>>> len(D[8]) == 128
			True
		"""
		if self.test:
			test = copy.deepcopy(self.test)
			def is_sum(p):
				return all(test(q) for q in p.sum_decomposition())
		else:
			C = copy.deepcopy(self)
			def is_sum(p):
				return all(q in C for q in p.sum_decomposition())
		return PermClass.class_from_test(is_sum, max_len=max_len, has_all_syms=has_all_syms)

if __name__ == "__main__":
	pass
