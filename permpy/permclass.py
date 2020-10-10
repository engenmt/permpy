import copy
import logging
import time
from math import factorial

from .permutation import Permutation
from .permset import PermSet


logging.basicConfig(level=logging.INFO)


class PermClass(list):

	@staticmethod
	def class_from_test(test, max_len=8, has_all_syms=False):
		"""Return the smallest PermClass of all permutations which satisfy the test.

		Args:
			test (func): function which accepts a permutation and returns a Boolean. Should be closed downward.
			max_len (int): maximum length to be included in class
			has_all_syms (Boolean): whether the class is closed under all symmetries.

		Returns:
			PermClass: class of permutations that satisfy the test.
		"""

		C = [PermSet(Permutation())] # List consisting of just the PermSet containing the empty Permutation
		for length in range(1,max_len+1):
			if len(C[length-1]) == 0:
				return PermClass(C)

			new_set = PermSet()
			to_check = PermSet(set.union(*[p.covered_by() for p in C[length-1]]))
			to_check = PermSet(p for p in to_check if PermSet(p.covers()).issubset(C[length-1]))
			
			while to_check:
				p = to_check.pop()

				if test(p):
					if has_all_syms:
						syms = PermSet(p.symmetries())
						new_set += syms
						to_check -= syms
					else:
						logging.info(f"Keeping p = {p}, as it passed the test.")
						new_set.add(p)
				else:
					logging.info(f"Throwing out p = {p}, as it failed the test.")
					if has_all_syms:
						to_check -= PermSet(p.symmetries())

			C.append(new_set)

		return PermClass(C, test)

	def __init__(cls, C, test=None):
		super(PermClass, cls).__init__(C)
		cls.test = test

	def __contains__(self, p):
		if len(p) > len(self):
			return self.test(p)

		return p in self[len(p)]

	def filter_by(self, test):
		"""Modify self by removing those permutations which fail the test.

		Note:
			Does not actually ensure the result is a class.
		"""
		for i in range(0, len(self)):
			D = list(self[i])
			for P in D:
				if not test(P):
					self[i].remove(P)

	def guess_basis(self, max_length=6, search_mode=False):
		"""Guess a basis for the class up to "max_length" by iteratively
		generating the class with basis elements known so far (initially {})
		and adding elements which should be avoided to the basis.

		Search mode goes up to the max length in the class and prints out the 
		number of basis elements of each length on the way.
		"""
		assert max_length < len(self), 'Class not big enough to check that far!'

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
		"""
		Todos: 
			Test that this extend properly, even if `self` and `other` are modified. 
		"""
		return PermClass([S_1.union(S_2) for S_1, S_2 in zip(self, other)], 
			test=lambda p: self.test(p) or other.test(p))

	def extend(self, t):
		for i in range(t):
			self.append(self[-1].right_extensions(test=self.test))

	def extended(self, t):
		C = copy.deepcopy(self)
		C.extend(t)
		return C

	def heatmap(self, **kwargs):
		permset = PermSet(set.union(*self)) # Collect all perms in self into one PermSet
		permset.heatmap(**kwargs)

	def sum_closure(self, max_len=8, has_all_syms=False):
		"""
		Notes:
			This will raise an IndexError if the resulting class is extended.
		Todos: 
			Check that the `test` works properly, even if `self` is modified. 
		"""
		def is_sum(p):
			return all(self.test(q) for q in p.sum_decomposition())
		return PermClass.class_from_test(is_sum, max_len=max_len, has_all_syms=has_all_syms)

if __name__ == "__main__":
	pass
