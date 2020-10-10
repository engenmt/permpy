from math import factorial
import logging
import sys
import types


from .permutation import Permutation
from .permset import PermSet
from .permclass import PermClass

class AvClass(PermClass):
	"""Object representing an avoidance class.

	Examples:
		>>> B = [123]
		>>> A = AvClass(B, length=4)
		>>> for S in A:
		...    print(S)
		... 
		Set of 0 permutations
		Set of 1 permutations
		Set of 2 permutations
		Set of 5 permutations
		Set of 14 permutations
	"""
	def __init__(self, basis, length=8, verbose=0):

		list.__init__(self, [PermSet()])
		self.basis = [Permutation(b) for b in basis]
		self.test = lambda p: all(b not in p for b in basis)

		p = Permutation([0], clean=True)
		if length >= 1:
			if p not in self.basis:
				self.append(PermSet(p))
				self.length = 1
				self.extend_to_length(length,verbose=verbose)
			else:
				for _ in range(length):
					self.append(PermSet())

	def extend_by_one(self, test=None, verbose=0, return_info=False):
		logging.info(f"Calling extend_by_one({self}, test={test}, verbose={verbose}, return_info={return_info})")
		self.length += 1
		if test is None:
			upset = self[-1].upset(basis=self.basis, verbose=verbose)
		else:
			upset, failures = self[-1].upset_with_test(test, basis=self.basis, return_failures=True, verbose=verbose)
			# print(f"upset = (length {len(upset)})\n{sorted([perm for perm in upset])}")
			# print(f"failures = (length {len(failures)})\n{sorted([perm for perm in failures])}")
			self.basis.extend(failures)
		self.append(upset)
		if return_info:
			return f"{self.length:2}: {len(self[-1]):10}"

	def extend_to_length(self, length, test=None, verbose=0):
		if length <= self.length:
			return

		old_length = self.length
		for n in range(old_length+1, length+1):
			self.extend_by_one(test=test, verbose=verbose)

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

	def contains(self, other):
		"""Check if `self` contains `other` as a permutation class using their bases.
		"""
		for p in self.basis:
			for q in other.basis:
				if p in q:
					break
			else:
				# If we're here, then `p` is not involved in any of the basis elements of `q`, so
				# the permutation `p` lies in `other` but not `self`.
				return False
		return True

if __name__ == "__main__":
	print()

	B = [123]
	A = AvClass(B, 12)
	for idx, S in enumerate(A):
		print(S)


