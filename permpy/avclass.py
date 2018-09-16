from math import factorial
import types
import sys

from permpy.permutation import Permutation
from permpy.permset import PermSet
from permpy.permclass import PermClass

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

		P = Permutation([0],clean=True)
		if length >= 1 and P not in self.basis:
			self.append(PermSet([P]))
			self.length = 1
			self.extend_to_length(length,verbose)

	def extend_by_one(self, verbose=0):
		self.length += 1
		self.append(self[-1].upset(basis=self.basis, verbose=verbose))

	def extend_to_length(self, length, verbose=0):
		if length <= self.length:
			return

		old_length = self.length
		for n in range(old_length+1, length+1):
			self.extend_by_one(verbose=verbose)

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
		"""Check if `self` contains `C` as a permutation class using their bases.

		ME: Done!
		"""
		for P in self.basis:
			for Q in C.basis:
				if P.involved_in(Q):
					break
			else:
				# If we're here, then `P` is not involved in any of the basis elements of `C`, so
				# the permutation `P` lies in `C` but not `self`.
				return False
		return True

if __name__ == "__main__":
	print()

	B = [123]
	A = AvClass(B, 10)
	for idx, S in enumerate(A):
		print(S)


