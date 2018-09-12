from math import factorial
import types

import permutation
import permset
import permclass

# from permutation import Permutation
# from permset import PermSet
# from permclass import PermClass

class AvClass(permclass.PermClass):
	"""Object representing an avoidance class.

	Examples:
		>>> B = [123]
		>>> A = AvClass(B, length=4)
		>>> print(A)
		Set of 0 permutations
		Set of 1 permutations
		Set of 2 permutations
		Set of 5 permutations
		Set of 14 permutations
	"""
	def __init__(self, basis, length=8, verbose=0):

		list.__init__(self, [permset.PermSet()])
		self.basis = [permutation.Permutation(b) for b in basis]

		P = permutation.Permutation([0],clean=True)
		if length >= 1 and P not in self.basis:
			self.append(permset.PermSet([P]))
			self.length = 1
			self.extend_to_length(length,verbose)

	def extend_to_length(self, length, verbose=0):

		if length <= self.length:
			return
		for i in range(self.length+1, length+1):
			self.append(permset.PermSet())

		old_length = self.length
		self.length = length
		for n in range(old_length+1, length+1):
			prev_count = len(self[n-1])
			for k, P in enumerate(self[n-1]):

				if verbose > 0 and k % verbose == 0:
					print(f"\tRight Extensions: {k}/{prev_count}\t(length {n})")

				new_insertion_values = P.insertion_values
				to_add = set()
				for Q in P.right_extensions():
					new_val = Q[-1]
					for B in self.basis:
						if B.involved_in(Q,last_require=2):
							# Here, last_require = 2, since if Q contains B using the new value
							# and not the previous, then this value would have been flagged as bad.
							try: 
								new_insertion_values.remove(new_val)
							except ValueError:
								pass
							break
					else:
						# If we're here, then none of the basis elements were involved in Q,
						# i.e., the loop never broke.
						to_add.add(Q)

				for Q in to_add:
					new_val = Q[-1]
					Q.insertion_values = [val if val < new_val else val+1 for val in new_insertion_values]
					Q.insertion_values.append(new_val)
					self[n].add(Q)

	def right_juxtaposition(self, C, generate_perms=True):
		A = permset.PermSet()
		max_length = max([len(P) for P in self.basis]) + max([len(P) for P in C.basis])
		for n in range(2, max_length+1):
			for i in range(0, factorial(n)):
				P = permutation.Permutation(i,n)
				for Q in self.basis:
					for R in C.basis:
						if len(Q) + len(R) == n:
							if (Q == permutation.Permutation(P[0:len(Q)]) and R == permutation.Permutation(P[len(Q):n])):
								A.add(P)
						elif len(Q) + len(R) - 1 == n:
							if (Q == permutation.Permutation(P[0:len(Q)]) and permutation.Permutation(R) == permutation.Permutation(P[len(Q)-1:n])):
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


