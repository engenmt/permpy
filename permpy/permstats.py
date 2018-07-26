from collections import Counter

class PermutationStatsMixin:

	def num_fixed_points(self):
		return len(self.fixed_points())

	def num_descents(self):
		return len(self.descents())

	def num_ascents(self):
		return len(self.ascents())

	def num_peaks(self):
		return len(self.peaks())

	def num_valleys(self):
		return len(self.valleys())

	def num_ltr_min(self):
		return len(self.ltr_min())

	def num_rtl_min(self):
		return len(self.rtl_min())

	def num_ltr_max(self):
		return len(self.ltr_max())

	def num_rtl_max(self):
		return len(self.rtl_max())

	def trivial(self):
		return 0

	def num_inversions(self):
		return len(self.inversions())

	def num_noninversions(self):
		return len(self.noninversions())

	def major_index(self):
		"""Return the major index of `self`.

		ME: Done!
		"""
		return sum(self.descents())

	def len_max_run(self):
		""" Returns the length of the longest sequence of montone consecutive entries.
		
		ME: Done!
		"""
		return max(self.max_ascending_run()[1], self.max_descending_run()[1])

	# def christiecycles(self):
	#     # builds a permutation induced by the black and gray edges separately, and
	#     # counts the number of cycles in their product. used for transpositions
	#     p = list(self)
	#     n = self.__len__()
	#     q = [0] + [p[i] + 1 for i in range(n)]
	#     grayperm = range(1,n+1) + [0]
	#     blackperm = [0 for i in range(n+1)]
	#     for i in range(n+1):
	#         ind = q.index(i)
	#         blackperm[i] = q[ind-1]
	#     newperm = []
	#     for i in range(n+1):
	#         k = blackperm[i]
	#         j = grayperm[k]
	#         newperm.append(j)
	#     return Permutation(newperm).numcycles()

	# def othercycles(self):
	#     # builds a permutation induced by the black and gray edges separately, and
	#     # counts the number of cycles in their product
	#     p = list(self)
	#     n = self.__len__()
	#     q = [0] + [p[i] + 1 for i in range(n)]
	#     grayperm = [n] + range(n)
	#     blackperm = [0 for i in range(n+1)]
	#     for i in range(n+1):
	#         ind = q.index(i)
	#         blackperm[i] = q[ind-1]
	#     newperm = []
	#     for i in range(n+1):
	#         k = blackperm[i]
	#         j = grayperm[k]
	#         newperm.append(j)
	#     return Permutation(newperm).numcycles()

	# def sumcycles(self):
	# 	return self.othercycles() + self.christiecycles()

	# def maxcycles(self):
	# 	return max(self.othercycles() - 1,self.christiecycles())

	def is_involution(self):
		"""Determine if the permutation is an involution, i.e., is equal to it's
		own inverse. 

		ME: Done!
		"""
		for idx, val in enumerate(self):
			if idx != self[val]: 
				return False
		return True

	def is_identity(self):
		"""Determine if the permutation is the identity.

		>>> p = Permutation.random(10)
		>>> (p * p.inverse()).is_identity()
		True

		ME: Done!
		"""
		for idx, val in enumerate(self):
			if idx != val:
				return False
		return True

	def is_simple(self):
		"""Determine if `self` is simple.

		ME: TODO!
		"""
		(i,j) = self.simple_location()
		return i == 0

	def is_strongly_simple(self):
		return self.is_simple() and all([p.is_simple() for p in self.children()])

	def pattern_counts(self, k):
		"""Return a Counter (dictionary) counting the occurrences of each perm of length `k` in `self`.

		ME: Done!
		"""
		C = Counter()
		for vals in itertools.combinations(self,k):
			C[ Permutation(vals) ] += 1
		return C

	def num_copies_of(self, other):
		"""Return the number of copies of `other` in `self`.

		ME: Done!
		"""
		return len(self.copies_of(other))

	def num_immediate_copies_of(self, other):
		"""Return the number of copies of `other` in `self`.

		ME: Done!
		"""
		return len(self.immediate_copies_of(other))







