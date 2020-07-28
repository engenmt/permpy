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
		"""
		return sum(self.descents())

	def len_max_run(self):
		"""Return the length of the longest monotone contiguous subsequence of entries."""
		return max(self.max_ascending_run()[1], self.max_descending_run()[1])

	def is_involution(self):
		"""Determine if the permutation is an involution, i.e., is equal to it's own inverse. """
		for idx, val in enumerate(self):
			if idx != self[val]: 
				return False
		return True

	def is_identity(self):
		"""Determine if the permutation is the identity.
		
		Examples:
			>>> p = Permutation.random(10)
			>>> (p * p.inverse()).is_identity()
			True

		"""
		for idx, val in enumerate(self):
			if idx != val:
				return False
		return True

	def is_simple(self):
		"""Determine if `self` is simple.

		Todo: Implement this better, if possible.
		"""
		(i,j) = self.simple_location()
		return i == 0

	def is_strongly_simple(self):
		return self.is_simple() and all([p.is_simple() for p in self.children()])

	def num_copies(self, other):
		"""Return the number of copies of `other` in `self`.
		"""
		return len(self.copies(other))

	def num_contiguous_copies_of(self, other):
		"""Return the number of contiguous copies of `other` in `self`.
		"""
		return len(self.contiguous_copies(other))







