
from .permdeprecated import deprecated

class PermSetDeprecatedMixin:

	@deprecated
	def layer_down(self, verbose=0):
		"""Return the PermSet of those permutations which are covered by an element of `self`."""
		return self.covers(verbose=0)

	@deprecated
	def all_syms(self):
		return self.symmetries()

	@deprecated
	def threepats(self):
		return {str(p): count for p, count in self.pattern_counts(3).items()}

	@deprecated
	def fourpats(self):
		return {str(p): count for p, count in self.pattern_counts(4).items()}

