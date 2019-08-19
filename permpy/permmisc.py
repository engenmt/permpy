from math import gcd

def lcm(L):
	result = 1
	for val in L:
		result *= val // gcd(result, val)
	return result

class PermutationMiscMixin:

	def cycle_decomp(self):
		"""Calculate the cycle decomposition of the permutation. Return as a 
		list of cycles, each of which is represented as a list.

		>>> Permutation(53814276).cycle_decomp()
		[[4, 3, 0], [6], [7, 5, 1, 2]]

		ME: Done!
		"""
		not_seen = set(self)
		cycles = []
		while len(not_seen) > 0:
			a = max(not_seen)
			cyc = [a]
			not_seen.remove(a)
			b = self[a]
			while b in not_seen:
				not_seen.remove(b)
				cyc.append(b)
				b = self[b]
			cycles.append(cyc)
		return cycles[::-1]

	def cycles(self):
		"""Returns the cycle notation representation of the permutation."""
		stringlist = ['( ' + ' '.join([str(x+1) for x in cyc]) + ' )'
		                    for cyc in self.cycle_decomp()]
		return ' '.join(stringlist)

	
	def order(self):
		L = map(len, self.cycle_decomp())
		return lcm(L)
