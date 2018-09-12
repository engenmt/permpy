import random
import fractions
from functools import reduce

from collections import Counter, defaultdict

import permpy.permutation
# from permutation import Permutation
# import permpy.permclass
import permpy.avclass

try:
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	mpl_imported = True
except ImportError:
	mpl_imported = False


class PermSet(set):
	"""Represents a set of permutations, and allows statistics to be computed
	across the set."""

	def __repr__(self):
		return 'Set of {} permutations'.format(len(self))

	def __add__(self, other):
		"""Returns the union of the two permutation sets. Does not modify in
		place.

		Example
		-------
		>>> S = PermSet.all(3) + PermSet.all(4)
		>>> len(S)
		30
		"""
		result = PermSet()
		result.update(self); result.update(other)
		return result

	@classmethod
	def all(cls, length):
		"""Builds the set of all permutations of a given length.

		Parameters:
		-----------
		length : int
			the length of the permutations

		Examples
		--------
		>>> p = Permutation(12); q = Permutation(21)
		>>> PermSet.all(2) == PermSet([p, q])
		True
		"""
		return PermSet(permutation.Permutation.genall(length))

	def get_random(self):
		"""Returns a random element from the set.

		Example
		-------
		>>> p = PermSet.all(4).get_random()
		>>> p in PermSet.all(4) and len(p) == 4
		True
		"""

		return random.sample(self, 1)[0]


	def get_length(self, length=None):
		"""Returns the subset of permutations which have the specified length.

		Parameters
		----------
		length : int
			lenght of permutations to be returned

		Example
		-------
		>>> S = PermSet.all(4) + PermSet.all(3)
		>>> S.get_length(3) == PermSet.all(3)
		True
		"""
		return PermSet(p for p in self if len(p) == length)

	def heatmap(self, only_length=None, ax=None, blur=False, gray=False, **kwargs):
		"""Visalization of a set of permutations, which, for each length, shows
		the relative frequency of each value in each position.

		Paramters
		---------
		only_length : int or None
			if given, restrict to the permutations of this length
		"""
		if not mpl_imported:
			err = 'heatmap requires matplotlib to be imported'
			raise NotImplementedError(err)
		try:
			import numpy as np
		except ImportError as e:
			err = 'heatmap function requires numpy'
			raise e(err)
		# first group permutations by length
		total_size = len(self)
		perms_by_length = {}
		for perm in self:
			n = len(perm)
			if n in perms_by_length:
				perms_by_length[n].add(perm)
			else:
				perms_by_length[n] = PermSet([perm])
		# if given a length, ignore all other lengths
		if only_length:
			perms_by_length = {only_length: perms_by_length[only_length]}
		lengths = list(perms_by_length.keys())
		def lcm(l):
			"""Returns the least common multiple of the list l."""
			lcm = reduce(lambda x,y: x*y // fractions.gcd(x,y), l)
			return lcm
		grid_size = lcm(lengths)
		grid = np.zeros((grid_size, grid_size))
		def inflate(a, n):
			"""Inflates a k x k array A by into a nk x nk array by inflating
			each entry from A into a n x n matrix."""
			ones = np.ones((n, n))
			c = np.multiply.outer(a, ones)
			c = np.concatenate(np.concatenate(c, axis=1), axis=1)
			return c
		for length, permset in perms_by_length.items():
			small_grid = np.zeros((length, length))
			for p in permset:
				for idx, val in enumerate(p):
					small_grid[length-val-1, idx] += 1
			mul = grid_size // length
			inflated = inflate(small_grid, mul)
			num_perms = len(permset)
			inflated /= inflated.max()
			grid += inflated

		if not ax:
			ax = plt.gca()
		if blur:
			interpolation = 'bicubic'
		else:
			interpolation = 'nearest'
		def get_cubehelix(gamma=1, start=1, rot=-1, hue=1, light=1, dark=0):
			"""Get a cubehelix palette."""
			cdict = mpl._cm.cubehelix(gamma, start, rot, hue)
			cmap = mpl.colors.LinearSegmentedColormap("cubehelix", cdict)
			x = np.linspace(light, dark, 256)
			pal = cmap(x)
			cmap = mpl.colors.ListedColormap(pal)
			return cmap
		if gray:
			cmap = get_cubehelix(start=.5, rot=1, light=1, dark=.2, hue=0)
		else:
			cmap = get_cubehelix(start=.5, rot=-.5, light=1, dark=.2)

		ax.imshow(grid, cmap=cmap, interpolation=interpolation)
		ax.set_aspect('equal')
		ax.set(**kwargs)
		ax.axis('off')
		return ax

	def show_all(self):
		"""The default representation doesn't print the entire set, this
		function does."""
		return set.__repr__(self)

	def minimal_elements(self):
		"""Return the elements of `self` which are minimal with respect to the 
		permutation pattern order.

		ME: TODO... Is there a better way?
		"""

		L = list(self)
		shortest_perms  = [L[0]]
		shortest_length = len(L[0])

		# Find the shortest permutations in the set.
		for P in L[1:]:
			if len(P) < shortest_length:
				shortest_perms = [P]
				shortest_length = len(P)
			elif len(P) == shortest_length:
				shortest_perms.append(P)

		# Remove them from the rest.
		for P in shortest_perms:
			L.remove(P)

		remaining = PermSet()

		# Keep only those which avoid all the shortest permutations.
		for P in L:
			for Q in shortest_perms:
				if Q.involved_in(P):
					break
			else:
				# If we're here, we never broke out of the for loop above!
				remaining.add(P)

		minimal = PermSet(shortest_perms)
		minimal.update(minimal_elements(remaining))
		return minimal



		B = list(self)
		B = sorted(B, key=len)
		C = B[:]
		n = len(B)
		for (i,b) in enumerate(B):
			# if i % 1 == 0:
				# print i,'/',n
			if b not in C:
				continue
			for j in range(i+1,n):
				if B[j] not in C:
					continue
				if b.involved_in(B[j]):
					C.remove(B[j])
		return PermSet(C)

	def all_syms(self):
		"""Return the PermSet of all symmetries of all permutations in `self`.

		ME: Done!
		"""
		S = set(self)
		S.update([P.reverse() for P in S])
		S.update([P.complement() for P in S])
		S.update([P.inverse() for P in S])
		return PermSet(S)

	def layer_down(self, verbose=0):
		"""Return the PermSet of those permutations which are covered by an element of `self`.

		ME: Done!
		"""
		S = PermSet()

		if verbose:
			n = len(self)
			for idx, P in enumerate(self):
				if idx % verbose == 0:
					print('\t{idx} of {n}. Now with {lenS}.'.format(idx=idx, n=n, lenS=len(S)))
				S.update(P.covers())
		else:
			for P in self:
				S.update(P.covers())

		return S

	def cover(self, verbose=0):
		"""Return the PermSet of those permutations which cover an element of `self`.

		ME: Done!
		"""
		S = PermSet()

		if verbose:
			n = len(self)
			for idx, P in enumerate(self):
				if idx % verbose == 0:
					print('\t{idx} of {n}. Now with {lenS}.'.format(idx=idx, n=n, lenS=len(S)))
				S.update(P.covered_by())
		else:
			for P in self:
				S.update(P.covered_by())

		return S

	# def upset(self, )

	def downset(self, return_class=False):
		"""
			ME: TODO!
		"""
		bottom_edge = PermSet()
		bottom_edge.update(self)

		done = PermSet(bottom_edge)
		while len(bottom_edge) > 0:
			oldsize = len(done)
			next_layer = bottom_edge.layer_down()
			done.update(next_layer)
			del bottom_edge
			bottom_edge = next_layer
			del next_layer
			newsize = len(done)
			# print '\t\tDownset currently has',newsize,'permutations, added',(newsize-oldsize),'in the last run.'
		if not return_class:
			return done
		cl = [PermSet([])]
		max_length = max([len(P) for P in done])
		for i in range(1,max_length+1):
			cl.append(PermSet([P for P in done if len(P) == i]))
		return permclass.PermClass(cl)


	def total_statistic(self, statistic):
		""" Returns the sum of the given statistic over all perms in `self`.

			ME: Done!
		"""
		return sum(statistic(p) for p in self)

	def pattern_counts(self, k):
		""" Returns a dictionary counting the occurrences of each perm of length `k` in each 
			permutation in `self`.

			ME: Done!
		"""
		C = Counter()
		for pi in self:
			for vals in itertools.combinations(pi, k):
				C[permutation.Permutation(vals)] += 1
		return C

	def stack_inverse(self):
		A = [tuple([val+1 for val in pi]) for pi in self]
		# print(A)
		n = len(A[0])
		assert all(len(pi)==n for pi in self), "Not designed to handle this, unfortunately!"
		L = [set() for _ in range(n+1)]
		L[n].update((pi, tuple(), tuple()) for pi in A)
		for k in range(n)[::-1]:
			# print("k={}".format(k))
			# print("L[{}]={}".format(k+1,L[k+1]))
			unpop_temp = set(unpop(state) for state in L[k+1])
			L[k].update(state for state in unpop_temp if state is not None)
			old = L[k]
			# print("init_old = {}".format(old))

			unpush_temp = set(unpush(state) for state in old)
			new = set(state for state in unpush_temp if state is not None)
			while new:
				L[k].update(new)
				old = new
				unpush_temp = set(unpush(state) for state in old)
				new = set(state for state in unpush_temp if state is not None)
		# 	print(unpush_temp)
		# for a in L[0]:
		# 	print(tuple(a))
		return PermSet(permutation.Permutation(state[2]) for state in L[0] if state is not None and not state[1])

def unpop(state):
	"""Given the before, stack, and after tuples, returns the (one-step) preimage.
	"""
	after,stack,before = state
	if after and after[-1] and (not stack or after[-1] < stack[-1]):
		return (after[:-1], stack+tuple([after[-1]]), before)
	else:
		return

def unpush(state):
	"""Given the before, stack, and after tuples, returns the (one-step) preimage.
	"""
	after,stack,before = state
	if stack:
		return (after, stack[:-1], tuple([stack[-1]]) + before)
	else:
		return

if __name__ == "__main__":
	a = 0








