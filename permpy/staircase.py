from __future__ import print_function

from collections import Counter, defaultdict
from itertools import combinations_with_replacement as cwr

def pretty_out(pi, k, vert_line = True, by_lines=False, width = 2):
	"""Return a nice string to visualize `pi`.
	If `by_lines == True`, then will return the list of strings by line,
	in case you want to append some stuff to each line.
	"""
	print(pi, k)
	lines = []
	n = len(pi)
	
	max_width = len(str(n+1)) # This is the width of each value.
	if max_width > width:
		width = max_width

	blank = " " * width
	for val in range(n)[::-1]:
		idx = pi.index(val)
		line = (blank * (idx) \
				+ str(val+1).rjust(width)
				+ blank * (n-idx-1))
		lines.append(line)

	if vert_line:
		if k == 0:
			for idx in range(n):
				lines[idx] += " |"
		else:
			for idx in range(n):
				lines[idx] = lines[idx][:-width*k] + " |" + lines[idx][-width*k:]
	else:
		lines.insert(k, "-"*(width*n))

	if by_lines:
		return lines
	else:
		return "\n".join(lines)

def gen_compositions(n, k=0):
	"""Generate all compositions (as lists) of `n` into `k` parts.
	If `k == 0`, then generate all compositions of `n`.
	"""
	assert n >= k, "Need weight to be more than length: {} > {}".format(n, k)

	if k == 0:
		for i in xrange(1,n+1):
			for c in compositions(n,i):
				yield c
	else:
		if k == 1:
			yield [n]
		elif n == k:
			yield [1]*n
		else:
			for i in xrange(1,n-k+2):
				for c in compositions(n-i,k-1):
					yield c+[i]

def gen_weak_compositions(n, k):
	"""Generate all weak compositions (as lists) of `n` into `k` parts.
	"""
	for c in compositions(n+k,k):
		yield [part-1 for part in c]

def gen_interval_divisions(m, k, shift = 0, reverse=False):
	"""Generate all ways of splitting the interval `[1, m]` shifted up by `shift` into `k` pieces.

	Example:
		>>> list(gen_interval_divisions(4, 2))
		[[ ()          , (0, 1, 2, 3) ], 
		 [ (0,)        ,    (1, 2, 3) ], 
		 [ (0, 1)      ,       (2, 3) ], 
		 [ (0, 1, 2)   ,          (3,)], 
		 [ (0, 1, 2, 3),            ()]
		]

	"""
	if reverse:
		direction = -1
	else:
		direction = +1

	# print("calling gid: {}, {}, {}, {}".format(m, k, shift, reverse))
	L = range(shift, shift+m)[::direction]
	# print(L)
	for c in cwr(range(m+1),k-1):
		# For each choice of divisions...
		
		# print("c = {} ->".format(c), end = " ")
		c = (0,) + c + (m,)
		# print("d = {} ->".format([c[i+1]-c[i] for i in range(k)]), end = " ")

		yield [tuple(val for val in L[c[i]:c[i+1]]) for i in range(k)]

def all_vertical_extensions(pi, m, k, verbose = False):
	"""Given a permutation `pi`, generate all ways to add an increasing sequence 
	of length `m` above its right `k` points.
	"""
	n = len(pi)

	# Split pi on its last k elements.
	if k == 0:
		prefix = pi
		suffix = ()
	else:
		prefix = pi[:-k]
		suffix = pi[-k:]

	if verbose:
		print("vertically extending (pi, m, k) = {}".format((pi,m,k)))

		print("prefix = {}".format(prefix))
		print("suffix = {}".format(suffix))

	for uppers in gen_interval_divisions(m,k+1,shift = n):
		# assert len(uppers) == k+1
		# assert len(sum(uppers,())) == m
		new_suffix = sum([uppers[i] + (suffix[i],) for i in range(k)], ()) + uppers[-1]

		if verbose:
			print("uppers = {:20}, new_suffix = {:20}".format(uppers,new_suffix))
			print("yielding {}".format(prefix + new_suffix))

		yield prefix + new_suffix

def all_horizontal_extensions(pi, m, k, verbose=False):
	"""Given a permutation `pi`, generate all ways to add an decreasing sequence 
	of length `m` to the right of its upper `k` points.
	"""

	tau = inverse(pi)
	n = len(tau)

	if k == 0:
		prefix = tau
		suffix = ()
	else:
		prefix = tau[:-k]
		suffix = tau[-k:]

	if verbose:
		print("horizontally extending (pi, m, k) = {}".format((pi,m,k)))

		print("prefix = {}".format(prefix))
		print("suffix = {}".format(suffix))

	for uppers in gen_interval_divisions(m,k+1,shift = n,reverse=True):
		new_suffix = sum([uppers[i] + (suffix[i],) for i in range(k)], ()) + uppers[-1]
		
		if verbose:
			print("uppers = {:20}, new_suffix = {:20}".format(uppers,new_suffix))
			print("yielding the inverse of {}".format(prefix + new_suffix))

		yield inverse(prefix + new_suffix)

def inverse(pi):
	# print("taking inverse of pi = {}".format(pi))
	q = tuple(pi.index(val) for val in range(len(pi)))
	return q

def first_two_cells(n):
	"""Return the set of initial configurations of points in the first two cells.
	"""

	initial = ((), 0)
	R = set([initial]) # The set containing the empty tuple.

	S = set()

	for pi, k in R:
		for m in range(0,n+1):
			S.update((tau, m) for tau in all_vertical_extensions(pi, m, k))
			# S.update((pi, k) for pi in all_horizontal_extensions(initial, k, 0))

	# E = defaultdict(list)
	# for tau, k in S:
	# 	E[len(tau)].append((tau, k))

	# for idx, val in sorted(E.items()):
	# 	for perm in val:
	# 		print(pretty_out(*perm, vert_line=False))
	# 		print("="*10)

	T = set()

	for pi, k in S:
		if k == 0 and len(pi) != 0:
			T.add((pi, 0))
		else:
			for m in range(0,n-len(pi)+1):
				T.update((tau, m) for tau in all_horizontal_extensions(pi, m, k))

	# E = defaultdict(list)
	# for tau, m in T:
	# 	E[len(tau)].append((tau, m))

	# for idx, val in sorted(E.items()):
	# 	for perm in val:
	# 		print(pretty_out(*perm, vert_line=True))
	# 		print("="*10)

	return T

def add_two_cells(R, n, verbose=False):
	# if verbose:
	# 	print("Adding two cells!")
	# 	print("Current state of affairs: ")


	S = set()
	for pi, k in R:
		S.add((pi, 0))	
		for m in range(1,n-len(pi)+1):
			S.update((tau, m) for tau in all_vertical_extensions(pi, m, k))

	# E = defaultdict(list)
	# for tau, k in S:
	# 	E[len(tau)].append((tau, k))

	# for idx, val in sorted(E.items()):
	# 	for perm in val:
	# 		print(pretty_out(*perm, vert_line=False))
	# 		print("="*10)

	T = set()
	for pi, k in S:
		T.add((pi, 0))
		for m in range(1,n-len(pi)+1):
			T.update((tau, m) for tau in all_horizontal_extensions(pi, m, k))

	return T

if __name__ == "__main__":

	for pi in all_vertical_extensions((0, 1), 1, 0):
		print(pretty_out(pi, 1))
		print("-"*6)

	# n = 12
	# A = first_two_cells(n)
	# # print("before running, A = ")

	# for total_cells in range(2,n+6,2):

	# 	S = set()
	# 	D = Counter()
	# 	E = defaultdict(list)

	# 	for tau, m in A:
	# 		S.add(tau)
	# 		E[len(tau)].append((tau, m))
	# 		D[len(tau)] += 1

	# 	# for idx, val in sorted(E.items()):
	# 	# 	for perm in sorted(val):
	# 	# 		print(pretty_out(*perm))
	# 	# 		print("-"*10)

	# 	C = Counter()

	# 	# print("cells = {:2},".format(total_cells), end = "")

	# 	# print(S)

	# 	for s in S:
	# 		C[len(s)] += 1

	# 	for idx in [total_cells-4, total_cells-3]:
	# 		if 0 <= idx <= n:
	# 			print("cells = {:2}, n = {:2}, grids = {:8}, perms = {:8}".format(total_cells, idx, D[idx], C[idx]))

	# 	A = add_two_cells(A, n)
	# 	# print(A)

	# # print(S)


















