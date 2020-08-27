import logging

from collections import Counter, defaultdict
from itertools import combinations_with_replacement as cwr

def pretty_out(pi, k, vert_line = True, by_lines=False, width = 2):
	"""Return a nice string to visualize `pi`.
	If `by_lines == True`, then will return the list of strings by line,
	in case you want to append some stuff to each line.
	"""
	print(f"{pi}, {k}")
	lines = []
	n = len(pi)
	
	width = max(len(str(n+1)), width) # This is the width of each value.

	blank = " " * width
	for val in range(n-1,-1,-1): # val from n-1 to 0
		idx = pi.index(val)
		line = blank * idx + f"{val+1:>{width}}" + blank * (n-idx-1)
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

def gen_interval_divisions(m, k, shift = 0, increasing=True):
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
	direction = +1 if increasing else -1

	logging.debug(f"calling gid: {m}, {k}, {shift}, {increasing}")
	L = range(shift, shift+m)[::direction]
	logging.debug(L)
	for c in cwr(range(m+1),k-1):
		# For each choice of divisions...
	
		c = (0,) + c + (m,)

		yield [tuple(val for val in L[c[i]:c[i+1]]) for i in range(k)]

def all_vertical_extensions(pi, m, k, increasing=True):
	"""Given a permutation `pi`, generate all ways to add a monotone sequence  
	of length `m` above its right `k` points.
	
	Example:
		Here is permutation p, drawn with k = 2.
		+-------+-----+
		|       | x   |
		| x     |     |
		|     x |     |
		|       |   x |
		|   x   |     |
		+-------+-----+
		all_vertical_extensions(p, 3, 2, True) would generate each way of 
		inserting 3 points in increasing position that lie above every point in p 
		and, together with the k = 2 rightmost points of p, occupy the 3+2 rightmost 
		positions of the new permutation. Here is one example:
		        +-----------+
		        |         x |
		        |     x     |
		        | x         |
		+-------+-----------+
		|       |   x       |
		| x     |           |
		|     x |           |
		|       |       x   |
		|   x   |           |
		+-------+-----------+
	"""
	n = len(pi)

	# Split pi on its last k elements.
	if k == 0:
		prefix = pi
		suffix = ()
	else:
		prefix = pi[:-k]
		suffix = pi[-k:]

	logging.debug("vertically extending (pi, m, k) = ({pi}, {m}, {k})")
	logging.debug("prefix = {prefix}")
	logging.debug("suffix = {suffix}")

	for uppers in gen_interval_divisions(m,k+1,shift = n):
		# assert len(uppers) == k+1
		# assert len(sum(uppers,())) == m
		new_suffix = sum([uppers[i] + (suffix[i],) for i in range(k)], ()) + uppers[-1]

		logging.debug(f"uppers = {uppers}, new_suffix = {new_suffix}")
		logging.debug(f"yielding {prefix + new_suffix}")

		yield prefix + new_suffix

def all_horizontal_extensions(pi, m, k):
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

	logging.debug(f"horizontally extending (pi, m, k) = ({pi}, {m}, {k})")
	logging.debug("prefix = {prefix}")
	logging.debug("suffix = {suffix}")

	for uppers in gen_interval_divisions(m, k+1, shift=n, reverse=True):
		new_suffix = sum([uppers[i] + (suffix[i],) for i in range(k)], ()) + uppers[-1]
		
		logging.debug(f"uppers = {uppers}, new_suffix = {new_suffix}")
		logging.debug(f"yielding {prefix + new_suffix}")
		logging.debug(f"yielding the inverse of {prefix + new_suffix}")

		yield inverse(prefix + new_suffix)

def inverse(pi):
	return tuple(pi.index(val) for val in range(len(pi)))

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


















