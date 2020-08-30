import logging

from collections import Counter, defaultdict
from itertools import combinations_with_replacement as cwr

logging.basicConfig(level=10)

def pretty_out(pi, k, vert_line = True, by_lines=False, width = 2):
	"""Return a nice string to visualize `pi`.
	If `by_lines == True`, then will return the list of strings by line,
	in case you want to append some stuff to each line.
	"""
	logging.debug(f"{pi}, {k}")
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
	logging.debug(f"Called gen_interval_divisions({m}, {k}, shift={shift}, increasing={increasing})")
	max_val = shift+m
	if increasing:
		if k == 1:
			yield (tuple(range(shift+1, max_val+1)),)
		else:
			for last_interval_size in range(0,m+1):
				last_interval = (tuple(range(max_val+1-last_interval_size,max_val+1)),)
				for division in gen_interval_divisions(m-last_interval_size, k-1, shift=shift, increasing=increasing):
					yield division + last_interval
	else:
		if k == 1:
			yield (tuple(range(max_val, shift, -1)),)
		else:
			for last_interval_size in range(0,m+1):
				first_interval = (tuple(range(max_val,max_val-last_interval_size,-1)),)
				for division in gen_interval_divisions(m-last_interval_size, k-1, shift=shift, increasing=increasing):
					yield first_interval + division

def all_vertical_extensions(pi, m, k, increasing=True):
	"""Given a permutation `pi`, generate all ways to add a monotone sequence  
	of length `m` above its right `k` points.
	
	Example:
		Here is permutation p, drawn with k = 2.
		 +-------+-----+
		4|       | x   |
		3| x     |     |
		2|     x |     |
		1|       |   x |
		0|   x   |     |
		 +-------+-----+
		all_vertical_extensions(p, 3, 2, True) would generate each way of 
		inserting 3 points in increasing position that lie above every point in p 
		and, together with the k = 2 rightmost points of p, occupy the 3+2 rightmost 
		positions of the new permutation. Here is one example:
		         +-----------+
		7        |         x |
		6        |     x     |
		5        | x         |
		 +-------+-----------+
		4|       |   x       |
		3| x     |           |
		2|     x |           |
		1|       |       x   |
		0|   x   |           |
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

	logging.debug(f"vertically extending (pi, m, k) = ({pi}, {m}, {k})")
	logging.debug(f"\tprefix = {prefix}")
	logging.debug(f"\tsuffix = {suffix}")

	for uppers in gen_interval_divisions(m, k+1, shift=n-1, increasing=increasing):
		# assert len(uppers) == k+1
		# assert len(sum(uppers,())) == m
		new_suffix = sum((uppers[i] + (pt,) for i, pt in enumerate(suffix)), ()) + uppers[-1]

		logging.debug(f"\tuppers = {uppers}, new_suffix = {new_suffix}")
		logging.debug(f"\tyielding {prefix + new_suffix}")

		yield prefix + new_suffix

def all_horizontal_extensions(pi, m, k, increasing=True):
	"""Given a permutation `pi`, generate all ways to add a monotone sequence 
	of length `m` to the right of its upper `k` points.
		
	Example:
		Here is permutation p, drawn with k = 2.
		+-----------+
		|       x   |
		| x         |
		+-----------+
		|     x     |
		|         x |
		|   x       |
		+-----------+
		all_horizontal_extensions(p, 3, 2, True) would generate each way of 
		inserting 3 points in increasing position that lie east every point in p 
		and, together with the k = 2 uppermost points of p, occupy the 3+2 uppermost 
		positions of the new permutation. Here is one example:
		+-----------+-------+
		|           |     x |
		|       x   |       |
		|           |   x   |
		| x         |       |
		|           | x     |
		+-----------+-------+
		|     x     |
		|         x |
		|   x       |
		+-----------+
	"""
	logging.debug(f"horizontally extending (pi, m, k) = ({pi}, {m}, {k})")
	tau = inverse(pi)
	for tau_ext in all_vertical_extensions(tau, m, k, increasing=increasing):
		yield inverse(tau_ext)

def inverse(pi):
	return tuple(pi.index(val) for val in range(len(pi)))
	
def one_cell(max_len, increasing):
	if increasing:
		return {tuple(range(m)): m for m in range(max_len+1)}
	else:
		return {tuple(range(m-1,-1,-1)): m for m in range(max_len+1)}

def add_horizontal_cell(C, max_len, increasing):
	D = defaultdict(int)
	for pi, k in C.items():
		logging.debug(f"\tOld perm: {pi}, {k}.")
		if pi not in D:
			D[pi] = 0
		
		if not k:
			continue
		
		for num_pts in range(1, max_len-len(pi)+1):			
			for tau in all_horizontal_extensions(pi, num_pts, k, increasing=increasing):
				logging.debug(f"\t\tNew perm: {tau}, {num_pts}")
				D[tau] = max(D[tau], num_pts)
	
	return D

def add_vertical_cell(C, max_len, increasing):
	D = defaultdict(int)
	for pi, k in C.items():
		logging.debug(f"\tOld perm: {pi}, {k}.")
		if pi not in D:
			D[pi] = 0
		
		if not k:
			continue
		
		for num_pts in range(1, max_len-len(pi)+1):			
			for tau in all_vertical_extensions(pi, num_pts, k, increasing=increasing):
				logging.debug(f"\t\tNew perm: {tau}, {num_pts}")
				D[tau] = max(D[tau], num_pts)
	
	return D

def staircase(n, cell_seq):
	cell_seq = [val == 1 for val in cell_seq]
	
	C = one_cell(n, increasing=cell_seq[0])
	by_length = [set() for _ in range(n+1)]
	for pi, v in C.items():
		by_length[len(pi)].add((pi, v))
	
	logging.info(f" 1 cell,  {[len(S) for S in by_length]}")
	for S in by_length:
		logging.debug(f"\t{sorted(S)}")
	
	for cell_idx, cell_dir in enumerate(cell_seq[1:]):
		# cell_idx = 0 corresponds to the second cell and so on.
		
		if cell_idx % 2:
			# Odd cell, add vertically
			C = add_vertical_cell(C, max_len=n, increasing=cell_dir)
		else:
			# Even cell, add horizontally
			C = add_horizontal_cell(C, max_len=n, increasing=cell_dir)
		by_length = [set() for _ in range(n+1)]
		for pi, v in C.items():
			by_length[len(pi)].add((pi, v))
		
		logging.info(f"{cell_idx+2:>2} cells, {[len(S) for S in by_length]}")
		for S in by_length:
			logging.debug(f"\t{sorted(S)}")

def increasing_staircase(n):
	increasing_seq = [1 for _ in range(n//2+1)]
	return staircase(n, increasing_seq)

def decreasing_staircase(n):
	decreasing_seq = [-1 for _ in range(n)]
	return staircase(n, decreasing_seq)

if __name__ == "__main__":
	n = 4
	S = decreasing_staircase(n)
	
# 	print(list(all_vertical_extensions((0,), 2, 1, increasing=True)))
	
