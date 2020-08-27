import types

def av_test(p):
	"""Return a function that is True if the input avoids p.
	"""
	return lambda q: p not in q

def copy_func(f, name=None):
    """Return a function with same code, globals, defaults, closure, and 
    name (or provide a new name).
    """
    fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
        f.__defaults__, f.__closure__)
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__) 
    return fn

def gen_compositions(n, k=0):
	"""Generate all compositions (as lists) of `n` into `k` parts.
	If `k == 0`, then generate all compositions of `n`.
	"""
	assert n >= k, f"Need weight to be more than length: {n} > {k}"

	if k == 0:
		for i in range(1,n+1):
			for c in gen_compositions(n,i):
				yield c
	else:
		if k == 1:
			yield [n]
		elif n == k:
			yield [1]*n
		else:
			for i in range(1,n-k+2):
				for c in gen_compositions(n-i,k-1):
					yield c+[i]

def gen_weak_compositions(n, k):
	"""Generate all weak compositions (as lists) of `n` into `k` parts.
	"""
	for c in gen_compositions(n+k,k):
		yield [part-1 for part in c]


if __name__ == "__main__":
	pass