from permpy.permutation import Permutation

# stack


def main1():
    pi = Permutation(3214765)

    iota = Permutation(12345)

    for n in range(len(pi), 10):
        print("n = {}".format(n))
        L = pi.optimizers(n)
        for tau in L:
            if tau.involves(Permutation(312)) or tau.involves(Permutation(231)):
                print("tau = {} is not layered!".format(tau))
            else:
                print([len(layer) for layer in tau.sum_decomposition()])
            if tau.involves(iota):
                print("tau = {} contains {}!".format(tau, iota))
                copies = tau.copies_of(iota)
                for copy in copies:
                    rest_of_tau = [
                        val for idx, val in enumerate(tau) if idx not in copy
                    ]
                    for sigma in L:
                        rest_of_sigma = [
                            val for idx, val in enumerate(sigma) if idx not in copy
                        ]
                        if rest_of_sigma == rest_of_tau:
                            print("tau = {}".format(tau))
                            print("sigma = {}".format(sigma))
                            print("change to = {}".format([sigma[idx] for idx in copy]))


def main2():
    qpi = Permutation(3214765)
    iota = Permutation(12345)

    a = Permutation(312)
    b = Permutation(231)

    height = 10
    L = iota.upset(height, stratified=True)
    for n in range(len(iota), height + 1):
        S = L[n]
        for tau in S:
            if tau.involves(a) or tau.involves(b):
                continue
            num_copies = len(tau.copies_of(pi))
            if num_copies == 0:
                continue
            print("{:30s} {:10d}".format("tau = {}".format(tau), num_copies))
            copies = tau.copies_of(iota)
            for copy in copies:
                print("\tcopy = {}".format(" ".join(str(val + 1) for val in copy)))
                sys.stdout.flush()

                lower_deletion = tau.delete(values=copy[1])
                lower_inflation = lower_deletion.insert(tau.index(copy[0]), copy[0] + 1)
                lower_num_copies = len(lower_inflation.copies_of(pi))

                print(
                    "{:30s} {:6d}".format(
                        "\tnew = {}".format(lower_inflation), lower_num_copies
                    ),
                    end="",
                )

                upper_deletion = tau.delete(values=copy[3])
                upper_inflation = upper_deletion.insert(tau.index(copy[4]) - 1, copy[4])
                upper_num_copies = len(upper_inflation.copies_of(pi))

                print(
                    "{:30s} {:6d}".format(
                        "\tnew = {}".format(upper_inflation), upper_num_copies
                    )
                )
                sys.stdout.flush()

                new_num_copies = max(lower_num_copies, upper_num_copies)

                if 0 < new_num_copies < num_copies:
                    break
            else:
                continue

            print("Something went wrong!")
            sys.stdout.flush()
            break

    def stack_sorted(self):
        """Return self after one deterministic stack sort."""
        before = list(self)
        stack = []  # bottom to top, i.e. stack[0] is the bottom, stack[-1] is the top.
        after = []

        while before:
            val = before.pop(0)

            while stack and val > stack[-1]:
                after.append(stack.pop(-1))

            stack.append(val)

        after += stack[::-1]
        return Permutation(after, clean=True)

    def gen_stack_sorted(self):
        """Yield self after one deterministic stack sort."""
        before = list(self)
        stack = []  # bottom to top, i.e. stack[0] is the bottom, stack[-1] is the top.

        while before:
            val = before.pop(0)

            while stack and val > stack[-1]:
                yield stack.pop(-1)

            stack.append(val)

        for val in stack[::-1]:
            yield val

    # def sort_sequence(self):
    # 	"""Return the sequence obtained by (deterministically) stack sorting self.
    # 	"""
    # 	p = Permutation(self, clean=True)
    # 	L = [self]
    # 	while not p.is_identity():
    # 		p = p.stack_sorted()
    # 		L.append(p)
    # 	return L

    # def create_tree(self):
    # 	"""Given a nonempty permutation `p`, return the associated decreasing binary plane tree.
    # 	"""
    # 	assert self
    # 	L = list(self)

    # 	max_val = max(self)
    # 	max_idx = self.index(max_val)

    # 	before = L[:max_idx]
    # 	after  = L[max_idx+1:]

    # 	T = Node(max_val) #
    # 	if len(before):
    # 		T.add_left(create_tree(before))
    # 	if len(after):
    # 		T.add_right(create_tree(after))

    # 	return T

    def inverse_stack(self):
        L = []
        for pi in Permutation.genall(len(self)):
            for idx, val in enumerate(pi.gen_stack_sorted()):
                if not self[idx] == val:
                    break
            else:
                # For loop didn't break!
                L.append(pi)
        return L

    def stack_preimage(self, k):
        S = permset.PermSet(self)
        for _ in range(k):
            S = S.inverse_stack()
        return S
