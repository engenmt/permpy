import itertools

from math import gcd


def lcm(L):
    result = 1
    for val in L:
        result *= val // gcd(result, val)
    return result


class PermutationMiscMixin:
    """Contains various functions for Permutation to inherit."""

    @classmethod
    def one_cycles(cls, n):
        """Generate those permutations of length n that consist of one cycle."""
        for pi in itertools.permutations(range(n - 1)):
            cycle = [n - 1] + list(pi)
            tau = [None for _ in range(n)]
            for idx, val in enumerate(cycle[:-1]):
                tau[val] = cycle[idx + 1]
            tau[cycle[-1]] = cycle[0]
            yield (Permutation(tau), cycle)

    def cycle_decomp(self):
        """Return the cycle decomposition of the permutation.
        Return as a list of cycles, each of which is represented as a list.

        Examples:
                >>> Permutation(53814276).cycle_decomp()
                [[4, 3, 0], [6], [7, 5, 1, 2]]

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
        """Return the cycle notation representation of the permutation."""
        stringlist = [
            "( " + " ".join([str(x + 1) for x in cyc]) + " )"
            for cyc in self.cycle_decomp()
        ]
        return " ".join(stringlist)

    def order(self):
        """Return the grou-theotric order of self."""
        L = map(len, self.cycle_decomp())
        return lcm(L)

    def children(self):
        """Return all patterns contained in self of length one less than the permutation."""
        return self.covers()

    def shrink_by_one(self):
        """Return all patterns contained in self of length one less than the permutation."""
        return self.covers()

    def all_right_extensions(self, max_length, l, S):
        if l == max_length:
            return S
        else:
            re = self.right_extensions()
        for p in re:
            S.add(p)
            S = p.all_right_extensions(max_length, l + 1, S)
        return S

    def sum_indecomposable_sequence(self):
        S = self.downset()
        return [
            len([p for p in S if len(p) == i and not p.sum_decomposable()])
            for i in range(1, max([len(p) for p in S]) + 1)
        ]

    # def sum_indec_bdd_by(self, n):
    #     l = [1]
    #     S = list(self.children())
    #     while len(S) > 0 and len(S[0]) > 0:
    #         l = [len([s for s in S if not s.sum_decomposable()])] + l
    #         if l[0] > n:
    #             return False
    #         S = list(permset.PermSet(S).layer_down())
    #     return True

    def contains_locations(self, Q):
        locs = []
        sublocs = itertools.combinations(range(len(self)), len(Q))
        for subloc in sublocs:
            if Permutation([self[i] for i in subloc]) == Q:
                locs.append(subloc)

        return locs

    def rank_val(self, i):
        return len([j for j in range(i + 1, len(self)) if self[j] < self[i]])

    def rank_encoding(self):
        return [self.rank_val(i) for i in range(len(self))]

    def num_rtlmax_ltrmin_layers(self):
        return len(self.rtlmax_ltrmin_decomposition())

    def rtlmax_ltrmin_decomposition(self):
        P = Permutation(self)
        num_layers = 0
        layers = []
        while len(P) > 0:
            num_layers += 1
            positions = sorted(list(set(P.rtlmax() + P.ltrmin())))
            layers.append(positions)
            P = Permutation([P[i] for i in range(len(P)) if i not in positions])
        return layers

    def contains_locations(self, Q):
        locs = []
        sublocs = itertools.combinations(range(len(self)), len(Q))
        for subloc in sublocs:
            if Permutation([self[i] for i in subloc]) == Q:
                locs.append(subloc)

        return locs

    def rank_val(self, i):
        return len([j for j in range(i + 1, len(self)) if self[j] < self[i]])

    def rank_encoding(self):
        return [self.rank_val(i) for i in range(len(self))]

    def num_rtlmax_ltrmin_layers(self):
        return len(self.rtlmax_ltrmin_decomposition())

    def rtlmax_ltrmin_decomposition(self):
        P = Permutation(self)
        num_layers = 0
        layers = []
        while len(P) > 0:
            num_layers += 1
            positions = sorted(list(set(P.rtlmax() + P.ltrmin())))
            layers.append(positions)
            P = Permutation([P[i] for i in range(len(P)) if i not in positions])
        return layers

    def num_inc_bonds(self):
        return len([i for i in range(len(self) - 1) if self[i + 1] == self[i] + 1])

    def num_dec_bonds(self):
        return len([i for i in range(len(self) - 1) if self[i + 1] == self[i] - 1])

    def num_bonds(self):
        return len(
            [
                i
                for i in range(len(self) - 1)
                if self[i + 1] == self[i] + 1 or self[i + 1] == self[i] - 1
            ]
        )

    def contract_inc_bonds(self):
        P = Permutation(self)
        while P.num_inc_bonds() > 0:
            for i in range(0, len(P) - 1):
                if P[i + 1] == P[i] + 1:
                    P = Permutation(P[:i] + P[i + 1 :])
                    break
        return P

    def contract_dec_bonds(self):
        P = Permutation(self)
        while P.num_dec_bonds() > 0:
            for i in range(0, len(P) - 1):
                if P[i + 1] == P[i] - 1:
                    P = Permutation(P[:i] + P[i + 1 :])
                    break
        return P

    def contract_bonds(self):
        P = Permutation(self)
        while P.num_bonds() > 0:
            for i in range(0, len(P) - 1):
                if P[i + 1] == P[i] + 1 or P[i + 1] == P[i] - 1:
                    P = Permutation(P[:i] + P[i + 1 :])
                    break
        return P
