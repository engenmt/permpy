import itertools
import os
import random
import subprocess
import sys
import time

try:
    import matplotlib.pyplot as plt

    mpl_imported = True
except ImportError:
    mpl_imported = False


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
        cycle_list = [
            "( " + " ".join(f"{x+1}" for x in cyc) + " )" for cyc in self.cycle_decomp()
        ]
        return " ".join(cycle_list)

    def order(self):
        """Return the group-theotric order of self."""
        L = [len(cycle) for cycle in self.cycle_decomp()]
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

        for p in self.right_extensions():
            S.add(p)
            S = p.all_right_extensions(max_length, l + 1, S)
        return S

    def sum_indecomposable_sequence(self):
        S = self.downset()
        return [
            len([p for p in S if len(p) == i and not p.sum_decomposable()])
            for i in range(1, max([len(p) for p in S]) + 1)
        ]

    def contains_locations(self, Q):
        locs = []
        sublocs = itertools.combinations(range(len(self)), len(Q))
        for subloc in sublocs:
            if Permutation([self[i] for i in subloc]) == Q:
                locs.append(subloc)

        return locs

    def rank_val(self, i):
        return len([j for j in range(i + 1, len(self)) if self[j] < self[i]])

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

    def rtlmax_ltrmin_decomposition(self):
        P = Permutation(self)
        layers = []
        while len(P) > 0:
            positions = sorted(list(set(P.rtlmax() + P.ltrmin())))
            layers.append(positions)
            P = Permutation([P[i] for i in range(len(P)) if i not in positions])
        return layers

    def rtlmin_ltrmax_decomposition(self):
        P = Permutation(self)
        layers = []
        while len(P) > 0:
            positions = sorted(list(set(P.rtlmax() + P.ltrmin())))
            layers.append(positions)
            P = Permutation([P[i] for i in range(len(P)) if i not in positions])
        return layers

    def dec_bonds(self):
        return [i for i in range(len(self) - 1) if self[i + 1] == self[i] - 1]

    def inc_bonds(self):
        return [i for i in range(len(self) - 1) if self[i + 1] == self[i] + 1]

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

    def plot(self, show=True, ax=None, use_mpl=True, fname=None, **kwargs):
        """Draw a matplotlib plot of the permutation. Can be used for both
        quick visualization, or to build a larger figure. Unrecognized
        arguments are passed as options to the axes object to allow for
        customization (i.e., setting a figure title, or setting labels on the
        axes). Falls back to an ascii_plot if matplotlib isn't found, or if
        use_mpl is set to False.
        """
        if not mpl_imported or not use_mpl:
            return self._ascii_plot()
        xs = [idx + Permutation._BASE for idx in range(len(self))]
        ys = [val + Permutation._BASE for val in self]
        if not ax:
            ax = plt.gca()
        ax.scatter(xs, ys, s=40, c="k")
        ax_settings = {
            "xticks": xs,
            "yticks": ys,
            "xticklabels": "",
            "yticklabels": "",
            "xlim": (min(xs) - 1, max(xs) + 1),
            "ylim": (min(ys) - 1, max(ys) + 1),
        }
        ax.set(**ax_settings)
        ax.set(**kwargs)
        ax.set_aspect("equal")
        if fname:
            fig = plt.gcf()
            fig.savefig(fname, dpi=300)
        if show:
            plt.show()
        return ax

    def _show(self):
        if sys.platform == "linux2":
            opencmd = "gnome-open"
        else:
            opencmd = "open"

        s = "\n\n".join(
            [
                r"\documentclass{standalone}",
                r"\usepackage{tikz}",
                r"\begin{document}",
                self.to_tikz(),
                r"\end{document}",
            ]
        )

        dir = f"t_{random.randint(1000, 9999)}"
        os.system(f"mkdir {dir}")
        with open(f"{dir}/t.tex", "w") as f:
            f.write(s)

        subprocess.call(
            [
                "pdflatex",
                f"-output-directory={dir}",
                f"{dir}/t.tex",
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        # os.system('pdflatex -output-directory=t_'+str(dname)+' t_'+str(dname)+'/t.tex')
        subprocess.call(
            [opencmd, f"{dir}/t.pdf"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        time.sleep(1)
        if sys.platform != "linux2":
            subprocess.call(["rm", "-r", f"{dir}/"])

    def to_tikz(self):
        """Return a pure-tikz simple plot of `self`."""
        n = len(self)
        s = "\n\t".join(
            [
                r"\begin{tikzpicture}[scale=.3,baseline=(current bounding box.center)]",
                rf"\draw[ultra thick] (1,0) -- ({n},0);",
                rf"\draw[ultra thick] (0,1) -- (0,{n});",
                r"\foreach \x in {1,...," + f"{n}" + r"} {",
                "\t" + r"\draw[thick] (\x,.09)--(\x,-.5);",
                "\t" + r"\draw[thick] (.09,\x)--(-.5,\x);",
                r"}",
            ]
            + [
                rf"\draw[fill=black] ({i+1},{e+1}) circle (5pt);"
                for (i, e) in enumerate(self)
            ]
        )

        s += "\n" + r"\end{tikzpicture}"
        return s
