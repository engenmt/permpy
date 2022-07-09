""" This is working code that may or may not be added to a class at some point."""

from __future__ import print_function
from permpy import *
from math import factorial
import itertools

from permpy.RestrictedContainer import *


def expected_basis(B):
    return PermSet([expected_basis_element(P) for P in B]).minimal_elements()


def expected_basis_element(P):
    if P[0] == len(P) - 1:
        return P
    else:
        return Permutation([len(P) + 2] + list(P))


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def check_combos(max_length, size_of_combos=0, verbose=False):
    bad_ones = 0
    so_far = 0
    permlist = []
    for i in range(2, max_length + 1):
        permlist.extend(PermSet.all(i))
    if size_of_combos == 0:
        to_check = powerset(permlist)
    else:
        to_check = itertools.combinations(permlist, size_of_combos)
    for basis in to_check:
        if len(basis) < 1:
            continue
        so_far += 1
        print("----------------------------------------------------")
        expected = expected_basis(basis)
        B = make_restricted_container_class(basis).guess_basis()
        if B == expected:
            print(f"{basis}  -->  {B}  :  ok")
        else:
            bad_ones += 1
            print(f"{basis}  -->  {B}  : WHOA!")
        if verbose and so_far % 10 == 0:
            print(f"----------------------------------------------------")
            print(f"----------------------------------------------------")
            print(f" === SO FAR: {bad_ones} BAD BASES OUT OF {so_far} === ")
            print(f"----------------------------------------------------")
    print(f"----------------------------------------------------")
    print(f"----------------------------------------------------")
    print(f"----------------------------------------------------")
    print(f"=== THERE WERE {bad_ones} BAD BASES OUT OF {so_far} ===")
    print(f"----------------------------------------------------")


def check_singleton_restricted_containers():
    for length in range(2, 6):
        print("----------------------------------------------------")
        for index in range(0, factorial(length)):
            print("----------------------------------------------------")
            P = Permutation(index, length)
            expected = expected_basis(P)
            B = make_restricted_container_class([P]).guess_basis()
            if B == expected:
                print(f"{P}  -->  {B}  :  ok")
            else:
                print(f"{P}  -->  {B}  : WHOA!")
    print("----------------------------------------------------")


def make_restricted_container_class(basis, length=7):
    basis = [Permutation(b) for b in basis]
    D = [PermSet([])]
    for i in range(1, length + 1):
        A = RestrictedContainer(basis, i)
        D.append(generate_from_restricted_container(A))
    return PermClass(D)


def generate_from_restricted_container(container):
    perms_generated = PermSet()
    queue = [container]
    while len(queue) > 0:
        machine = queue.pop(0)
        container_size = machine.container_size()
        input_size = machine.input_size()
        if container_size == 0 and input_size == 0:
            output = machine.output()
            output_permutation = Permutation(output)
            perms_generated.add(output_permutation)
        else:
            if container_size > 0:
                queue.append(machine.pop())
            for open_spot in range(0, container_size + 1):
                new_machine = machine.push(open_spot)
                if new_machine != -1:
                    queue.append(new_machine)
    return perms_generated


def color_4231(p):
    red_entries = []
    blue_entries = []

    for entry in p:
        if Permutation(blue_entries + [entry]).involves(Permutation(312)):
            red_entries.append(entry)
        else:
            blue_entries.append(entry)

    return (red_entries, blue_entries)


def label_4231(p, new_version=False):
    (red_entries, blue_entries) = color_4231(p)
    word = ""
    for (index, entry) in enumerate(p):
        if new_version and index in p.ltrmax():
            word += "D"
        elif new_version and index in p.rtlmin():
            word += "A"
        elif entry in red_entries:
            word += "A" if index in p.rtlmin() else "B"
        else:
            word += "D" if index in p.ltrmax() else "C"
    return (word, "".join([word[i] for i in p.inverse()]))


def color_1324(p):
    red_entries = []
    blue_entries = []

    for entry in p:
        if (len(blue_entries) > 0 and entry > min(blue_entries)) or Permutation(
            red_entries + [entry]
        ).involves(Permutation(132)):
            blue_entries.append(entry)
        else:
            red_entries.append(entry)

    return (red_entries, blue_entries)


def label_1324(p, new_version=False):
    (red_entries, blue_entries) = color_1324(p)
    word = ""
    for (index, entry) in enumerate(p):
        if new_version and index in p.rtlmax():
            word += "D"
        elif entry in red_entries:
            word += "A" if index in p.ltrmin() else "B"
        else:
            word += "D" if index in p.rtlmax() else "C"
    return (word, "".join([word[i] for i in p.inverse()]))


def check_pattern(word_pairs, pattern):
    n = [0, 0]
    for (x, y) in word_pairs:
        if x.find(pattern) != -1:
            n[0] += 1
        if y.find(pattern) != -1:
            n[1] += 1
    if n[0] == 0 or n[1] == 0:
        print(f"\n\n{'!'*40}\n\t\t{pattern}: ({str(n[0])}, {str(n[1])})\n\n")
    return n


def check_pattern_list(word_pairs, patterns):
    d = dict()
    for pattern in patterns:
        d[pattern] = check_pattern(word_pairs, pattern)
    return d


def allstrings(alphabet, length):
    c = []
    for i in range(length):
        c = [[x] + y for x in alphabet for y in c or [[]]]
    return ["".join(a) for a in c]


def nc_contains(w, u):
    seen = 0
    for i in range(0, len(w)):
        if w[i] == u[seen]:
            seen += 1
            if seen == len(u):
                return True
    return False


def check_nc_pattern(word_pairs, pattern):
    n = [0, 0]
    for (x, y) in word_pairs:
        if nc_contains(x, pattern):
            n[0] += 1
        if nc_contains(y, pattern):
            n[1] += 1
    return n


def check_nc_pattern_list(word_pairs, patterns):
    d = dict()
    for pattern in patterns:
        d[pattern] = check_nc_pattern(word_pairs, pattern)
    return d


# implements rsk:
class Tab(object):
    def __init__(self, P, Q):
        self.P = P
        self.Q = Q

    def __repr__(self):
        l = len(self.P[0])
        h = len(self.P)
        sP = "P = |"
        sQ = "Q = |"
        for i in range(h):
            for j in range(len(self.P[i])):
                sP += ("%2i" % self.P[i][j]) + "|"
                sQ += ("%2i" % self.Q[i][j]) + "|"
            sP += "\n    |"
            sQ += "\n    |"
        return sP[:-6] + "\n" + sQ[:-6]


def rsk(perm):
    p = [i + 1 for i in perm[::-1]]
    q = 1
    P = [[]]
    Q = [[]]

    def insert(k, r, q1):
        if len(P) < r + 1:
            P.append([])
            Q.append([])
        if not P[r] or k > P[r][-1]:
            P[r].append(k)
            Q[r].append(q1)
        else:
            n = len(P[r])
            i = n - 1
            while True:
                b = P[r][i]
                if k < b:
                    if i > 0:
                        i -= 1
                    else:
                        P[r][0] = k
                        break
                if k > b:
                    b = P[r][i + 1]
                    P[r][i + 1] = k
                    break
            insert(b, r + 1, q)

    for i in range(len(perm) - 1, -1, -1):
        insert(p.pop(), 0, q)
        q += 1
    return Tab(P, Q)


def rsk_shape(perm):
    P = rsk(perm).P
    return [len(P[i]) for i in range(0, len(P))]


def shape_contains(A, B):
    return len(A) >= len(B) and all(b <= a for a, b in zip(A, B))


def check_stat(f, L1, L2, l=8):
    C = [sorted([f(P) for P in T[l]]) for T in L1]
    D = [sorted([f(P) for P in T[l]]) for T in L2]
    return [(i, j) for i in range(len(L1)) for j in range(len(L2)) if C[i] == D[j]]


def check_stat_list(fs, L1, L2, l=8):
    C = [sorted([tuple([f(P) for f in fs]) for P in T[l]]) for T in L1]
    D = [sorted([tuple([f(P) for f in fs]) for P in T[l]]) for T in L2]
    # done = []
    # vals = []
    # for thing in C[0]:
    # if thing in done:
    # continue
    # done.append(thing)
    # vals.append(C[0].count(thing))
    # print C[0].count(thing),'\t->\t',thing
    # print max(vals)
    return [(i, j) for i in range(len(L1)) for j in range(len(L2)) if C[i] == D[j]]


def size_value_blocks(L):
    if len(L) == 0:
        return []
    L = sorted(list(L))
    T = []
    s = 0
    for i in range(len(L) - 1):
        s += 1
        if L[i] + 1 not in L:
            T.append(s)
            s = 0
    s += 1
    T.append(s)
    return sorted(T)


def rhs_blocks(P):
    right_side = P[P.index(0) :]
    rhs_sorted = sorted(right_side)
    sorted_lookup = sorted(enumerate(right_side), key=lambda i: i[1])
    rhs_blocks = []
    b_sizes = []
    this_block_min = 0
    this_block_size = 1
    for i in range(len(rhs_sorted) - 1):
        if rhs_sorted[i] + 1 in rhs_sorted:
            this_block_size += 1
        else:
            rhs_blocks.append(
                [
                    t[1]
                    for t in sorted(
                        [
                            sorted_lookup[i]
                            for i in range(
                                this_block_min, this_block_min + this_block_size
                            )
                        ],
                        key=lambda i: i[0],
                    )
                ]
            )
            this_block_min = i + 1
            b_sizes.append(this_block_size)
            this_block_size = 1
    b_sizes.append(this_block_size)
    # if this_block_size == 1:

    rhs_blocks.append(
        [
            t[1]
            for t in sorted(
                [
                    sorted_lookup[i]
                    for i in range(this_block_min, this_block_min + this_block_size)
                ],
                key=lambda i: i[0],
            )
        ]
    )
    return rhs_blocks


def size_LR_max_contig_blocks(P):
    rhs_bs = rhs_blocks(P)
    sizes = []
    for i in range(1, len(rhs_bs)):
        b = rhs_bs[i]
        P = Perm(b)
        vs = [P[i] for i in P.ltrmax()]
        v = max(vs)
        j = 1
        while v - j in vs:
            j += 1
        sizes.append(j)
    return sizes


def num_value_blocks(L):
    if len(L) == 0:
        return 0
    L = sorted(list(L))
    n = 1
    for i in range(len(L) - 1):
        if L[i] + 1 not in L:
            n += 1
    return n


def longest_init_asc(L):
    n = 1
    for i in range(1, len(L)):
        if L[i] > L[i - 1]:
            n += 1
        else:
            return n
    return n


def longest_init_desc(L):
    return longest_init_asc(Perm(L).complement())


def split_and_check_perm(P):
    (a, b) = (P[: P.index(0)], P[P.index(0) :])
    return (num_value_blocks(a), num_value_blocks(b))


def split_size_value_blocks(P):
    (a, b) = (P[: P.index(0)], P[P.index(0) :])
    return (size_value_blocks(a), size_value_blocks(b))


def up_jump_lengths(P):
    L = []
    for i in range(len(P) - 1):
        if P[i + 1] - P[i] > 0:
            L.append(P[i + 1] - P[i])
        else:
            L.append(0)
    return L


def check_stats(L1, L2, l=8, add_syms=False, tups=1):
    """Not sure what this does.

    ME: I've updated some of this, but I don't know what many of these even do.
    """
    if add_syms:
        L1 = [
            item
            for sublist in [list(PermSet(l1).all_syms()) for l1 in L1]
            for item in sublist
        ]
        L2 = [
            item
            for sublist in [list(PermSet(l2).all_syms()) for l2 in L2]
            for item in sublist
        ]
    stats = [
        ("number of descents", Perm.num_descents),
        ("position of descents", Perm.descents),
        ("number of ascents", Perm.num_ascents),
        ("position of ascents", Perm.ascents),
        ("number of ltrmin", Perm.num_ltr_min),
        ("positions of ltrmin", Perm.ltr_min),
        ("values of ltrmin", lambda P: [P[i] for i in P.ltr_min()]),
        ("number of ltrmax", Perm.num_ltr_max),
        ("positions of ltrmax", Perm.ltr_max),
        ("values of ltrmax", lambda P: [P[i] for i in P.ltr_max()]),
        ("number of rtlmin", Perm.num_rtl_min),
        ("positions of rtlmin", Perm.rtl_min),
        ("values of rtlmin", lambda P: [P[i] for i in P.rtl_min()]),
        ("number of rtlmax", Perm.num_rtl_max),
        ("positions of rtlmax", Perm.rtl_max),
        ("values of rtlmax", lambda P: [P[i] for i in P.rtl_max()]),
        ("is sum decomposable", Perm.sum_decomposable),
        ("number of sum components", lambda P: len(P.sum_decomposition())),
        ("is skew decomposable", Perm.skew_decomposable),
        ("number of skew components", lambda P: len(P.skew_decomposition())),
        (
            "number of inflations of simples",
            lambda P: P.skew_decomposable() or P.sum_decomposable(),
        ),
        ("length of simple quotient", lambda P: len(P.decomposition()[0])),
        ("number of simple permutations", Perm.is_simple),
        ("number of bonds", Perm.bonds),
        (
            "+ bonds",
            lambda P: len([i for i in range(len(P) - 1) if P[i + 1] == P[i] + 1]),
        ),
        (
            "- bonds",
            lambda P: len([i for i in range(len(P) - 1) if P[i + 1] == P[i] - 1]),
        ),
        (
            "position of + bonds",
            lambda P: [i for i in range(len(P) - 1) if P[i] + 1 == P[i + 1]],
        ),
        (
            "position of - bonds",
            lambda P: [i for i in range(len(P) - 1) if P[i] - 1 == P[i + 1]],
        ),
        ("number of bends", Perm.bends),
        (
            "position of bends",
            lambda P: [
                i
                for i in range(1, len(P) - 1)
                if (P[i - 1] > P[i] and P[i + 1] > P[i])
                or (P[i - 1] < P[i] and P[i + 1] < P[i])
            ],
        ),
        (
            "number of valleys",
            lambda P: len(
                [i for i in range(1, len(P) - 1) if P[i - 1] > P[i] and P[i + 1] > P[i]]
            ),
        ),
        (
            "position of valleys",
            lambda P: [
                i for i in range(1, len(P) - 1) if P[i - 1] > P[i] and P[i + 1] > P[i]
            ],
        ),
        (
            "number of peaks",
            lambda P: len(
                [i for i in range(1, len(P) - 1) if P[i - 1] < P[i] and P[i + 1] < P[i]]
            ),
        ),
        (
            "position of peaks",
            lambda P: [
                i for i in range(1, len(P) - 1) if P[i - 1] < P[i] and P[i + 1] < P[i]
            ],
        ),
        (
            "number of exceedances",
            lambda P: len([i for i in range(len(P)) if P[i] > i]),
        ),
        ("position of exceedances", lambda P: [i for i in range(len(P)) if P[i] > i]),
        ("value of first entry", lambda P: P[0]),
        ("value of second entry", lambda P: P[1]),
        ("value of second to last entry", lambda P: P[len(P) - 2]),
        ("value of last entry", lambda P: P[len(P) - 1]),
        ("position of 1", lambda P: [i for i in range(len(P)) if P[i] == 0]),
        ("position of 2", lambda P: [i for i in range(len(P)) if P[i] == 1]),
        ("position of n-1", lambda P: [i for i in range(len(P)) if P[i] == len(P) - 2]),
        ("position of n", lambda P: [i for i in range(len(P)) if P[i] == len(P) - 1]),
        ("major index", Perm.majorindex),
        ("longest ascending run", Perm.longestrunA),
        ("longest descending run", Perm.longestrunD),
        ("longest run", Perm.longestrun),
        ("number of inversions", Perm.inversions),
        (
            "number of top steps",
            lambda P: len(
                [
                    i
                    for i in [i for i in range(len(P) - 1) if P[i] > P[i + 1]]
                    if i in P.rtlmax() and i + 1 in P.rtlmax() and P[i] - 1 == P[i + 1]
                ]
            ),
        ),
        (
            "position of top steps",
            lambda P: [
                i
                for i in [i for i in range(len(P) - 1) if P[i] > P[i + 1]]
                if i in P.rtlmax() and i + 1 in P.rtlmax() and P[i] - 1 == P[i + 1]
            ],
        ),
        (
            "number of fixed points",
            lambda P: len([i for i in range(len(P)) if P[i] == i]),
        ),
        ("position of fixed points", lambda P: [i for i in range(len(P)) if P[i] == i]),
        ("number of disjoint cycles", Perm.num_disjoint_cycles),
        ("number of involutions", Perm.is_involution),
        ("rank encodings", Perm.rank_encoding),
        ("max rank", lambda P: max(P.rank_encoding())),
        # ('num rtlmax / ltrmin layers', Perm.num_rtlmax_ltrmin_layers),
        # ('num ltrmax / rtlmin layers', lambda P : P.reverse().num_rtlmax_ltrmin_layers()),
        # ('num rtlmax / ltrmin layers contracted', lambda P : P.contract_inc_bonds().num_rtlmax_ltrmin_layers()),
        ("occurrences of 123", lambda P: P.occurrences(Perm(123))),
        ("occurrences of 132", lambda P: P.occurrences(Perm(132))),
        ("occurrences of 213", lambda P: P.occurrences(Perm(213))),
        ("occurrences of 231", lambda P: P.occurrences(Perm(231))),
        ("occurrences of 312", lambda P: P.occurrences(Perm(312))),
        ("occurrences of 321", lambda P: P.occurrences(Perm(321))),
        ("occurrences of 2413", lambda P: P.occurrences(Perm(2413))),
        # ('split and count value-blocks', split_and_check_perm),
        # ('size_LR_max_contig_blocks', size_LR_max_contig_blocks),
    ]

    C1 = [Av(B, l) for B in L1]
    C2 = [Av(B, l) for B in L2]

    print("")
    if tups == 1:
        for (name, f) in stats:
            equivs = check_stat(f, C1, C2, l)
            print(name, ":")
            if len(equivs) == 0:
                print("\tnone\n")
                continue
            for (x, y) in equivs:
                print("\t", L1[x], "~", L2[y])
            print("")
    else:
        fs = itertools.combinations(stats, tups)
        for stat_set in fs:
            equivs = check_stat_list(
                [stat_set[i][1] for i in range(len(stat_set))], C1, C2, l
            )
            print("")
            print(
                "[", (" / ".join([stat_set[i][0] for i in range(len(stat_set))])), "]"
            )
            if len(equivs) == 0:
                print("\tnone\n")
                continue
            for (x, y) in equivs:
                print("\t", L1[x], "~", L2[y])


def kill_syms(bases):
    if len(bases) == 0:
        return bases
    bases = list(bases)
    new_bases = [PermSet(bases[0])]

    for B in bases[1:]:
        syms = [PermSet(K) for K in PermSet(B).all_syms()]
        if not any([BSym in new_bases for BSym in syms]):
            new_bases.append(PermSet(B))
    return new_bases


def kill_auts(basis_so_far, additions):
    auts = [
        lambda P: P.reverse(),
        lambda P: P.complement(),
        lambda P: P.reverse().complement(),
        lambda P: P.inverse(),
        lambda P: P.inverse().reverse(),
        lambda P: P.inverse().complement(),
        lambda P: P.inverse().reverse().complement(),
    ]

    good_auts = [a for a in auts if all([P == a(P) for P in basis_so_far])]

    keep_additions = []
    for beta in additions:
        if not any([a(beta) in keep_additions for a in good_auts]):
            keep_additions.append(beta)

    return keep_additions


def hunt_for_enumeration(enum, max_len=8, check_ahead=2, verbose=False):
    bases = [[]]
    done_bases = []
    # on_length = 1

    while len(bases) > 0:
        B = bases[0]
        Bshow = list(B)
        Bshow.sort()
        Bshow.sort(key=len)
        bases = bases[1:]
        print("Evaluating basis: ", list(Bshow), "\t(", len(bases), "more in queue )")
        C = AvClass(B, 6)
        diffs = [i for i in range(1, len(C)) if len(C[i]) != enum[i - 1]]
        if len(diffs) == 0:
            for i in range(7, max_len + 1):
                C.extend_to_length(i)
                diffs = [i for i in range(1, len(C)) if len(C[i]) != enum[i - 1]]
                if len(diffs) > 0:
                    break
        if len(diffs) == 0:
            # Now check all the way
            C.extend_to_length(max_len + check_ahead)
            if all([len(C[i]) == enum[i - 1] for i in range(1, len(C))]):
                print("\tPerfect match!")
                done_bases.append(B)
        else:
            first_diff = min(diffs)
            if len(C[first_diff]) <= enum[first_diff - 1]:
                if verbose:
                    print("\tNo chance.")
                continue
            if verbose:
                print("\tCurrent Enum:", [len(C[i]) for i in range(1, len(C))])
            if len(B) == 0:
                to_try = sorted([P for P in C[first_diff] if P.is_representative()])
            elif max([len(b) for b in B]) < first_diff:
                # try all basis elements in lex order
                to_try = sorted(list(C[first_diff]))
                if verbose:
                    print("\tPossible new basis elts:", to_try)
            else:
                # try all basis elements after last one (lex) of cur length in B
                to_try = sorted(list(C[first_diff]))
                to_try = [
                    b for b in to_try if b > max([t for t in B if len(t) == first_diff])
                ]
                if verbose:
                    print("\tPossible new basis elts:", to_try)
            if len(to_try) == 0:
                # nothing else to try of this length
                continue
            to_try = kill_auts(B, to_try)
            k = len(to_try)
            ci = 0
            for new_b in to_try:
                ci += 1
                new_B = PermSet(B).union(PermSet([new_b]))
                if verbose:
                    print(
                        "\t\tTrying basis:", sorted(list(new_B)), "\t(", ci, "/", k, ")"
                    )
                elif ci % 25 == 0:
                    print("\t(", ci, "/", k, ")")
                new_C = AvClass(new_B, first_diff + check_ahead)
                if all(
                    [
                        len(new_C[i]) >= enum[i - 1]
                        for i in range(first_diff + 1, first_diff + check_ahead + 1)
                    ]
                ):
                    if verbose:
                        print(
                            "\t\t\tOkay. New enum:",
                            [len(new_C[i]) for i in range(1, len(new_C))],
                        )

                        # print("\t\t\t\tMatches check_ahead: checking all the way.")
                        # new_C.extend_to_length(max_len+check_ahead)
                        # if all([len(new_C[i]) == enum[i-1] for i in range(1,max_len+check_ahead+1)]):
                        #   print("\t\t\t\t\tMatches all the way!")
                        #   done_bases.append(new_B)
                        #   continue
                        # else:
                        #   print("\t\t\t\tNot yet a perfect match.")
                    bases.append(new_B)
                else:
                    if verbose:
                        print("\t\t\tToo Small.")
        # Kill syms in list
        old_size = len(bases)
        bases = kill_syms(bases)
        new_size = len(bases)
        if verbose:
            print("Killed off symmetries: ", old_size, "->", new_size)

    return [list(B) for B in done_bases]


# def all_rtlmax_end(P,bound=1):
#   L = P.rtlmax()
#   return L == range(len(P)-len(L), len(P)) and len(L) >= bound

# from collections import defaultdict
# import csv

# ltd = 10

# def prof(P):
#   return tuple(tupify([f[1](P) for f in stats]))

# def tupify(L):
#   S = []
#   for p in L:
#     if isinstance(p,list):
#       S.append(tuple(p))
#     else:
#       S.append(p)
#   return tuple(S)

# def sf(P):
#   return str(P).replace(' ', '')

# def pos_of_least_val_block_inc(P):
#   L = [P.index(0)]
#   v = 1
#   while P.index(v) > P.index(v-1) and v < len(P):
#     L.append(P.index(v))
#     v += 1
#   return L

# def pos_of_greatest_val_block_inc(P):
#   L = [P.index(len(P)-1)]
#   v = len(P)-1
#   while P.index(v) > P.index(v-1):
#     L.append(P.index(v-1))
#     v -= 1
#   return L[::-1]


# stats = [
#   ('position of descents', lambda P : [i for i in range(len(P)-1) if P[i] > P[i+1]]),
#   ('positions of ltrmin', Perm.ltrmin),
#   ('positions of rtlmax', Perm.rtlmax),
#   ('position of + bonds', lambda P : [i for i in range(len(P)-1) if P[i]+1==P[i+1]]),
#   # ('least_val_block', pos_of_least_val_block_inc),
#   # ('greatest_val_block', pos_of_greatest_val_block_inc),
# ]

# av_2143 = defaultdict(set)
# av_3142 = defaultdict(set)

# C = Av([3142], ltd)
# D = Av([2143,415263], ltd)

# for P in C[ltd]:
#   if not P.skew_decomposable() and P not in D[ltd] and all_rtlmax_end(P,2) and P.num_inc_bonds() == 0:
#     av_3142[prof(P)].add(P)

# for P in D[ltd]:
#   if not P.skew_decomposable() and P not in C[ltd] and all_rtlmax_end(P,2) and P.num_inc_bonds() == 0:
#     av_2143[prof(P)].add(P)

# S = set(av_2143.keys()+av_3142.keys())

# with open('data.csv', 'wb') as cf:
#   w = csv.writer(cf)
#   for k in S:
#     w.writerow(list(k)+[' / '.join([sf(P) for P in av_2143[k]])]+[' / '.join([sf(P) for P in av_3142[k]])])
