from ..permutation import Permutation
from ..avclass import AvClass
from ..permset import PermSet
from .configuration import Configuration
import sympy


class InsertionScheme:
    _tree = {}
    _basis = []
    _max_basis_length = 0
    _automaton = {}
    _root = None
    _configs_checked = set()
    _automaton_ready = False
    _has_inssch = False
    _reductions = {}
    _class = []

    def __init__(self, basis, quiet=False):
        self._basis = basis
        self._max_basis_length = max([len(b) for b in self._basis])
        self._class = AvClass(basis, 1)
        self._configs_checked = set()
        self._automaton_ready = False
        self._root = Configuration((0,), self._basis, "m")
        self._perm = Configuration((1,), self._basis, "f")
        self._tree = {self._root: self._root.children()}
        self._automaton = {self._root: list(self._root.children())}
        self._has_inssch = False
        self._reductions = {}
        self._automaton_ready = False
        if self.has_inssch():
            self._has_inssch = True
        else:
            syms = PermSet(self._basis).all_syms()
            for S in syms:
                self._basis = S
                if self.has_inssch():
                    self._has_inssch = True
                    if not quiet:
                        print(
                            "Av({}) does not have an insertion scheme, but its symmetry Av({}) does.".format(
                                basis, S
                            )
                        )
                    break
            if not self._has_inssch:
                if not quiet:
                    print(
                        "Neither this class nor its symmetries has an insertion scheme."
                    )

    def has_inssch(self):
        types = [
            PermSet([Permutation(321), Permutation(2143), Permutation(2413)]),
            PermSet([Permutation(123), Permutation(3142), Permutation(3412)]),
            PermSet([Permutation(132), Permutation(312)]),
            PermSet([Permutation(213), Permutation(231)]),
        ]
        return all(
            [any([all([b.avoids(p) for p in B]) for b in self._basis]) for B in types]
        )

    def follow_reduce(self, config):
        while config in self._reductions.keys():
            config = self._reductions[config]
        return config

    def build_rules(self, verbose=True, make_class=False, class_bound=100):
        configs_to_check = [Configuration((0,), self._basis)]
        while len(configs_to_check) > 0:
            if verbose:
                # print '\tstates to check:',len(configs_to_check),',   nodes in tree:',len(self._tree.keys()),',   states:', len(self._automaton)
                s_to_print = (
                    "\tstates to check: {}, nodes in tree: {}, states: {}".format(
                        len(configs_to_check),
                        len(self._tree.keys()),
                        len(self._automaton),
                    )
                )
                print(s_to_print)
            configs_to_check.sort(key=len)
            current_config = configs_to_check[0]
            if verbose:
                print("\nChecking: ", current_config)
            self._configs_checked.add(current_config)

            if current_config.is_permutation():
                if verbose:
                    print(current_config, " is not reducible")
                configs_to_check.remove(current_config)
                continue

            length_to_check = self._max_basis_length + current_config.num_slots() - 2

            reducible = False
            order_to_check = sorted(
                range(len(current_config)),
                key=lambda i: current_config[i],
                reverse=True,
            )
            checked_already = list()
            for i in order_to_check:
                if current_config[i] == 0 or (
                    i > 0
                    and i < len(current_config) - 1
                    and current_config[i - 1] == 0
                    and current_config[i + 1] == 0
                ):
                    continue
                second_config = Configuration(
                    current_config[:i] + current_config[i + 1 :], self._basis, "?"
                )
                if second_config in checked_already:
                    continue
                checked_already.append(second_config)
                if verbose:
                    # print 'Checking isomorphism (depth='+str(length_to_check)+'):',current_config, '<->', second_config
                    print(
                        "Checking isomorphism (depth={}): {} <-> {}".format(
                            length_to_check, current_config, second_config
                        )
                    )
                if self.check_isomorphism(
                    current_config,
                    second_config,
                    length_to_check,
                    make_class,
                    class_bound,
                ):
                    self._reductions[current_config] = second_config
                    if verbose:
                        print(current_config, " is reducible to ", second_config)
                    if current_config in self._automaton.keys():
                        del self._automaton[current_config]
                    for config in self._automaton.keys():
                        if current_config in self._automaton[config]:
                            self._automaton[config].remove(current_config)
                            self._automaton[config].append(
                                self.follow_reduce(second_config)
                            )
                    configs_to_check.append(second_config)
                    reducible = True
                    break
            if not reducible:
                if verbose:
                    print(current_config, "is not reducible")
                if (
                    make_class
                    and len(self._class) < len(current_config) + 2
                    and len(current_config) + 2 <= class_bound
                ):
                    if verbose:
                        # print '\t\t\tExtending class length from ', (len(self._class) - 1), 'to', (len(current_config) + 2), 'for', current_config, '(!)'
                        print(
                            "\t\t\tExtending class length from {} to {} for {} (!)".format(
                                len(self._class - 1),
                                len(current_config) + 2,
                                current_config,
                            )
                        )
                    self._class.extend_to_length(len(current_config) + 2)
                    if verbose:
                        print("\t\t\t\tDone!")

                # print '3'
                TTT = current_config.valid_children(self._class)
                self._automaton[current_config] = list(TTT)
                configs_to_check.extend(TTT)

            configs_to_check.remove(current_config)
            configs_to_check = list(
                set(configs_to_check).difference(self._configs_checked)
            )

        self.standardize_perms()
        self._automaton_ready = True

    def standardize_perms(self):
        for config in self._automaton.keys():
            for c in self._automaton[config]:
                if c.is_permutation():
                    self._automaton[config].remove(c)
                    self._automaton[config].append(self._perm)

    def check_isomorphism(self, c1, c2, depth, make_class=False, class_bound=100):
        # print '\t\tCI:',depth,c1,c2,len(self._tree.keys())
        if depth == 0 or c1.is_permutation() or c2.is_permutation():
            if c1.is_permutation() != c2.is_permutation():
                pass
                # print c1,c2
            return c1.is_permutation() == c2.is_permutation()
        if c1 not in self._tree.keys():
            if (
                make_class
                and len(self._class) < len(c1) + 3
                and len(c1) + 2 <= class_bound
            ):
                print(
                    "\t\t\tExtending class length from ",
                    (len(self._class) - 1),
                    "to",
                    (len(c1) + 2),
                    "for",
                    c1,
                    "(@)",
                )
                self._class.extend_to_length(len(c1) + 2)
                print("\t\t\t\tDone!")
            # print '1',c1,len(c1)
            self._tree[c1] = c1.valid_children(self._class)
        if c2 not in self._tree.keys():
            if (
                make_class
                and len(self._class) < len(c2) + 3
                and len(c2) + 2 <= class_bound
            ):
                print(
                    "\t\t\tExtending class length from ",
                    (len(self._class) - 1),
                    "to",
                    (len(c2) + 2),
                    "for",
                    c2,
                    "(#)",
                )
                self._class.extend_to_length(len(c2) + 2)
                print("\t\t\t\tDone!")
            # print '2'
            self._tree[c2] = c2.valid_children(self._class)
        c1kids = {c._type: c for c in self._tree[c1]}
        c2kids = {c._type: c for c in self._tree[c2]}
        if set(c1kids.keys()) != set(c2kids.keys()):
            # print c1,c2,c1kids,c2kids
            return False
        # if random.randint(1,1000) == 535:
        # print '\tnum states:',len(self._tree), '\tdepth:',depth
        return all(
            [
                self.check_isomorphism(
                    c1kids[i], c2kids[i], depth - 1, make_class, class_bound
                )
                for i in c1kids.keys()
            ]
        )

    def gf(self, verbose=False, show_series=False):
        if not self._automaton_ready:
            self.build_rules(verbose=verbose)
        states = self._automaton.keys()
        states.append(self._perm)

        x = sympy.Symbol("x")
        transitions = []
        for s1 in states:
            t = []
            for s2 in states:
                if s1 in self._automaton.keys() and s2 in self._automaton[s1]:
                    t.append(self._automaton[s1].count(s2) * x)
                else:
                    t.append(0)
            transitions.append(t)
        M = (sympy.eye(len(states)) - sympy.Matrix(transitions)) ** -1
        f = (1 + M[states.index(self._root), states.index(self._perm)]).factor()
        if show_series:
            print(f.series(x, 0, 11))
        return f
