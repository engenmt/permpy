from ..permutation import *


class Configuration(tuple):
    _basis = set()
    _type = ""
    _children_computed = False
    _children = set()

    def __new__(cls, t, basis=set(), childtype="?"):
        entries = list(t)
        nonzero = [x for x in entries if x != 0]
        assert len(set(nonzero)) == len(nonzero), "Nonzero entries must be distinct."
        for i in range(len(entries) - 1):
            assert (
                entries[i] != 0 or entries[i + 1] != 0
            ), "Two slots cannot be adjacent."
        standardization = [0] * len(entries)
        values = sorted(nonzero)
        for i in range(1, len(values) + 1):
            standardization[entries.index(values[i - 1])] = i
        return tuple.__new__(cls, standardization)

    def __init__(self, t, basis=set(), childtype="?"):
        self._basis = basis
        self._type = childtype
        if childtype == "?":
            t = ""
            if len(self) == 1:
                t = "f1"
            else:
                loc = self.index(max(self))
                if loc == 0:
                    if self[1] == 0:
                        t = "l1"
                    else:
                        t = "f1"
                elif loc == len(self) - 1:
                    if self[len(self) - 2] == 0:
                        t = "r" + str(self.num_slots())
                    else:
                        t = "f" + str(self.num_slots() + 1)
                else:
                    if self[loc - 1] == 0 and self[loc + 1] == 0:
                        t = "m" + str(len([k for k in self[:loc] if k == 0]))
                    elif self[loc - 1] == 0:
                        t = "r" + str(len([k for k in self[:loc] if k == 0]))
                    elif self[loc + 1] == 0:
                        t = "l" + str(len([k for k in self[:loc] if k == 0]) + 1)
                    else:
                        t = "f" + str(len([k for k in self[: loc + 1] if k == 0]) + 1)
            self._type = t

    def num_slots(self):
        return len([x for x in self if x == 0])

    def slot_locs(self):
        return [i for i in range(len(self)) if self[i] == 0]

    def children(self):
        if self._children_computed:
            return self._children
        S = set()
        max_entry = len(self) + 1
        L = list(self)
        sl = self.slot_locs()
        for i in sl:
            S.add(
                Configuration(
                    L[:i] + [max_entry] + L[i + 1 :],
                    self._basis,
                    "f" + str(sl.index(i) + 1),
                )
            )
            S.add(
                Configuration(
                    L[: i + 1] + [max_entry] + L[i + 1 :],
                    self._basis,
                    "r" + str(sl.index(i) + 1),
                )
            )
            S.add(
                Configuration(
                    L[:i] + [max_entry] + L[i:], self._basis, "l" + str(sl.index(i) + 1)
                )
            )
            S.add(
                Configuration(
                    L[: i + 1] + [max_entry] + L[i:],
                    self._basis,
                    "m" + str(sl.index(i) + 1),
                )
            )
        self._children = S
        self._children_computed = True
        return self._children

    def is_permutation(self):
        return self.num_slots() == 0

    def to_perm(self):
        if self.is_permutation():
            return Permutation(self)
        else:
            return False

    def has_valid_filling(self, C=[]):
        if self.is_permutation():
            return True
        if len(C) - 1 >= self.num_slots():
            filling_entries = C[self.num_slots()]
        else:
            filling_entries = Permutation.listall(self.num_slots())
        for P in filling_entries:
            L = list(self)
            slot_locs = self.slot_locs()
            for (p, v) in enumerate(P):
                L[slot_locs[p]] = v + len(L)
            filled_perm = Permutation(L)
            if len(C) < len(filled_perm) + 1:
                # if len(C) > 2:
                # print 'needed length',(len(filled_perm)),'but only had',(len(C)-1),'[for',self,'/',filled_perm,']'
                if filled_perm.avoids_set(self._basis):
                    return True
            else:
                if filled_perm in C[len(filled_perm)]:
                    return True
        return False

    def valid_children(self, C=[]):
        return set([c for c in self.children() if c.has_valid_filling(C)])
