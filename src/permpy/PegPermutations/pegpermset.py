import time
import gc

from itertools import combinations_with_replacement

from sympy.core.cache import clear_cache

from .pegpermutation import PegPermutation
from .vectorset import VectorSet
from ..permset import PermSet


class PegPermSet(PermSet):

    # ===== Static Methods to build rearrangement generators =====#

    @staticmethod
    def sortable_by_block_transposition(d):
        assert d == int(d), "must be given an integer"
        assert d >= 0, "must be given a nonnegative integer"
        start = PegPermutation(1, "+")
        sortable = PegPermSet(start)
        for i in range(0, d):
            copy = set(sortable)
            for P in copy:
                n = len(P)
                for indices in combinations_with_replacement(range(0, n), 3):
                    PP = P.split(list(indices))

                    entries = (
                        PP[: indices[0] + 1]
                        + PP[indices[1] + 2 : indices[2] + 3]
                        + PP[indices[0] + 1 : indices[1] + 2]
                        + PP[indices[2] + 3 :]
                    )
                    signs = (
                        PP.signs[: indices[0]]
                        + PP.signs[indices[1] + 1 : indices[2] + 2]
                        + PP.signs[indices[0] : indices[1] + 1]
                        + PP.signs[indices[2] + 2 :]
                    )

                    sortable.add(PegPermutation(entries, signs))
        return sortable

    @staticmethod
    def sortable_by_prefix_block_transposition(d):
        assert d == int(d), "must be given an integer"
        assert d >= 0, "must be given a nonnegative integer"
        start = PegPermutation(1, "+")
        sortable = PegPermSet(start)
        for i in range(0, d):
            copy = set(sortable)
            for P in copy:
                n = len(P)
                for indices in combinations_with_replacement(range(0, n), 2):
                    PP = P.split(list(indices))

                    entries = (
                        PP[indices[0] + 1 : indices[1] + 2]
                        + PP[: indices[0] + 1]
                        + PP[indices[1] + 2 :]
                    )
                    signs = (
                        PP.signs[indices[0] + 1 : indices[1] + 2]
                        + PP.signs[: indices[0] + 1]
                        + PP.signs[indices[1] + 2 :]
                    )

                    sortable.add(PegPermutation(entries, signs))
        return sortable

    @staticmethod
    def flip_signs(s):
        for i in range(0, len(s)):
            if s[i] == "-":
                s[i] = "+"
            elif s[i] == "+":
                s[i] = "-"
        return s

    @staticmethod
    def sortable_by_reversal(d):
        assert d == int(d), "must be given an integer"
        assert d >= 0, "must be given a nonnegative integer"
        start = PegPermutation(1, "+")
        sortable = PegPermSet([start])
        for i in range(0, d):
            copy = set(sortable)
            for P in copy:
                n = len(P)
                for indices in combinations_with_replacement(range(0, n), 2):
                    PP = P.split(list(indices))

                    entries = (
                        PP[: indices[0] + 1]
                        + PP[indices[0] + 1 : indices[1] + 2][::-1]
                        + PP[indices[1] + 2 :]
                    )
                    signs = (
                        PP.signs[: indices[0] + 1]
                        + PegPermSet.flip_signs(
                            PP.signs[indices[0] + 1 : indices[1] + 2][::-1]
                        )
                        + PP.signs[indices[1] + 2 :]
                    )

                    sortable.add(PegPermutation(entries, signs))
        return sortable

    @staticmethod
    def sortable_by_prefix_reversal(d):
        assert d == int(d), "must be given an integer"
        assert d >= 0, "must be given a nonnegative integer"
        start = PegPermutation(1, "+")
        sortable = PegPermSet([start])
        for i in range(0, d):
            copy = set(sortable)
            for P in copy:
                n = len(P)
                for indices in combinations_with_replacement(range(0, n), 1):
                    PP = P.split(list(indices))

                    entries = PP[: indices[0] + 1][::-1] + PP[indices[0] + 1 :]
                    signs = (
                        PegPermSet.flip_signs(PP.signs[: indices[0] + 1][::-1])
                        + PP.signs[indices[0] + 1 :]
                    )

                    sortable.add(PegPermutation(entries, signs))
        return sortable

    @staticmethod
    def sortable_by_block_interchange(d):
        assert d == int(d), "must be given an integer"
        assert d >= 0, "must be given a nonnegative integer"
        start = PegPermutation(1, "+")
        sortable = PegPermSet([start])
        for i in range(0, d):
            copy = set(sortable)
            for P in copy:
                n = len(P)
                for indices in combinations_with_replacement(range(0, n), 4):
                    PP = P.split(list(indices))

                    entries = (
                        PP[: indices[0] + 1]
                        + PP[indices[2] + 3 : indices[3] + 4]
                        + PP[indices[1] + 2 : indices[2] + 3]
                        + PP[indices[0] + 1 : indices[1] + 2]
                        + PP[indices[3] + 4 :]
                    )
                    signs = (
                        PP.signs[: indices[0] + 1]
                        + PP.signs[indices[2] + 3 : indices[3] + 4]
                        + PP.signs[indices[1] + 2 : indices[2] + 3]
                        + PP.signs[indices[0] + 1 : indices[1] + 2]
                        + PP.signs[indices[3] + 4 :]
                    )

                    sortable.add(PegPermutation(entries, signs))
        return sortable

    @staticmethod
    def sortable_by_cut_and_paste(d):
        assert d == int(d), "must be given an integer"
        assert d >= 0, "must be given a nonnegative integer"
        start = PegPermutation(1, "+")
        sortable = PegPermSet([start])
        for i in range(0, d):
            copy = set(sortable)
            for P in copy:
                n = len(P)
                for indices in combinations_with_replacement(range(0, n), 3):
                    PP = P.split(list(indices))

                    entries = (
                        PP[: indices[0] + 1]
                        + PP[indices[1] + 2 : indices[2] + 3]
                        + PP[indices[0] + 1 : indices[1] + 2]
                        + PP[indices[2] + 3 :]
                    )

                    signs = (
                        PP.signs[: indices[0] + 1]
                        + PP.signs[indices[1] + 2 : indices[2] + 3]
                        + PP.signs[indices[0] + 1 : indices[1] + 2]
                        + PP.signs[indices[2] + 3 :]
                    )

                    sortable.add(PegPermutation(entries, signs))

                    entries = (
                        PP[: indices[0] + 1]
                        + PP[indices[1] + 2 : indices[2] + 3]
                        + PP[indices[0] + 1 : indices[1] + 2][::-1]
                        + PP[indices[2] + 3 :]
                    )

                    signs = (
                        PP.signs[: indices[0] + 1]
                        + PP.signs[indices[1] + 2 : indices[2] + 3]
                        + PegPermSet.flip_signs(
                            PP.signs[indices[0] + 1 : indices[1] + 2][::-1]
                        )
                        + PP.signs[indices[2] + 3 :]
                    )

                    sortable.add(PegPermutation(entries, signs))

                    entries = (
                        PP[: indices[0] + 1]
                        + PP[indices[1] + 2 : indices[2] + 3][::-1]
                        + PP[indices[0] + 1 : indices[1] + 2]
                        + PP[indices[2] + 3 :]
                    )

                    signs = (
                        PP.signs[: indices[0] + 1]
                        + PegPermSet.flip_signs(
                            PP.signs[indices[1] + 2 : indices[2] + 3][::-1]
                        )
                        + PP.signs[indices[0] + 1 : indices[1] + 2]
                        + PP.signs[indices[2] + 3 :]
                    )

                    sortable.add(PegPermutation(entries, signs))
        return sortable

    # ===== end Static Methods =====#

    def involves_set(self, P):
        for Q in self:
            if P.involved_in(Q):
                return True
        return False

    def sum_gfs_no_basis(self, S, only_clean=False):
        i = 0
        gf = 0
        n = len(S)
        t = time.time()
        print("\t\tComputing GF.")
        for PP in S:
            i += 1
            if i % 100000 == 0 and i > 0:
                print("\t\t\t", i, "of", n, "\ttook", (time.time() - t), "seconds.")
                t = time.time()
            if not PP.is_compact():
                continue
            if only_clean and not PP.is_compact_and_clean():
                continue
            if i % 10000 == 0 and i > 0:
                gf = gf.simplify()
            if i % 50000 == 0 and i > 0:
                clear_cache()
            gf += PP.csgf([])
            # print 'adding gf for',PP,'with basis []'

        return gf

    def alt_downset(self):

        topset = PegPermSet(self)

        bottom_edge = PegPermSet()
        keyssofar = PegPermSet()
        unclean = dict()

        for PP in self:
            if PP.is_compact() and not PP.is_compact_and_clean():
                cleaned = PP.clean()
                if cleaned in keyssofar:
                    unclean[cleaned].add(PP)
                else:
                    unclean[cleaned] = PegPermSet([PP])
                    keyssofar.add(cleaned)

        bottom_edge.update(self)
        n = len(bottom_edge)
        gf = self.sum_gfs_no_basis(bottom_edge, only_clean=True)

        while len(bottom_edge) > 0:
            oldsize = n
            n = len(bottom_edge)
            next_layer = PegPermSet()

            i = 0
            num_built = 0
            t = time.time()
            while len(bottom_edge) > 0:
                i += 1
                P = bottom_edge.pop()
                next_layer.update(P.shrink_by_one())
                del P
                if i % 100000 == 0:
                    clear_cache()
                    print(
                        "\t",
                        i,
                        "of",
                        n,
                        ". Now with",
                        len(next_layer),
                        ". Took",
                        (time.time() - t),
                        "seconds.",
                    )
                    t = time.time()

            del bottom_edge
            clear_cache()
            n += len(next_layer)

            print("\t\tScanning permutations for cleanliness!")
            i = 0
            num_unclean = 0
            nll = len(next_layer)
            for PP in next_layer:
                i += 1
                if i % 200000 == 0:
                    print("\t\t\tScanned", i, "of", nll, ".")
                if PP.is_compact() and not PP.is_compact_and_clean():
                    cleaned = PP.clean()
                    num_unclean += 1
                    if cleaned in keyssofar:
                        unclean[cleaned].add(PP)
                    else:
                        unclean[cleaned] = PegPermSet([PP])
                        keyssofar.add(cleaned)

            print("\t\tScanning permutations for unnecessary uncleans!")
            i = 0
            nll = len(next_layer)
            for PP in next_layer:
                i += 1
                if i % 200000 == 0:
                    print("\t\t\tScanned", i, "of", nll, ".")
                if unclean.get(PP):
                    del unclean[PP]

            print(
                "\tOut of",
                len(next_layer),
                "permutations in this layer,",
                num_unclean,
                "were unclean.",
            )
            gf += self.sum_gfs_no_basis(next_layer, only_clean=True)

            bottom_edge = next_layer
            del next_layer
            clear_cache()
            newsize = n
            print("\t\tDownset currently has", newsize, "permutations.")

        return (gf, unclean)

    def compactify(self):
        copy = PermSet(self)
        for P in copy:
            if not P.is_compact():
                self.remove(P)

    # def downset(self):
    #   return PegPermSet(super(PegPermSet, self).downset())

    def cross_sections(self):
        print("Starting to compute downset.")
        generating_set = PegPermSet(self.downset())
        print("\tDownset done. Contains", len(generating_set), "peg permutations.")
        print("Starting to compactify.")
        generating_set.compactify()
        print(
            "\tDone compactifying. Now contains",
            len(generating_set),
            "peg permutations.",
        )
        print("Starting to compute cross sections.")
        cross_sections = dict()
        pairs = list()
        print("\tCleaning, finding bases, and loading pairs.")
        i = 0
        n = len(generating_set)
        for P in generating_set:
            if i % 20000 == 0:
                print("\t\t", i, "of", n, "...")
            cp = P.clean()
            b = P.clean_basis()
            pairs.append((cp, b))
            cross_sections[cp] = VectorSet([-1])
            i += 1
        i = 0
        del generating_set
        print("\tUnioning bases.")
        for (cleaned_perm, V) in pairs:
            if i % 200000 == 0:
                print("\t\t", i, "of", n, "... dict_size =", len(cross_sections))
            # if cleaned_perm in cross_sections.keys():
            cross_sections[cleaned_perm] = V.basis_union(cross_sections[cleaned_perm])
            # else:
            # cross_sections[cleaned_perm] = V
            i += 1
        del pairs
        return cross_sections

    def enumerate(self, cross_sections=None):
        if cross_sections is None:
            cross_sections = self.cross_sections()
        gc.collect()
        print(
            "\tDone computing cross_sections. There are",
            len(cross_sections),
            "cross sections.",
        )
        print("Starting to compute generating function.")
        gf = 0
        i = 0
        n = len(cross_sections)
        t = time.time()
        for clean_perm in cross_sections.keys():
            if i % 10000 == 0 and i > 0:
                gf = gf.simplify()
            if i % 50000 == 0 and i > 0:
                clear_cache()
            if i % 1000 == 0 and i > 0:
                print("\t\t", i, "of", n, "\ttook", (time.time() - t), "seconds.")
                t = time.time()
            gf += clean_perm.csgf(cross_sections[clean_perm])
            i += 1
        print("\tDone!")
        return gf.simplify()

    def alt_cross_sections(self):
        print("Starting to compute downset.")
        (gf, uc) = self.alt_downset()
        unclean = PegPermSet()
        uncleanlist = list()
        for PP in uc.keys():
            uncleanlist.extend(uc[PP])
        unclean = set(uncleanlist)
        print("\tDownset done. Contains", len(unclean), "unclean peg permutations.")
        # print 'Starting to compactify.'
        # unclean.compactify()
        # print '\tDone compactifying. Now contains',len(unclean),'peg permutations.'
        print("Starting to compute cross sections of UNCLEAN peg permutations.")
        cross_sections = dict()
        i = 1
        pairs = list()
        print("\tCleaning, finding bases, and loading pairs.")
        n = len(unclean)
        for P in unclean:
            if i % 20000 == 0:
                print("\t\t", i, "of", n, "...")
            cp = P.clean()
            b = P.clean_basis()
            pairs.append((cp, b))
            cross_sections[cp] = VectorSet([-1])
            i += 1
        i = 1
        del unclean
        clear_cache()
        print("\tUnioning bases.")
        for (cleaned_perm, V) in pairs:
            if i % 200000 == 0:
                print("\t\t", i, "of", n, "... dict_size =", len(cross_sections))
            # if cleaned_perm in cross_sections.keys():
            cross_sections[cleaned_perm] = V.basis_union(cross_sections[cleaned_perm])
            # else:
            # cross_sections[cleaned_perm] = V
            i += 1
        del pairs
        clear_cache()
        return (gf.simplify(), cross_sections)

    def alt_enumerate(self, cross_sections=None):
        """only works when the set is a generating set for sortables and the top layer has all the same length!!!"""
        ml = max([len(s) for s in self])
        PPS = PegPermSet([s for s in self if len(s) == ml])
        (gf, cross_sections) = PPS.alt_cross_sections()
        gc.collect()
        print(
            "\tDone computing cross_sections. There are",
            len(cross_sections),
            "cross sections.",
        )
        print("Starting to compute generating function for uncleans.")
        i = 0
        n = len(cross_sections)
        t = time.time()
        # print 'clean gf:',gf.simplify()
        for clean_perm in cross_sections.keys():
            if i % 10000 == 0 and i > 0:
                gf = gf.simplify()
            if i % 50000 == 0 and i > 0:
                clear_cache()
            if i % 10000 == 0 and i > 0:
                print("\t\t", i, "of", n, "\ttook", (time.time() - t), "seconds.")
                t = time.time()
            # gf -= clean_perm.csgf([])
            # print 'subtracting gf for',clean_perm,'with basis []'
            # print 'adding gf for',clean_perm,'with basis',cross_sections[clean_perm]
            gf += clean_perm.csgf(cross_sections[clean_perm])
            i += 1
        print("\tDone!")
        return gf.simplify()
