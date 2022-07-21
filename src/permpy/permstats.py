class PermutationStatsMixin:
    def num_fixed_points(self):
        return len(self.fixed_points())

    def num_descents(self):
        return len(self.descents())

    def num_ascents(self):
        return len(self.ascents())

    def num_peaks(self):
        return len(self.peaks())

    def num_valleys(self):
        return len(self.valleys())

    def num_ltr_min(self):
        return len(self.ltr_min())

    def num_rtl_min(self):
        return len(self.rtl_min())

    def num_ltr_max(self):
        return len(self.ltr_max())

    def num_rtl_max(self):
        return len(self.rtl_max())

    def num_rtlmin_ltrmax_layers(self):
        return len(self.rtlmin_ltrmax_decomposition())

    def num_rtlmax_ltrmin_layers(self):
        return len(self.rtlmax_ltrmin_decomposition())

    def trivial(self):
        return 0

    def num_inversions(self):
        return len(self.inversions())

    def num_noninversions(self):
        return len(self.noninversions())

    def major_index(self):
        """Return the major index of `self`."""
        return sum(self.descents())

    def len_max_run(self):
        """Return the length of the longest monotone contiguous subsequence of entries."""
        return max(self.max_ascending_run()[1], self.max_descending_run()[1])

    def is_involution(self):
        """Determine if the permutation is an involution, i.e., is equal to it's own inverse."""
        for idx, val in enumerate(self):
            if idx != self[val]:
                return False
        return True

    def is_increasing(self):
        """Determine if the permutation is increasing."""
        return all(idx == val for idx, val in enumerate(self))

    def is_decreasing(self):
        """Determine if the permutation is increasing."""
        return all(idx == val for idx, val in enumerate(self[::-1]))

    def is_identity(self):
        """Wrapper for is_increasing."""
        return self.is_increasing()

    def is_simple(self):
        """Determine if `self` is simple.

        Todo:
            Implement this better, if possible.

        """
        (i, _) = self.simple_location()
        return i == 0

    def is_strongly_simple(self):
        return self.is_simple() and all([p.is_simple() for p in self.children()])

    def num_bonds(self):
        return len(self.bonds())

    def num_inc_bonds(self):
        return len(self.inc_bonds())

    def num_dec_bonds(self):
        return len(self.dec_bonds())

    def num_copies(self, other):
        """Return the number of copies of `other` in `self`."""
        return len(self.copies(other))

    def num_contiguous_copies_of(self, other):
        """Return the number of contiguous copies of `other` in `self`."""
        return len(self.contiguous_copies(other))
