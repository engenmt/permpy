import itertools
import logging
from itertools import combinations

from .permutation import Permutation
from .permset import PermSet
from .permclass import PermClass


logging.basicConfig(level=10)


class BadMatrixException(Exception):
    pass


class BadWordException(Exception):
    pass


class GeometricGridClass(PermClass):
    def __init__(self, M, col=None, row=None, max_length=8, generate=True):
        """
        
        Args:
            M (list of lists of ints): A 2D matrix to build the GGC from.
                Goes from left-to-right, bottom-to-top. That is, M[0] is the
                leftmost column, and M[0][0] is the lowest entry of this column.
                Entries should be -1, 0, +1, or 2. A 0 represents an empty cell,
                and a 2 represents a cell that has at most one point in it.
                Otherwise, a +1 represents an increasing cell, and
                a -1 represents a decreasing cell.
            col (list of ints, optional): A vector representing the orientation
                of the columns. Entries should be +1 or -1. If col[i] is +1,
                then the i'th column is oriented from left to right, and
                if col[i] is -1, then it is oriented from right to left.
            row (list of ints, optional): A vector representing the orientation
                of the rows. Entries should be +1 or -1. If row[j] is +1,
                then the j'th row is oriented from bottom to top, and
                if row[j] is -1, then it is oriented from top to bottom.

        """
        self.M = M

        self.col, self.row = col, row

        if col is None or row is None:
            self.compute_signs()

        # Our alphabet consists of Cartesian coordinates of cells
        self.alphabet = [
            (col_idx, row_idx)
            for col_idx, col in enumerate(self.M)
            for row_idx, val in enumerate(col)
            if val
        ]

        self.dots = [(x, y) for x, y in self.alphabet if self.M[x][y] == 2]

        # We will only use words that do _not_ contain these as factors.
        self.commuting_pairs = [
            pair
            for pair in combinations(self.alphabet, 2)  # Each pair of letters
            if all(
                coord_1 != coord_2 for coord_1, coord_2 in zip(*pair)
            )  # where all coordinates differ
        ]

        if generate:
            L = self.build_perms(max_length)
        else:
            L = [PermSet() for _ in range(max_length + 1)]

        PermClass.__init__(self, L)

    def find_word_for_perm(self, p):

        all_words = itertools.product(self.alphabet_indices, repeat=len(p))

        for word in all_words:
            perm = self.dig_word_to_perm(word)
            if perm == p:
                return word

    def compute_signs(self):
        col_signs = self.col or [0 for _ in range(len(self.M))]
        row_signs = self.row or [0 for _ in range(len(self.M[0]))]

        unsigned_vals = {
            0,
            2,
        }  # These represent empty cells and point-cells respectively

        for col_idx, col in enumerate(self.M):
            if all(val in unsigned_vals for val in col):
                # This column has no entries that need a sign, so we set it arbitrarily.
                col_signs[col_idx] = 1

        for row_idx in range(len(row_signs)):
            if all(col[row_idx] in unsigned_vals for col in self.M):
                # This row has no entries that need a sign, so we set it arbitrarily.
                row_signs[row_idx] = 1

        while not (all(col_signs) and all(row_signs)):
            # This loop will continue until all col_signs and row_signs are non-zero
            # It will make at most one "arbitrary" column assignment per loop.
            logging.debug("Starting loop again.")
            logging.debug(f"\tself.M = {self.M}")
            logging.debug(f"\tcol_signs = {col_signs}")
            logging.debug(f"\trow_signs = {row_signs}")
            choice_made = False

            for col_idx, col in enumerate(self.M):
                if col_signs[col_idx]:
                    # This column has a sign already.
                    continue

                for row_idx, (row_sign, entry) in enumerate(zip(row_signs, col)):
                    if entry in unsigned_vals:
                        continue

                    if not row_sign:
                        continue

                    # If we're here, then:
                    # - there's a signed entry in entry = self.M[col_idx][row_idx]
                    # - row_sign = row_signs[row_idx] is defined.
                    col_signs[col_idx] = entry * row_sign
                    break
                else:
                    # If we're here, then col_signs[col_idx] is undefined.
                    if not choice_made:
                        # Make our arbitrary choice.
                        col_signs[col_idx] = 1
                        choice_made = True

                if col_signs[col_idx]:
                    for row_idx, entry in enumerate(col):
                        if entry in unsigned_vals:
                            continue
                        if row_signs[row_idx]:
                            assert (
                                row_signs[row_idx] == entry * col_signs[col_idx]
                            ), f"The signs are all messed up now: {self.M}, {col_signs}, {row_signs} ({col_idx}, {row_idx})"
                        else:
                            row_signs[row_idx] = entry * col_signs[col_idx]

        # This verifies that everything is consistent.
        for col_idx, (col, col_sign) in enumerate(zip(self.M, col_signs)):
            for row_idx, (entry, row_sign) in enumerate(zip(col, row_signs)):
                if entry not in unsigned_vals:
                    if entry != col_sign * row_sign:
                        raise BadMatrixException(
                            f"Signs can't be computed for this matrix: {self.M}"
                        )

        self.col = col_signs
        self.row = row_signs

    def build_perms(self, max_length):

        L = [PermSet.all(length) for length in range(2)]
        # Include all the length-0 and length-1 perms.

        for length in range(2, max_length + 1):
            # Try all words of length 'length' with alphabet equal to the cell alphabet of M.
            this_length = PermSet()

            for word in itertools.product(self.alphabet, repeat=length):
                p = self.dig_word_to_perm(word)
                if p:
                    this_length.add(p)

            L.append(this_length)

        return L

    def dig_word_to_perm(self, word, ignore_bad=False):
        if not ignore_bad:
            for letter in self.dots:
                if word.count(letter) > 1:
                    return False
            if not self.is_valid_word(word):
                return False

        # Let's build a permutation in the Geometric Grid Class.
        # Imagine each "signed" cell having a line segment at 45º either
        # oriented up-and-to-the-right if the cell has a positive sign or
        # oriented down-and-to-the-right if the cell has a negative sign with
        # len(word)+1 open slots on it.
        points = []
        height = len(word) + 2
        for position, (letter_x, letter_y) in enumerate(word):

            if self.col[letter_x] == 1:
                x_point = letter_x * height + position
            else:
                x_point = (letter_x + 1) * height - position

            if self.row[letter_y] == 1:
                y_point = letter_y * height + position
            else:
                y_point = (letter_y + 1) * height - position

            points.append((x_point, y_point))

        return Permutation([y for x, y in sorted(points)])

    def is_valid_word(self, word):
        return all(
            word[i : i + 2] not in self.commuting_pairs for i in range(len(word) - 1)
        )
