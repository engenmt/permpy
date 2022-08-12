from permpy import GeometricGridClass


def test_init():
    """
    Notes:
        The following example represents the matrix
            +-+-+-+
            | | |/|
            +-+-+-+
        M = | |/|/|
            +-+-+-+
            |/|/| |
            +-+-+-+

    """
    M = [[1, 0, 0], [1, 1, 0], [0, 1, 1]]  # Partial increasing staircase.
    G = GeometricGridClass(M)  # This will the same as Av(321) until length 9.
    result = [len(S) for S in G]
    expected = [1, 1, 2, 5, 14, 42, 132, 429, 1430]
    assert result == expected, "Generating the six-cell positive staircase failed!"


def test_compute_signs():
    r"""
    Notes:
        The following matrix example represents
            +-+-+-+
            | |/|\|
        M = +-+-+-+
            |/| |/|
            +-+-+-+

        It should have signs:

            +-+-+-+
            | |/|\|↓
        M = +-+-+-+
            |/| |/|↑
            +-+-+-+
            → ← →

        Meaning col = [1, -1, 1] and row = [1, -1].

    """
    M = [[1, 0], [0, 1], [1, -1]]
    G = GeometricGridClass(M, generate=False)
    expected_col = [1, -1, 1]
    assert (
        G.col == expected_col
    ), "GeometricGridClass computed column signs incorrectly."

    expected_row = [1, -1]
    assert G.row == expected_row, "GeometricGridClass computed row signs incorrectly."
