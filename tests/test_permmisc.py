from permpy import Permutation as Perm


def test_cycle_decomp():
    p = Perm(53814276)
    expected = [[4, 3, 0], [6], [7, 5, 1, 2]]
    result = p.cycle_decomp()
    assert result == expected, (
        f"Perm({p}).cycle_decomp() returned {result},"
        f" but it should return {expected}."
    )
