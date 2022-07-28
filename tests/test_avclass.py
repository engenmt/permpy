from permpy import AvClass


def test_all():
    B = [123]
    C = AvClass(B, max_len=4)
    expected = [1, 1, 2, 5, 14]
    result = [len(S) for S in C]
    assert result == expected, f"AvClass({B}) is incorrect through length 4."
