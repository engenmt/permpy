class Vector(tuple):
  def __new__(cls, v):
    for i, val in enumerate(v):
      v[i] = int(val)
      assert v[i] >= 0, 'Vector entries must be nonnegative integers!'
    return tuple.__new__(cls, v)

  def __contains__(self, other):
    """Return True if `other` is contained in `self`.
    """
    assert len(v) == len(self), 'Vectors must be same length to check containment!'
    return all(self_val >= other_val for self_val, other_val in zip(self, other))

  def meet(self, v):
  	assert len(v) == len(self), 'Vectors must be same length to meet!'
  	return Vector([max(self_val, v_val) for self_val, v_val in zip(self, v)])

  def contained_in(self, other):
    return other.__contains__(self)

  def norm(self):
    return sum(self)
