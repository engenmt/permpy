class Vector(tuple):
    def __new__(cls, v):
        for idx, val in enumerate(v):
            v[idx] = int(val)
            assert v[idx] >= 0, "Vector entries must be nonnegative integers!"
        return tuple.__new__(cls, v)

    def meet(self, other):
        assert len(self) == len(other), "Vectors must be same length to meet!"
        return Vector([max(*pair) for pair in zip(self, other)])

    def contained_in(self, other):
        assert len(self) == len(
            other
        ), "Vectors must be same length to check containment"
        return all(val_self <= val_other for val_self, val_other in zip(self, other))

    def norm(self):
        return sum(self)
