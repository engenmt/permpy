class Vector(tuple):
    def __new__(cls, v):
        for i in range(0, len(v)):
            v[i] = int(v[i])
            assert v[i] >= 0, "vector entries must be nonnegative integers"
        return tuple.__new__(cls, v)

    def meet(self, V):
        assert len(V) == len(self), "vectors must be same length to meet"
        new_vector = []
        for i in range(0, len(V)):
            new_vector.append(max(V[i], self[i]))
        return Vector(new_vector)

    def contained_in(self, V):
        assert len(V) == len(self), "vectors must be same length to check containment"
        for i in range(0, len(V)):
            if self[i] > V[i]:
                return False
        return True

    def norm(self):
        return sum(self)
