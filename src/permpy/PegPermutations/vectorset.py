from .vector import Vector


class VectorSet(list):
    def basis_union(self, B):
        if len(B) > 0 and B[0] == -1:
            return self
        if len(self) > 0 and self[0] == -1:
            return B
        S = set(V.meet(W) for V in self for W in B)
        return VectorSet(S).minimal_elements()

    def minimal_elements(self):
        C = self[:]
        for V in self:
            for W in self:
                if V == W:
                    continue
                if W.contained_in(V):
                    C.remove(V)
                    break
        return C

    def meet_all(self):
        if len(self) == 0:
            return Vector([])
        assert (
            len(set(len(V) for V in self)) == 1
        ), "Not all vector lengths are the same!"
        return Vector([max(*vals) for vals in zip(*self)])
