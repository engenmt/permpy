from .vector import *


class VectorSet(list):
    def basis_union(self, B):
        if len(B) > 0 and B[0] == -1:
            return self
        if len(self) > 0 and self[0] == -1:
            return B
        S = set()
        for V in self:
            for W in B:
                S.add(V.meet(W))
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

        l = len(self[0])
        V = Vector([1] * l)
        for W in self:
            V = V.meet(W)
        return V
