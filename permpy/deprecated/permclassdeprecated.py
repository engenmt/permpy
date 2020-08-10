from .permdeprecated import deprecated


class PermClassDeprecatedMixin:
    @deprecated
    def plus_class(self, t):
        return self.extended(t)

    @deprecated
    def plus_one_class(self):
        return self.extended(1)
