from .permdeprecated import deprecated


class PermClassDeprecatedMixin:
    def plus_class(self, t):
        return self.extended(t)

    def plus_one_class(self):
        return self.extended(1)
