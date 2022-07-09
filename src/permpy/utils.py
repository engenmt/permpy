import types


def av_test(p):
    """Return a function that is True if the input avoids p."""
    return lambda q: p not in q


def copy_func(f, name=None):
    """Return a function with same code, globals, defaults, closure, and
    name (or provide a new name)
    """
    fn = types.FunctionType(
        f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__
    )
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__)
    return fn


if __name__ == "__main__":
    pass
