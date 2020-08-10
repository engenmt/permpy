from .avclass import AvClass
from .geometricgridclass import GeometricGridClass
from .InsertionEncoding import InsertionScheme
from .pegpermset import PegPermSet
from .pegpermutation import PegPermutation
from .permclass import PermClass
from .permset import PermSet
from .permutation import Permutation
from .RestrictedContainer import RestrictedContainer

try:
    import matplotlib as mpl

    mpl.rc("axes", fc="E5E5E5", ec="white", lw="1", grid="True", axisbelow="True")
    mpl.rc("grid", c="white", ls="-")
    mpl.rc("figure", fc="white")
    mpl_imported = True
except ImportError:
    print("Install matplotlib for extra plotting functionality.")
    pass

Perm = Permutation
Av = AvClass

__all__ = [
    "AvClass",
    "GeometricGridClass",
    "InsertionScheme",
    "PegPermSet",
    "PegPermutation",
    "Perm",
    "PermClass",
    "PermSet",
    "Permutation",
    "RestrictedContainer",
]
