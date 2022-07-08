from .permutation import Permutation
from .permset import PermSet
from .permclass import PermClass
from .propertyclass import PropertyClass
from .avclass import AvClass

# from .pegpermutation import PegPermutation
# from .pegpermset import PegPermSet
from .geometricgridclass import GeometricGridClass

# from .InsertionEncoding import *
# from .RestrictedContainer import *

# try:
#     import matplotlib as mpl

#     mpl.rc("axes", fc="E5E5E5", ec="white", lw="1", grid="True", axisbelow="True")
#     mpl.rc("grid", c="white", ls="-")
#     mpl.rc("figure", fc="white")
#     mpl_imported = True
# except ImportError:
#     print("Install matplotlib for extra plotting functionality.")
#     pass

Perm = Permutation
Av = AvClass
