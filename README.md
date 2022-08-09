# permpy

PermPy is a user-friendly Python library for maniupulating permutation patterns and permutation classes. See [Wikipedia](https://en.wikipedia.org/wiki/Permutation_pattern) for an introduction to permutation patterns.

## Installation

Install `permpy` with `pip`:

```bash
$ python -m pip install permpy
```

## Usage 

`permpy` contains a number of useful Python classes including `permpy.Permutation`, which represents a permutation and can determine containment.
```python
>>> from permpy import Permutation
>>> p = Permutation(1324)
>>> q = Permutation(123)
>>> q <= p
True
>>> r = Permutation(321)
>>> r <= p
False
>>> S = pp.PermSet.all(6)
>>> S
Set of 720 permutations
>>> S.total_statistic(pp.Perm.num_inversions)
5400
>>> S.total_statistic(pp.Perm.num_descents)
1800
>>> from permpy import AvClass
>>> A = AvClass([132])
>>> for S in A:
...     print(S)
... 
Set of 1 permutations
Set of 1 permutations
Set of 2 permutations
Set of 5 permutations
Set of 14 permutations
Set of 42 permutations
Set of 132 permutations
Set of 429 permutations
Set of 1430 permutations 
```

## Build Instructions
For a summary of how PermPy is built, go [here](https://py-pkgs.org/03-how-to-package-a-python#summary-and-next-steps).
```bash
$ python -m poetry build
$ python -m poetry publish
```

## Test Instructions

To run tests, run
```bash
$ python -m poetry build
$ python -m poetry shell
$ python -m pytest tests/
```

To build and install locally, run
```bash
$ python -m poetry install
$ python -m poetry shell
$ python
>>> import permpy
>>>
```