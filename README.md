permpy
=======

## A Python Permutations Class

Contains Various tools for working interactively with permutations. 
Easily extensible.

### Examples:
```python
>>> import permpy as pp
>>> 
>>> p = pp.Perm.random(8)
>>> p
 5 4 7 1 6 2 3 8 
>>> p.cycles()
'( 6 2 4 1 5 ) ( 7 3 ) ( 8 )'
>>> p.order()
10 
>>> p ** p.order()
 1 2 3 4 5 6 7 8
>>>
>>> S = pp.PermSet.all(6)
>>> S
Set of 720 permutations
>>> S.total_statistic(pp.Perm.num_inversions)
5400
>>> S.total_statistic(pp.Perm.num_descents)
1800
>>> 
>>> A = pp.AvClass([ 132 ])
>>> A
[Set of 0 permutations, 
 Set of 1 permutations, 
 Set of 2 permutations, 
 Set of 5 permutations, 
 Set of 14 permutations, 
 Set of 42 permutations, 
 Set of 132 permutations, 
 Set of 429 permutations, 
 Set of 1430 permutations]
>>> 
```

## Build Instructions
For a summary of how PermPy is built, go [here](https://py-pkgs.org/03-how-to-package-a-python#summary-and-next-steps).
```bash
$ python -m poetry build
$ python -m poetry publish -r test-pypi
```