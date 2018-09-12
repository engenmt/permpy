permpy
=======

## A Python Permutations Class

Contains Various tools for working interactively with permutaions. 
Easily extensible.

### Examples:
```python
>>>
>>> import permpy as pp
>>> 
>>> 
>>> p = pp.Perm.random(8)
>>> 
>>> p
 5 4 7 1 6 2 3 8 
>>> 
>>> 
>>> p.cycles()
'( 6 2 4 1 5 ) ( 7 3 ) ( 8 )'
>>> 
>>> p.order()
10
>>> 
>>> p ** 10
 1 2 3 4 5 6 7 8
>>>

>>> S = pp.PermSet.all(6)
>>> 
>>> S
 Set of 720 permutations
>>> 
>>> S.total_statistic(Perm.inversions)
 5400
>>> 
>>> S.total_statistic(Perm.descents)
 1800
>>> 

>>> 
>>> A = pp.AvClass([ 132 ])
>>> 
>>> A
[Set of 0 permutations,
 Set of 1 permutations,
 Set of 2 permutations,
 Set of 6 permutations,
 Set of 24 permutations,
 Set of 120 permutations,
 Set of 720 permutations,
 Set of 5040 permutations,
 Set of 40320 permutations]
>>> 
>>> 
>>> 
```
