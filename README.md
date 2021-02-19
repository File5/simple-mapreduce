# Python microlibrary for map-reduce testing/prototyping

Map-reduce approach is used in distributed computing, however, the deployment of real map-reduce tools like Hadoop is too complicated for those who just want to practice solving simple tasks with this approach.

This library makes use of Python built-in `map()`, `functools.reduce()` and Python generators to implement map-reduce pipeline.
It includes additional simple feature which is suited for learning and debugging - printing execution step-by-step.
It is not intended for production usage.

## Installation

You can copy single file mapreduce/mapreduce.py into your project, there are no dependencies.

Or, alternatively:
```
pip install -e git+https://github.com/File5/simple-mapreduce#egg=simple-mapreduce
```
then, to uninstall:
```
pip uninstall simple-mapreduce
```

## Usage

Example task which finds the number with the largest number of repetitions
```python
from mapreduce import MapReduceTask

# actually, (verbose=True, lazy=False) are default parameters
t = MapReduceTask(verbose=True, lazy=False)

# the order matters
@t.map
def m1(k, v):
    return v, 1

@t.reduce
def r1(k, v):
    return k, sum(v)

@t.map
def m2(k, v):
    return 'all', (k, v)

@t.reduce
def r2(k, v):
    km, vm = None, None
    for ki, vi in v:
        if vm is None or vi > vm:
            km, vm = ki, vi
    return 'max', (km, vm)

x = [1,2,3,1,2,1,4,5,6]
print(list(t(x)))
```

The output is the following
```
m1: (0, 1) -> (1, 1)
m1: (1, 2) -> (2, 1)
m1: (2, 3) -> (3, 1)
m1: (3, 1) -> (1, 1)
m1: (4, 2) -> (2, 1)
m1: (5, 1) -> (1, 1)
m1: (6, 4) -> (4, 1)
m1: (7, 5) -> (5, 1)
m1: (8, 6) -> (6, 1)
r1: (1, [1, 1, 1]) -> (1, 3)
r1: (2, [1, 1]) -> (2, 2)
r1: (3, [1]) -> (3, 1)
r1: (4, [1]) -> (4, 1)
r1: (5, [1]) -> (5, 1)
r1: (6, [1]) -> (6, 1)
m2: (1, 3) -> ('all', (1, 3))
m2: (2, 2) -> ('all', (2, 2))
m2: (3, 1) -> ('all', (3, 1))
m2: (4, 1) -> ('all', (4, 1))
m2: (5, 1) -> ('all', (5, 1))
m2: (6, 1) -> ('all', (6, 1))
r2: ('all', [(1, 3), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1)]) -> ('max', (1, 3))
[('max', (1, 3))]
```
