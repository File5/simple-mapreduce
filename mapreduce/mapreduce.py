# MIT License
# Copyright (c) 2021 Aleksandr Zuev
# See LICENSE for further information

from collections import defaultdict
from functools import reduce
import itertools

def mapdict():
    return defaultdict(list)

_initial_missing = object()
def lazy_reduce(function, iterable, initializer=_initial_missing):
    if initializer is _initial_missing:
        yield reduce(function, iterable)
    else:
        yield reduce(function, iterable, initializer)

def items_of_single(x):
    for i in next(x).items():
        yield i

class MapReduceTask:
    def __init__(self, verbose=True, lazy=False):
        self.verbose = verbose
        self.lazy = lazy
        self.steps = []

    def map(self, function):
        if not callable(function):
            # it's decorator with () call
            return self.map

        def f(x):
            def map_func(i):
                result = function(i[0], i[1])

                if self.verbose:
                    print('{}: {} -> {}'.format(function.__name__, i, result))
                return result
            result = map(map_func, x)

            if self.lazy:
                return result
            else:
                return list(result)  # evaluate

        self.steps.append(f)
        return function

    def reduce(self, function):
        if not callable(function):
            # it's decorator with () call
            return self.reduce

        def f(x):
            def reduce_func(a, b):
                a[b[0]].append(b[1])
                return a
            x = lazy_reduce(reduce_func, x, mapdict())  # x is single item iterable
            x = items_of_single(x)

            def map_func(i):
                result = function(i[0], i[1])

                if self.verbose:
                    print('{}: {} -> {}'.format(function.__name__, i, result))
                return result
            result = map(map_func, x)
            if self.lazy:
                return result
            else:
                return list(result)  # evaluate

        self.steps.append(f)
        return function

    def eval(self, input_val):
        x = enumerate(input_val)
        for func in self.steps:
            x = func(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)
