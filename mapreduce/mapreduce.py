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

def flat_map(function, iterable):
    results = map(function, iterable)
    for map_iterable in results:
        for i in map_iterable:
            yield i

def items_of_single(x):
    for i in next(x).items():
        yield i

class MapReduceSteps:
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
                    result = list(result)  # need to materialize, can't run iterable twice
                    for j in result:
                        print('{}: {} -> {}'.format(function.__name__, i, j))
                return result
            result = flat_map(map_func, x)

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
                    result = list(result)  # need to materialize, can't run iterable twice
                    for j in result:
                        print('{}: {} -> {}'.format(function.__name__, i, j))
                return result
            result = flat_map(map_func, x)
            if self.lazy:
                return result
            else:
                return list(result)  # evaluate

        self.steps.append(f)
        return function

class StopRepeated(Exception):
    pass

class Repeated(MapReduceSteps):
    def __init__(self, times=-1):
        super().__init__()
        self.remaining = times

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def break_map(self, function):
        if not callable(function):
            # it's decorator with () call
            return self.break_map

        def break_if(k, v):
            if function(k, v):
                raise StopRepeated
            yield k, v
        break_if.__name__ = function.__name__
        self.map(break_if)
        return function

    def break_reduce(self, function):
        if not callable(function):
            # it's decorator with () call
            return self.break_reduce

        def break_if(k, v):
            if function(k, v):
                raise StopRepeated
            yield k, v
        break_if.__name__ = function.__name__
        self.reduce(break_if)
        return function

    def eval(self, x):
        try:
            i = 0
            while self.remaining != 0:
                if self.verbose:
                    print('-' * 10, 'Repeat {}'.format(i), '-' * 10)
                for func in self.steps:
                    x = func(x)
                if self.remaining > 0:
                    self.remaining -= 1
                i += 1
        except StopRepeated:
            if self.verbose:
                print('-' * 10, 'break'.format(i), '-' * 10)
        return x

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

class MapReduceTask(MapReduceSteps):
    def __init__(self, verbose=True, lazy=False):
        super().__init__(verbose, lazy)

    def repeated(self, times=-1):
        r = Repeated(times)
        self.steps.append(r)
        return r

    def eval(self, input_val):
        x = enumerate(input_val)
        for func in self.steps:
            x = func(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)
