from mapreduce import MapReduceTask

def test_flat_map():
    def f(x):
        yield 1
        yield 2

    from mapreduce.mapreduce import flat_map
    assert list(flat_map(f, [1, 2])) == [1, 2, 1, 2]

# run pytest -s to see the output
def test_readme_example():
    t = MapReduceTask(verbose=True, lazy=False)

    # the order matters
    @t.map
    def m1(k, v):
        yield v, 1

    @t.reduce
    def r1(k, v):
        yield k, sum(v)

    @t.map
    def m2(k, v):
        yield 'all', (k, v)

    @t.reduce
    def r2(k, v):
        km, vm = None, None
        for ki, vi in v:
            if vm is None or vi > vm:
                km, vm = ki, vi
        yield 'max', (km, vm)

    x = [1,2,3,1,2,1,4,5,6]
    # print newline, so the output will be on the new line when run by pytest
    print('')
    assert list(t(x)) == [('max', (1, 3))]

def test_readme_example2():
    t = MapReduceTask(verbose=True, lazy=False)

    # the order matters
    @t.map
    def m1(k, v):
        for word in v.split(' '):
            yield word, 1

    @t.reduce
    def r1(k, v):
        yield k, sum(v)

    x = ["hello world word world of words"]
    # print newline, so the output will be on the new line when run by pytest
    print('')
    assert list(t(x)) == [('hello', 1), ('world', 2), ('word', 1), ('of', 1), ('words', 1)]


def test_readme_example_lazy():
    t = MapReduceTask(verbose=True, lazy=True)

    # the order matters
    @t.map
    def m1(k, v):
        yield v, 1 / 0  # will raise ZeroDivisionError if evaluated

    @t.reduce
    def r1(k, v):
        yield k, sum(v)

    @t.map
    def m2(k, v):
        yield 'all', (k, v)

    @t.reduce
    def r2(k, v):
        km, vm = None, None
        for ki, vi in v:
            if vm is None or vi > vm:
                km, vm = ki, vi
        yield 'max', (km, vm)

    x = [1,2,3,1,2,1,4,5,6]
    # print newline, so the output will be on the new line when run by pytest
    print('')
    try:
        t(x)
    except ZeroDivisionError:
        assert False, "should not be evaluated"
    else:
        assert True


def test_repeated_n_times():
    t = MapReduceTask(verbose=True, lazy=False)

    @t.map
    def m1(k, v):
        nonlocal start_node
        n, neighbors = v
        state = 1 if n == start_node else 0
        yield n, (state, neighbors)

    with t.repeated(3) as repeated:
        @repeated.reduce
        def r1(n, l):
            state = 0
            neighbors = []
            for i in l:
                state = max(state, i[0])  # i.state
                neighbors += i[1]  # i.neighbors
            if state == 1:
                for o in neighbors:
                    yield o, (1, [])
                state = 2
            yield n, (state, neighbors)

    start_node = 'x'
    x = {
        'x': ['a', 'b', 'c'],
        'a': ['e'],
        'b': ['d'],
        'c': ['d', 'x'],
    }
    assert list(t(x.items())) == [
        ('e', (2, [])),
        ('a', (2, ['e'])),
        ('d', (2, [])),
        ('b', (2, ['d'])),
        ('x', (2, ['a', 'b', 'c'])),
        ('c', (2, ['d', 'x']))
    ]


def test_repeated_inf_break():
    t = MapReduceTask(verbose=True, lazy=False)

    @t.map
    def m1(k, v):
        nonlocal start_node
        n, neighbors = v
        state = 1 if n == start_node else 0
        yield n, (state, neighbors)

    with t.repeated() as repeated:
        @repeated.reduce
        def r1(n, l):
            state = 0
            neighbors = []
            for i in l:
                state = max(state, i[0])  # i.state
                neighbors += i[1]  # i.neighbors
            if state == 1:
                for o in neighbors:
                    yield o, (1, [])
                state = 2
            yield n, (state, neighbors)

        @repeated.map
        def m2_break(k, v):
            yield 'all', (k, v)

        @repeated.reduce
        def break_reduce(k, v):
            for n, l in v:
                if l[0] == 1:
                    break
            else:
                repeated.stop()

            for ki, vi in v:
                yield ki, vi

    @t.reduce
    def r2_back(k, v):
        for ki, vi in v:
            yield ki, vi

    start_node = 'x'
    x = {
        'x': ['a', 'b', 'c'],
        'a': ['e'],
        'b': ['d'],
        'c': ['d', 'x'],
    }
    # print newline, so the output will be on the new line when run by pytest
    print('')
    assert list(t(x.items())) == [
        ('e', (2, [])),
        ('a', (2, ['e'])),
        ('d', (2, [])),
        ('b', (2, ['d'])),
        ('x', (2, ['a', 'b', 'c'])),
        ('c', (2, ['d', 'x']))
    ]


def test_repeated_n_times_lazy():
    t = MapReduceTask(verbose=True, lazy=True)

    @t.map
    def m1(k, v):
        x = 1 / 0  # will raise ZeroDivisionError if evaluated
        nonlocal start_node
        n, neighbors = v
        state = 1 if n == start_node else 0
        yield n, (state, neighbors)

    with t.repeated(3) as repeated:
        @repeated.reduce
        def r1(n, l):
            state = 0
            neighbors = []
            for i in l:
                state = max(state, i[0])  # i.state
                neighbors += i[1]  # i.neighbors
            if state == 1:
                for o in neighbors:
                    yield o, (1, [])
                state = 2
            yield n, (state, neighbors)

    start_node = 'x'
    x = {
        'x': ['a', 'b', 'c'],
        'a': ['e'],
        'b': ['d'],
        'c': ['d', 'x'],
    }
    # print newline, so the output will be on the new line when run by pytest
    print('')
    try:
        t(x)
    except ZeroDivisionError:
        assert False, "should not be evaluated"
    else:
        assert True


def test_repeated_inf_break_lazy():
    t = MapReduceTask(verbose=True, lazy=True)

    @t.map
    def m1(k, v):
        x = 1 / 0  # will raise ZeroDivisionError if evaluated
        nonlocal start_node
        n, neighbors = v
        state = 1 if n == start_node else 0
        yield n, (state, neighbors)

    with t.repeated() as repeated:
        @repeated.reduce
        def r1(n, l):
            state = 0
            neighbors = []
            for i in l:
                state = max(state, i[0])  # i.state
                neighbors += i[1]  # i.neighbors
            if state == 1:
                for o in neighbors:
                    yield o, (1, [])
                state = 2
            yield n, (state, neighbors)

        @repeated.map
        def m2_break(k, v):
            yield 'all', (k, v)

        @repeated.reduce
        def break_reduce(k, v):
            for n, l in v:
                if l[0] == 1:
                    break
            else:
                repeated.stop()

            for ki, vi in v:
                yield ki, vi

    @t.reduce
    def r2_back(k, v):
        for ki, vi in v:
            yield ki, vi

    start_node = 'x'
    x = {
        'x': ['a', 'b', 'c'],
        'a': ['e'],
        'b': ['d'],
        'c': ['d', 'x'],
    }
    # print newline, so the output will be on the new line when run by pytest
    print('')
    try:
        t(x)
    except ZeroDivisionError:
        assert False, "should not be evaluated"
    else:
        assert True
