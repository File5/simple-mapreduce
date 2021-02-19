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
