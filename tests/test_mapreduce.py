from mapreduce import MapReduceTask

# run pytest -s to see the output
def test_readme_example():
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
    # print newline, so the output will be on the new line when run by pytest
    print('')
    assert list(t(x)) == [('max', (1, 3))]
