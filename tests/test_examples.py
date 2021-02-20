from mapreduce import MapReduceTask

# run pytest -s to see the output

def test_find_number_max_repeating():
    """
    Find the number which is repeated the largest number of times
    """
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


# Lecture 14
def test_word_count():
    """
    Word count
    """
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


def test_page_rank():
    t = MapReduceTask(verbose=True, lazy=False)

    @t.map
    def m0(k, v):
        nonlocal N
        url, outgoing = v
        l = []
        for t in outgoing:
            l.append(t)
        yield url, (1 / N, l)

    with t.repeated(4) as repeated:
        @repeated.reduce
        def r1(url, values):
            nonlocal alpha
            rank = 0
            l = []
            for v in values:
                rank += v[0] * (1 - alpha)
                l += v[1]  # in practice, all the list will be empty except for 1
            yield url, (rank, l)

        # we can stop after reduce phase to get final answer
        @repeated.map
        def check_stop(url, value):
            if repeated.remaining < 2:
                repeated.stop()
            yield url, value

        @repeated.map
        def m1(url, value):
            yield url, (alpha / N, value[1])
            len_value1 = len(value[1])
            for t in value[1]:
                yield t, (value[0] / len_value1, [])

    N = 4
    alpha = 0.5
    x = {
        'p1': ['p2', 'p3', 'p4'],
        'p2': ['p1'],
        'p3': ['p2'],
        'p4': ['p2'],
    }
    # print newline, so the output will be on the new line when run by pytest
    print('')
    answer = list(t(x.items()))
    print(answer)
    expected = dict([
        ('p1', (0.15, ['p2', 'p3', 'p4'])),
        ('p2', (0.175, ['p1'])),
        ('p3', (0.0875, ['p2'])),
        ('p4', (0.0875, ['p2']))
    ])
    sum_error = 0
    for page, values in answer:
        rank, outgoing = values
        error = abs(rank - expected[page][0])
        assert error < 0.01
        sum_error += error
    print('sum error =', sum_error)


def test_join_two_relations():
    st = MapReduceTask()
    @st.map
    def map_s(r, t):  # recond_id, tuple
        yield t.x, (t.a, 's')

    rt = MapReduceTask()
    @rt.map
    def map_r(r, t):  # recond_id, tuple
        yield t.y, (t.b, 'r')

    t = MapReduceTask()
    # input is enumerated, so we just need values
    @t.map
    def map_all(k, v):
        yield v[0], v[1]

    @t.reduce
    def reduce_all(key, values):
        s = []
        r = []
        for v in values:
            if v[1] == 's':
                s.append(v[0])
            else:
                r.append(v[0])
        for t1 in r:
            for t2 in s:
                yield key, (t1, t2)

    from collections import namedtuple
    nts = namedtuple('nts', ['x', 'a'])
    s = [
        nts(2, 3),
        nts(2, 4),
        nts(3, 6),
        nts(3, 5),
        nts(0, 1),
        nts(1, 2),
    ]
    ntr = namedtuple('ntr', ['y', 'b'])
    r = [
        ntr(2, 2),
        ntr(2, 3),
        ntr(3, 3),
        ntr(3, 4),
        ntr(3, 5),
        ntr(1, 0),
    ]
    # print newline, so the output will be on the new line when run by pytest
    print('')
    import itertools
    map_s = st(s)
    map_r = rt(r)
    reduce_all = t(itertools.chain(map_s, map_r))
    answer = list(reduce_all)
    print('[', ',\n '.join(map(str, answer)), ']', sep='')  # pretty print

    expected = [
        (2, (2, 3)),
        (2, (2, 4)),
        (2, (3, 3)),
        (2, (3, 4)),
        (3, (3, 6)),
        (3, (3, 5)),
        (3, (4, 6)),
        (3, (4, 5)),
        (3, (5, 6)),
        (3, (5, 5)),
        (1, (0, 2))
    ]
    assert len(expected) == len(answer)
    for t in answer:
        assert t in expected


# Lecture 15
def test_bfs():
    t = MapReduceTask(verbose=True, lazy=False)

    @t.map
    def m1(k, v):
        nonlocal start_node
        n, neighbors = v
        state = 1 if n == start_node else 0
        yield n, (state, neighbors)

    # repeated(3) would work in our case, but this approach works for any depth
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

        # changes made in the following block are discarded after exit
        # useful for testing condition for stopping repeated block
        with repeated.discarded(verbose=True) as discarded:
            # map everything to 1 key
            @discarded.map
            def m2_break(k, v):
                yield 'all', (k, v)

            # check if there are nodes with state 1
            @discarded.reduce
            def break_reduce(k, v):
                for n, l in v:
                    if l[0] == 1:  # i.state
                        break  # just exit, continue repeated block
                else:
                    # if there were no nodes with state 1 - stop repeated block
                    repeated.stop()
                # should have yield statement for the function to be generator
                if False: yield 1
                # or, can yield anything, the changes are discarded after exit
                # yield None, None
        # changes are discarded here

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


def test_k_means():
    import random
    random.seed(1)
    tk = MapReduceTask()
    @tk.map
    def m0(key, v):
        nonlocal k
        yield random.randint(1, k), v  # split between machines

    @tk.reduce
    def r0(key, l):
        nonlocal k
        # flatten list if needed
        try:
            iter(l[0])
            l = [i for sublist in l for i in sublist]
        except TypeError:
            pass
        random.shuffle(l)
        l = l[:k]
        yield 0, l

    # execute the same reduce function 2nd time
    # now, the entries are on the same machine in the case of distributed computation
    tk.reduce(r0)

    t = MapReduceTask()
    # phase 2
    @t.map
    def m1(key, n):
        nonlocal k, c
        d = 1e8
        b = 0
        for i in range(k):
            delta = abs(n - c[i])
            if delta < d:
                d = delta
                b = i
        yield b, n

    # phase 3
    @t.reduce
    def r1(i, l):
        s = 0
        for n in l:
            s += n
        s /= len(l)
        yield 0, (i, s)

    @t.reduce
    def r2(key, l):
        l.sort(key=lambda x: x[0])
        l = [i[1] for i in l]
        yield 0, l

    k = 3
    d = [1, 2, 3, 7, 8, 9, 12, 13, 14]
    random.shuffle(d)

    print('')
    c = list(tk(d))[0][1]  # must be available on all machines
    print(c)
    for i in range(3):
        print('-' * 10, 'Repeat {}'.format(i), '-' * 10)
        new_c = list(t(d))[0][1]
        print(new_c)
        c = new_c

    expected = [2, 8, 13]
    assert len(c) == k
    sum_error = 0
    for actual, t in zip(sorted(c), expected):
        error = abs(actual - t)
        assert error < 0.1
        sum_error += error
    print('sum error =', sum_error)


# Sheet8
def test_multi_sets_equal_fields():
    tx = MapReduceTask()
    @tx.map
    def m0_x(k, v):
        yield v[0], v[1]

    @tx.map
    def map_x(a, b):
        yield b, ('A', a, b)

    ty = MapReduceTask()
    @ty.map
    def m0_y(k, v):
        yield v[0], v[1]

    @ty.map
    def map_y(c, d):
        yield c, ('B', c, d)

    import itertools
    t = MapReduceTask()
    @t.map
    def map_all(k, v):
        yield v[0], v[1]

    @t.reduce
    def reduce_all(k, l):
        l_a = filter(lambda x: x[0] == 'A', l)
        l_b = filter(lambda x: x[0] == 'B', l)
        for x in itertools.product(l_a, l_b):
            yield 'result', x

    print('')
    x = [
        (2, 1),
        (2, 1),
        (5, 4),
        (1, 0),
        (4, 3),
        (3, 2),
    ]
    y = [
        (4, 5),
        (2, 3),
        (5, 6),
        (5, 6),
        (1, 2),
        (1, 2),
        (3, 4),
    ]
    map_x = tx(x)
    map_y = ty(y)
    answer = t(itertools.chain(map_x, map_y))
    print('[', ',\n '.join(map(str, answer)), ']', sep='')  # pretty print
    expected = [
        (('A', 2, 1), ('B', 1, 2)),
        (('A', 2, 1), ('B', 1, 2)),
        (('A', 2, 1), ('B', 1, 2)),
        (('A', 2, 1), ('B', 1, 2)),
        (('A', 5, 4), ('B', 4, 5)),
        (('A', 4, 3), ('B', 3, 4)),
        (('A', 3, 2), ('B', 2, 3))
    ]
    assert len(expected) == len(answer)
    for k, t in answer:
        assert t in expected


def test_docs_common_words():
    tc = MapReduceTask()
    @tc.map
    def map_count(k, v):
        doc, words = v
        yield 'count', 1

    @tc.reduce
    def get_count(k, l):
        yield k, sum(l)

    t = MapReduceTask()
    @t.map
    def m0(k, v):
        doc, words = v
        yield doc, words

    @t.map
    def map_words(doc, words):
        for word in words:
            yield word, doc

    @t.reduce
    def reduce_words(word, docs):
        yield word, len(set(docs))

    print('')
    d = {
        "doc1": ['word1', 'word2', 'word3', 'word2'],
        "doc2": ['word2', 'word3', 'word5'],
        "doc3": ['word1', 'word2', 'word3', 'word4'],
    }
    c = tc(d.items())
    nr_docs = list(c)[0][1]
    words = t(d.items())
    answer = list(filter(lambda x: x[1] == nr_docs, words))
    print(answer)
    assert answer == [('word2', 3), ('word3', 3)]


def test_matrix_product():
    ta = MapReduceTask()
    @ta.map
    def m0_a(k, v):
        yield (v[0], v[1]), v[2]

    @ta.map
    def map_a(k, value):
        in_row, in_col = k
        nonlocal p
        for out_col in range(1, p + 1):
            yield (in_row, out_col, in_col), value

    tb = MapReduceTask()
    @tb.map
    def m0_b(k, v):
        yield (v[0], v[1]), v[2]

    @tb.map
    def map_b(k, value):
        in_row, in_col = k
        nonlocal n
        for out_row in range(1, n + 1):
            yield (out_row, in_col, in_row), value

    t = MapReduceTask()
    import functools
    @t.map
    def m0_all(k, v):
        yield v[0], v[1]

    @t.reduce
    def reduce_mult(k, values):
        print(k, values)
        out_row, out_col, idx = k
        yield (out_row, out_col), functools.reduce(lambda a, b: a * b, values)

    @t.reduce
    def reduce_sum(k, values):
        # in the solutions it was (out_col, out_row) but that doesn't work
        out_row, out_col = k
        yield (out_row , out_col), sum(values)

    #{{7, 3}, {2, 5}, {6, 8}, {9, 0}}
    a = [
        [7, 3],
        [2, 5],
        [6, 8],
        [9, 0],
    ]
    n, m = len(a), len(a[0])
    #{{7, 4, 9}, {8, 1, 5}}
    b = [
        [7, 4, 9],
        [8, 1, 5],
    ]
    m2, p = len(b), len(b[0])
    assert m2 == m, "matrices have imcompatible shapes"

    la = []
    for row in range(1, n + 1):
        for col in range(1, m + 1):
            la.append((row, col, a[row - 1][col - 1]))
    lb = []
    for row in range(1, m + 1):
        for col in range(1, p + 1):
            lb.append((row, col, b[row - 1][col - 1]))

    import itertools
    map_a = ta(la)
    map_b = tb(lb)
    answer = t(itertools.chain(map_a, map_b))

    answer = list(answer)
    print(answer)
    ans = [[0] * 3 for i in range(4)]
    for ii, value in answer:
        row, col = ii
        try:
            ans[row - 1][col - 1] = value
        except IndexError:
            print(row, col, ans)
    print('[', ',\n '.join(map(str, ans)), ']', sep='')  # pretty print
    #{{73, 31, 78}, {54, 13, 43}, {106, 32, 94}, {63, 36, 81}}
    assert ans == [
        [73, 31, 78],
        [54, 13, 43],
        [106, 32, 94],
        [63, 36, 81]
    ]
