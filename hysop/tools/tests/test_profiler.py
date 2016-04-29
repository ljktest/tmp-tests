"""
Unitary tests for hysop.tools.profiler module
"""
from hysop.tools.profiler import Profiler, profile, FProfiler, ftime
from hysop.mpi import main_comm


class A_class(object):
    def __init__(self):
        self.name = 'A_class'
        self.profiler = Profiler(self, main_comm)
        self.profiler += FProfiler('manual')
        self.n = 0

    @profile
    def call(self):
        self.n += 1

    @profile
    def call_other(self):
        self.n += 10

    def func(self):
        t = ftime()
        self.n += 100
        self.profiler['manual'] += ftime() - t


def test_profilers():
    a = A_class()
    assert len(a.profiler._elems.keys()) == 1
    assert a.n == 0
    a.call()
    assert len(a.profiler._elems.keys()) == 2
    assert a.n == 1  # the function have been called
    assert a.profiler['call'].n == 1
    a.call()
    a.call_other()
    assert len(a.profiler._elems.keys()) == 3
    assert a.n == 12  # the call and call_other functions have been called
    assert a.profiler['call'].n == 2
    assert a.profiler['call_other'].n == 1
    a.func()
    assert len(a.profiler._elems.keys()) == 3
    assert a.n == 112  # the call and call_other functions have been called
    assert a.profiler['call'].n == 2
    assert a.profiler['call_other'].n == 1
    assert a.profiler['manual'].n == 1
