"""
@file test_simulation.py
tests simulation incr and io_utils writer
"""
from hysop.problem.simulation import Simulation
from hysop.tools.io_utils import Writer, IOParams, IO

simu = Simulation(tinit=0.0, tend=1.0, nbIter=10)


def test_simu_incr():
    io_params = IOParams(filename='temp_test', frequency=2,
                         fileformat=IO.ASCII)
    wr = Writer(io_params)
    assert wr.do_write(simu.currentIteration)

    simu.initialize()

    assert not wr.do_write(simu.currentIteration)

    count = 1
    while not simu.isOver:
        if count % 2 == 0:
            assert wr.do_write(simu.currentIteration)
        else:
            assert not wr.do_write(simu.currentIteration)
        simu.printState()
        simu.advance()
        count += 1
    assert simu.currentIteration == 10
    simu.finalize()
    assert wr.do_write(simu.currentIteration)


def test_simu_incr2():
    io_params = IOParams(filename='temp_test', frequency=3,
                         fileformat=IO.ASCII)
    wr = Writer(io_params)
    assert wr.do_write(simu.currentIteration)
    simu.timeStep = 0.10000000001
    simu.initialize()

    assert not wr.do_write(simu.currentIteration)

    count = 1
    while not simu.isOver:
        if count % 3 == 0:
            assert wr.do_write(simu.currentIteration)
        else:
            assert not wr.do_write(simu.currentIteration)
        simu.printState()
        simu.advance()
        count += 1
    assert simu.currentIteration == 10
    simu.finalize()
    assert wr.do_write(simu.currentIteration)
