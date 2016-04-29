# -*- coding: utf-8 -*-
import hysop as pp
from hysop.operator.adapt_timestep import AdaptTimeStep
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop.mpi import main_comm
import numpy as np
import hysop.tools.numpywrappers as npw
import os
sin = np.sin
cos = np.cos

d3d = Discretization([33, 33, 33], [2, 2, 2])


def computeVel(res, x, y, z, t):
    res[0][...] = sin(x * t) * cos(y) * cos(z)
    res[1][...] = - cos(x) * sin(y) * cos(z)
    res[2][...] = 0.
    return res


def computeVort(res, x, y, z, t):
    res[0][...] = 0.
    res[1][...] = 0.
    res[2][...] = sin(x) * cos(y * t) * cos(z)
    return res


def init():
    box = pp.Box(length=[2.] * 3, origin=[0.0] * 3)
    velo = pp.Field(domain=box, formula=computeVel,
                    name='Velocity', is_vector=True)
    vorti = pp.Field(domain=box, formula=computeVort,
                     name='Vorticity', is_vector=True)
    return velo, vorti


def test_adapt():
    """
    Todo : write proper tests.
    Here we just check if discr/setup/apply process goes well.
    """
    velo, vorti = init()
    simu = Simulation(nbIter=2)
    op = AdaptTimeStep(velo, vorti, simulation=simu,
                       discretization=d3d, lcfl=0.125, cfl=0.5)
    op.discretize()
    op.setup()
    op.apply()
    op.wait()


def test_adapt_2():
    """
    The same but with file output
    """
    velo, vorti = init()
    simu = Simulation(nbIter=2)
    op = AdaptTimeStep(velo, vorti, simulation=simu, io_params=True,
                       discretization=d3d, lcfl=0.125, cfl=0.5)
    op.discretize()
    op.setup()
    op.apply()
    op.wait()
    filename = op.io_params.filename
    assert os.path.exists(filename)


def test_adapt_3():
    """
    The same but with external work vector
    """
    velo, vorti = init()
    simu = Simulation(nbIter=2)
    op = AdaptTimeStep(velo, vorti, simulation=simu, io_params=True,
                       discretization=d3d, lcfl=0.125, cfl=0.5)
    op.discretize()
    wk_p = op.get_work_properties()
    rwork = []
    wk_length = len(wk_p['rwork'])
    for i in xrange(wk_length):
        memshape = wk_p['rwork'][i]
        rwork.append(npw.zeros(memshape))

    op.setup(rwork=rwork)
    op.apply(simu)
    op.wait()
    filename = op.io_params.filename
    assert os.path.exists(filename)


def test_adapt_4():
    """
    The same but with external work vector
    """
    # MPI procs are distributed among two tasks
    GPU = 4
    CPU = 1
    VISU = 12
    from hysop.mpi.main_var import main_size
    proc_tasks = [CPU, ] * main_size

    if main_size > 4:
        proc_tasks[-1] = GPU
        proc_tasks[2] = GPU
        proc_tasks[1] = VISU

    dom = pp.Box(dimension=3, proc_tasks=proc_tasks)
    velo = pp.Field(domain=dom, formula=computeVel,
                    name='Velocity', is_vector=True)
    vorti = pp.Field(domain=dom, formula=computeVort,
                     name='Vorticity', is_vector=True)

    from hysop.tools.parameters import MPIParams
    cpu_task = MPIParams(comm=dom.comm_task, task_id=CPU)
    simu = Simulation(nbIter=4)
    op = AdaptTimeStep(velo, vorti, simulation=simu, io_params=True,
                       discretization=d3d, lcfl=0.125, cfl=0.5,
                       mpi_params=cpu_task)
    simu.initialize()
    if dom.is_on_task(CPU):
        op.discretize()
        op.setup()
        vorti.initialize()

    while not simu.isOver:
        if dom.is_on_task(CPU):
            op.apply()
        op.wait()
        simu.advance()
        refval = 0
        if dom.is_on_task(CPU):
            refval = simu.timeStep
        refval = main_comm.bcast(refval, root=0)
        assert refval == simu.timeStep
