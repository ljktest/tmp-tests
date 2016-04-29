"""
Testing hysop.domain.box.Box
"""
from hysop.constants import PERIODIC, DEFAULT_TASK_ID
from hysop.domain.box import Box
from numpy import allclose, ones_like, zeros_like
import hysop.tools.numpywrappers as npw
from hysop.mpi import main_size, main_rank


def test_create_box1():
    dom = Box()
    assert dom.dimension == 3
    assert allclose(dom.length, ones_like(dom.length))
    assert allclose(dom.origin, zeros_like(dom.origin))
    assert [b == PERIODIC for b in dom.boundaries]
    cond = [dom.tasks_on_proc(i) == DEFAULT_TASK_ID for i in range(main_size)]
    cond = npw.asboolarray(cond)
    assert cond.all()


def test_create_box2():
    length = npw.asrealarray([1., 2.5])
    ori = npw.asrealarray([1, 2.1])
    dom = Box(length=length, origin=ori)
    assert dom.dimension == 2
    assert allclose(dom.length, length)
    assert allclose(dom.origin, ori)
    assert [b == PERIODIC for b in dom.boundaries]
    cond = [dom.tasks_on_proc(i) == DEFAULT_TASK_ID for i in range(main_size)]
    cond = npw.asboolarray(cond)
    assert cond.all()


def test_create_box3():
    length = [1, 2, 4.]
    dom = Box(length=length)
    assert dom.dimension == 3
    assert allclose(dom.length, npw.asrealarray(length))
    assert allclose(dom.origin, zeros_like(length))
    assert [b == PERIODIC for b in dom.boundaries]
    cond = [dom.tasks_on_proc(i) == DEFAULT_TASK_ID for i in range(main_size)]
    cond = npw.asboolarray(cond)
    assert cond.all()


def test_create_box4():
    length = [1, 2, 4.]
    tasks = [CPU] * main_size
    if main_size > 1:
        tasks[-1] = GPU
    dom = Box(length=length, proc_tasks=tasks)

    last = main_size - 1
    if main_size > 1:
        if main_rank != last:
            assert dom.current_task() == CPU
        else:
            assert dom.current_task() == GPU
    else:
        assert dom.current_task() == CPU


# Test topology creation ...
N = 33
from hysop.tools.parameters import Discretization, MPIParams
r3D = Discretization([N, N, 17])  # No ghosts
r3DGh = Discretization([N, N, 17], [2, 2, 2])  # Ghosts

CPU = 12
GPU = 29
proc_tasks = [CPU] * main_size
if main_size > 1:
    proc_tasks[-1] = GPU
from hysop.mpi import main_comm
comm_s = main_comm.Split(color=proc_tasks[main_rank], key=main_rank)
mpCPU = MPIParams(comm=comm_s, task_id=CPU)
mpGPU = MPIParams(comm=comm_s, task_id=GPU)

from hysop.mpi.topology import Cartesian


def test_topo_standard():
    dom = Box()
    topo = dom.create_topology(discretization=r3D)
    assert len(dom.topologies) == 1
    assert isinstance(topo, Cartesian)
    assert topo is dom.topologies.values()[0]
    topo2 = dom.create_topology(discretization=r3DGh)
    assert len(dom.topologies) == 2
    assert isinstance(topo2, Cartesian)
    topo3 = dom.create_topology(discretization=r3DGh)
    assert len(dom.topologies) == 2
    assert topo3 is topo2


def test_topo_multi_tasks():
    dom = Box(proc_tasks=proc_tasks)
    if dom.is_on_task(CPU):
        topo = dom.create_topology(discretization=r3D)
    elif dom.is_on_task(GPU):
        topo = dom.create_topology(discretization=r3DGh, dim=2)
    assert len(dom.topologies) == 1
    assert isinstance(topo, Cartesian)
    assert topo is dom.topologies.values()[0]
    if dom.is_on_task(CPU):
        assert not topo.has_ghosts()
    elif dom.is_on_task(GPU):
        assert topo.has_ghosts()


def test_topo_plane():
    # e.g. for advectionDir
    dom = Box()
    topo = dom.create_topology(discretization=r3D,
                               cutdir=[False, True, False])
    assert len(dom.topologies) == 1
    assert isinstance(topo, Cartesian)
    assert topo is dom.topologies.values()[0]
    assert topo.dimension == 1
    assert topo.shape[1] == main_size


def test_topo_from_mesh():
    # e.g. for fftw
    dom = Box(proc_tasks=proc_tasks)
    from hysop.f2hysop import fftw2py
    if dom.is_on_task(CPU):
        localres, global_start = fftw2py.init_fftw_solver(
            r3D.resolution, dom.length, comm=comm_s.py2f())
        print localres, global_start
        topo = dom.create_plane_topology_from_mesh(localres=localres,
                                                   global_start=global_start,
                                                   discretization=r3D)
    elif dom.is_on_task(GPU):
        topo = dom.create_topology(discretization=r3DGh, dim=2)
    if dom.is_on_task(CPU):
        assert (topo.mesh.resolution == localres).all()
        assert (topo.mesh.start() == global_start).all()
        assert topo.dimension == 1
        assert (topo.shape == [1, 1, comm_s.Get_size()]).all()
    elif dom.is_on_task(GPU):
        assert topo.size == 1
