import hysop as pp
from hysop.domain.box import Box
from hysop.constants import DEFAULT_TASK_ID
from hysop.tools.parameters import Discretization
from hysop.mpi import main_size
import numpy as np
import hysop.tools.numpywrappers as npw


N = 33
r1D = Discretization([N])
r2D = Discretization([N, 17])  # No ghosts
r3D = Discretization([N, N, 17])  # No ghosts
r3DGh = Discretization([N, N, 17], [2, 2, 2])  # Ghosts

CPU = DEFAULT_TASK_ID
GPU = 29
proc_tasks = [CPU] * main_size
if main_size > 2:
    proc_tasks[-1] = GPU
    proc_tasks[0] = GPU

dom3D = Box(proc_tasks=proc_tasks)
dom2D = Box(dimension=2, proc_tasks=proc_tasks)
dom3D_notask = Box()

# A mesh of reference for comparion.
# Obviously we assume that default topo constructor works well...
toporef = dom3D.create_topology(r3DGh, dim=1)
refmesh = toporef.mesh
toporef_notask = dom3D_notask.create_topology(r3DGh, dim=1)
toporef2d = dom2D.create_topology(r2D, dim=1)
refmesh2d = toporef2d.mesh


def check2D(topo):
    assert topo.size == main_size
    assert topo.task_id() == DEFAULT_TASK_ID
    assert np.allclose(topo.mesh.discretization.resolution,
                       r2D.resolution)


def check3D(topo):
    assert topo.size == main_size
    assert topo.task_id() == DEFAULT_TASK_ID
    assert np.allclose(topo.mesh.discretization.resolution,
                       r3D.resolution)


# ===== 2D domains ====
# Default:
def test_create_default_topology_2d():
    dom = Box(dimension=2)
    topo = dom.create_topology(r2D)
    assert topo.domain is dom
    check2D(topo)


# Test taskid
def test_create_default_topology2_2d():
    dom = Box(dimension=2, proc_tasks=proc_tasks)
    topo = dom.create_topology(r2D)
    assert topo.domain == dom
    assert topo.size == dom2D.comm_task.Get_size()
    if dom.is_on_task(CPU):
        assert topo.task_id() == CPU
    if dom.is_on_task(GPU):
        assert topo.task_id() == GPU


# Input : dimension
def test_create_topologyFromDim_2d():
    dom = Box(dimension=2)
    topo1 = dom.create_topology(r2D, dim=1)
    check2D(topo1)
    topo2 = dom.create_topology(r2D, dim=2)
    check2D(topo2)


# Input : shape
def test_create_topologyFromShape_2d():
    dom = Box(dimension=2)
    if main_size == 8:
        topoShape = npw.asdimarray([2, 4])
        topo = dom.create_topology(r2D, shape=topoShape)
        assert topo.domain == dom
        assert topo.dimension == 2
        assert topo.size == pp.mpi.main_size
        assert (topo.shape == topoShape).all()
        assert (topo.mesh.resolution == [16, 4]).all()

    else:
        shape = [main_size, 1]
        topo = dom.create_topology(r2D, shape=shape)
        assert (topo.shape == shape).all()
        assert topo.dimension == 1
        check2D(topo)


# Input = cutdir
def test_create_topologyFromCutdir_2d():
    dom = Box(dimension=2)
    if main_size >= 4:
        topo = dom.create_topology(r2D, cutdir=[False, True])
        assert topo.domain == dom
        assert topo.dimension == 1
        assert topo.size == pp.mpi.main_size
        assert (topo.shape == [1, main_size]).all()

    topo2 = dom.create_topology(r2D, cutdir=[True, False])
    assert (topo2.shape == [main_size, 1]).all()
    assert topo2.dimension == 1
    check2D(topo2)


# plane topo with input mesh
def test_create_planetopology_2d():
    dom = Box(dimension=2)
    offs = refmesh2d.start()
    lres = refmesh2d.resolution
    topo = dom.create_plane_topology_from_mesh(global_start=offs,
                                               localres=lres,
                                               discretization=r2D,
                                               )
    assert topo.domain == dom
    assert topo.dimension == 1
    assert topo.size == pp.mpi.main_size
    assert (topo.shape == [1, main_size]).all()
    assert topo.mesh == refmesh2d
    topo2 = dom.create_plane_topology_from_mesh(discretization=r2D,
                                                global_start=offs,
                                                localres=lres, cdir=0)
    assert topo2.domain == dom
    assert topo2.dimension == 1
    assert topo2.size == pp.mpi.main_size
    assert (topo2.shape == [main_size, 1]).all()


# ===== 3D domains ====
# Default:
def test_create_default_topology():
    dom = Box()
    topo = dom.create_topology(r3D)
    assert topo.domain is dom
    check3D(topo)


# Test taskid
def test_create_default_topology2():
    dom = Box(proc_tasks=proc_tasks)
    topo = dom.create_topology(r3D)
    assert topo.domain == dom
    assert topo.size == dom3D.comm_task.Get_size()
    if dom.is_on_task(CPU):
        assert topo.task_id() == CPU
    if dom.is_on_task(GPU):
        assert topo.task_id() == GPU


# Input : dimension
def test_create_topologyFromDim():
    dom = Box()
    topo1 = dom.create_topology(r3D, dim=1)
    check3D(topo1)
    topo2 = dom.create_topology(r3D, dim=2)
    check3D(topo2)
    topo3 = dom.create_topology(r3D, dim=3)
    check3D(topo3)


# Input : shape
def test_create_topologyFromShape():
    dom = Box()
    if main_size == 8:
        topoShape = npw.asdimarray([2, 2, 2])
        topo = dom.create_topology(r3D, shape=topoShape)
        assert topo.domain == dom
        assert topo.dimension == 3
        assert topo.size == pp.mpi.main_size
        assert (topo.shape == topoShape).all()
        assert (topo.mesh.resolution == [16, 16, 8]).all()

    else:
        shape = [main_size, 1, 1]
        topo = dom.create_topology(r3D, shape=shape)
        assert (topo.shape == shape).all()
        assert topo.dimension == 1
        check3D(topo)


# Input = cutdir
def test_create_topologyFromCutdir():
    dom = Box()
    if main_size == 8:
        topo = dom.create_topology(r3D, cutdir=[False, True, True])
        assert topo.domain == dom
        assert topo.dimension == 2
        assert topo.size == pp.mpi.main_size
        assert (topo.shape == [1, 2, 4]).all()

    topo2 = dom.create_topology(r3D, cutdir=[False, True, False])
    assert (topo2.shape == [1, main_size, 1]).all()
    assert topo2.dimension == 1
    check3D(topo2)


# plane topo with input mesh
def test_create_planetopology():
    dom = Box()
    offs = refmesh.start()
    lres = refmesh.resolution
    topo = dom.create_plane_topology_from_mesh(discretization=r3DGh,
                                               global_start=offs,
                                               localres=lres)
    assert topo.domain == dom
    assert topo.dimension == 1
    assert topo.size == pp.mpi.main_size
    assert (topo.shape == [1, 1, main_size]).all()
    assert topo.mesh == refmesh
    topo2 = dom.create_plane_topology_from_mesh(discretization=r3DGh,
                                                global_start=offs,
                                                localres=lres, cdir=1)
    assert topo2.domain == dom
    assert topo2.dimension == 1
    assert topo2.size == pp.mpi.main_size
    assert (topo2.shape == [1, main_size, 1]).all()


def test_operator_equal():
    dom = Box()
    topoDims = [main_size, 1, 1]
    topo = dom.create_topology(r3DGh, shape=topoDims)
    mesh = toporef_notask.mesh
    topo2 = Box().create_plane_topology_from_mesh(
        discretization=r3DGh, global_start=mesh.start(),
        localres=mesh.resolution, cdir=2)
    # Same as topo2 but the discretization
    topo3 = Box().create_plane_topology_from_mesh(
        discretization=r3D, global_start=mesh.start(),
        localres=mesh.resolution, cdir=2)
    assert topo2.mesh == mesh
    assert (topo2.shape == toporef_notask.shape).all()
    assert topo2.domain == toporef_notask.domain
    assert topo2 == toporef_notask
    assert not topo2 == topo3
    if main_size > 1:
        assert not topo == topo2
    else:
        assert topo == topo2

    # test not equal ...
    assert topo2 != topo3
