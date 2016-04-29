from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization, MPIParams
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.multiresolution_filter import MultiresolutionFilter
import hysop.tools.numpywrappers as npw
import numpy as np
from hysop.methods_keys import Remesh, Support, ExtraArgs
from hysop.methods import Rmsh_Linear, L2_1
# In parallel we need to use as many threads as gpu
from hysop.mpi.main_var import main_size, main_rank
import pyopencl as cl
n_gpu = len(cl.get_platforms()[0].get_devices(
    device_type=cl.device_type.GPU))
PROC_TASKS = [0, ] * main_size
if main_rank < n_gpu:
    PROC_TASKS[main_rank] = 1


L = [1., 1., 1.]
O = [0., 0., 0.]
simu = Simulation(tinit=0., tend=0.1, nbIter=1)
PY_COMPARE = True


def func(res, x, y, z, t=0):
    res[0][...] = np.cos(2. * np.pi * x) * \
                  np.sin(2. * np.pi * y) * np.cos(4. * np.pi * z)
    return res


def test_filter_linear():
    """This test compares the GPU linear filter with python implementation"""
    box = Box(length=L, origin=O, proc_tasks=PROC_TASKS)
    mpi_p = MPIParams(comm=box.comm_task, task_id=1)
    f = Field(box, formula=func, is_vector=False, name='f1')
    d_fine = Discretization([513, 513, 513])
    d_coarse = Discretization([257, 257, 257], ghosts=[1, 1, 1])
    op = MultiresolutionFilter(d_in=d_fine, d_out=d_coarse,
                               variables={f: d_coarse},
                               method={Remesh: Rmsh_Linear,
                                       Support: 'gpu',
                                       ExtraArgs: {'device_id': main_rank, }},
                               mpi_params=mpi_p)
    if box.is_on_task(1):
        op.discretize()
        op.setup()
        topo_coarse = op.discreteFields[f].topology
        topo_fine = [t for t in f.discreteFields.keys()
                     if not t is topo_coarse][0]
        f.initialize(topo=topo_fine)
        f_out = f.discreteFields[topo_coarse]
        f_out.toDevice()
        op.apply(simu)
        f_out.toHost()
        f_out.wait()
        valid = [npw.zeros(f_out[0].shape), ]
        valid = func(valid, *topo_coarse.mesh.coords)
        assert np.allclose(valid[0][topo_coarse.mesh.iCompute],
                           f_out[0][topo_coarse.mesh.iCompute],
                           atol=1e-4, rtol=1e-3), \
            np.max(np.abs(valid[0][topo_coarse.mesh.iCompute] -
                          f_out[0][topo_coarse.mesh.iCompute]))
        if PY_COMPARE:
            f_py = Field(box, formula=func, name='fpy')
            op_py = MultiresolutionFilter(d_in=d_fine, d_out=d_coarse,
                                          variables={f_py: d_coarse},
                                          method={Remesh: Rmsh_Linear, },
                                          mpi_params=mpi_p)
            op_py.discretize()
            op_py.setup()
            f_py.initialize(topo=topo_fine)
            op_py.apply(simu)
            valid = f_py.discreteFields[topo_coarse]
            assert np.allclose(valid[0][topo_coarse.mesh.iCompute],
                               f_out[0][topo_coarse.mesh.iCompute]), \
                np.max(np.abs(valid[0][topo_coarse.mesh.iCompute] -
                              f_out[0][topo_coarse.mesh.iCompute]))


def test_filter_L2_1():
    """
    This test compares the GPU L2_1 filter with the expected result
    on the coarse grid and with python implementation.
    """
    box = Box(length=L, origin=O, proc_tasks=PROC_TASKS)
    mpi_p = MPIParams(comm=box.comm_task, task_id=1)
    f = Field(box, formula=func, name='f1')
    d_fine = Discretization([513, 513, 513])
    d_coarse = Discretization([257, 257, 257], ghosts=[2, 2, 2])
    if box.is_on_task(1):
        op = MultiresolutionFilter(d_in=d_fine, d_out=d_coarse,
                                   variables={f: d_coarse},
                                   method={Remesh: L2_1,
                                           Support: 'gpu',
                                           ExtraArgs: {'device_id': main_rank, }},
                                   mpi_params=mpi_p)
        op.discretize()
        op.setup()
        topo_coarse = op.discreteFields[f].topology
        topo_fine = [t for t in f.discreteFields.keys()
                     if not t is topo_coarse][0]
        f.initialize(topo=topo_fine)
        f_out = f.discreteFields[topo_coarse]
        f_out.toDevice()
        op.apply(simu)
        f_out.toHost()
        f_out.wait()
        valid = [npw.zeros(f_out[0].shape), ]
        valid = func(valid, *topo_coarse.mesh.coords)
        assert np.allclose(valid[0][topo_coarse.mesh.iCompute],
                           f_out[0][topo_coarse.mesh.iCompute]), \
            np.max(np.abs(valid[0][topo_coarse.mesh.iCompute] -
                          f_out[0][topo_coarse.mesh.iCompute]))
        if PY_COMPARE:
            f_py = Field(box, formula=func, name='fpy')
            op_py = MultiresolutionFilter(d_in=d_fine, d_out=d_coarse,
                                          variables={f_py: d_coarse},
                                          method={Remesh: L2_1, })
            op_py.discretize()
            op_py.setup()
            f_py.initialize(topo=topo_fine)
            op_py.apply(simu)
            valid = f_py.discreteFields[topo_coarse]
            assert np.allclose(valid[0][topo_coarse.mesh.iCompute],
                               f_out[0][topo_coarse.mesh.iCompute]), \
                np.max(np.abs(valid[0][topo_coarse.mesh.iCompute] -
                              f_out[0][topo_coarse.mesh.iCompute]))
