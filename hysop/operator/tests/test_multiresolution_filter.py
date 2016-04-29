from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.multiresolution_filter import MultiresolutionFilter
import hysop.tools.numpywrappers as npw
import numpy as np
from hysop.methods_keys import Remesh
from hysop.methods import Rmsh_Linear, L2_1
from hysop.mpi.main_var import main_size


L = [1., 1., 1.]
O = [0., 0., 0.]
simu = Simulation(tinit=0., tend=0.1, nbIter=1)


def func_periodic_X(res, x, y, z, t=0):
    res[0][...] = np.sin(2. * np.pi * x)
    return res


def func_periodic_Y(res, x, y, z, t=0):
    res[0][...] = np.sin(2. * np.pi * y)
    return res


def func_periodic_Z(res, x, y, z, t=0):
    res[0][...] = np.sin(2. * np.pi * z)
    return res


def func_periodic_XY(res, x, y, z, t=0):
    res[0][...] = np.sin(2. * np.pi * x) * np.cos(2. * np.pi * y)
    return res


def filter(d_fine, d_coarse, func, method, atol=1e-8, rtol=1e-5):
    box = Box(length=L, origin=O)
    f = Field(box, formula=func, name='f0')
    op = MultiresolutionFilter(d_in=d_fine, d_out=d_coarse,
                               variables={f: d_coarse},
                               method=method)
    op.discretize()
    op.setup()
    topo_coarse = op.discreteFields[f].topology
    topo_fine = [t for t in f.discreteFields.keys()
                 if not t is topo_coarse][0]
    f.initialize(topo=topo_fine)
    # f_in = f.discreteFields[topo_fine]
    f_out = f.discreteFields[topo_coarse]
    op.apply(simu)
    valid = [npw.zeros(f_out[0].shape), ]
    valid = func(valid, *topo_coarse.mesh.coords)
    e = np.max(np.abs(valid[0][topo_coarse.mesh.iCompute] -
                      f_out[0][topo_coarse.mesh.iCompute]))
    err = atol + rtol * np.max(np.abs(valid[0][topo_coarse.mesh.iCompute]))
    return np.allclose(f_out[0][topo_coarse.mesh.iCompute],
                       valid[0][topo_coarse.mesh.iCompute],
                       atol=atol, rtol=rtol), e, err


def test_linear_X():
    b, e, err = filter(d_coarse=Discretization([513, 5, 5],
                                               ghosts=[1, 1, 1]),
                       d_fine=Discretization([1025, 5, 5]),
                       method={Remesh: Rmsh_Linear, },
                       func=func_periodic_X)
    assert b, "max(|error|)=" + str(e) + " <= " + str(err)


def test_linear_Y():
    b, e, err = filter(d_coarse=Discretization([5, 513, 5],
                                               ghosts=[1, 1, 1]),
                       d_fine=Discretization([5, 1025, 5]),
                       method={Remesh: Rmsh_Linear, },
                       func=func_periodic_Y)
    assert b, "max(|error|)=" + str(e) + " <= " + str(err)


def test_linear_Z():
    b, e, err = filter(d_coarse=Discretization([5, 5, 513],
                                               ghosts=[1, 1, 1]),
                       d_fine=Discretization([5, 5, 1025]),
                       method={Remesh: Rmsh_Linear, },
                       func=func_periodic_Z)
    assert b, "max(|error|)=" + str(e) + " <= " + str(err)


def test_linear_XY():
    b, e, err = filter(d_coarse=Discretization([1025, 1025, 5],
                                               ghosts=[1, 1, 1]),
                       d_fine=Discretization([2049, 2049, 5]),
                       func=func_periodic_XY,
                       method={Remesh: Rmsh_Linear, },
                       atol=1e-3, rtol=1e-3)
    assert b, "max(|error|)=" + str(e) + " <= " + str(err)


def order_linear():
    e_old = 1.
    for i in (128, 256, 512, 1024, 2048):
        b, e, err = filter(d_coarse=Discretization([i + 1, 5, 5],
                                                   ghosts=[1, 1, 1]),
                           d_fine=Discretization([2 * i + 1, 5, 5]),
                           method={Remesh: Rmsh_Linear, },
                           func=func_periodic_X)
        if i > 128:
            print i, e_old / e
        e_old = e


def test_L21_X():
    b, e, err = filter(d_coarse=Discretization([513, 5, 5],
                                               ghosts=[2, 2, 2]),
                       d_fine=Discretization([1025, 5, 5]),
                       method={Remesh: L2_1, },
                       func=func_periodic_X)
    assert b, "max(|error|)=" + str(e) + " <= " + str(err)


def test_L21_Y():
    b, e, err = filter(d_coarse=Discretization([5, 513, 5],
                                               ghosts=[2, 2, 2]),
                       d_fine=Discretization([5, 1025, 5]),
                       method={Remesh: L2_1, },
                       func=func_periodic_X)
    assert b, "max(|error|)=" + str(e) + " <= " + str(err)


def test_L21_Z():
    b, e, err = filter(d_coarse=Discretization([5, 5, 513],
                                               ghosts=[2, 2, 2]),
                       d_fine=Discretization([5, 5, 1025]),
                       method={Remesh: L2_1, },
                       func=func_periodic_X)
    assert b, "max(|error|)=" + str(e) + " <= " + str(err)


def test_L21_XY():
    b, e, err = filter(d_coarse=Discretization([1025, 1025, 5],
                                               ghosts=[2, 2, 2]),
                       d_fine=Discretization([2049, 2049, 5]),
                       func=func_periodic_XY,
                       method={Remesh: L2_1, },
                       atol=1e-3, rtol=1e-3)
    assert b, "max(|error|)=" + str(e) + " <= " + str(err)


def order_L21():
    e_old = 1.
    for i in (128, 256, 512, 1024, 2048):
        b, e, err = filter(d_coarse=Discretization([i + 1, 5, 5],
                                                   ghosts=[1, 1, 1]),
                           d_fine=Discretization([2 * i + 1, 5, 5]),
                           method={Remesh: Rmsh_Linear, },
                           func=func_periodic_X)
        if i > 128:
            print i, e_old / e
        e_old = e


def func(res, x, y, z, t=0):
    res[0][...] = np.cos(2. * np.pi * x) * \
                  np.sin(2. * np.pi * y) * np.cos(4. * np.pi * z)
    return res


def test_filter_linear():
    """This test compares the GPU linear filter with python implementation"""
    box = Box(length=L, origin=O)
    f = Field(box, formula=func, name='f0')
    d_fine = Discretization([513, 513, 513])
    d_coarse = Discretization([257, 257, 257], ghosts=[1, 1, 1])
    op = MultiresolutionFilter(d_in=d_fine, d_out=d_coarse,
                               variables={f: d_coarse},
                               method={Remesh: Rmsh_Linear, })
    op.discretize()
    op.setup()
    topo_coarse = op.discreteFields[f].topology
    topo_fine = [t for t in f.discreteFields.keys()
                 if not t is topo_coarse][0]
    f.initialize(topo=topo_fine)
    f_out = f.discreteFields[topo_coarse]
    op.apply(simu)
    valid = [npw.zeros(f_out[0].shape), ]
    valid = func(valid, *topo_coarse.mesh.coords)
    assert np.allclose(valid[0][topo_coarse.mesh.iCompute],
                       f_out[0][topo_coarse.mesh.iCompute],
                       atol=1e-4, rtol=1e-3), \
        np.max(np.abs(valid[0][topo_coarse.mesh.iCompute] -
                      f_out[0][topo_coarse.mesh.iCompute]))


def test_filter_l2_1():
    """This test compares the GPU linear filter with python implementation"""
    box = Box(length=L, origin=O)
    f = Field(box, formula=func, name='f0')
    d_fine = Discretization([513, 513, 513])
    d_coarse = Discretization([257, 257, 257], ghosts=[2, 2, 2])
    op = MultiresolutionFilter(d_in=d_fine, d_out=d_coarse,
                               variables={f: d_coarse},
                               method={Remesh: L2_1, })
    op.discretize()
    op.setup()
    topo_coarse = op.discreteFields[f].topology
    topo_fine = [t for t in f.discreteFields.keys()
                 if not t is topo_coarse][0]
    f.initialize(topo=topo_fine)
    f_out = f.discreteFields[topo_coarse]
    op.apply(simu)
    valid = [npw.zeros(f_out[0].shape), ]
    valid = func(valid, *topo_coarse.mesh.coords)
    assert np.allclose(valid[0][topo_coarse.mesh.iCompute],
                       f_out[0][topo_coarse.mesh.iCompute]), \
        np.max(np.abs(valid[0][topo_coarse.mesh.iCompute] -
                      f_out[0][topo_coarse.mesh.iCompute]))

