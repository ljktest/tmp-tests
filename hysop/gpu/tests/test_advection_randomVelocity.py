"""
@file hysop.gpu.tests.test_advection_randomVelocity
Testing advection kernels with a random velocity field.
"""
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.advection import Advection
from hysop.constants import np
from hysop.problem.simulation import Simulation
from hysop.methods_keys import TimeIntegrator, Interpolation, Remesh, \
    Support, Splitting
from hysop.numerics.integrators.runge_kutta2 import RK2
from hysop.numerics.interpolation import Linear
from hysop.numerics.remeshing import L2_1, L4_2, M8Prime
from hysop.tools.parameters import Discretization
import hysop.tools.numpywrappers as npw


def setup_2D():
    box = Box(length=[1., 1.], origin=[0., 0.])
    scal = Field(domain=box, name='Scalar')
    velo = Field(domain=box, name='Velocity', is_vector=True)
    return scal, velo


def setup_3D():
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar')
    velo = Field(domain=box, name='Velocity', is_vector=True)
    return scal, velo


def assertion_2D_withPython(scal, velo, advec, advec_py):
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    velo_d.data[0][...] = np.asarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = np.asarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[1])
    scal_d.toDevice()
    velo_d.toDevice()

    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    py_res = scal_d.data[0].copy()
    scal_d.toHost()

    advec.finalize()
    err_m = np.max(np.abs(py_res - scal_d.data[0]))
    assert np.allclose(py_res, scal_d.data[0], rtol=1e-04, atol=1e-07), str(err_m)


def assertion_3D_withPython(scal, velo, advec, advec_py):
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    velo_d.data[0][...] = npw.zeros_like(
        scal_d.data[0]) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.zeros_like(
        scal_d.data[0]) / (2. * scal_d.resolution[1])
    velo_d.data[2][...] = npw.zeros_like(
        scal_d.data[0]) / (2. * scal_d.resolution[2])
    scal_d.toDevice()
    velo_d.toDevice()

    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    py_res = scal_d.data[0].copy()
    scal_d.toHost()

    advec.finalize()
    err_m = np.max(np.abs(py_res - scal_d.data[0]))
    assert np.allclose(py_res, scal_d.data[0], rtol=1e-04, atol=1e-07), str(err_m)

d2d = Discretization([33, 33])

# M6 testing
def test_2D_m6_1k():
    """
    Testing M6 remeshing formula in 2D, 1 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_2k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_2D_m6_2k():
    """
    Testing M6 remeshing formula in 2D, 2 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_2k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_2D_m6_1k_sFH():
    """
    Testing M6 remeshing formula in 2D, 1 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_2k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_2D_m6_2k_sFH():
    """
    Testing M6 remeshing formula in 2D, 2 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_2k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_3D_m6_1k():
    """
    Testing M6 remeshing formula in 3D, 1 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_1k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_3D_m6_2k():
    """
    Testing M6 remeshing formula in 3D, 2 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_2k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2'}
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_3D_m6_1k_sFH():
    """
    Testing M6 remeshing formula in 3D, 1 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_1k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_3D_m6_2k_sFH():
    """
    Testing M6 remeshing formula in 3D, 2 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_2k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


# M4 testing
def test_2D_m4_1k():
    """
    Testing M4 remeshing formula in 2D, 1 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L2_1,
                              Support: 'gpu_1k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_2D_m4_2k():
    """
    Testing M4 remeshing formula in 2D, 2 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L2_1,
                              Support: 'gpu_2k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_2D_m4_1k_sFH():
    """
    Testing M4 remeshing formula in 2D, 1 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L2_1,
                              Support: 'gpu_1k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_2D_m4_2k_sFH():
    """
    Testing M4 remeshing formula in 2D, 2 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L2_1,
                              Support: 'gpu_2k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_3D_m4_1k():
    """
    Testing M4 remeshing formula in 3D, 1 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L2_1,
                              Support: 'gpu_1k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_3D_m4_2k():
    """
    Testing M4 remeshing formula in 3D, 2 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L2_1,
                              Support: 'gpu_2k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_3D_m4_1k_sFH():
    """
    Testing M4 remeshing formula in 3D, 1 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L2_1,
                              Support: 'gpu_1k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_3D_m4_2k_sFH():
    """
    Testing M4 remeshing formula in 3D, 2 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L2_1,
                              Support: 'gpu_2k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


# M8 testing
def test_2D_m8_1k():
    """
    Testing M8 remeshing formula in 2D, 1 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: M8Prime,
                              Support: 'gpu_1k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_2D_m8_2k():
    """
    Testing M8 remeshing formula in 2D, 2 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: M8Prime,
                              Support: 'gpu_2k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_2D_m8_1k_sFH():
    """
    Testing M8 remeshing formula in 2D, 1 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: M8Prime,
                              Support: 'gpu_1k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)


def test_2D_m8_2k_sFH():
    """
    Testing M8 remeshing formula in 2D, 2 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: M8Prime,
                              Support: 'gpu_2k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d2d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_2D_withPython(scal, velo, advec, advec_py)

d3d = Discretization([17, 17, 17])

def test_3D_m8_1k():
    """
    Testing M8 remeshing formula in 3D, 1 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: M8Prime,
                              Support: 'gpu_1k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_3D_m8_2k():
    """
    Testing M8 remeshing formula in 3D, 2 kernel, simple precision,
    o2 splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: M8Prime,
                              Support: 'gpu_2k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_3D_m8_1k_sFH():
    """
    Testing M8 remeshing formula in 3D, 1 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: M8Prime,
                              Support: 'gpu_1k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_3D_m8_2k_sFH():
    """
    Testing M8 remeshing formula in 3D, 2 kernel, simple precision,
    o2_FullHalf splitting.
    """
    scal, velo = setup_3D()

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: M8Prime,
                              Support: 'gpu_2k',
                              Splitting: 'o2_FullHalf'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    assertion_3D_withPython(scal, velo, advec, advec_py)


def test_rectangular_domain2D():
    box = Box(length=[1., 1.], origin=[0., 0.])
    scal = Field(domain=box, name='Scalar')
    velo = Field(domain=box, name='Velocity', is_vector=True)

    advec = Advection(velo, scal, discretization=Discretization([65, 33]),
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_1k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=Discretization([65, 33]),
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(np.random.random(scal_d.data[0].shape))
    velo_d.data[0][...] = np.asarray(
        np.random.random(velo_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = np.asarray(
        np.random.random(velo_d.data[1].shape)) / (2. * scal_d.resolution[1])
    scal_d.toDevice()
    velo_d.toDevice()

    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    py_res = scal_d.data[0].copy()
    scal_d.toHost()

    err_m = np.max(np.abs(py_res - scal_d.data[0]))
    assert np.allclose(py_res, scal_d.data[0], rtol=1e-04, atol=1e-07), str(err_m)
    advec.finalize()


def test_rectangular_domain3D():
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar')
    velo = Field(domain=box, name='Velocity', is_vector=True)

    advec = Advection(velo, scal, discretization=Discretization([65, 33, 17]),
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_1k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=Discretization([65, 33, 17]),
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(np.random.random(scal_d.data[0].shape))
    velo_d.data[0][...] = npw.asrealarray(
        np.random.random(velo_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.asrealarray(
        np.random.random(velo_d.data[1].shape)) / (2. * scal_d.resolution[1])
    velo_d.data[2][...] = npw.asrealarray(
        np.random.random(velo_d.data[2].shape)) / (2. * scal_d.resolution[2])
    scal_d.toDevice()
    velo_d.toDevice()

    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    py_res = scal_d.data[0].copy()
    scal_d.toHost()

    err_m = np.max(np.abs(py_res - scal_d.data[0]))
    assert np.allclose(py_res, scal_d.data[0], rtol=1e-04, atol=1e-07), str(err_m) + " " + \
        str(np.where(np.abs(py_res - scal_d.data[0]) > 1e-07+1e-04*np.abs(scal_d.data[0])))
    advec.finalize()


def test_vector_2D():
    box = Box(length=[1., 1.], origin=[0., 0.])
    scal = Field(domain=box, name='Scalar', is_vector=True)
    velo = Field(domain=box, name='Velocity', is_vector=True)

    advec = Advection(velo, scal, discretization=Discretization([129, 129]),
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_1k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=Discretization([129, 129]),
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(np.random.random(scal_d.data[0].shape))
    scal_d.data[1][...] = npw.asrealarray(np.random.random(scal_d.data[1].shape))
    velo_d.data[0][...] = npw.asarray(
        np.random.random(velo_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.asarray(
        np.random.random(velo_d.data[1].shape)) / (2. * scal_d.resolution[1])
    scal_d.toDevice()
    velo_d.toDevice()

    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    py_res_X = scal_d.data[0].copy()
    py_res_Y = scal_d.data[1].copy()
    scal_d.toHost()

    err_m = np.max(np.abs(py_res_X - scal_d.data[0]))
    assert np.allclose(py_res_X, scal_d.data[0], rtol=1e-04, atol=1e-07), str(err_m) + " " + \
        str(np.where(np.abs(py_res_X - scal_d.data[0]) > 1e-07+1e-04*np.abs(scal_d.data[0])))
    err_m = np.max(np.abs(py_res_Y - scal_d.data[1]))
    assert np.allclose(py_res_Y, scal_d.data[1], rtol=1e-04, atol=1e-07), str(err_m) + " "+ \
        str(np.where(np.abs(py_res_Y - scal_d.data[1]) > 1e-07+1e-04*np.abs(scal_d.data[1])))
    advec.finalize()


def test_vector_3D():
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar', is_vector=True)
    velo = Field(domain=box, name='Velocity', is_vector=True)

    advec = Advection(velo, scal, discretization=d3d,
                      method={TimeIntegrator: RK2,
                              Interpolation: Linear,
                              Remesh: L4_2,
                              Support: 'gpu_1k',
                              Splitting: 'o2'}
                      )
    advec_py = Advection(velo, scal, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Support: '',
                                 Splitting: 'o2'},
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    velo_d.data[0][...] = npw.asrealarray(
        np.random.random(velo_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.asrealarray(
        np.random.random(velo_d.data[1].shape)) / (2. * scal_d.resolution[1])
    velo_d.data[2][...] = npw.asrealarray(
        np.random.random(velo_d.data[2].shape)) / (2. * scal_d.resolution[2])
    scal_d.toDevice()
    velo_d.toDevice()

    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    py_res_X = scal_d.data[0].copy()
    py_res_Y = scal_d.data[1].copy()
    py_res_Z = scal_d.data[2].copy()
    scal_d.toHost()

    err_m = np.max(np.abs(py_res_X - scal_d.data[0]))
    assert np.allclose(py_res_X, scal_d.data[0], rtol=1e-04, atol=1e-07), str(err_m) + " " + \
        str(np.where(np.abs(py_res_X_- scal_d.data[0]) > 1e-07+1e-04*np.abs(scal_d.data[0])))
    err_m = np.max(np.abs(py_res_Y - scal_d.data[0]))
    assert np.allclose(py_res_Y, scal_d.data[1], rtol=1e-04, atol=1e-07), str(err_m) + " " + \
        str(np.where(np.abs(py_res_Y_- scal_d.data[1]) > 1e-07+1e-04*np.abs(scal_d.data[1])))
    err_m = np.max(np.abs(py_res_Z - scal_d.data[0]))
    assert np.allclose(py_res_Z, scal_d.data[2], rtol=1e-04, atol=1e-07), str(err_m) + " " + \
        str(np.where(np.abs(py_res_Z_- scal_d.data[2]) > 1e-07+1e-04*np.abs(scal_d.data[2])))
    advec.finalize()
