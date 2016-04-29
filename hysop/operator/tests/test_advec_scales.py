"""
Testing Scales advection operator.
"""
import numpy as np
from hysop.methods_keys import Scales, TimeIntegrator, Interpolation,\
    Remesh, Support, Splitting
from hysop.methods import RK2, L2_1, L4_2, M8Prime, Linear
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.advection import Advection
from hysop.problem.simulation import Simulation
import hysop.tools.numpywrappers as npw

from hysop.tools.parameters import Discretization
d3d = Discretization([17, 17, 17])


def test_nullVelocity_m4():
    """Basic test with random velocity. Using M4prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar')
    scal_ref = Field(domain=box, name='Scalar_ref')
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z, t: (0., 0., 0.), is_vector=True,
                 vectorize_formula=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M4'})
    advec_py = Advection(velo, scal_ref, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Splitting: 'o2_FullHalf',
                                 Support: ''}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()
    scal_d = scal.discreteFields.values()[0]
    scal_ref_d = scal_ref.discreteFields.values()[0]

    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_ref_d.data[0][...] = scal_d.data[0][...]
    topo = scal_d.topology
    assert (velo.norm(topo) == 0).all()
    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    print (np.max(np.abs(scal_ref_d.data[0] - scal_d.data[0])))
    assert np.allclose(scal_ref_d.data[0], scal_d.data[0])


def test_nullVelocity_vec_m4():
    """Basic test with random velocity and vector field. Using M4prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar', is_vector=True)
    scal_ref = Field(domain=box, name='Scalar_ref', is_vector=True)
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z, t: (0., 0., 0.), is_vector=True,
                 vectorize_formula=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M4'})
    advec_py = Advection(velo, scal_ref, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Splitting: 'o2_FullHalf',
                                 Support: ''}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_ref_d = scal_ref.discreteFields.values()[0]

    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[1].shape))
    scal_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[2].shape))
    scal_ref_d.data[0][...] = scal_d.data[0][...]
    scal_ref_d.data[1][...] = scal_d.data[1][...]
    scal_ref_d.data[2][...] = scal_d.data[2][...]

    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    assert np.allclose(scal_ref_d.data[0], scal_d.data[0])
    assert np.allclose(scal_ref_d.data[1], scal_d.data[1])
    assert np.allclose(scal_ref_d.data[2], scal_d.data[2])


def test_nullVelocity_m6():
    """Basic test with null velocity. Using M6prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar')
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z, t: (0., 0., 0.), is_vector=True,
                 vectorize_formula=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M6'}
                      )
    advec.discretize()
    advec.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_d.data[0][...] = np.asarray(np.random.random(scal_d.data[0].shape))
    scal_init = npw.copy(scal_d.data[0])

    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    assert np.allclose(scal_init, scal_d.data[0])


def test_nullVelocity_vec_m6():
    """Basic test with null velocity and vector field. Using M6prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar', is_vector=True)
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z, t: (0., 0., 0.), is_vector=True,
                 vectorize_formula=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M6'}
                      )
    advec.discretize()
    advec.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[1].shape))
    scal_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[2].shape))
    scal_init0 = npw.copy(scal_d.data[0])
    scal_init1 = npw.copy(scal_d.data[1])
    scal_init2 = npw.copy(scal_d.data[2])

    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    assert np.allclose(scal_init0, scal_d.data[0])
    assert np.allclose(scal_init1, scal_d.data[1])
    assert np.allclose(scal_init2, scal_d.data[2])


def test_nullVelocity_m8():
    """Basic test with random velocity. Using M4prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar')
    scal_ref = Field(domain=box, name='Scalar_ref')
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z, t: (0., 0., 0.), is_vector=True,
                 vectorize_formula=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M8'}
                      )
    advec_py = Advection(velo, scal_ref, discretization=d3d, method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Splitting: 'o2_FullHalf',
                                 Support: ''}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()
    topo = advec.discreteFields[velo].topology
    scal_d = scal.discreteFields[topo]
    scal_ref_d = scal_ref.discreteFields[topo]

    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_ref_d.data[0][...] = scal_d.data[0]

    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    assert np.allclose(scal_ref_d.data[0], scal_d.data[0])


def test_nullVelocity_vec_m8():
    """Basic test with random velocity and vector field. Using M4prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar', is_vector=True)
    scal_ref = Field(domain=box, name='Scalar_ref', is_vector=True)
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z, t: (0., 0., 0.), is_vector=True,
                 vectorize_formula=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M8'}
                      )
    advec_py = Advection(velo, scal_ref, discretization=d3d, method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Splitting: 'o2_FullHalf',
                                 Support: ''}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()
    scal_d = scal.discreteFields.values()[0]
    scal_ref_d = scal_ref.discreteFields.values()[0]
    topo = scal_d.topology
    assert (velo.norm(topo) == 0).all()
    for i in xrange(box.dimension):
        scal_d.data[i][...] = \
            npw.asrealarray(np.random.random(scal_d.data[i].shape))
        scal_ref_d.data[i][...] = scal_d.data[i][...]

    advec.apply(Simulation(tinit=0., tend=0.075, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.075, nbIter=1))

    for i in xrange(box.dimension):
        assert np.allclose(scal_ref_d.data[i], scal_d.data[i])


def _randomVelocity_m4():
    """Basic test with random velocity. Using M4prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar')
    scal_ref = Field(domain=box, name='Scalar_ref')
    velo = Field(domain=box, name='Velocity', is_vector=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M4'}
                      )
    advec_py = Advection(velo, scal_ref, discretization=d3d, method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Splitting: 'o2_FullHalf',
                                 Support: ''}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_ref_d = scal_ref.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]

    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_ref_d.data[0][...] = scal_d.data[0][...]
    velo_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[1])
    velo_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[1])

    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    assert np.allclose(scal_ref_d.data[0], scal_d.data[0])


def _randomVelocity_vec_m4():
    """Basic test with random velocity vector Field. Using M4prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar', is_vector=True)
    scal_ref = Field(domain=box, name='Scalar_ref', is_vector=True)
    velo = Field(domain=box, name='Velocity', is_vector=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M4'}
                      )
    advec_py = Advection(velo, scal_ref, discretization=d3d, method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L2_1,
                                 Splitting: 'o2_FullHalf',
                                 Support: ''}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_ref_d = scal_ref.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]

    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[1].shape))
    scal_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[2].shape))
    scal_ref_d.data[0][...] = scal_d.data[0][...]
    scal_ref_d.data[1][...] = scal_d.data[1][...]
    scal_ref_d.data[2][...] = scal_d.data[2][...]
    velo_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[1].shape)) / (2. * scal_d.resolution[1])
    velo_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[2].shape)) / (2. * scal_d.resolution[1])

    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    assert np.allclose(scal_ref_d.data[0], scal_d.data[0])
    assert np.allclose(scal_ref_d.data[1], scal_d.data[1])
    assert np.allclose(scal_ref_d.data[2], scal_d.data[2])


def test_randomVelocity_m6():
    """Basic test with random velocity. Using M6prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar')
    scal_ref = Field(domain=box, name='Scalar_ref')
    velo = Field(domain=box, name='Velocity', is_vector=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M6'}
                      )
    advec_py = Advection(velo, scal_ref, discretization=d3d, method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Splitting: 'o2_FullHalf',
                                 Support: ''}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_ref_d = scal_ref.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]

    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_ref_d.data[0][...] = scal_d.data[0][...]
    velo_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[1])
    velo_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[1])

    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    assert np.allclose(scal_ref_d.data[0], scal_d.data[0])


def test_randomVelocity_vec_m6():
    """Basic test with random velocity vector Field. Using M6prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar', is_vector=True)
    scal_ref = Field(domain=box, name='Scalar_ref', is_vector=True)
    velo = Field(domain=box, name='Velocity', is_vector=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M6'}
                      )
    advec_py = Advection(velo, scal_ref, discretization=d3d, method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: L4_2,
                                 Splitting: 'o2_FullHalf',
                                 Support: ''}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_ref_d = scal_ref.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]

    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[1].shape))
    scal_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[2].shape))
    scal_ref_d.data[0][...] = scal_d.data[0][...]
    scal_ref_d.data[1][...] = scal_d.data[1][...]
    scal_ref_d.data[2][...] = scal_d.data[2][...]
    velo_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[1].shape)) / (2. * scal_d.resolution[1])
    velo_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[2].shape)) / (2. * scal_d.resolution[1])

    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    assert np.allclose(scal_ref_d.data[0], scal_d.data[0])
    assert np.allclose(scal_ref_d.data[1], scal_d.data[1])
    assert np.allclose(scal_ref_d.data[2], scal_d.data[2])


def test_randomVelocity_m8():
    """Basic test with random velocity. Using M8prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar')
    scal_ref = Field(domain=box, name='Scalar_ref')
    velo = Field(domain=box, name='Velocity', is_vector=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M8'}
                      )
    advec_py = Advection(velo, scal_ref, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Splitting: 'o2_FullHalf',
                                 Support: ''}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_ref_d = scal_ref.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]

    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_ref_d.data[0][...] = scal_d.data[0][...]
    velo_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[1])
    velo_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[1])

    advec.apply(Simulation(tinit=0., tend=0.1, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.1, nbIter=1))

    assert np.allclose(scal_ref_d.data[0], scal_d.data[0], atol=1e-07)


def test_randomVelocity_vec_m8():
    """Basic test with random velocity vector Field. Using M8prime"""
    box = Box(length=[1., 1., 1.], origin=[0., 0., 0.])
    scal = Field(domain=box, name='Scalar', is_vector=True)
    scal_ref = Field(domain=box, name='Scalar_ref', is_vector=True)
    velo = Field(domain=box, name='Velocity', is_vector=True)
    advec = Advection(velo, scal, discretization=d3d, method={Scales: 'p_M8'}
                      )
    advec_py = Advection(velo, scal_ref, discretization=d3d,
                         method={TimeIntegrator: RK2,
                                 Interpolation: Linear,
                                 Remesh: M8Prime,
                                 Support: '',
                                 Splitting: 'o2_FullHalf'}
                         )
    advec.discretize()
    advec_py.discretize()
    advec.setup()
    advec_py.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_ref_d = scal_ref.discreteFields.values()[0]
    velo_d = velo.discreteFields.values()[0]

    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[1].shape))
    scal_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[2].shape))
    scal_ref_d.data[0][...] = scal_d.data[0][...]
    scal_ref_d.data[1][...] = scal_d.data[1][...]
    scal_ref_d.data[2][...] = scal_d.data[2][...]
    velo_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape)) / (2. * scal_d.resolution[0])
    velo_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[1].shape)) / (2. * scal_d.resolution[1])
    velo_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[2].shape)) / (2. * scal_d.resolution[1])

    advec.apply(Simulation(tinit=0., tend=0.01, nbIter=1))
    advec_py.apply(Simulation(tinit=0., tend=0.01, nbIter=1))

    assert np.allclose(scal_ref_d.data[0], scal_d.data[0], atol=1e-07)
    assert np.allclose(scal_ref_d.data[1], scal_d.data[1], atol=1e-07)
    assert np.allclose(scal_ref_d.data[2], scal_d.data[2], atol=1e-07)
