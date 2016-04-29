"""Testing pure python particle advection with null velocity.
"""
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.advection import Advection
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
import hysop.tools.numpywrappers as npw
import numpy as np


d2d = Discretization([17, 17])
d3d = Discretization([17, 17, 17])



def setup_2D():
    box = Box(length=[2., 2.], origin=[-1., -1.])
    scal = Field(domain=box, name='Scalar')
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y: (0., 0.), is_vector=True)
    return scal, velo


def setup_vector_2D():
    box = Box(length=[2., 2.], origin=[-1., -1.])
    scal = Field(domain=box, name='Vector', is_vector=True)
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y: (0., 0.), is_vector=True)
    return scal, velo


def setup_list_2D():
    box = Box(length=[2., 2.], origin=[-1., -1.])
    scal1 = Field(domain=box, name='Scalar1')
    scal2 = Field(domain=box, name='Scalar2')
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y: (0., 0.), is_vector=True)
    return [scal1, scal2], velo


def setup_3d():
    """Build fields inside a 3d domain
    """
    box = Box(length=[2., 4., 1.], origin=[-1., -2., 0.])
    scal = Field(domain=box, name='Scalar')
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z: (0., 0., 0.), is_vector=True)
    return scal, velo


def setup_vector_3D():
    box = Box(length=[2., 4., 1.], origin=[-1., -2., 0.])
    scal = Field(domain=box, name='Scalar', is_vector=True)
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z: (0., 0., 0.), is_vector=True)
    return scal, velo


def setup_list_3D():
    box = Box(length=[2., 4., 1.], origin=[-1., -2., 0.])
    scal1 = Field(domain=box, name='Scalar1')
    scal2 = Field(domain=box, name='Scalar2')
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z: (0., 0., 0.), is_vector=True)
    return [scal1, scal2], velo


def setup_dict_3D():
    box = Box(length=[2., 4., 1.], origin=[-1., -2., 0.])
    scal1 = Field(domain=box, name='Scalar1')
    scal2 = Field(domain=box, name='Scalar2')
    velo = Field(domain=box, name='Velocity',
                 formula=lambda x, y, z: (0., 0., 0.), is_vector=True)
    return {scal1: d3d, scal2: d3d}, velo



def assertion(scal, advec):
    advec.discretize()
    advec.setup()
    scal_d = scal.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_init = npw.copy(scal_d.data[0])

    advec.apply(Simulation(tinit=0., tend=0.01, nbIter=1))
    return np.allclose(scal_init, scal_d.data[0])


def assertion_vector2D(scal, advec):
    advec.discretize()
    advec.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_d.data[1][...] = npw.asrealarray(
        np.random.random(scal_d.data[1].shape))
    scal1_init = npw.copy(scal_d.data[0])
    scal2_init = npw.copy(scal_d.data[1])

    advec.apply(Simulation(tinit=0., tend=0.01, nbIter=1))
    print (np.max(np.abs((scal1_init - scal_d.data[0]))))
    print (np.max(np.abs((scal2_init - scal_d.data[1]))))
    return np.allclose(scal1_init, scal_d.data[0]) and \
        np.allclose(scal2_init, scal_d.data[1])


def assertion_vector3D(scal, advec):
    advec.discretize()
    advec.setup()

    scal_d = scal.discreteFields.values()[0]
    scal_d.data[0][...] = npw.asrealarray(
        np.random.random(scal_d.data[0].shape))
    scal_d.data[1][...] = npw.asrealarray(np.random.random(
        scal_d.data[1].shape))
    scal_d.data[2][...] = npw.asrealarray(
        np.random.random(scal_d.data[2].shape))
    scal1_init = npw.copy(scal_d.data[0])
    scal2_init = npw.copy(scal_d.data[1])
    scal3_init = npw.copy(scal_d.data[2])

    advec.apply(Simulation(tinit=0., tend=0.01, nbIter=1))
    return np.allclose(scal1_init, scal_d.data[0]) and \
        np.allclose(scal2_init, scal_d.data[1]) and \
        np.allclose(scal3_init, scal_d.data[2])


def assertion_list(scal, advec):
    advec.discretize()
    advec.setup()

    scal1_d = scal[0].discreteFields.values()[0]
    scal2_d = scal[1].discreteFields.values()[0]
    scal1_d.data[0][...] = npw.asrealarray(
        np.random.random(scal1_d.data[0].shape))
    scal2_d.data[0][...] = npw.asrealarray(
        np.random.random(scal2_d.data[0].shape))
    scal1_init = npw.copy(scal1_d.data[0])
    scal2_init = npw.copy(scal2_d.data[0])

    advec.apply(Simulation(tinit=0., tend=0.01, nbIter=1))
    print (scal1_init, scal1_d.data[0])
    print (scal2_init, scal2_d.data[0])
    return np.allclose(scal1_init, scal1_d.data[0]) and \
        np.allclose(scal2_init, scal2_d.data[0])


def test_nullVelocity_2D():
    """2D case, advection with null velocity, single resolution.
    """
    scal, velo = setup_2D()

    advec = Advection(velo, scal, discretization=d2d)

    assert assertion(scal, advec)


def test_nullVelocity_vector_2D():
    """
    """
    scal, velo = setup_vector_2D()

    advec = Advection(velo, scal, discretization=d2d)
    assert assertion_vector2D(scal, advec)


def test_nullVelocity_list_2D():
    """
    """
    scal, velo = setup_list_2D()

    advec = Advection(velo, scal, discretization=d2d)
    assert assertion_list(scal, advec)


def test_nullVelocity_3D():
    """
    """
    scal, velo = setup_3d()

    advec = Advection(velo, scal, discretization=d3d)
    assert assertion(scal, advec)


def test_nullVelocity_vector_3D():
    """
    """
    scal, velo = setup_vector_3D()
    advec = Advection(velo, scal, discretization=d3d)

    assert assertion_vector3D(scal, advec)


def test_nullVelocity_list_3D():
    """
    """
    scal, velo = setup_list_3D()

    advec = Advection(velo, scal, discretization=d3d)

    assert assertion_list(scal, advec)


def test_nullVelocity_dict_3D():
    scal, velo = setup_dict_3D()

    advec = Advection(velocity=velo, variables=scal, discretization=d3d)

    assert assertion_list(scal.keys(), advec)
