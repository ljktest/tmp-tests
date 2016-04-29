"""
Testing transport problem.
"""
import numpy as np
import hysop.tools.numpywrappers as npw
from math import sqrt, pi, cos
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.advection import Advection
from hysop.problem.transport import TransportProblem
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization


def cosinus_product_2D(x, y, t):
    return cos(2. * pi * x) * cos(pi * y)


def cosinus_product_3D(x, y, z, t):
    return cos(2. * pi * x) * cos(pi * y) * cos(4. * pi * z)


def gaussian_scalar_3D(x, y, z, t):
    r = sqrt(x * x + y * y + z * z)
    if r < 1:
        return (1. - r * r) ** 6
    else:
        return 0.


def rotating_velocity_3D(x, y, z, t):
    r = sqrt(x * x + y * y + z * z)
    c = cos(3. * pi * r / 2.)
    return -c * y, c * x, c * x


def gaussian_scalar_2D(x, y, t):
    r = sqrt(x * x + y * y)
    if r < 1:
        return (1. - r * r) ** 6
    else:
        return 0.


def rotating_velocity_2D(x, y, t):
    r = sqrt(x * x + y * y)
    c = cos(3. * pi * r / 2.)
    return -c * y, c * x


def assertion(dim, boxLength, boxMin, nbElem, finalTime, timeStep,
              s, v, rtol=1e-05, atol=1e-08):
    box = Box(length=boxLength, origin=boxMin)
    print "domain init ...", id(box)
    scal = Field(domain=box, formula=s, vectorize_formula=True, name='Scalar')
    velo = Field(domain=box, formula=v, vectorize_formula=True,
                 name='Velocity', is_vector=True)
    advec = Advection(velo, scal, discretization=Discretization(nbElem))
    simu = Simulation(tinit=0.0, tend=finalTime, timeStep=timeStep, iterMax=1)
    print "velo dom ...", id(velo.domain)
    print "scal dom ...", id(scal.domain)
    pb = TransportProblem([advec], simu)
    pb.setup()
    initial_scalar = npw.copy(scal.discreteFields.values()[0].data[0])
    pb.solve()
    return np.allclose(initial_scalar, scal.discreteFields.values()[0].data[0],
                       rtol, atol)


def test_nullVelocity_2D():
    dim = 2
    nb = 33
    boxLength = [1., 1.]
    boxMin = [0., 0.]
    nbElem = [nb, nb]
    timeStep = 0.01
    finalTime = timeStep
    assert assertion(dim, boxLength, boxMin,
                     nbElem, finalTime, timeStep,
                     lambda x, y, t: np.random.random(),
                     lambda x, y, t: (0., 0.))


def test_nullVelocity_3D():
    dim = 3
    nb = 17
    boxLength = [1., 1., 1.]
    boxMin = [0., 0., 0.]
    nbElem = [nb, nb, nb]
    timeStep = 0.01
    finalTime = timeStep
    assert assertion(dim, boxLength, boxMin,
                     nbElem, finalTime, timeStep,
                     lambda x, y, z, t: np.random.random(),
                     lambda x, y, z, t: (0., 0., 0.))


def test_gaussian_2D():
    dim = 2
    nb = 33
    boxLength = [2., 2.]
    boxMin = [-1., -1.]
    nbElem = [nb, nb]
    timeStep = 0.001
    finalTime = timeStep
    assert assertion(dim, boxLength, boxMin,
                     nbElem, finalTime, timeStep,
                     gaussian_scalar_2D, rotating_velocity_2D,
                     rtol=1e-04, atol=1e-05)


def test_cosinus_translation_2D():
    dim = 2
    nb = 33
    boxLength = [2., 2.]
    boxMin = [-1., -1.]
    nbElem = [nb, nb]
    timeStep = 1.
    finalTime = 1.
    assert assertion(dim, boxLength, boxMin,
                     nbElem, finalTime, timeStep,
                     cosinus_product_2D,
                     lambda x, y, t: (1., 2.))


def test_cosinus_translation_3D():
    dim = 3
    nb = 17
    boxLength = [2., 2., 2.]
    boxMin = [-1., -1., -1.]
    nbElem = [nb, nb, nb]
    timeStep = 1.
    finalTime = timeStep
    assert assertion(dim, boxLength, boxMin,
                     nbElem, finalTime, timeStep,
                     cosinus_product_3D,
                     lambda x, y, z, t: (1., 2., 0.5))
