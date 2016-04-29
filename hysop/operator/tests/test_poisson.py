# -*- coding: utf-8 -*-

import hysop as pp
from hysop.operator.poisson import Poisson
from hysop.operator.analytic import Analytic
from hysop.operator.reprojection import Reprojection
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop.methods_keys import SpaceDiscretisation, \
    GhostUpdate, Formulation
import numpy as np
import hysop.tools.numpywrappers as npw
import math
from hysop.domain.subsets import SubBox

pi = math.pi
sin = np.sin
cos = np.cos

## Physical Domain description
dim = 3
LL = 2 * pi * npw.ones((dim))
# formula to compute initial vorticity field
coeff = 4 * pi ** 2 * (LL[1] ** 2 * LL[2] ** 2 + LL[0] ** 2 * LL[2] ** 2 +
                       LL[0] ** 2 * LL[1] ** 2) / (LL[0] ** 2 * LL[1] ** 2
                                                   * LL[2] ** 2)
cc = 2 * pi / LL
d3D = Discretization([33, 257, 257])
d2D = Discretization([33, 33])
uinf = 1.0


def computeVort(res, x, y, z, t):
    res[0][...] = coeff * sin(x * cc[0]) * sin(y * cc[1]) * cos(z * cc[2])
    res[1][...] = coeff * cos(x * cc[0]) * sin(y * cc[1]) * sin(z * cc[2])
    res[2][...] = coeff * cos(x * cc[0]) * cos(y * cc[1]) * sin(z * cc[2])
    return res

def computePressure(res, x, y, z, t):
    res[0][...] = -3.0 * sin(x * cc[0]) * cos(y * cc[1]) * cos(z * cc[2])
    return res

def computeRefPressure(res, x, y, z, t):
    res[0][...] = sin(x * cc[0]) * cos(y * cc[1]) * cos(z * cc[2])
    return res


# ref. field
def computeRef(res, x, y, z, t):
    res[0][...] = -2. * pi / LL[1] * \
        (cos(x * cc[0]) * sin(y * cc[1]) * sin(z * cc[2])) \
        - 2. * pi / LL[2] * (cos(x * cc[0]) * sin(y * cc[1]) * cos(z * cc[2]))

    res[1][...] = -2. * pi / LL[2] * \
        (sin(x * cc[0]) * sin(y * cc[1]) * sin(z * cc[2])) \
        + 2. * pi / LL[0] * (sin(x * cc[0]) * cos(y * cc[1]) * sin(z * cc[2]))

    res[2][...] = -2. * pi / LL[0] * \
        (sin(x * cc[0]) * sin(y * cc[1]) * sin(z * cc[2])) \
        - 2. * pi / LL[1] * (sin(x * cc[0]) * cos(y * cc[1]) * cos(z * cc[2]))

    return res


# ref. field
def computeRef_with_correction(res, x, y, z, t):
    res[0][...] = -2. * pi / LL[1] * \
        (cos(x * cc[0]) * sin(y * cc[1]) * sin(z * cc[2])) \
        - 2. * pi / LL[2] * (cos(x * cc[0]) * sin(y * cc[1]) * cos(z * cc[2]))\
        + uinf

    res[1][...] = -2. * pi / LL[2] * \
        (sin(x * cc[0]) * sin(y * cc[1]) * sin(z * cc[2])) \
        + 2. * pi / LL[0] * (sin(x * cc[0]) * cos(y * cc[1]) * sin(z * cc[2]))

    res[2][...] = -2. * pi / LL[0] * \
        (sin(x * cc[0]) * sin(y * cc[1]) * sin(z * cc[2])) \
        - 2. * pi / LL[1] * (sin(x * cc[0]) * cos(y * cc[1]) * cos(z * cc[2]))

    return res


def computeVort2D(res, x, y, t):
    # todo ...
    res[0][...] = 4 * pi ** 2 * (cos(x * cc[0]) * sin(y * cc[1])) * \
        (1. / LL[0] ** 2 + 1. / LL[1] ** 2)
    return res


# ref. field
def computeRef2D(res, x, y, t):
    res[0][...] = 2. * pi / LL[1] * (cos(x * cc[0]) * cos(y * cc[1]))
    res[1][...] = 2. * pi / LL[0] * (sin(x * cc[0]) * sin(y * cc[1]))

    return res

def test_Poisson_Pressure_3D():
    dom = pp.Box(length=LL)

    # Fields
    ref = pp.Field(domain=dom, name='Ref')
    pressure = pp.Field(domain=dom, formula=computePressure, name='Pressure')

    # Definition of the Poisson operator
    poisson = Poisson(pressure, pressure, discretization=d3D,
                      method={SpaceDiscretisation: 'fftw',
                              GhostUpdate: True,
                              Formulation: 'pressure'})

    poisson.discretize()
    poisson.setup()
    topo = poisson.discreteFields[pressure].topology
    # Analytic operator to compute the reference field
    refOp = Analytic(variables={ref: topo}, formula=computeRefPressure)
    simu = Simulation(nbIter=10)
    refOp.discretize()
    refOp.setup()
    pressure.initialize(topo=topo)
    poisson.apply(simu)
    refOp.apply(simu)
    assert np.allclose(ref.norm(topo), pressure.norm(topo))
    refD = ref.discretize(topo)
    prsD = pressure.discretize(topo)
    assert np.allclose(prsD[0], refD[0])
    poisson.finalize()


def test_Poisson3D():
    dom = pp.Box(length=LL)

    # Fields
    velocity = pp.Field(domain=dom, is_vector=True, name='Velocity')
    vorticity = pp.Field(domain=dom, formula=computeVort,
                         name='Vorticity', is_vector=True)

    # Definition of the Poisson operator
    poisson = Poisson(velocity, vorticity, discretization=d3D)

    poisson.discretize()
    poisson.setup()
    topo = poisson.discreteFields[vorticity].topology
    # Analytic operator to compute the reference field
    ref = pp.Field(domain=dom, name='reference', is_vector=True)
    refOp = Analytic(variables={ref: topo}, formula=computeRef)
    simu = Simulation(nbIter=10)
    refOp.discretize()
    refOp.setup()
    vorticity.initialize(topo=topo)
    poisson.apply(simu)
    refOp.apply(simu)
    assert np.allclose(ref.norm(topo), velocity.norm(topo))
    refD = ref.discretize(topo)
    vd = velocity.discretize(topo)
    for i in range(dom.dimension):
        assert np.allclose(vd[i], refD[i])
    poisson.finalize()


def test_Poisson2D():
    dom = pp.Box(length=[2. * pi, 2. * pi], origin=[0., 0.])

    # Fields
    velocity = pp.Field(domain=dom, is_vector=True, name='Velocity')
    vorticity = pp.Field(domain=dom, formula=computeVort2D, name='Vorticity')

    # Definition of the Poisson operator
    poisson = Poisson(velocity, vorticity, discretization=d2D)

    poisson.discretize()
    poisson.setup()
    topo = poisson.discreteFields[vorticity].topology
    # Analytic operator to compute the reference field
    ref = pp.Field(domain=dom, name='reference', is_vector=True)
    refOp = Analytic(variables={ref: topo}, formula=computeRef2D)
    simu = Simulation(nbIter=10)
    refOp.discretize()
    refOp.setup()
    vorticity.initialize(topo=topo)
    poisson.apply(simu)
    refOp.apply(simu)

    assert np.allclose(ref.norm(topo), velocity.norm(topo))
    refD = ref.discretize(topo)
    vd = velocity.discretize(topo)

    assert np.allclose(ref.norm(topo), velocity.norm(topo))
    for i in range(dom.dimension):
        assert np.allclose(vd[i], refD[i])
    poisson.finalize()


def test_Poisson3D_correction():
    dom = pp.Box(length=LL)

    # Fields
    velocity = pp.Field(domain=dom, is_vector=True, name='Velocity')
    vorticity = pp.Field(domain=dom, formula=computeVort,
                         name='Vorticity', is_vector=True)

    # Definition of the Poisson operator
    ref_rate = npw.zeros(3)
    ref_rate[0] = uinf * LL[1] * LL[2]
    rate = pp.VariableParameter(data=ref_rate, name='flowrate')
    poisson = Poisson(velocity, vorticity, discretization=d3D, flowrate=rate)

    poisson.discretize()
    poisson.setup()
    topo = poisson.discreteFields[vorticity].topology
    # Analytic operator to compute the reference field
    ref = pp.Field(domain=dom, name='reference', is_vector=True)
    refOp = Analytic(variables={ref: topo}, formula=computeRef_with_correction)
    simu = Simulation(nbIter=10)
    refOp.discretize()
    refOp.setup()
    vorticity.initialize(topo=topo)

    poisson.apply(simu)
    refOp.apply(simu)
    refD = ref.discretize(topo)
    vd = velocity.discretize(topo)
    surf = SubBox(parent=dom, origin=dom.origin,
                  length=[0., LL[1], LL[2]])
    surf.discretize(topo)
    assert np.allclose(ref.norm(topo), velocity.norm(topo))
    assert np.allclose(ref.norm(topo), velocity.norm(topo))
    for i in range(dom.dimension):
        assert np.allclose(vd[i], refD[i])
    poisson.finalize()


def test_Poisson3D_projection_1():
    dom = pp.Box(length=LL)

    # Fields
    velocity = pp.Field(domain=dom, is_vector=True, name='Velocity')
    vorticity = pp.Field(domain=dom, formula=computeVort,
                         name='Vorticity', is_vector=True)

    # Definition of the Poisson operator
    poisson = Poisson(velocity, vorticity, discretization=d3D, projection=4)

    poisson.discretize()
    poisson.setup()
    topo = poisson.discreteFields[vorticity].topology
    # Analytic operator to compute the reference field
    ref = pp.Field(domain=dom, name='reference', is_vector=True)
    refOp = Analytic(variables={ref: topo}, formula=computeRef)
    simu = Simulation(nbIter=10)
    refOp.discretize()
    refOp.setup()
    vorticity.initialize(topo=topo)
    poisson.apply(simu)
    refOp.apply(simu)
    assert np.allclose(ref.norm(topo), velocity.norm(topo))
    refD = ref.discretize(topo)
    vd = velocity.discretize(topo)

    assert np.allclose(ref.norm(topo), velocity.norm(topo))
    for i in range(dom.dimension):
        assert np.allclose(vd[i], refD[i])

    poisson.finalize()


def test_Poisson3D_projection_2():
    dom = pp.Box(length=LL)

    # Fields
    velocity = pp.Field(domain=dom, is_vector=True, name='Velocity')
    vorticity = pp.Field(domain=dom, formula=computeVort,
                         name='Vorticity', is_vector=True)
    d3dG = Discretization([33, 33, 33], [2, 2, 2])
    # Definition of the Poisson operator
    proj = Reprojection(vorticity, threshold=0.05, frequency=4,
                        discretization=d3dG, io_params=True)

    poisson = Poisson(velocity, vorticity, discretization=d3D,
                      projection=proj)
    proj.discretize()
    poisson.discretize()
    poisson.setup()
    proj.setup()
    topo = poisson.discreteFields[vorticity].topology
    # Analytic operator to compute the reference field
    ref = pp.Field(domain=dom, name='reference', is_vector=True)
    refOp = Analytic(variables={ref: topo}, formula=computeRef)
    simu = Simulation(nbIter=10)
    refOp.discretize()
    refOp.setup()
    vorticity.initialize(topo=topo)
    poisson.apply(simu)
    refOp.apply(simu)
    assert np.allclose(ref.norm(topo), velocity.norm(topo))
    refD = ref.discretize(topo)
    vd = velocity.discretize(topo)

    assert np.allclose(ref.norm(topo), velocity.norm(topo))
    for i in range(dom.dimension):
        assert np.allclose(vd[i], refD[i])
    poisson.finalize()
