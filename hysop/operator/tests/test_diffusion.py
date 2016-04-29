# -*- coding: utf-8 -*-

import hysop as pp
from hysop.operator.diffusion import Diffusion
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
import numpy as np
import hysop.tools.numpywrappers as npw
import math
pi = math.pi
sin = np.sin
cos = np.cos
## Physical Domain description
dim = 3
LL = 2 * pi * npw.ones((dim))
cc = 2 * pi / LL
d3D = Discretization([33, 33, 33])
d2D = Discretization([33, 33])


def computeVort(res, x, y, z, t):
    res[0][...] = sin(x * cc[0]) * sin(y * cc[1]) * cos(z * cc[2])
    res[1][...] = cos(x * cc[0]) * sin(y * cc[1]) * sin(z * cc[2])
    res[2][...] = cos(x * cc[0]) * cos(y * cc[1]) * sin(z * cc[2])
    return res


def computeVort2D(res, x, y, t):
    # todo ...
    res[0][...] = 4 * pi ** 2 * (cos(x * cc[0]) * sin(y * cc[1])) * \
        (1. / LL[0] ** 2 + 1. / LL[1] ** 2)
    return res


def test_Diffusion3D():
    dom = pp.Box(length=LL)

    # Fields
    vorticity = pp.Field(domain=dom, formula=computeVort,
                         name='Vorticity', is_vector=True)

    # Definition of the Poisson operator
    diff = Diffusion(viscosity=0.3, vorticity=vorticity, discretization=d3D)
    diff.discretize()
    diff.setup()
    topo = diff.discreteFields[vorticity].topology
    simu = Simulation(nbIter=10)
    vorticity.initialize(topo=topo)
    diff.apply(simu)
    diff.finalize()


def test_Diffusion3D_2():
    dom = pp.Box(length=LL)

    # Fields
    vorticity = pp.Field(domain=dom, formula=computeVort,
                         name='Vorticity', is_vector=True)

    # Definition of the Poisson operator
    diff = Diffusion(viscosity=0.3, variables={vorticity: d3D})
    diff.discretize()
    diff.setup()
    topo = diff.discreteFields[vorticity].topology
    simu = Simulation(nbIter=10)
    vorticity.initialize(topo=topo)
    diff.apply(simu)
    diff.finalize()


def test_Diffusion2D():
    dom = pp.Box(length=LL[:2])

    # Fields
    vorticity = pp.Field(domain=dom, formula=computeVort2D, name='Vorticity')

    # Definition of the Poisson operator
    diff = Diffusion(viscosity=0.3, vorticity=vorticity, discretization=d2D)
    diff.discretize()
    diff.setup()
    topo = diff.discreteFields[vorticity].topology
    simu = Simulation(nbIter=10)
    vorticity.initialize(topo=topo)
    diff.apply(simu)
    diff.finalize()

