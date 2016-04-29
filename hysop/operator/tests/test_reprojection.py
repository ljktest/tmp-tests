# -*- coding: utf-8 -*-

import hysop as pp
from hysop.operator.reprojection import Reprojection
from hysop.problem.simulation import Simulation
import numpy as np
from hysop.tools.parameters import Discretization
pi = np.pi
cos = np.cos
sin = np.sin
# Upstream flow velocity
uinf = 1.0
tol = 1e-12


## Function to compute TG velocity
def computeVel(res, x, y, z, t):
    res[0][...] = sin(x) * cos(y) * cos(z)
    res[1][...] = - cos(x) * sin(y) * cos(z)
    res[2][...] = 0.
    return res


## Function to compute reference vorticity
def computeVort(res, x, y, z, t):
    res[0][...] = - cos(x) * sin(y) * sin(z)
    res[1][...] = - sin(x) * cos(y) * sin(z)
    res[2][...] = 2. * sin(x) * sin(y) * cos(z)
    return res

## Global resolution
d3D = Discretization([33, 33, 33], [2, 2, 2])


def test_reprojection():
    # Domain
    box = pp.Box(length=[2.0 * pi, pi, pi])
    # Vector Fields
    vorti = pp.Field(domain=box, formula=computeVort,
                     name='Vorticity', is_vector=True)

    # Usual Cartesian topology definition
    topo = box.create_topology(dim=box.dimension, discretization=d3D)

    op = Reprojection(vorti, threshold=0.05, frequency=4,
                      discretization=topo, io_params=True)
    op.discretize()
    op.setup()
    # === Simulation setup ===
    simu = Simulation(nbIter=8)
    # init fields
    vorti.initialize(topo=topo)
    # Apply correction
    simu.initialize()
    while not simu.isOver:
        op.apply(simu)
        simu.advance()
