# -*- coding: utf-8 -*-
import hysop as pp
from hysop.operator.density import DensityVisco
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop import Field

d3d = Discretization([129, 129, 129])


def computeVisco(res, x, y, z, t):
    res[0][...] = x + y + z * t
    return res


def test_density():
    """
    Todo : write proper tests.
    Here we just check if discr/setup/apply process goes well.
    """
    box = pp.Box(length=[2., 1., 0.9], origin=[0.0, -1., -0.43])
    density = Field(domain=box, name='density')
    viscosity = Field(domain=box, formula=computeVisco, name='visco')
    op = DensityVisco(density=density, viscosity=viscosity, discretization=d3d)
    op.discretize()
    op.setup()
    topo = op.variables[viscosity]
    viscosity.initialize(topo=topo)
    #    simu = Simulation(nbIter=2)
    # op.apply(simu)  ## --> need to be reviewed
