# -*- coding: utf-8 -*-
import hysop as pp
from hysop.operator.energy_enstrophy import EnergyEnstrophy
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop import Field
import numpy as np
try:
    from scipy.integrate import nquad

except:
    def nquad(func, coords_range):
        coords = []
        nbsteps = 1000
        for x in coords_range:
            coords.append(np.linspace(x[0], x[1], nbsteps))
        coords = tuple(coords)
        hstep = coords[0][1] - coords[0][0]
        ll = [coords_range[i][1] - coords_range[i][0] for i in xrange(1, 3)]
        return [np.sum(func(*coords)[:-1]) * hstep * np.prod(ll)]

sin = np.sin
cos = np.cos

d3d = Discretization([129, 129, 129])


def computeVel(res, x, y, z, t):
    res[0][...] = x
    res[1][...] = y
    res[2][...] = z
    return res


def computeVort(res, x, y, z, t):
    res[0][...] = x
    res[1][...] = y
    res[2][...] = z
    return res


def energy_ref(x, y, z):
    return x ** 2


def init():
    box = pp.Box(length=[2., 1., 0.9], origin=[0.0, -1., -0.43])
    velo = Field(domain=box, formula=computeVel,
                 name='Velocity', is_vector=True)
    vorti = Field(domain=box, formula=computeVort,
                  name='Vorticity', is_vector=True)
    return velo, vorti


def test_energy_enstrophy():
    """
    Todo : write proper tests.
    Here we just check if discr/setup/apply process goes well.
    """
    dim = 3
    velo, vorti = init()
    op = EnergyEnstrophy(velo, vorti, discretization=d3d)
    op.discretize()
    op.setup()
    topo = op.variables[velo]
    velo.initialize(topo=topo)
    vorti.initialize(topo=topo)
    simu = Simulation(nbIter=2)
    op.apply(simu)
    intrange = []
    box = topo.domain
    invvol = 1. / np.prod(box.length)
    for i in xrange(dim):
        origin = box.origin[i]
        end = origin + box.length[i]
        intrange.append([origin, end])
    intrange = 2 * intrange
    eref = nquad(energy_ref, intrange[:dim])[0]
    eref += nquad(energy_ref, intrange[1:dim + 1])[0]
    eref += nquad(energy_ref, intrange[2:dim + 2])[0]
    eref *= invvol
    tol = (topo.mesh.space_step).max() ** 2
    assert (op.energy() - eref * 0.5) < tol
    assert (op.enstrophy() - eref) < tol
