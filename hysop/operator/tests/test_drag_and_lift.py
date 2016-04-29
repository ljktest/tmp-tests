# -*- coding: utf-8 -*-
from hysop.domain.subsets import Sphere
from hysop.operator.penalization import PenalizeVorticity
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop.tools.io_utils import IOParams, IO
from hysop.mpi.topology import Cartesian
import numpy as np
import os
from hysop import Field, Box
# from hysop.operator.hdf_io import HDF_Reader
from hysop.operator.drag_and_lift import MomentumForces, NocaForces
cos = np.cos
sin = np.sin


def v2d(res, x, y, t):
    res[0][...] = 1.
    res[1][...] = 1.
    return res


def s2d(res, x, y, t):
    res[0][...] = 1.
    return res


def v3d(res, x, y, z, t):
    res[0][...] = t * sin(x) * cos(y) * cos(z)
    res[1][...] = - t * cos(x) * sin(y) * cos(z)
    res[2][...] = 0.
    return res


def vorticity_f(res, x, y, z, t):
    res[0][...] = - t * cos(x) * sin(y) * sin(z)
    res[1][...] = - t * sin(x) * cos(y) * sin(z)
    res[2][...] = 2 * t * sin(x) * sin(y) * cos(z)
    return res


def drag_vol(x, y, z, t):
    res = np.zeros(3)
    res[0] = -2 * cos(x) * sin(z) * (sin(y) - y * cos(y))\
        - 2 * t * cos(x) * sin(y) * (sin(z) - z * cos(z))
    res[1] = 2 * cos(y) * sin(z) * (sin(x) - x * cos(x)) +\
        sin(x) * cos(y) * (sin(z) - z * cos(z))
    res[2] = 2 * sin(y) * cos(z) * t * (sin(x) - x * cos(x))\
        - sin(x) * cos(z) * (sin(y) - y * cos(y))
    return res


def s3d(res, x, y, z, t):
    res[0][...] = 1.
    return res


Nx = 32
Ny = 106
Nz = 64
g = 2


ldef = [0.3, 0.4, 1.0]
discr3D = Discretization([Nx + 1, Ny + 1, Nz + 1], [g, g, g])
discr2D = Discretization([Nx + 1, Ny + 1], [g, g])
xdom = np.asarray([0.1, -0.3, 0.5])
import math
ldom = np.asarray([math.pi * 2., ] * 3)
xdef = xdom + 0.2
xpos = ldom * 0.5
xpos[-1] += 0.1
working_dir = os.getcwd() + '/'
xdom = np.asarray([0., 0., 0.])
ldom = np.asarray([1., ] * 3)

def init(discr, vform=v3d, wform=vorticity_f):
    Cartesian.reset_counter()
    dim = len(discr.resolution)
    dom = Box(dimension=dim, length=ldom[:dim],
              origin=xdom[:dim])
    topo = dom.create_topology(discr)
    velo = Field(domain=topo.domain, formula=vform,
                 name='Velo', is_vector=True)
    vorti = Field(domain=topo.domain, formula=wform, name='Vorti',
                  is_vector=dim == 3)
    rd = ldom[0] * 0.2
    hsphere = Sphere(parent=topo.domain, origin=xpos, radius=rd)
    hsphere.discretize(topo)
    # penalisation
    penal = PenalizeVorticity(velocity=velo, vorticity=vorti,
                              discretization=topo,
                              obstacles=[hsphere], coeff=1e8)
    penal.discretize()
    penal.setup()
    velo.initialize(topo=topo)
    vorti.initialize(topo=topo)
    return topo, penal


def test_momentum_forces_3d():
    topo, op = init(discr3D)
    # Forces
    velo = op.velocity
    obst = op.obstacles
    dg = MomentumForces(velocity=velo, discretization=topo,
                        obstacles=obst, penalisation_coeff=[1e8])
    dg.discretize()
    dg.setup()
    simu = Simulation(nbIter=3)
    op.apply(simu)
    dg.apply(simu)


def build_noca(formulation):
    """
    Compute drag/lift in 3D, flow around a sphere
    """
    topo, op = init(discr3D)
    # Velocity field
    velo = op.velocity
    vorti = op.vorticity
    obst = op.obstacles
    dg = NocaForces(velocity=velo, vorticity=vorti, discretization=topo,
                    nu=0.3, obstacles=obst, surfdir=[2, 3],
                    formulation=formulation)
    dg.discretize()
    dg.setup()
    return dg, op


def test_noca1():
    dg, op = build_noca(1)
    simu = Simulation(nbIter=3)
    op.apply(simu)
    dg.apply(simu)


def test_noca2():
    dg, op = build_noca(2)
    simu = Simulation(nbIter=3)
    op.apply(simu)
    dg.apply(simu)


def test_noca3():
    dg, op = build_noca(3)
    simu = Simulation(nbIter=3)
    op.apply(simu)
    dg.apply(simu)


def test_all_drags():
    topo, op = init(discr3D)
    velo = op.velocity
    vorti = op.vorticity
    obst = op.obstacles
    dg = {}
    dg['mom'] = MomentumForces(velocity=velo, discretization=topo,
                               obstacles=obst, penalisation_coeff=[1e8],
                               io_params=IOParams('Heloise',
                                                  fileformat=IO.ASCII))
    sdir = [2]
    dg['noca_1'] = NocaForces(velocity=velo, vorticity=vorti,
                              discretization=topo, surfdir=sdir,
                              nu=0.3, obstacles=obst, formulation=1,
                              io_params=IOParams('Noca1',
                                                 fileformat=IO.ASCII))
    dg['noca_2'] = NocaForces(velocity=velo, vorticity=vorti,
                              discretization=topo, surfdir=sdir,
                              nu=0.3, obstacles=obst, formulation=2,
                              io_params=IOParams('Noca2',
                                                 fileformat=IO.ASCII))
    dg['noca_3'] = NocaForces(velocity=velo, vorticity=vorti,
                              discretization=topo, surfdir=sdir,
                              nu=0.3, obstacles=obst, formulation=3,
                              io_params=IOParams('Noca3',
                                                 fileformat=IO.ASCII))
    for drag in dg.values():
        drag.discretize()
        drag.setup()

    simu = Simulation(timeStep=1e-4, tend=1e-2)

    op.apply(simu)
    for drag in dg:
        dg[drag].apply(simu)
    simu.initialize()
    while not simu.isOver:
        velo.initialize(simu.time, topo)
        vorti.initialize(simu.time, topo)
        for drag in dg:
            dg[drag].apply(simu)
        op.apply(simu)
        simu.advance()
    #assert False
