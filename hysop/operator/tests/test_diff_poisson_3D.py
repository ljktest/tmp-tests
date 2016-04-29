# -*- coding: utf-8 -*-
import hysop as pp
from hysop.operator.poisson import Poisson
from hysop.operator.diffusion import Diffusion
from math import sqrt, pi, exp
from hysop.problem.simulation import Simulation


def computeVel(x, y, z):
    vx = 0.
    vy = 0.
    vz = 0.
    return vx, vy, vz


def computeVort(x, y, z):
    xc = 1. / 2.
    yc = 1. / 2.
    zc = 1. / 4.
    R = 0.2
    sigma = R / 2.
    Gamma = 0.0075
    dist = sqrt((x - xc) ** 2 + (y - yc) ** 2)
    s2 = (z - zc) ** 2 + (dist - R) ** 2
    wx = 0.
    wy = 0.
    wz = 0.
    if (dist != 0.):
        cosTheta = (x - xc) / dist
        sinTheta = (y - yc) / dist
        wTheta = Gamma / (pi * sigma ** 2) * \
            exp(-(s2 / sigma ** 2))
        wx = - wTheta * sinTheta
        wy = wTheta * cosTheta
        wz = 0.
    return wx, wy, wz


def test_Diff_Poisson():
    # Parameters
    nb = 33
    boxLength = [1., 1., 1.]
    boxMin = [0., 0., 0.]
    from hysop.tools.parameters import Discretization
    d3D = Discretization([nb, nb, nb])

    ## Domain
    box = pp.Box(length=boxLength, origin=boxMin)

    ## Fields
    velo = pp.Field(domain=box, formula=computeVel,
                    name='Velocity', is_vector=True)
    vorti = pp.Field(domain=box, formula=computeVort,
                     name='Vorticity', is_vector=True)

    ## FFT Diffusion operators and FFT Poisson solver
    diffusion = Diffusion(variables={vorti: d3D}, viscosity=0.002)
    poisson = Poisson(velo, vorti, discretization=d3D)

    diffusion.discretize()
    poisson.discretize()

    diffusion.setup()
    poisson.setup()

    simu = Simulation(tinit=0.0, tend=10., timeStep=0.002,
                      iterMax=1000000)
    diffusion.apply(simu)
    poisson.apply(simu)
