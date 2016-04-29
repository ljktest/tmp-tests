# -*- coding: utf-8 -*-
import hysop as pp
import numpy as np
from hysop.fields.continuous import Field
from hysop.operator.stretching import Stretching, \
    StretchingLinearized
from hysop.problem.simulation import Simulation
from hysop.methods_keys import TimeIntegrator, Formulation,\
    SpaceDiscretisation
from hysop.methods import Euler, RK3, FD_C_4, Conservative
from hysop.tools.parameters import Discretization
import hysop.tools.numpywrappers as npw
pi = np.pi
cos = np.cos
sin = np.sin


def computeVel(res, x, y, z, t):
    amodul = cos(pi * 1. / 3.)
    pix = pi * x
    piy = pi * y
    piz = pi * z
    pi2x = 2. * pix
    pi2y = 2. * piy
    pi2z = 2. * piz
    res[0][...] = 2. * sin(pix) * sin(pix) \
        * sin(pi2y) * sin(pi2z) * amodul
    res[1][...] = - sin(pi2x) * sin(piy) \
        * sin(piy) * sin(pi2z) * amodul
    res[2][...] = - sin(pi2x) * sin(piz) \
        * sin(piz) * sin(pi2y) * amodul
    return res


def computeVort(res, x, y, z, t):
    amodul = cos(pi * 1. / 3.)
    pix = pi * x
    piy = pi * y
    piz = pi * z
    pi2x = 2. * pix
    pi2y = 2. * piy
    pi2z = 2. * piz
    res[0][...] = 2. * pi * sin(pi2x) * amodul *\
        (- cos(pi2y) * sin(piz) * sin(piz)
         + sin(piy) * sin(piy) * cos(pi2z))

    res[1][...] = 2. * pi * sin(pi2y) * amodul *\
        (2. * cos(pi2z) * sin(pix) * sin(pix)
         + sin(piz) * sin(piz) * cos(pi2x))

    res[2][...] = -2. * pi * sin(pi2z) * amodul *\
        (cos(pi2x) * sin(piy) * sin(piy)
         + sin(pix) * sin(pix) * cos(pi2y))

    return res

def computeVelBF(res, x, y, z, t):
    amodul = cos(pi * 1. / 3.)
    pix = pi * x
    piy = pi * y
    piz = pi * z
    pi2x = 2. * pix
    pi2y = 2. * piy
    pi2z = 2. * piz
    res[0][...] = 2. * sin(pix) * sin(pix) \
        * sin(pi2y) * sin(pi2z) * amodul
    res[1][...] = - sin(pi2x) * sin(piy) \
        * sin(piy) * sin(pi2z) * amodul
    res[2][...] = - sin(pi2x) * sin(piz) \
        * sin(piz) * sin(pi2y) * amodul
    return res


def computeVortBF(res, x, y, z, t):
    amodul = cos(pi * 1. / 3.)
    pix = pi * x
    piy = pi * y
    piz = pi * z
    pi2x = 2. * pix
    pi2y = 2. * piy
    pi2z = 2. * piz
    res[0][...] = 2. * pi * sin(pi2x) * amodul *\
        (- cos(pi2y) * sin(piz) * sin(piz)
         + sin(piy) * sin(piy) * cos(pi2z))
    
    res[1][...] = 2. * pi * sin(pi2y) * amodul *\
        (2. * cos(pi2z) * sin(pix) * sin(pix)
         + sin(piz) * sin(piz) * cos(pi2x))
    
    res[2][...] = -2. * pi * sin(pi2z) * amodul *\
        (cos(pi2x) * sin(piy) * sin(piy)
         + sin(pix) * sin(pix) * cos(pi2y))
    
    return res


def test_stretching():

    # Parameters
    nb = 33
    boxLength = [1., 1., 1.]
    boxMin = [0., 0., 0.]
    nbElem = Discretization([nb, nb, nb], [2, 2, 2])
    timeStep = 0.05

    # Domain
    box = pp.Box(length=boxLength, origin=boxMin)

    # Fields
    velo = Field(
        domain=box, formula=computeVel,
        name='Velocity', is_vector=True)
    vorti = Field(
        domain=box, formula=computeVort,
        name='Vorticity', is_vector=True)

    # Operators
    #method = {TimeIntegrator: RK3, Formulation: Conservative,
    #          SpaceDiscretisation: FD_C_4}
    stretch = Stretching(velo, vorti, discretization=nbElem)
    stretch.discretize()
    topo = stretch.discreteFields[velo].topology
    velo.initialize(topo=topo)
    vorti.initialize(topo=topo)
    stretch.setup()
    simulation = Simulation(tinit=0, tend=1., timeStep=timeStep)
    stretch.apply(simulation)

def test_stretchingLinearized():
    
    # Parameters
    nb = 33
    boxLength = [1., 1., 1.]
    boxMin = [0., 0., 0.]
    nbElem = Discretization([nb, nb, nb], [2, 2, 2])
    timeStep = 0.05
    
    # Domain
    box = pp.Box(length=boxLength, origin=boxMin)
    
    # Fields
    velo = Field(
        domain=box, formula=computeVel,
        name='Velocity', is_vector=True)
    vorti = Field(
        domain=box, formula=computeVort,
        name='Vorticity', is_vector=True)
    veloBF = Field(
        domain=box, formula=computeVelBF,
        name='VelocityBF', is_vector=True)
    vortiBF = Field(
        domain=box, formula=computeVortBF,
        name='VorticityBF', is_vector=True)
        
    # Operators
    # Usual stretching operator
    stretch1 = Stretching(velo, vorti, discretization=nbElem)
    # Linearized stretching operator
    stretch2 = StretchingLinearized(velocity=velo,
                                    vorticity=vorti,
                                    velocity_BF=veloBF,
                                    vorticity_BF=vortiBF,
                                    discretization=nbElem)
    stretch1.discretize()
    stretch2.discretize()
    topo = stretch1.discreteFields[velo].topology
    velo.initialize(topo=topo)
    vorti.initialize(topo=topo)
    veloBF.initialize(topo=topo)
    vortiBF.initialize(topo=topo)
    stretch1.setup()
    stretch2.setup()
    simulation = Simulation(tinit=0, tend=1., timeStep=timeStep)
    stretch1.apply(simulation)
    print 'norm vorti (usual):', vorti.norm(topo)

    stretch2.apply(simulation)
    print 'norm vorti (lin):', vorti.norm(topo)


def test_stretching_external_work():
    # Parameters
    nb = 33
    boxLength = [1., 1., 1.]
    boxMin = [0., 0., 0.]
    nbElem = Discretization([nb, nb, nb], [2, 2, 2])
    timeStep = 0.05

    # Domain
    box = pp.Box(length=boxLength, origin=boxMin)

    # Fields
    velo = Field(
        domain=box, formula=computeVel,
        name='Velocity', is_vector=True)
    vorti = Field(
        domain=box, formula=computeVort,
        name='Vorticity', is_vector=True)

    # Operators
    #method = {TimeIntegrator: RK3, Formulation: Conservative,
    #          SpaceDiscretisation: FD_C_4}
    stretch = Stretching(velo, vorti, discretization=nbElem)
    stretch.discretize()
    wk_p = stretch.get_work_properties()
    rwork = []
    wk_length = len(wk_p['rwork'])
    for i in xrange(wk_length):
        memshape = wk_p['rwork'][i]
        rwork.append(npw.zeros(memshape))

    topo = stretch.discreteFields[velo].topology
    velo.initialize(topo=topo)
    vorti.initialize(topo=topo)
    stretch.setup(rwork=rwork)
    simulation = Simulation(tinit=0, tend=1., timeStep=timeStep)
    stretch.apply(simulation)
