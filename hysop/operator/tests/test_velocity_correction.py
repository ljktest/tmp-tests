# -*- coding: utf-8 -*-

import hysop as pp
from hysop.operator.velocity_correction import VelocityCorrection
from hysop.problem.simulation import Simulation
import numpy as np
import hysop.tools.numpywrappers as npw
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


## Function to compute TG velocity
def computeVel2D(res, x, y, t):
    res[0][...] = sin(x) * cos(y)
    res[1][...] = - cos(x) * sin(y)
    return res


## Function to compute reference vorticity
def computeVort2D(res, x, y, t):
    res[0][...] = - cos(x) * sin(y)
    return res

## Global resolution
g = 0
d2D = Discretization([33, 33], [g, g])
d3D = Discretization([33, 33, 33], [g, g, g])


def test_velocity_correction_3D():
    # Domain
    box = pp.Box(length=[2.0 * pi, pi, pi])
    # Vector Fields
    velo = pp.Field(domain=box, formula=computeVel,
                    name='Velocity', is_vector=True)
    vorti = pp.Field(domain=box, formula=computeVort,
                     name='Vorticity', is_vector=True)

    # Usual Cartesian topology definition
    topo = box.create_topology(discretization=d3D)

    ref_rate = npw.zeros(3)
    ref_rate[0] = uinf * box.length[1] * box.length[2]
    rate = pp.VariableParameter(data=ref_rate, name='flowrate')
    op = VelocityCorrection(velo, vorti, req_flowrate=rate,
                            discretization=topo, io_params={})
    op.discretize()
    op.setup()
    # === Simulation setup ===
    simu = Simulation(tinit=0.0, tend=5., timeStep=0.005, iterMax=1000000)
    # init fields
    velo.initialize(topo=topo)
    vorti.initialize(topo=topo)
    # Apply correction
    op.apply(simu)
    # check new flowrate values
    sref = op.cb.surf[0]
    flowrate = sref.integrate_field_allc(velo, topo)
    assert (np.abs(flowrate - ref_rate) < tol).all()


def test_velocity_correction_2D():
    ## Domain
    box = pp.Box(length=[2.0 * pi, pi], origin=[0., 0.])

    ## Vector Fields
    velo = pp.Field(domain=box, formula=computeVel2D,
                    name='Velocity', is_vector=True)
    vorti = pp.Field(domain=box, formula=computeVort2D,
                     name='Vorticity', is_vector=False)

    ## Usual Cartesian topology definition
    topo = box.create_topology(discretization=d2D)

    ref_rate = npw.zeros(2)
    ref_rate[0] = uinf * box.length[1]
    rate = pp.VariableParameter(data=ref_rate, name='flowrate')
    op = VelocityCorrection(velo, vorti, req_flowrate=rate,
                            discretization=topo)
    op.discretize()
    op.setup()
    # === Simulation setup ===
    simu = Simulation(tinit=0.0, tend=5., timeStep=0.005, iterMax=1000000)
    # init fields
    velo.initialize(topo=topo)
    vorti.initialize(topo=topo)

    # Apply correction
    op.apply(simu)
    # check new flowrate values
    sref = op.cb.surf[0]
    flowrate = sref.integrate_field_allc(velo, topo)
    assert (np.abs(flowrate - ref_rate) < tol).all()
