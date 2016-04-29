# -*- coding: utf-8 -*-
from hysop.methods import Euler, RK2, RK3, RK4
import hysop.tools.numpywrappers as npw
import math
import numpy as np
from hysop.tools.parameters import Discretization
pi = math.pi
sin = np.sin
cos = np.cos

# Grid resolution for tests
nb = 9
# Initial time
tinit = 0.
# Final time
tend = 0.2
# Time step
#dt = 1e-3
# time sequence
#time = npu.seq(tinit, tend, dt)
#nbSteps = time.size
import hysop as pp
d1 = Discretization([nb + 1])
box = pp.Box(length=[2.0 * pi], origin=[0.])
topo = box.create_topology(dim=1, discretization=d1)


# A set of tests and reference functions

def func1D(t, y, sol, work=None):
    sol[0][...] = -y[0]
    return sol


def func2D(t, y, sol, work=None):

    sol[0][...] = y[1]
    sol[1][...] = -y[0]
    return sol


def func3D(t, y, work=None):
    sol = []
    sol.append(cos(t * y[0]))
    sol.append(sin(t * y[1]))
    sol.append(cos(t * y[2] / 2))
    return sol


def analyticalSolution(t, y):
    sx = (t * np.exp(t) + 1.) * np.exp(-t)
    sy = (t * np.exp(t) + 1.) * np.exp(-t)
    sz = (t * np.exp(t) + 1.) * np.exp(-t)
    return [sx, sy, sz]


def f(t, u):
    fx = -u[0][:, :, :] + t + 1.
    fy = -u[1][:, :, :] + t + 1.
    fz = -u[2][:, :, :] + t + 1.
    return [fx, fy, fz]


# -- 1D cases --
def integrate(integ, nbSteps):
    """
    Integration with hysop
    """
    t = tinit
    time_points = np.linspace(tinit, tend, nbSteps)
    dtt = time_points[1] - time_points[0]
    y = [npw.ones(nb) * math.exp(-tinit)]
    res = [npw.zeros(nb)]
    # work = None
    i = 1
    ref = npw.zeros((nbSteps, nb))
    ref[0, :] = y[0][:]
    while i < nbSteps:
        res = integ(t, y, dtt, res)
        y[0][...] = res[0]
        ref[i, :] = y[0][:]
        t += dtt
        i += 1
    err = 0.0
    for i in xrange(nb):
        err = max(err, (np.abs(ref[:, i] - np.exp(-time_points))).max())
    return dtt, err


def run_integ(integrator, order):
    nbSteps = 100
    lwork = integrator.getWorkLengths(1)
    work = [npw.zeros(nb) for i in xrange(lwork)]
    dt, err = integrate(integrator(1, work, topo, func1D), nbSteps)
    assert err < dt ** order


def test_Euler_1D():
    run_integ(Euler, 1)


def test_RK2_1D():
    run_integ(RK2, 2)


def test_RK3_1D():
    run_integ(RK3, 3)


def test_RK4_1D():
    run_integ(RK4, 4)
