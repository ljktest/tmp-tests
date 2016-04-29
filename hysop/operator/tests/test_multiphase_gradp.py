"""Testing gradp operator"""
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.multiphase_gradp import MultiphaseGradP
from hysop.numerics.finite_differences import FD_C_4, FD_C_2
import hysop.tools.numpywrappers as npw
import numpy as np
from hysop.methods_keys import Support, SpaceDiscretisation, ExtraArgs
from hysop.constants import HYSOP_REAL
pi, sin, cos = np.pi, np.sin, np.cos

VISCOSITY = 1e-4


def compute_true_res_formula():
    """Tool to compute true res from sympy"""
    import sympy as sp
    from sympy.vector import CoordSysCartesian, gradient
    R = CoordSysCartesian('R')
    u = (sp.sin(2*sp.pi*R.x)*sp.cos(2*sp.pi*R.y)*sp.cos(2*sp.pi*R.z)) * R.i + \
        (-sp.cos(2*sp.pi*R.x)*sp.sin(2*sp.pi*R.y)*sp.cos(2*sp.pi*R.z)) * R.j
    n = sp.symbols('n')
    res = lambda r, c: sp.simplify(u.dot(gradient(u.components[r], R)) -
                                   n * sp.diff(u.components[r], c, c))
    for r, c in zip((R.i, R.j), (R.x, R.y)):
        print str(res(r, c)).replace('R.', '')


def velo_func(res, x, y, z, t):
    res[0][...] = sin(2. * pi * x) * \
        cos(2. * pi * y) * cos(2. * pi * z)
    res[1][...] = - cos(2. * pi * x) * \
        sin(2. * pi * y) * cos(2. * pi * z)
    res[2][...] = 0.
    return res


def res_func(res, x, y, z, t):
    res[0][...] = pi*(4*pi*VISCOSITY*cos(2*y*pi) + cos(pi*(2*x - 2*z)) +
                      cos(pi*(2*x + 2*z)))*sin(2*x*pi)*cos(2*z*pi)
    res[1][...] = pi*(-4*pi*VISCOSITY*cos(2*x*pi) + cos(pi*(2*y - 2*z)) +
                      cos(pi*(2*y + 2*z)))*sin(2*y*pi)*cos(2*z*pi)
    res[2][...] = 9.81
    return res


def test_gradp():
    """Testing gradp operator against analytical result"""
    simu = Simulation(tinit=0.0, tend=1.0, timeStep=0.1, iterMax=1)
    box = Box()
    velo = Field(box, is_vector=True, formula=velo_func, name='v0')
    res = Field(box, is_vector=True, name='res')
    true_res = Field(box, is_vector=True, formula=res_func, name='vres')
    d = Discretization([129,129,129])
    op = MultiphaseGradP(velocity=velo, gradp=res, viscosity=VISCOSITY,
                         discretization=d)
    op.discretize()
    op.setup()
    topo = op.discreteFields[velo].topology
    iCompute = topo.mesh.iCompute
    velo.initialize(topo=topo)
    op.initialize_velocity()
    res.initialize(topo=topo)
    true_res.initialize(topo=topo)
    op.apply(simu)
    d_res = res.discreteFields[topo]
    d_true_res = true_res.discreteFields[topo]
    assert np.allclose(d_res[0][iCompute], d_true_res[0][iCompute], atol=8e-3)
    assert np.allclose(d_res[1][iCompute], d_true_res[1][iCompute], atol=8e-3)
    assert np.allclose(d_res[2][iCompute], d_true_res[2][iCompute])
