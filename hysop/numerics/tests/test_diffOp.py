# -*- coding: utf-8 -*-
from hysop import Field, Box
import numpy as np
import hysop.numerics.differential_operations as diffop
import hysop.tools.numpywrappers as npw
from hysop.tools.parameters import Discretization
import math as m
pi = m.pi
cos = np.cos
sin = np.sin


def computeVel(res, x, y, z, t):
    res[0][...] = sin(x) * cos(y) * cos(z)
    res[1][...] = - cos(x) * sin(y) * cos(z)
    res[2][...] = 0.
    return res


def computeVort(res, x, y, z, t):
    res[0][...] = - cos(x) * sin(y) * sin(z)
    res[1][...] = - sin(x) * cos(y) * sin(z)
    res[2][...] = 2. * sin(x) * sin(y) * cos(z)
    return res


def laplacian_func(res, x, y, z, t):
    res[0][...] = 3 * cos(x) * sin(y) * sin(z)
    res[1][...] = 3 * sin(x) * cos(y) * sin(z)
    res[2][...] = -6. * sin(x) * sin(y) * cos(z)
    return res


def computeVel2(res, x, y, t):
    res[0][...] = sin(x) * cos(y)
    res[1][...] = - cos(x) * sin(y)
    return res


def computeVort2(res, x, y, t):
    res[0][...] = 2. * sin(x) * sin(y)
    return res


def analyticalDivWV(res, x, y, z, t):
    res[0][...] = - sin(y) * cos(y) * sin(z) * cos(z) * \
        (- sin(x) * sin(x) + cos(x) * cos(x)) + \
        2. * cos(x) * cos(x) * sin(z) * cos(z) * sin(y) * cos(y)
    res[1][...] = - 2. * cos(y) * cos(y) * sin(z) * cos(z) * sin(x) * cos(x) +\
        sin(x) * cos(x) * sin(z) * cos(z) * (cos(y) * cos(y) - sin(y) * sin(y))
    res[2][...] = 0.
    return res


def analyticalDivWV2D(res, x, y, t):
    res[0][...] = 0.
    return res

def analyticalGradVxW(res, x, y, z, t):
    res[0][...] = - sin(y) * cos(y) * sin(z) * cos(z)
    res[1][...] = sin(x) * cos(x) * sin(z) * cos(z)
    res[2][...] = 0.
    return res


def analyticalDivStressTensor(res, x, y, z, t):
    res[0][...] = - 3. * sin(x) * cos(y) * cos(z)
    res[1][...] = 3. * cos(x) * sin(y) * cos(z)
    res[2][...] = 0.
    return res

def analyticalDivAdvection(res, x, y, z, t):
    res[0][...] = - cos(z) * cos(z) * (cos(x) * cos(x) - sin(x) * sin(x)) - \
        cos(z) * cos(z) * (cos(y) * cos(y) - sin(y) * sin(y))
    return res


Nx = 128
Ny = 110
Nz = 162
g = 2
discr3D = Discretization([Nx + 1, Ny + 1, Nz + 1], [g, g, g])
discr2D = Discretization([Nx + 1, Ny + 1], [g, g])
ldom = npw.asrealarray([pi * 2., ] * 3)
xdom = [0., 0., 0.]


def init(discr, vform, wform):
    dim = len(discr.resolution)
    dom = Box(dimension=dim, length=ldom[:dim],
              origin=xdom[:dim])
    topo = dom.create_topology(discr)
    work = []
    shape = tuple(topo.mesh.resolution)
    for _ in xrange(3):
        work.append(npw.zeros(shape))
    velo = Field(domain=dom, formula=vform,
                 name='Velocity', is_vector=True)
    vorti = Field(domain=dom, formula=wform,
                  name='Vorticity', is_vector=dim == 3)
    vd = velo.discretize(topo)
    wd = vorti.discretize(topo)
    velo.initialize(topo=topo)
    vorti.initialize(topo=topo)
    return vd, wd, topo, work


# Init 2D and 3D
v3d, w3d, topo3, work3 = init(discr3D, computeVel, computeVort)
v2d, w2d, topo2, work2 = init(discr2D, computeVel2, computeVort2)
ic3 = topo3.mesh.iCompute
ic2 = topo2.mesh.iCompute


def build_op(op_class, len_result, topo, work):
    lwk = op_class.get_work_length()
    op = op_class(topo=topo, work=work[:lwk])
    memshape = tuple(topo.mesh.resolution)
    result = [npw.zeros(memshape) for _ in xrange(len_result)]
    return op, result


def test_curl():
    op, result = build_op(diffop.Curl, 3, topo3, work3)
    result = op(v3d.data, result)
    rtol = np.max(topo3.mesh.space_step ** 2)
    for i in xrange(3):
        assert np.allclose(w3d.data[i][ic3], result[i][ic3], rtol=rtol)


def test_curl_2d():
    op, result = build_op(diffop.Curl, 1, topo2, work2)
    result = op(v2d.data, result)
    rtol = np.max(topo3.mesh.space_step ** 2)
    assert np.allclose(w2d.data[0][ic2], result[0][ic2], rtol=rtol)


def test_laplacian():
    ref = Field(domain=topo3.domain, formula=laplacian_func,
                name='Analytical', is_vector=True)
    rd = ref.discretize(topo3)
    ref.initialize(topo=topo3)

    op, result = build_op(diffop.Laplacian, 3, topo3, work3)
    result = op(w3d.data, result)
    tol = np.max(topo3.mesh.space_step ** 2)
    for i in xrange(3):
        assert np.allclose(rd.data[i][ic3], result[i][ic3], rtol=tol)


def test_div_rho_v_2d():
    # Reference field
    ref = Field(domain=topo2.domain, formula=analyticalDivWV2D,
                name='Analytical', is_vector=False)
    rd = ref.discretize(topo2)
    ref.initialize(topo=topo2)
    op, result = build_op(diffop.DivRhoV, 1, topo2, work2)
    tol = np.max(topo2.mesh.space_step ** 4)
    result = op(v2d.data, w2d.data[0:1], result)
    # Numerical VS analytical
    assert np.allclose(rd[0][ic2], result[0][ic2], atol=tol)


def test_div_rho_v():
    # Reference field
    ref = Field(domain=topo3.domain, formula=analyticalDivWV,
                name='Analytical', is_vector=True)
    rd = ref.discretize(topo3)
    ref.initialize(topo=topo3)
    op, result = build_op(diffop.DivRhoV, 3, topo3, work3)
    tol = np.max(topo3.mesh.space_step ** 4)
    for i in xrange(3):
        result[i:i + 1] = op(v3d.data, w3d.data[i:i + 1], result[i:i + 1])
        # Numerical VS analytical
        assert np.allclose(rd[i][ic3], result[i][ic3], atol=tol)


def test_divwv():
    # Reference field
    ref = Field(domain=topo3.domain, formula=analyticalDivWV,
                name='Analytical', is_vector=True)
    rd = ref.discretize(topo3)
    ref.initialize(topo=topo3)
    op, result = build_op(diffop.DivWV, 3, topo3, work3)
    result = op(v3d.data, w3d.data, result)

    # Numerical VS analytical
    tol = np.max(topo3.mesh.space_step ** 2)
    for i in xrange(3):
        assert np.allclose(rd[i][ic3], result[i][ic3], atol=tol)


def test_grad_vxw():
    # Reference field
    ref = Field(domain=topo3.domain, formula=analyticalGradVxW,
                name='Analytical', is_vector=True)
    rd = ref.discretize(topo3)
    ref.initialize(topo=topo3)
    op, result = build_op(diffop.GradVxW, 3, topo3, work3)
    diag = npw.zeros(2)
    result, diag = op(v3d.data, w3d.data, result, diag)

    # Numerical VS analytical
    tol = np.max(topo3.mesh.space_step ** 2)
    for i in xrange(3):
        assert np.allclose(rd[i][ic3], result[i][ic3], atol=tol)


# test for RHS of Pressure-Poisson equation
def test_div_advection():
    # Reference scalar field
    ref = Field(domain=topo3.domain, formula=analyticalDivAdvection,
                name='Analytical')
    rd = ref.discretize(topo3)
    ref.initialize(topo=topo3)
    op, result = build_op(diffop.DivAdvection, 1, topo3, work3)
    result = op(v3d.data, result)

    # Numerical VS analytical
    errx = (topo3.domain.length[0] / (Nx - 1)) ** 4
    assert np.allclose(rd[0][ic3], result[0][ic3], rtol=errx)

