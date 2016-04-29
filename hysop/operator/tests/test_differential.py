"""Tests for differential operators.
"""

import numpy as np
from hysop.domain.box import Box
from hysop.fields.continuous import Field
import hysop.tools.numpywrappers as npw
from hysop.methods_keys import SpaceDiscretisation
from hysop.methods import FD_C_4, FD_C_2
from hysop.operator.differential import Curl, Grad, DivAdvection
from hysop.tools.parameters import Discretization

# Domain and topologies definitions

nb = 65
import math

Lx = Ly = Lz = 2. * math.pi
box1_3d = Box(length=[Lx, Ly, Lz], origin=[0., 0., 0.])
box1_2d = Box(length=[Lx, Ly], origin=[0., 0.])

d1_3d = Discretization([nb, nb, nb], [2, 2, 2])
d1_2d = Discretization([nb, nb], [2, 2])
topo1_3d = box1_3d.create_topology(d1_3d)
topo1_2d = box1_2d.create_topology(d1_2d)

cos = np.cos
sin = np.sin

Nx = 128
Ny = 96
Nz = 102
g = 2
ldef = npw.asrealarray([0.3, 0.4, 1.0])
d3_3d = Discretization([Nx + 1, Ny + 1, Nz + 1], [g, g, g])
d3_2d = Discretization([Nx + 1, Ny + 1], [g, g])
xdom = npw.asrealarray([0.1, -0.3, 0.5])
ldom = npw.asrealarray([math.pi * 2., ] * 3)
xdef = npw.asrealarray(xdom + 0.2)
box3_3d = Box(length=ldom, origin=xdom)
box3_2d = Box(length=ldom[:2], origin=xdom[:2])
topo3_3d = box3_3d.create_topology(d3_3d)
topo3_2d = box3_2d.create_topology(d3_2d)



def velocity_f(res, x, y, z, t):
    res[0][...] = sin(x) * cos(y) * cos(z)
    res[1][...] = - cos(x) * sin(y) * cos(z)
    res[2][...] = 0.
    return res


def vorticity_f(res, x, y, z, t):
    res[0][...] = - cos(x) * sin(y) * sin(z)
    res[1][...] = - sin(x) * cos(y) * sin(z)
    res[2][...] = 2. * sin(x) * sin(y) * cos(z)
    return res


def velocity_f2d(res, x, y, t):
    res[0][...] = sin(x) * cos(y)
    res[1][...] = - cos(x) * sin(y)
    return res


def vorticity_f2d(res, x, y, t):
    res[0][...] = 2. * sin(x) * sin(y)
    return res


def grad_velo(res, x, y, z, t):
    res[0][...] = cos(x) * cos(y) * cos(z)
    res[1][...] = -sin(x) * sin(y) * cos(z)
    res[2][...] = -sin(x) * cos(y) * sin(z)
    res[3][...] = sin(x) * sin(y) * cos(z)
    res[4][...] = - cos(x) * cos(y) * cos(z)
    res[5][...] = cos(x) * sin(y) * sin(z)
    res[6][...] = 0.0
    res[7][...] = 0.0
    res[8][...] = 0.0
    return res


def grad_velo_2d(res, x, y, t):
    res[0][...] = cos(x) * cos(y)
    res[1][...] = -sin(x) * sin(y)
    res[2][...] = sin(x) * sin(y)
    res[3][...] = -cos(x) * cos(y)
    return res


def check(op, ref_formula, topo, op_dim=3, order=4):
    # Reference field
    ref = Field(domain=topo.domain, formula=ref_formula, nb_components=op_dim,
                name='reference')
    ref_d = ref.discretize(topo)
    ref.initialize(topo=topo)
    velo = op.invar
    result = op.outvar
    velo.initialize(topo=topo)
    op.apply()
    res_d = result.discreteFields[topo]

    # Compare results with reference
    ind = topo.mesh.iCompute
    err = topo.mesh.space_step ** order
    dim = topo.domain.dimension
    for i in xrange(result.nb_components):
        print ('err = O(h**order) =', err[i % dim])
        print (np.max(np.abs(res_d[i][ind] - ref_d[i][ind])))
        assert np.allclose(res_d[i][ind], ref_d[i][ind],
                           rtol=err[i % dim])
    op.finalize()


def call_op(class_name, ref_formula, topo, use_work=False,
            op_dim=3, method=None, order=4, vform=velocity_f):
    # Velocity and result fields
    velo = Field(domain=topo.domain, formula=vform, is_vector=True,
                 name='velo')
    result = Field(domain=topo.domain, nb_components=op_dim, name='result')
    # Differential operator
    op = class_name(invar=velo, outvar=result, discretization=topo,
                    method=method)

    op.discretize()
    work = None
    if use_work:
        work_prop = op.get_work_properties()['rwork']
        work = []
        for l in xrange(len(work_prop)):
            shape = work_prop[l]
            work.append(npw.zeros(shape))

    op.setup(rwork=work)
    check(op, ref_formula, topo, op_dim, order)


def call_op_fft(class_name, ref_formula, dom, discr,
                op_dim=3, method=None, order=4, vform=velocity_f):
    # Velocity and result fields
    velo = Field(domain=dom, formula=vform, is_vector=True,
                 name='velo')
    result = Field(domain=dom, nb_components=op_dim, name='result')
    # Differential operator
    op = class_name(invar=velo, outvar=result, discretization=discr,
                    method=method)

    op.discretize()
    op.setup()
    topo = op.discreteFields[velo].topology
    check(op, ref_formula, topo, op_dim, order)


def test_curl_fd_1():
    method = {SpaceDiscretisation: FD_C_4}
    call_op(Curl, vorticity_f, topo1_3d, method=method)


def test_curl_fd_2():
    method = {SpaceDiscretisation: FD_C_2}
    call_op(Curl, vorticity_f, topo1_3d, method=method, order=2)


def test_curl_fd_1_2d():
    method = {SpaceDiscretisation: FD_C_4}
    call_op(Curl, vorticity_f2d, topo1_2d, method=method,
            op_dim=1, vform=velocity_f2d)


def test_curl_fd_3():
    method = {SpaceDiscretisation: FD_C_4}
    call_op(Curl, vorticity_f, topo3_3d, method=method)


def test_curl_fd_3_2d():
    method = {SpaceDiscretisation: FD_C_4}
    call_op(Curl, vorticity_f2d, topo3_2d, op_dim=1,
            method=method, vform=velocity_f2d)


def test_curl_fft_1():
    method = {SpaceDiscretisation: 'fftw'}
    d1_3d_nog = Discretization([nb, nb, nb])
    call_op_fft(Curl, vorticity_f, box1_3d, d1_3d_nog, method=method, order=6)


# def test_curl_fft_1_2d():
#     method = {SpaceDiscretisation: 'fftw'}
#     d1_2d_nog = Discretization([nb, nb])
#     call_op_fft(Curl, vorticity_f2d, box1_2d, d1_2d_nog, op_dim=1,
#                 method=method, order=6, vform=velocity_f2d)

#def test_curl_fft_ghosts():
#    from hysop.methods_keys import SpaceDiscretisation
#    from hysop.operator.differential import Curl
#    method = {SpaceDiscretisation: 'fftw'}
#    call_op(Curl, vorticity_f, method=method, order=6, discretization=d3D)


# def test_curl_fft_2():
#     method = {SpaceDiscretisation: 'fftw'}
#     d2_3d_nog = Discretization([2 * nb, nb, nb])
#     box2_3d = Box(length=[2. * Lx, Ly, Lz], origin=[0., 0., 0.])
#     call_op_fft(Curl, vorticity_f, box2_3d, d2_3d_nog, method=method, order=6)


# def test_curl_fft_2_2d():
#     method = {SpaceDiscretisation: 'fftw'}
#     d2_2d_nog = Discretization([2 * nb, nb])
#     box2_2d = Box(length=[2. * Lx, Ly], origin=[0., 0.])
#     call_op_fft(Curl, vorticity_f, box2_2d, d2_2d_nog, method=method, order=6)


def test_grad_1():
    method = {SpaceDiscretisation: FD_C_2}
    call_op(Grad, grad_velo, topo1_3d, op_dim=9, method=method, order=2)


def test_grad_2():
    method = {SpaceDiscretisation: FD_C_4}
    call_op(Grad, grad_velo, topo1_3d, op_dim=9, method=method)


def test_grad_3():
    method = {SpaceDiscretisation: FD_C_2}
    call_op(Grad, grad_velo, topo3_3d, op_dim=9, method=method, order=2)


def test_grad_3_2d():
    method = {SpaceDiscretisation: FD_C_4}
    call_op(Grad, grad_velo_2d, topo3_2d, op_dim=4, method=method,
            vform=velocity_f2d)


def test_curl_fd_work():
    method = {SpaceDiscretisation: FD_C_4}
    call_op(Curl, vorticity_f, topo3_3d, use_work=True, method=method)


def divadvection_func(res, x, y, z, t):
    res[0][...] = - cos(z) * cos(z) * (cos(x) * cos(x) - sin(x) * sin(x)) - \
        cos(z) * cos(z) * (cos(y) * cos(y) - sin(y) * sin(y))
    return res


def test_div_advection():
    method = {SpaceDiscretisation: FD_C_4}
    call_op(DivAdvection, divadvection_func, topo3_3d, op_dim=1, method=method)
