from hysop.domain.control_box import ControlBox
from hysop.domain.subsets import SubBox
from hysop.tools.parameters import Discretization
from hysop import Field, Box
import numpy as np
import math
import hysop.tools.numpywrappers as npw


Nx = 128
Ny = 96
Nz = 102
g = 2


def v3d(res, x, y, z, t):
    res[0][...] = 1.
    res[1][...] = 1.
    res[2][...] = 1.
    return res


def v2d(res, x, y, t):
    res[0][...] = 1.
    res[1][...] = 1.
    return res


def f_test(x, y, z):
    return 1


def f_test_2(x, y, z):
    return math.cos(z)

ldef = [1.1, 0.76, 1.0]
discr3D = Discretization([Nx + 1, Ny + 1, Nz + 1], [g - 1, g - 2, g])
discr2D = Discretization([Nx + 1, Ny + 1], [g - 1, g])
orig = np.asarray([0.1, -0.3, 0.5])
ldom = [math.pi * 2., math.pi * 2., math.pi * 2.]
xdef = orig + 0.2


def check_control_box(discr, xr, lr):
    xr = npw.asrealarray(xr)
    lr = npw.asrealarray(lr)
    dim = len(discr.resolution)
    dom = Box(dimension=dim, length=ldom[:dim],
              origin=orig[:dim])
    # Starting point and length of the subdomain
    cb = ControlBox(origin=xr, length=lr, parent=dom)
    assert (cb.length == lr).all()
    assert (cb.origin == xr).all()
    # discretization of the dom and of its subset
    topo = dom.create_topology(discr)
    cb.discretize(topo)
    assert cb.mesh.values()[0].mesh == topo.mesh
    assert cb.mesh.keys()[0] == topo
    assert len(cb.surf) == dim * 2
    for s in cb.surf:
        assert isinstance(s, SubBox)
        ll = s.real_length[topo]
        assert len(np.where(ll == 0.0)) == 1
    return topo, cb


def test_cb_3d():
    topo, cb = check_control_box(discr3D, xdef, ldef)
    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cb.integrate_field_allc(velo, topo)
    vref = np.prod(cb.real_length[topo])
    assert (np.abs(i0 - vref) < 1e-6).all()
    nbc = velo.nb_components
    sref = npw.zeros(nbc)
    dirs = np.arange(nbc)
    for i in xrange(nbc):
        ilist = np.where(dirs != i)[0]
        sref = np.prod(cb.real_length[topo][ilist])
        isurf = cb.integrate_on_faces(velo, topo, [2 * i])
        assert np.abs(isurf - sref) < 1e-6
        isurf = cb.integrate_on_faces(velo, topo, [2 * i + 1])
        assert np.abs(isurf - sref) < 1e-6


def test_cb_3d_2():
    topo, cb = check_control_box(discr3D, xdef, ldef)
    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    velo.discretize(topo)
    velo.initialize(topo=topo)
    nbc = velo.nb_components
    sref = npw.zeros(nbc)
    dirs = np.arange(nbc)
    list_dir = np.arange(2 * nbc)
    sref = 0
    for i in xrange(nbc):
        ilist = np.where(dirs != i)[0]
        sref += np.prod(cb.real_length[topo][ilist])
    sref *= 2.
    isurf = cb.integrate_on_faces(velo, topo, list_dir)
    assert np.abs(isurf - sref) < 1e-6


def test_cb_3d_3():
    topo, cb = check_control_box(discr3D, xdef, ldef)
    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    velo.discretize(topo)
    velo.initialize(topo=topo)
    nbc = velo.nb_components
    sref = npw.zeros(nbc)
    dirs = np.arange(nbc)
    list_dir = np.arange(2 * nbc)
    sref = 0
    for i in xrange(nbc):
        ilist = np.where(dirs != i)[0]
        sref += np.prod(cb.real_length[topo][ilist])
    sref *= 2.
    isurf = cb.integrate_on_faces_allc(velo, topo, list_dir)
    assert (np.abs(isurf - sref) < 1e-6).all()


def test_cb_2d():
    topo, cb = check_control_box(discr2D, xdef[:2], ldef[:2])
    velo = Field(domain=topo.domain, is_vector=True, formula=v2d, name='velo')
    velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cb.integrate_field_allc(velo, topo)
    vref = np.prod(cb.real_length[topo])
    assert (np.abs(i0 - vref) < 1e-6).all()
    nbc = velo.nb_components
    sref = npw.zeros(nbc)
    dirs = np.arange(nbc)
    for i in xrange(nbc):
        ilist = np.where(dirs != i)[0]
        sref = np.prod(cb.real_length[topo][ilist])
        isurf = cb.integrate_on_faces(velo, topo, [2 * i])
        assert np.abs(isurf - sref) < 1e-6
        isurf = cb.integrate_on_faces(velo, topo, [2 * i + 1])
        assert np.abs(isurf - sref) < 1e-6


def test_cb_2d_2():
    topo, cb = check_control_box(discr2D, xdef[:2], ldef[:2])
    velo = Field(domain=topo.domain, is_vector=True, formula=v2d, name='velo')
    velo.discretize(topo)
    velo.initialize(topo=topo)
    nbc = velo.nb_components
    sref = npw.zeros(nbc)
    dirs = np.arange(nbc)
    list_dir = np.arange(2 * nbc)
    sref = 0
    for i in xrange(nbc):
        ilist = np.where(dirs != i)[0]
        sref += np.prod(cb.real_length[topo][ilist])
    sref *= 2.
    isurf = cb.integrate_on_faces(velo, topo, list_dir)
    assert np.abs(isurf - sref) < 1e-6


def test_cb_2d_3():
    topo, cb = check_control_box(discr2D, xdef[:2], ldef[:2])
    velo = Field(domain=topo.domain, is_vector=True, formula=v2d, name='velo')
    velo.discretize(topo)
    velo.initialize(topo=topo)
    nbc = velo.nb_components
    sref = npw.zeros(nbc)
    dirs = np.arange(nbc)
    list_dir = np.arange(2 * nbc)
    sref = 0
    for i in xrange(nbc):
        ilist = np.where(dirs != i)[0]
        sref += np.prod(cb.real_length[topo][ilist])
    sref *= 2.
    isurf = cb.integrate_on_faces_allc(velo, topo, list_dir)
    assert (np.abs(isurf - sref) < 1e-6).all()


def test_cb_3d_fulldomain():
    """
    A cb defined on the whole domain.
    Integrals on upper surfaces must fail,
    because of periodic bc.
    """
    topo, cb = check_control_box(discr3D, orig, ldom)
    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cb.integrate_field_allc(velo, topo)
    vref = np.prod(cb.real_length[topo])
    assert (np.abs(i0 - vref) < 1e-6).all()
    nbc = velo.nb_components
    sref = npw.zeros(nbc)
    dirs = np.arange(nbc)
    for i in xrange(nbc):
        ilist = np.where(dirs != i)[0]
        sref = np.prod(cb.real_length[topo][ilist])
        isurf = cb.integrate_on_faces(velo, topo, [2 * i])
        assert np.abs(isurf - sref) < 1e-6
        res = False
        try:
            isurf = cb.integrate_on_faces(velo, topo, [2 * i + 1])
        except:
            res = True
        assert res
