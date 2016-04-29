from hysop.domain.subsets import SubBox
from hysop.tools.parameters import Discretization
from hysop import Field, Box
import hysop.tools.numpywrappers as npw
import numpy as np
import math


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

ldef = [0.3, 0.4, 1.0]
discr3D = Discretization([Nx + 1, Ny + 1, Nz + 1], [g - 1, g - 2, g])
discr2D = Discretization([Nx + 1, Ny + 1], [g - 1, g])
xdom = np.asarray([0.1, -0.3, 0.5])
ldom = [math.pi * 2., ] * 3
xdef = xdom + 0.2


def check_subset(discr, xr, lr):
    xr = npw.asrealarray(xr)
    lr = npw.asrealarray(lr)
    dim = len(discr.resolution)
    dom = Box(dimension=dim, length=ldom[:dim],
              origin=xdom[:dim])
    # Starting point and length of the subdomain
    cub = SubBox(origin=xr, length=lr, parent=dom)
    assert (cub.length == lr).all()
    assert (cub.origin == xr).all()
    # discretization of the dom and of its subset
    topo = dom.create_topology(discr)
    cub.discretize(topo)
    assert cub.mesh.values()[0].mesh == topo.mesh
    assert cub.mesh.keys()[0] == topo
    return topo, cub


def test_subbox():
    check_subset(discr3D, xdef, ldef)


def test_integ():
    topo, cub = check_subset(discr3D, xdef, ldef)
    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    vd = velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cub.integrate_field_allc(velo, topo)
    i1 = cub.integrate_dfield_allc(vd)
    vref = np.prod(cub.real_length[topo])
    assert (np.abs(i0 - vref) < 1e-6).all()
    assert (np.abs(i1 - vref) < 1e-6).all()


def test_integ_2():
    topo, cub = check_subset(discr3D, xdef, ldef)
    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    vd = velo.discretize(topo)
    velo.initialize(topo=topo)
    vref = np.prod(cub.real_length[topo])
    for i in xrange(velo.nb_components):
        i0 = cub.integrate_field(velo, topo, component=i)
        assert np.abs(i0 - vref) < 1e-6
        i1 = cub.integrate_dfield(vd, component=i)
        assert np.abs(i1 - vref) < 1e-6


def test_integ_3():
    topo, cub = check_subset(discr3D, xdef, ldef)
    nbc = 1
    vref = np.prod(cub.real_length[topo])
    for i in xrange(1, nbc + 1):
        i0 = cub.integrate_func(f_test, topo, nbc=i)
        assert np.abs(i0 - vref) < 1e-6


def test_integ_4():
    xr = [math.pi * 0.5, ] * 3
    lr = [math.pi, ] * 3
    topo, cub = check_subset(discr3D, xr, lr)
    nbc = 1
    vref = - 2 * np.prod(cub.real_length[topo][:2])
    for i in xrange(1, nbc + 1):
        i0 = cub.integrate_func(f_test_2, topo, nbc=i)
        assert np.abs(i0 - vref) / np.abs(vref) < topo.mesh.space_step[2]


def test_subbox_2d():
    """
    subset == plane in 3D domain
    Test integration on this subset.
    """
    topo, cub = check_subset(discr3D, xdef, [0., 0.2, 0.8])
    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    vd = velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cub.integrate_field_allc(velo, topo)
    vref = np.prod(cub.real_length[topo][1:])
    assert (np.abs(i0 - vref) < 1e-6).all()
    i1 = cub.integrate_dfield_allc(vd)
    assert (np.abs(i1 - vref) < 1e-6).all()


def test_line():
    """
    subset == line in 3D domain
    Test integration on this subset.
    """
    topo, cub = check_subset(discr3D, xdef, [0., 0., 0.8])
    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    vd = velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cub.integrate_field_allc(velo, topo)
    vref = cub.real_length[topo][2]
    assert (np.abs(i0 - vref) < 1e-6).all()
    i1 = cub.integrate_dfield_allc(vd)
    assert (np.abs(i1 - vref) < 1e-6).all()


def test_2d_subbox():
    """
    subset in 2D domain
    """
    topo, cub = check_subset(discr2D, xdef[:2], ldef[:2])
    velo = Field(domain=topo.domain, is_vector=True, formula=v2d, name='velo')
    vd = velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cub.integrate_field_allc(velo, topo)
    vref = np.prod(cub.real_length[topo])
    assert (np.abs(i0 - vref) < 1e-6).all()
    i1 = cub.integrate_dfield_allc(vd)
    assert (np.abs(i1 - vref) < 1e-6).all()


def test_2d_line():
    """
    subset == line in 2D domain
    Test integration on this subset.
    """
    topo, cub = check_subset(discr2D, xdef[:2], lr=[0., 1.0])
    velo = Field(domain=topo.domain, is_vector=True, formula=v2d, name='velo')
    vd = velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cub.integrate_field_allc(velo, topo)
    vref = cub.real_length[topo][1]
    assert (np.abs(i0 - vref) < 1e-6).all()
    i1 = cub.integrate_dfield_allc(vd)
    assert (np.abs(i1 - vref) < 1e-6).all()


def test_integ_fulldom():
    topo, cub = check_subset(discr3D, xdom, ldom)
    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    vd = velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cub.integrate_field_allc(velo, topo)
    i1 = cub.integrate_dfield_allc(vd)
    vref = np.prod(cub.real_length[topo])
    assert (np.abs(i0 - vref) < 1e-6).all()
    assert (np.abs(i1 - vref) < 1e-6).all()


def test_integ_fulldom_2d():
    topo, cub = check_subset(discr2D, xdom[:2], ldom[:2])
    velo = Field(domain=topo.domain, is_vector=True, formula=v2d, name='velo')
    vd = velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cub.integrate_field_allc(velo, topo)
    i1 = cub.integrate_dfield_allc(vd)
    vref = np.prod(cub.real_length[topo])
    assert (np.abs(i0 - vref) < 1e-6).all()
    assert (np.abs(i1 - vref) < 1e-6).all()


def test_full2d_in3d():
    """
    subset == plane in 3D domain. This plane covers a whole
    boundary of the domain.
    Test integration on this subset.
    """
    ll = list(ldom)
    ll[0] = 0.
    print ll, xdom
    topo, cub = check_subset(discr3D, xdom, ll)
    print 'dims s,s,s ', cub.origin, cub.length

    velo = Field(domain=topo.domain, is_vector=True, formula=v3d, name='velo')
    vd = velo.discretize(topo)
    velo.initialize(topo=topo)
    i0 = cub.integrate_field_allc(velo, topo)
    vref = np.prod(cub.real_length[topo][1:])
    assert (np.abs(i0 - vref) < 1e-6).all()
    i1 = cub.integrate_dfield_allc(vd)
    assert (np.abs(i1 - vref) < 1e-6).all()
    print topo
    print cub.mesh[topo].iCompute
