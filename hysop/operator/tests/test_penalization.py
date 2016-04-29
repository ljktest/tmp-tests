# -*- coding: utf-8 -*-
from hysop.domain.subsets import HemiSphere, Sphere, Cylinder
from hysop.domain.porous import Porous
from hysop.operator.penalization import Penalization, PenalizeVorticity
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop.tools.io_utils import IOParams
from hysop.mpi.topology import Cartesian
import hysop.tools.numpywrappers as npw
import numpy as np
import os
from hysop import Field, Box
from hysop.operator.hdf_io import HDF_Reader
from hysop.domain.subsets import SubBox


def v2d(res, x, y, t):
    res[0][...] = 1.
    res[1][...] = 1.
    return res


def s2d(res, x, y, t):
    res[0][...] = 1.
    return res


def v3d(res, x, y, z, t):
    res[0][...] = 1.
    res[1][...] = 1.
    res[2][...] = 1.
    return res


def s3d(res, x, y, z, t):
    res[0][...] = 1.
    return res


def v2dw(res, x, y, t):
    res[0][...] = np.cos(x) + np.sin(y)
    res[1][...] = np.sin(x) + np.cos(y)
    return res


def v3dw(res, x, y, z, t):
    res[0][...] = np.cos(x) + np.sin(y) + np.cos(z)
    res[1][...] = np.sin(x) + np.cos(y)
    res[2][...] = np.cos(z) + np.sin(y) + np.cos(x)
    return res


Nx = 128
Ny = 96
Nz = 102
g = 2


ldef = npw.asrealarray([0.3, 0.4, 1.0])
discr3D = Discretization([Nx + 1, Ny + 1, Nz + 1], [g - 1, g - 2, g])
discr2D = Discretization([Nx + 1, Ny + 1], [g - 1, g])
xdom = npw.asrealarray([0.1, -0.3, 0.5])
import math
ldom = npw.asrealarray([math.pi * 2., ] * 3)
xdef = npw.asrealarray(xdom + 0.2)
xpos = npw.asrealarray(ldom * 0.5)
xpos[-1] += 0.1
working_dir = os.getcwd() + '/'


def init(discr, fileref):
    Cartesian.reset_counter()
    dim = len(discr.resolution)
    dom = Box(dimension=dim, length=ldom[:dim],
              origin=xdom[:dim])
    topo = dom.create_topology(discr)
    scalref = Field(domain=topo.domain, name='scalref')
    #    scalRef.hdf_load(topo, iop, restart=0)
    veloref = Field(domain=topo.domain, name='veloref', is_vector=True)
    # Read a reference file
    iop = IOParams(working_dir + fileref)
    reader = HDF_Reader(variables={scalref: topo, veloref: topo},
                        io_params=iop, restart=0)
    reader.discretize()
    reader.setup()
    reader.apply()
    reader.finalize()
    sdref = scalref.discretize(topo)
    vdref = veloref.discretize(topo)
    return topo, sdref, vdref


def check_penal(penal, sref, vref, scal, velo):
    penal.discretize()
    penal.setup()
    topo = penal.variables[scal]
    scal.initialize(topo=topo)
    velo.initialize(topo=topo)
    vd = velo.discretize(topo)
    sd = scal.discretize(topo)
    simu = Simulation(nbIter=3)
    penal.apply(simu)
    ind = topo.mesh.iCompute
    assert np.allclose(sd.data[0][ind], sref.data[0][ind])
    for d in xrange(vd.nb_components):
        assert np.allclose(vd.data[d][ind], vref.data[d][ind])


def test_penal_2d():
    """
    Penalization in 2D, obstacles = semi-cylinder (disk indeed ...)
    and a plate, fields=scalar and vector.
    """
    topo, sref, vref = init(discr2D, 'penal2d_sphere')
    # Obstacles
    rd = ldom[0] * 0.3
    # Fields to penalize
    scal = Field(domain=topo.domain, formula=s2d, name='Scalar')
    velo = Field(domain=topo.domain, formula=v2d, name='Velo', is_vector=True)
    hsphere = HemiSphere(parent=topo.domain, origin=xpos[:2], radius=rd)
    penal = Penalization(variables=[scal, velo], discretization=topo,
                         obstacles=[hsphere], coeff=1e6)
    hsphere.discretize(topo)
    check_penal(penal, sref, vref, scal, velo)


def test_penal_2d_multi():
    """
    Penalization in 2D, for several different obstacles
    """
    topo, sref, vref = init(discr2D, 'penal2d_multi')
    # Obstacles
    rd = ldom[0] * 0.1
    # Fields to penalize
    scal = Field(domain=topo.domain, formula=s2d, name='Scalar')
    velo = Field(domain=topo.domain, formula=v2d, name='Velo', is_vector=True)
    hsphere = Sphere(parent=topo.domain, origin=xpos[:2], radius=rd)
    newpos = list(xpos)
    newpos[1] += 1.
    hsphere2 = HemiSphere(parent=topo.domain, origin=newpos[:2],
                          radius=rd + 0.3)
    ll = topo.domain.length.copy()
    ll[1] = 0.
    downplane = SubBox(parent=topo.domain, origin=topo.domain.origin,
                       length=ll)
    penal = Penalization(variables=[scal, velo], discretization=topo,
                         obstacles=[hsphere, downplane, hsphere2], coeff=1e6)
    check_penal(penal, sref, vref, scal, velo)


def test_penal_3d():
    """
    Penalization in 3D, obstacles = semi-cylinder (disk indeed ...)
    and a plate, fields=scalar and vector.
    """
    topo, sref, vref = init(discr3D, 'penal3d_sphere')
    # Obstacles
    rd = ldom[0] * 0.3
    # Fields to penalize
    scal = Field(domain=topo.domain, formula=s3d, name='Scalar')
    velo = Field(domain=topo.domain, formula=v3d, name='Velo', is_vector=True)
    hsphere = HemiSphere(parent=topo.domain, origin=xpos, radius=rd)
    penal = Penalization(variables=[scal, velo], discretization=topo,
                         obstacles=[hsphere], coeff=1e6)
    check_penal(penal, sref, vref, scal, velo)


def test_penal_3d_multi():
    """
    Penalization in 3D, for several different obstacles
    """
    topo, sref, vref = init(discr3D, 'penal3d_multi')
    # Obstacles
    rd = ldom[0] * 0.1
    # Fields to penalize
    scal = Field(domain=topo.domain, formula=s3d, name='Scalar')
    velo = Field(domain=topo.domain, formula=v3d, name='Velo', is_vector=True)
    hsphere = Sphere(parent=topo.domain, origin=xpos, radius=rd)
    newpos = list(xpos)
    newpos[1] += 1.
    hsphere2 = HemiSphere(parent=topo.domain, origin=newpos,
                          radius=rd + 0.3)
    ll = topo.domain.length.copy()
    ll[1] = 0.
    downplane = SubBox(parent=topo.domain, origin=topo.domain.origin,
                       length=ll)
    penal = Penalization(variables=[scal, velo], discretization=topo,
                         obstacles=[hsphere, hsphere2, downplane], coeff=1e6)
    check_penal(penal, sref, vref, scal, velo)


def test_penal_3d_porous():
    """
    Penalization in 3D, with porous obstacles
    """
    topo, sref, vref = init(discr3D, 'penal3d_porous')
    # Fields to penalize
    scal = Field(domain=topo.domain, formula=s3d, name='Scalar')
    velo = Field(domain=topo.domain, formula=v3d, name='Velo', is_vector=True)
    newpos = list(xpos)
    newpos[1] += 1.
    psphere = Porous(parent=topo.domain, origin=newpos,
                     source=Sphere, layers=[0.5, 0.7, 0.3])
    ll = topo.domain.length.copy()
    ll[1] = 0.
    downplane = SubBox(parent=topo.domain, origin=topo.domain.origin,
                       length=ll)
    penal = Penalization(variables=[scal, velo], discretization=topo,
                         obstacles={psphere: [1e6, 1e2, 1e1], downplane: 1e7})
    check_penal(penal, sref, vref, scal, velo)


def test_penal_3d_porous_cyl():
    """
    Penalization in 3D, with porous obstacles
    """
    topo, sref, vref = init(discr3D, 'penal3d_porous_cyl')
    # Fields to penalize
    scal = Field(domain=topo.domain, formula=s3d, name='Scalar')
    velo = Field(domain=topo.domain, formula=v3d, name='Velo', is_vector=True)
    newpos = list(xpos)
    newpos[1] += 1.
    pcyl = Porous(parent=topo.domain, origin=newpos,
                  source=Cylinder, layers=[0.5, 0.7, 0.3])
    ll = topo.domain.length.copy()
    ll[1] = 0.
    downplane = SubBox(parent=topo.domain, origin=topo.domain.origin,
                       length=ll)
    penal = Penalization(variables=[scal, velo], discretization=topo,
                         obstacles={pcyl: [1e6, 0.1, 1e6], downplane: 1e7})
    check_penal(penal, sref, vref, scal, velo)


def test_penal_2d_porous():
    """
    Penalization in 2D, with porous obstacles
    """
    topo, sref, vref = init(discr2D, 'penal2d_porous')
    # Fields to penalize
    scal = Field(domain=topo.domain, formula=s2d, name='Scalar')
    velo = Field(domain=topo.domain, formula=v2d, name='Velo', is_vector=True)
    newpos = list(xpos)
    newpos[1] += 1.
    psphere = Porous(parent=topo.domain, origin=newpos[:2],
                     source=Sphere, layers=[0.5, 0.7, 0.3])
    ll = topo.domain.length.copy()
    ll[1] = 0.
    downplane = SubBox(parent=topo.domain, origin=topo.domain.origin,
                       length=ll)
    penal = Penalization(variables=[scal, velo], discretization=topo,
                         obstacles={psphere: [1e6, 1e2, 1e1], downplane: 1e7})
    check_penal(penal, sref, vref, scal, velo)


def init_vort(discr, fileref):
    Cartesian.reset_counter()
    dim = len(discr.resolution)
    dom = Box(dimension=dim, length=ldom[:dim],
              origin=xdom[:dim])
    topo = dom.create_topology(discr)
    wref = Field(domain=topo.domain, name='vortiref', is_vector=dim == 3)
    #    scalRef.hdf_load(topo, iop, restart=0)
    veloref = Field(domain=topo.domain, name='veloref', is_vector=True)
    # Read a reference file
    iop = IOParams(working_dir + fileref)
    reader = HDF_Reader(variables={wref: topo, veloref: topo},
                        io_params=iop, restart=0)
    reader.discretize()
    reader.setup()
    reader.apply()
    reader.finalize()
    wdref = wref.discretize(topo)
    vdref = veloref.discretize(topo)
    return topo, wdref, vdref


def check_penal_vort(penal, wref, vref, vorti, velo):
    penal.discretize()
    penal.setup()
    topo = penal.variables[vorti]
    vorti.initialize(topo=topo)
    velo.initialize(topo=topo)
    vd = velo.discretize(topo)
    wd = vorti.discretize(topo)
    ind = topo.mesh.iCompute

    simu = Simulation(nbIter=200)
    penal.apply(simu)
    for d in xrange(vd.nb_components):
        assert np.allclose(vd.data[d][ind], vref.data[d][ind])
    for d in xrange(wd.nb_components):
        assert np.allclose(wd.data[d][ind], wref.data[d][ind])


def test_penal_vort_2d():
    """
    Penalization + Curl in 2D, obstacles = semi-cylinder (disk indeed ...)
    and a plate, fields=scalar and vector.
    """
    d2d = Discretization([Nx + 1, Ny + 1], [g, g])
    topo, wref, vref = init_vort(d2d, 'penal_vort_2d_sphere')
    # Obstacles
    rd = ldom[0] * 0.3
    # Fields to penalize
    vorti = Field(domain=topo.domain, formula=s2d, name='Vorti')
    velo = Field(domain=topo.domain, formula=v2dw, name='Velo', is_vector=True)
    hsphere = HemiSphere(parent=topo.domain, origin=xpos[:2], radius=rd)
    penal = PenalizeVorticity(velocity=velo, vorticity=vorti,
                              discretization=topo,
                              obstacles=[hsphere], coeff=1e8)
    #hsphere.discretize(topo)
    check_penal_vort(penal, wref, vref, vorti, velo)


def test_penal_vort_3d():
    """
    Penalization in 3D, obstacles = semi-cylinder
    and a plate, fields=scalar and vector.
    """
    d3d = Discretization([Nx + 1, Ny + 1, Nz + 1], [g, g, g])
    topo, wref, vref = init_vort(d3d, 'penal_vort_3d_sphere')
    # Obstacles
    rd = ldom[0] * 0.3
    # Fields to penalize
    vorti = Field(domain=topo.domain, formula=v3d, name='Vorti',
                  is_vector=True)
    velo = Field(domain=topo.domain, formula=v3dw, name='Velo', is_vector=True)
    hsphere = HemiSphere(parent=topo.domain, origin=xpos, radius=rd)
    penal = PenalizeVorticity(velocity=velo, vorticity=vorti,
                              discretization=topo,
                              obstacles=[hsphere], coeff=1e8)
    check_penal_vort(penal, wref, vref, vorti, velo)


def test_penal_vort_multi_2d():
    """
    Penalization in 3D, obstacles = semi-cylinder
    and a plate, fields=scalar and vector.
    """
    d2d = Discretization([Nx + 1, Ny + 1], [g, g])
    topo, wref, vref = init_vort(d2d, 'penal_vort_2d_multi_sphere')
    # Fields to penalize
    vorti = Field(domain=topo.domain, formula=s2d, name='Vorti')
    velo = Field(domain=topo.domain, formula=v2dw, name='Velo', is_vector=True)
    hsphere = Porous(parent=topo.domain, source=HemiSphere,
                     origin=xpos[:2], layers=[0.5, 1.1, 1.])
    penal = PenalizeVorticity(velocity=velo, vorticity=vorti,
                              discretization=topo,
                              obstacles={hsphere: [1, 10, 1e8]})
    check_penal_vort(penal, wref, vref, vorti, velo)


def test_penal_vort_multi_3d():
    """
    Penalization in 3D, obstacles = semi-cylinder
    and a plate, fields=scalar and vector.
    """
    d3d = Discretization([Nx + 1, Ny + 1, Nz + 1], [g, g, g])
    topo, wref, vref = init_vort(d3d, 'penal_vort_3d_multi_sphere')
    # Fields to penalize
    vorti = Field(domain=topo.domain, formula=v3d, name='Vorti',
                  is_vector=True)
    velo = Field(domain=topo.domain, formula=v3dw, name='Velo', is_vector=True)
    hsphere = Porous(parent=topo.domain, source=HemiSphere,
                     origin=xpos, layers=[0.5, 1.1, 1.])
    penal = PenalizeVorticity(velocity=velo, vorticity=vorti,
                              discretization=topo,
                              obstacles={hsphere: [1, 10, 1e8]})
    check_penal_vort(penal, wref, vref, vorti, velo)

