""" Tests for porous subsets module"""

from hysop import Field, Box, Discretization, IOParams
from hysop.mpi.topology import Cartesian
from hysop.domain.subsets import HemiSphere, Sphere
from hysop.domain.subsets import Cylinder, HemiCylinder
import hysop.tools.numpywrappers as npw
import numpy as np
from hysop.domain.porous import BiPole, QuadriPole,\
    Ring, RingPole, Porous
from hysop.operator.hdf_io import HDF_Reader


Nx = Ny = Nz = 128
g = 2
d3d = Discretization([Nx + 1, Ny + 1, Nz + 1], [g, g, g])
d2d = Discretization([Nx + 1, Ny + 1], [g, g])
import math

ldom = npw.asrealarray([math.pi * 2., ] * 3)
xdom = npw.asrealarray([0.1, -0.3, 0.5])
import os
working_dir = os.getcwd() + "/"
xpos = npw.asrealarray(ldom * 0.5)
xpos[-1] += 0.1


def init_model(dref, filename):
    Cartesian.reset_counter()
    dim = len(dref.resolution)
    dom = Box(dimension=dim, length=ldom[:dim], origin=xdom[:dim])
    topo = dom.create_topology(dref)
    sref = Field(domain=dom, name='scal')
    iop = IOParams(working_dir + filename)
    # read sref from reference file
    wr = HDF_Reader(variables={sref: topo}, io_params=iop, restart=0)
    wr.discretize()
    wr.setup()
    wr.apply()

    return sref, topo


def assert_porous(dref, sourcename, filename):
    sref, topo = init_model(dref, filename)
    dim = sref.domain.dimension
    bp = Porous(parent=sref.domain, source=sourcename, origin=xpos[:dim],
                layers=[0.5, 1.5, 0.3])
    ind = bp.discretize(topo)
    scal = Field(domain=sref.domain, name='newscal')
    sd = scal.discretize(topo)
    scal.initialize(topo=topo)
    li = len(ind)
    for i in xrange(li):
        sd.data[0][ind[i]] = 10 * i + 1
    sdref = sref.discretize(topo)
    ic = topo.mesh.iCompute
    assert np.allclose(sd.data[0][ic], sdref.data[0][ic])


def test_porous_sphere_3d():
    assert_porous(d3d, Sphere, 'porous_sphere_3d')


def test_porous_hemisphere_3d():
    assert_porous(d3d, HemiSphere, 'porous_hemisphere_3d')


def test_porous_cylinder_3d():
    assert_porous(d3d, Cylinder, 'porous_cylinder_3d')


def test_porous_hemicylinder_3d():
    assert_porous(d3d, HemiCylinder, 'porous_hemicylinder_3d')


def test_porous_sphere_2d():
    assert_porous(d2d, Sphere, 'porous_sphere_2d')


def test_porous_hemisphere_2d():
    assert_porous(d2d, HemiSphere, 'porous_hemisphere_2d')


def assert_bipole(dref, sourcename, filename):
    sref, topo = init_model(dref, filename)
    dim = sref.domain.dimension
    bp = BiPole(parent=sref.domain, source=sourcename, origin=xpos[:dim],
                layers=[0.5, 1.5], poles_thickness=[0.5, 0.7])
    ind = bp.discretize(topo)
    scal = Field(domain=sref.domain, name='newscal')
    sd = scal.discretize(topo)
    scal.initialize(topo=topo)
    li = len(ind)
    for i in xrange(li):
        sd.data[0][ind[i]] = 10 * i + 1
    sdref = sref.discretize(topo)
    ic = topo.mesh.iCompute
    assert np.allclose(sd.data[0][ic], sdref.data[0][ic])


def test_bipole_sphere_3d():
    assert_bipole(d3d, Sphere, 'bipole_sphere_3d')


def test_bipole_hemisphere_3d():
    assert_bipole(d3d, HemiSphere, 'bipole_hemisphere_3d')


def test_bipole_cylinder_3d():
    assert_bipole(d3d, Cylinder, 'bipole_cylinder_3d')


def test_bipole_hemicylinder_3d():
    assert_bipole(d3d, HemiCylinder, 'bipole_hemicylinder_3d')


def test_bipole_sphere_2d():
    assert_bipole(d2d, Sphere, 'bipole_sphere_2d')


def test_bipole_hemisphere_2d():
    assert_bipole(d2d, HemiSphere, 'bipole_hemisphere_2d')


def assert_quadripole(dref, sourcename, filename):
    sref, topo = init_model(dref, filename)
    dim = sref.domain.dimension
    bp = QuadriPole(parent=sref.domain, source=sourcename, origin=xpos[:dim],
                    layers=[0.5, 1.5], poles_thickness=[0.5, 0.7])
    ind = bp.discretize(topo)
    scal = Field(domain=sref.domain, name='newscal')
    sd = scal.discretize(topo)
    scal.initialize(topo=topo)
    li = len(ind)
    for i in xrange(li):
        sd.data[0][ind[i]] = 10 * i + 1
    sdref = sref.discretize(topo)
    ic = topo.mesh.iCompute
    assert np.allclose(sd.data[0][ic], sdref.data[0][ic])


def test_quadripole_sphere_3d():
    assert_quadripole(d3d, Sphere, 'quadripole_sphere_3d')


def test_quadripole_hemisphere_3d():
    assert_quadripole(d3d, HemiSphere, 'quadripole_hemisphere_3d')


def assert_ring(dref, sourcename, filename):
    sref, topo = init_model(dref, filename)
    dim = sref.domain.dimension
    bp = Ring(parent=sref.domain, source=sourcename, origin=xpos[:dim],
              layers=[0.5, 1.5], ring_width=[0.5, 0.3])
    ind = bp.discretize(topo)
    scal = Field(domain=sref.domain, name='newscal')
    sd = scal.discretize(topo)
    scal.initialize(topo=topo)
    li = len(ind)
    for i in xrange(li):
        sd.data[0][ind[i]] = 10 * i + 1
    sdref = sref.discretize(topo)
    ic = topo.mesh.iCompute
    assert np.allclose(sd.data[0][ic], sdref.data[0][ic])


def test_ring_sphere_3d():
    assert_ring(d3d, Sphere, 'ring_sphere_3d')


def test_ring_hemisphere_3d():
    assert_ring(d3d, HemiSphere, 'ring_hemisphere_3d')


def test_ring_sphere_2d():
    assert_ring(d2d, Sphere, 'ring_sphere_2d')


def test_ring_hemisphere_2d():
    assert_ring(d2d, HemiSphere, 'ring_hemisphere_2d')


def assert_ringpole(dref, sourcename, filename):
    sref, topo = init_model(dref, filename)
    dim = sref.domain.dimension
    bp = RingPole(parent=sref.domain, source=sourcename, origin=xpos[:dim],
                  layers=[0.5, 1.5], ring_width=0.5)
    ind = bp.discretize(topo)
    scal = Field(domain=sref.domain, name='newscal')
    sd = scal.discretize(topo)
    scal.initialize(topo=topo)
    li = len(ind)
    for i in xrange(li):
        sd.data[0][ind[i]] = 10 * i + 1
    sdref = sref.discretize(topo)
    ic = topo.mesh.iCompute
    assert np.allclose(sd.data[0][ic], sdref.data[0][ic])


def test_ringpole_sphere_3d():
    assert_ringpole(d3d, Sphere, 'ringpole_sphere_3d')


def test_ringpole_hemisphere_3d():
    assert_ringpole(d3d, HemiSphere, 'ringpole_hemisphere_3d')


def test_ringpole_sphere_2d():
    assert_ringpole(d2d, Sphere, 'ringpole_sphere_2d')


def test_ringpole_hemisphere_2d():
    assert_ringpole(d2d, HemiSphere, 'ringpole_hemisphere_2d')
