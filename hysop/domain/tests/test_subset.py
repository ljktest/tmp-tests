"""Tests computation of subsets of
a given geometry/domain
"""
from hysop.domain.subsets import Sphere, HemiSphere
from hysop.domain.subsets import Subset, SubBox
from hysop.domain.subsets import Cylinder, HemiCylinder
from hysop.operator.hdf_io import HDF_Reader
from hysop.tools.parameters import Discretization
from hysop.tools.io_utils import IOParams
from hysop import Field, Box
from hysop.mpi.topology import Cartesian
import hysop.tools.numpywrappers as npw
import numpy as np
import math


Nx = 128
Ny = 96
Nz = 102
g = 2


ldef = npw.asrealarray([0.3, 0.4, 1.0])
discr3D = Discretization([Nx + 1, Ny + 1, Nz + 1], [g - 1, g - 2, g])
discr2D = Discretization([Nx + 1, Ny + 1], [g - 1, g])
xdom = npw.asrealarray([0.1, -0.3, 0.5])
ldom = npw.asrealarray([math.pi * 2., ] * 3)
xdef = npw.asrealarray(xdom + 0.2)
xpos = npw.asrealarray(ldom * 0.5)
xpos[-1] += 0.1
import os
working_dir = os.getcwd() + "/"


def check_subset(discr, fileref, class_name):
    Cartesian.reset_counter()
    dim = len(discr.resolution)
    dom = Box(dimension=dim, length=ldom[:dim],
              origin=xdom[:dim])
    topo = dom.create_topology(discr)
    vref = Field(domain=topo.domain, name='vref')
    vd = vref.discretize(topo)
    iop = IOParams(working_dir + fileref)
    reader = HDF_Reader(variables={vref: topo}, io_params=iop, restart=0)
    reader.discretize()
    reader.setup()
    reader.apply()
    reader.finalize()
    #return topo, dom, vd
    rd = ldom[0] * 0.4
    subs = class_name(origin=xpos[:dim], radius=rd, parent=dom)
    assert subs.radius == rd
    assert (subs.origin[:dim] == xpos[:dim]).all()
    assert hasattr(subs.chi, '__call__')
    subs.discretize(topo)
    assert subs.ind.keys()[0] == topo
    assert len(subs.ind[topo][0]) == dom.dimension
    scal = Field(domain=topo.domain, is_vector=False, name='s0')
    sd = scal.discretize(topo)
    sd[0][subs.ind[topo][0]] = 2.
    icompute = topo.mesh.iCompute
    return np.allclose(sd[0][icompute], vd[0][icompute])


def test_sphere_3d():
    assert check_subset(discr3D, 'sphere3d', Sphere)


def test_hemisphere_3d():
    assert check_subset(discr3D, 'hemisphere3d', HemiSphere)


def test_cylinder_3d():
    assert check_subset(discr3D, 'cylinder3d', Cylinder)


def test_hemicylinder_3d():
    assert check_subset(discr3D, 'hemicylinder3d', HemiCylinder)


def test_sphere_2d():
    assert check_subset(discr2D, 'sphere2d', Sphere)


def test_hemisphere_2d():
    assert check_subset(discr2D, 'hemisphere2d', HemiSphere)


def init_multi(discr, fileref):
    Cartesian.reset_counter()
    dim = len(discr.resolution)
    dom = Box(dimension=dim, length=ldom[:dim],
              origin=xdom[:dim])
    topo = dom.create_topology(discr)
    vref = Field(domain=topo.domain, name='vref', is_vector=False)
    vd = vref.discretize(topo)
    iop = IOParams(working_dir + fileref)
    reader = HDF_Reader(variables={vref: topo}, io_params=iop, restart=0)
    reader.discretize()
    reader.setup()
    reader.apply()
    reader.finalize()
    scal = Field(domain=topo.domain, is_vector=False, name='s1')
    sd = scal.discretize(topo)
    return topo, sd, vd


def init_obst_list(discr, fileref):
    topo, sd, vd = init_multi(discr, fileref)
    rd = ldom[0] * 0.1
    dim = len(discr.resolution)
    s1 = Sphere(origin=xpos[:dim], radius=rd * 1.4, parent=topo.domain)
    s2 = Cylinder(origin=xpos[:dim], radius=rd, parent=topo.domain)
    newpos = xpos.copy()
    newpos[1] += 3 * rd
    s3 = Sphere(origin=newpos[:dim], radius=1.3 * rd, parent=topo.domain)
    obs_list = [s1, s2, s3]
    for obs in obs_list:
        obs.discretize(topo)
    sd = Field(domain=topo.domain, is_vector=False, name='s2').discretize(topo)
    indices = Subset.union(obs_list, topo)
    sd[0][indices] = 2.
    icompute = topo.mesh.iCompute
    return np.allclose(sd[0][icompute], vd[0][icompute])


def test_union_3d():
    assert init_obst_list(discr3D, 'multi_obst_3d')


def test_union_2d():
    assert init_obst_list(discr2D, 'multi_obst_2d')


def init_subtract(discr, fileref):
    topo, sd, vd = init_multi(discr, fileref)
    dim = len(discr.resolution)
    rd = ldom[0] * 0.1
    s1 = Sphere(origin=xpos[:dim], radius=rd, parent=topo.domain)
    pos = topo.domain.origin + 0.05
    ll = topo.domain.length - 0.2
    box = SubBox(origin=pos, length=ll, parent=topo.domain)
    s1.discretize(topo)
    box.discretize(topo)
    sd = Field(domain=topo.domain, is_vector=False, name='s2').discretize(topo)
    indices = Subset.subtract(box, s1, topo)
    sd[0][indices] = 2.
    icompute = topo.mesh.iCompute
    return np.allclose(sd[0][icompute], vd[0][icompute])


def test_subtract_3d():
    assert init_subtract(discr3D, 'multi_obst_subs_3d')


def test_subtract_2d():
    assert init_subtract(discr2D, 'multi_obst_subs_2d')


def init_subtract_lists(discr, fileref):
    topo, sd, vd = init_multi(discr, fileref)
    dim = len(discr.resolution)
    rd = ldom[0] * 0.1
    obs = []
    obs.append(Sphere(origin=xpos[:dim], radius=rd, parent=topo.domain))
    pos = topo.domain.origin + 0.05
    ll = topo.domain.length - 0.2
    box = []
    box.append(SubBox(origin=pos, length=2 * ll / 3., parent=topo.domain))
    pos2 = topo.domain.origin + 0.05
    pos2[1] += topo.domain.length[1] / 5
    box.append(SubBox(origin=pos2, length=4 * ll / 5., parent=topo.domain))
    obs.append(Cylinder(origin=xpos[:dim], radius=rd * 0.5,
                        parent=topo.domain))
    newpos = xpos.copy()
    newpos[1] += 3 * rd
    obs.append(Sphere(origin=newpos[:dim], radius=1.3 * rd,
                      parent=topo.domain))
    for ob in obs:
        ob.discretize(topo)
    for ob in box:
        ob.discretize(topo)
    sd = Field(domain=topo.domain, is_vector=False, name='s2').discretize(topo)
    indices = Subset.subtract_list_of_sets(box, obs, topo)
    sd[0][indices] = 2.
    return np.allclose(sd[0][topo.mesh.iCompute], vd[0][topo.mesh.iCompute])


def test_subtract_list_3d():
    assert init_subtract_lists(discr3D, 'multi_obst_subslist_3d')


def test_subtract_list_2d():
    assert init_subtract_lists(discr2D, 'multi_obst_subslist_2d')

