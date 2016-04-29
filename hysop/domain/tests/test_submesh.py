"""Testing regular grids.
"""
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.tools.parameters import Discretization
from hysop.domain.submesh import SubMesh
from hysop.tools.misc import utils
import numpy as np
from hysop.mpi import main_rank

Nx = Ny = Nz = 32
g = 2


def check_submesh(discr):
    """
    Periodic mesh
    """
    dim = len(discr.resolution)
    dom = Box(dimension=dim, origin=[0.1, ] * dim)
    topo = dom.create_topology(discr)
    mesh = topo.mesh
    # Set a starting point and a length for the submesh
    xr = dom.origin + 3. * mesh.space_step
    # Find the indices of these points
    gstart = mesh.global_indices(xr)
    gend = mesh.global_indices(xr + 15 * mesh.space_step)
    # Create the submesh
    subm = SubMesh(mesh, gstart, gend)
    # check global params of the submesh
    assert subm.mesh is mesh
    assert (subm.substart == gstart).all()
    assert (subm.subend == gend).all()
    assert np.allclose(subm.global_origin,
                       dom.origin + gstart * mesh.space_step)
    # the global position we expect
    gpos = [slice(gstart[d], gend[d] + 1) for d in xrange(dim)]
    assert (subm.discretization.resolution == [gend[d] - gstart[d] + 1
                                               for d in xrange(dim)]).all()
    assert (subm.discretization.ghosts == 0).all()
    # check position of the local submesh, relative to the global grid
    # It must corresponds to the intersection of the expected position
    # of the submesh and the position of the parent mesh.
    if subm.on_proc:
        assert subm.position_in_parent == [s for s in
                                           utils.intersl(gpos, mesh.position)]

        r2 = mesh.convert2local([s for s
                                 in utils.intersl(gpos, mesh.position)])
        assert subm.iCompute == r2
        pos = subm.position_in_parent
        subpos = [slice(pos[d].start - gstart[d],
                        pos[d].stop - gstart[d]) for d in xrange(dim)]
        assert subm.position == subpos
        check_coords(mesh, subm)


def check_coords(m1, m2):
    dim = m1.domain.dimension
    x0 = np.zeros(dim)
    xend = np.zeros(dim)
    x0[0] = m2.cx()[0]
    xend[0] = m2.cx()[-1]
    if dim > 1:
        x0[1] = m2.cy()[0]
        xend[1] = m2.cy()[-1]
    if dim > 2:
        x0[2] = m2.cz()[0]
        xend[2] = m2.cz()[-1]
    req_orig = np.maximum(m1.origin + m1.discretization.ghosts * m1.space_step,
                          m1.domain.origin + m2.substart * m1.space_step)
    req_end = np.minimum(m1.end - m1.discretization.ghosts * m1.space_step,
                         m1.domain.origin + m2.subend * m1.space_step)
    assert np.allclose(x0, req_orig)
    assert np.allclose(xend, req_end)


def test_submesh_3d():
    """
    3D Periodic mesh
    """
    discr = Discretization([Nx + 1, Ny + 1, Nz + 1], [g, g, g])
    check_submesh(discr)


def test_submesh_2d():
    """
    2D Periodic mesh
    """
    discr = Discretization([Nx + 1, Nz + 1], [g, g])
    check_submesh(discr)


def test_integ_2d():
    discr = Discretization([Nx + 1, Ny + 1], [2, 1])
    gh = discr.ghosts
    dim = gh.size
    ll = [3.14, 12.8]
    dom = Box(dimension=dim, origin=[0.1, ] * dim, length=ll)
    topo = dom.create_topology(discr)
    scal = Field(domain=dom, name='scal')
    sd = scal.discretize(topo)
    sd[0][...] = 1.0
    mesh = topo.mesh
    xr = dom.origin + 3. * mesh.space_step
    gstart = mesh.global_indices(xr)
    gend = mesh.global_indices(xr + 15 * mesh.space_step)
    subm = SubMesh(mesh, gstart, gend)
    hh = np.prod(subm.mesh.space_step)
    i4i = subm.ind4integ
    ll = np.sum(sd[0][i4i]) * hh
    il = topo.comm.reduce(ll)
    tol = 1e-5
    vol = np.prod(subm.global_length)
    if main_rank == 0:
        assert np.abs(il - vol) < tol


def test_integ_3d():
    discr = Discretization([Nx + 1, Ny + 1, Nz + 1], [2, 1, 2])
    gh = discr.ghosts
    dim = gh.size
    ll = [3.14, 12.8, 1.09]
    dom = Box(dimension=dim, origin=[0.1, ] * dim, length=ll)
    topo = dom.create_topology(discr)
    scal = Field(domain=dom, name='scal')
    sd = scal.discretize(topo)
    sd[0][...] = 1.0
    mesh = topo.mesh
    xr = dom.origin + 3. * mesh.space_step
    gstart = mesh.global_indices(xr)
    gend = mesh.global_indices(xr + 15 * mesh.space_step)
    subm = SubMesh(mesh, gstart, gend)
    hh = np.prod(subm.mesh.space_step)
    i4i = subm.ind4integ
    ll = np.sum(sd[0][i4i]) * hh
    il = topo.comm.reduce(ll)
    tol = 1e-5
    vol = np.prod(subm.global_length)
    if main_rank == 0:
        assert np.abs(il - vol) < tol
