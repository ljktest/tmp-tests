"""Testing regular grids.
"""
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.tools.parameters import Discretization
from hysop.mpi import main_size, main_rank
import hysop.tools.numpywrappers as npw
import numpy as np

Nx = Ny = Nz = 32


def create_mesh(discr):
    """
    Periodic mesh
    """
    gh = discr.ghosts
    dim = gh.size
    dom = Box(dimension=dim, origin=[0.1, ] * dim)
    cutdir = [False, ] * dim
    cutdir[-1] = True
    topo = dom.create_topology(discr, cutdir=cutdir)
    mesh = topo.mesh
    # Test global grid
    assert mesh.discretization == discr
    assert (mesh.space_step == dom.length / (discr.resolution - 1)).all()
    # Position compared with global grid
    shift = npw.asrealarray([0, ] * dim)
    shift[-1] = Nz / main_size * main_rank
    resolution = (discr.resolution - 1) / topo.shape + 2 * gh
    end = shift + resolution - 2 * gh
    assert mesh.position == [slice(shift[d], end[d]) for d in xrange(dim)]
    assert (mesh.start() == shift).all()
    assert (mesh.stop() == end).all()
    # Test local grid
    assert (mesh.resolution == resolution).all()
    assert (mesh.origin == dom.origin + (shift - gh) * mesh.space_step).all()
    assert mesh.iCompute == [slice(gh[d], resolution[d] - gh[d])
                             for d in xrange(dim)]
    assert (mesh.global_indices(dom.origin) == [0, ] * dim).all()
    assert (mesh.local_indices(mesh.origin) == [0, ] * dim).all()
    if main_size == 0:
        assert mesh.is_inside([0.3, ] * dim)
    pt2 = [0., ] * dim
    pt2[-1] = - 0.8
    assert not mesh.is_inside(pt2)
    point = 6 * mesh.space_step + dom.origin
    point[-1] = (Nz - 1) * mesh.space_step[-1] + dom.origin[-1]
    req_point = [6, ] * dim
    req_point[-1] = Nz - 1
    assert (mesh.global_indices(point) == req_point).all()
    if main_rank == main_size - 1:
        assert mesh.is_inside(point)
        pos = npw.asrealarray(req_point)
        pos[-1] = Nz / main_size - 1
        pos += gh
        assert (mesh.local_indices(point) == pos).all()
    else:
        assert mesh.local_indices(point) is False


def test_mesh3d():
    discr = Discretization([Nx + 1, Ny + 1, Nz + 1])
    create_mesh(discr)


def test_mesh3d_ghost():
    discr = Discretization([Nx + 1, Ny + 1, Nz + 1], [2, 2, 2])
    create_mesh(discr)


def test_mesh2d():
    discr = Discretization([Nx + 1, Nz + 1])
    create_mesh(discr)


def test_mesh2d_ghost():
    discr = Discretization([Nx + 1, Nz + 1], [2, 2])
    create_mesh(discr)


def test_convert_local():
    discr = Discretization([Nx + 1, Ny + 1, Nz + 1], [2, 1, 2])
    gh = discr.ghosts
    dim = gh.size
    dom = Box(dimension=dim, origin=[0.1, ] * dim)
    topo = dom.create_topology(discr, cutdir=[False, False, True])
    mesh = topo.mesh

    # test conversion
    start = 2
    end = 10
    source = [slice(start, end), ] * dim
    res = mesh.convert2local(source)
    if main_size == 1:
        assert res == [slice(start + gh[d], end + gh[d]) for d in xrange(dim)]
    elif main_size == 8:
        newstart = start - topo.rank * Nz / main_size
        newend = end - topo.rank * Nz / main_size
        slref = slice(max(mesh.iCompute[-1].start, newstart + gh[-1]),
                      min(max(newend + gh[-1], gh[-1]),
                          mesh.iCompute[-1].stop))
        sl = [slice(start + gh[d], end + gh[d]) for d in xrange(dim)]
        sl[-1] = slref
        if res is not None:
            assert res == sl
        else:
            assert slref.stop - slref.start == 0


def test_convert_global():
    discr = Discretization([Nx + 1, Ny + 1, Nz + 1], [2, 1, 2])
    gh = discr.ghosts
    dim = gh.size
    dom = Box(dimension=dim, origin=[0.1, ] * dim)
    topo = dom.create_topology(discr, cutdir=[False, False, True])
    mesh = topo.mesh

    # test conversion
    start = 3
    end = 6
    source = [slice(start, end), ] * dim
    res = mesh.convert2global(source)
    if main_size == 1:
        assert res == [slice(start - gh[d], end - gh[d]) for d in xrange(dim)]
    elif main_size == 8:
        newstart = start + topo.rank * Nz / main_size
        newend = end + topo.rank * Nz / main_size
        slref = slice(newstart - gh[-1], newend - gh[-1])
        sl = [slice(start - gh[d], end - gh[d]) for d in xrange(dim)]
        sl[-1] = slref
        assert res == sl


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
    hh = np.prod(mesh.space_step)
    i4i = mesh.ind4integ
    ll = np.sum(sd[0][i4i]) * hh
    il = topo.comm.reduce(ll)
    tol = 1e-5
    vol = np.prod(dom.length)
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
    hh = np.prod(mesh.space_step)
    i4i = mesh.ind4integ
    ll = np.sum(sd[0][i4i]) * hh
    il = topo.comm.reduce(ll)
    tol = 1e-5
    vol = np.prod(dom.length)
    if main_rank == 0:
        assert np.abs(il - vol) < tol
