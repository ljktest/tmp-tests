# -*- coding: utf-8 -*-
"""Tests for reader/writer of fields in hdf5 format.
"""

from hysop import Box, Field
import numpy as np
import os
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop.operator.hdf_io import HDF_Writer, HDF_Reader
from hysop.tools.io_utils import IO, IOParams
from hysop.mpi import main_rank, main_size
from hysop.domain.subsets import SubBox
from hysop.testsenv import postclean

Lx = 2.
nb = 65
working_dir = os.getcwd() + '/test_hdf5/p' + str(main_size)
#IO.set_default_path(os.getcwd() + '/test_hdf5/')
if main_rank == 0:
    print 'Set I/O default path to ', IO.default_path()

cos = np.cos
sin = np.sin


def init1(dim):
    # Domain (cubic)
    dom = Box(length=[Lx] * dim, origin=[-1.] * dim)
    # global resolution for the grid
    resol = Discretization([nb] * dim, [2] * dim)
    topo = dom.create_topology(discretization=resol)
    return dom, topo


def init2():
    # Domain (not cubic)
    dom = Box(length=[Lx, 2 * Lx, 3.9 * Lx], origin=[-1., 2., 3.9])
    # global resolution for the grid
    resol = Discretization([nb, 2 * nb, nb + 8], [2, 0, 1])
    topo = dom.create_topology(discretization=resol)
    return dom, topo


def func3D(res, x, y, z, t):
    res[0][...] = cos(t * x) + sin(y) + z
    return res


def vec3D(res, x, y, z, t):
    res[0][...] = cos(t * x) + sin(y) + z + 0.2
    res[1][...] = sin(t * x) + sin(y) + z + 0.3
    res[2][...] = 3 * cos(2 * t * x) + sin(y) + y
    return res


def vort3D(res, x, y, z, t):
    res[0][...] = 3 * cos(2 * t * x) + cos(y) + z
    res[1][...] = sin(t * y) + x + 0.2
    res[2][...] = 3 * cos(t) + sin(y) + z
    return res


@postclean(working_dir)
def test_write_read_scalar_3D():
    dom, topo = init1(3)
    scal3D = Field(domain=dom, name='Scal3D')
    scalRef = Field(domain=dom, formula=func3D, name='ScalRef3D')

    filename = working_dir + '/testIO_scal'
    iop = IOParams(filename, fileformat=IO.HDF5)
    op = HDF_Writer(variables={scalRef: topo}, io_params=iop)
    simu = Simulation(nbIter=10)
    op.discretize()
    op.setup()
    simu.initialize()

    scalRef.initialize(simu.time, topo=topo)
    op.apply(simu)

    simu.advance()
    simu.advance()
    # Print scalRef for other iterations
    op.apply(simu)
    op.finalize()
    fullpath = iop.filename
    assert os.path.exists(fullpath + '.xmf')
    assert os.path.exists(fullpath + '_00000.h5')
    assert os.path.exists(fullpath + '_00002.h5')

    # Reader
    iop_read = IOParams(working_dir + '/testIO_scal_00002.h5',
                        fileformat=IO.HDF5)
    reader = HDF_Reader(variables=[scal3D], discretization=topo,
                        io_params=iop_read,
                        var_names={scal3D: 'ScalRef3D_' + str(topo.get_id())})
    reader.discretize()
    reader.setup()
    sc3d = scal3D.discretize(topo)
    scref = scalRef.discretize(topo)
    ind = topo.mesh.iCompute
    for d in xrange(scal3D.nb_components):
        sc3d.data[d][...] = 0.0
        assert not np.allclose(scref.data[d][ind], sc3d.data[d][ind])
    reader.apply()
    reader.finalize()

    for d in xrange(scal3D.nb_components):
        assert np.allclose(scref.data[d][ind], sc3d.data[d][ind])


@postclean(working_dir)
def test_write_read_scalar_3D_defaults():
    dom, topo = init1(3)
    scal3D = Field(domain=dom, name='Scal3D')
    scalRef = Field(domain=dom, formula=func3D, name='ScalRef3D')

    # Write a scalar field, using default configuration for output
    # names and location
    op = HDF_Writer(variables={scalRef: topo})
    simu = Simulation(nbIter=3)
    op.discretize()
    op.setup()
    scal3D.discretize(topo=topo)
    scalRef.initialize(simu.time, topo=topo)
    simu.initialize()
    while not simu.isOver:
        op.apply(simu)
        simu.advance()

    op.finalize()
    filename = scalRef.name
    fullpath = os.path.join(IO.default_path(), filename)

    assert os.path.exists(fullpath + '.xmf')
    assert os.path.exists(fullpath + '_00000.h5')
    assert os.path.exists(fullpath + '_00001.h5')

    sc3d = scal3D.discretize(topo)
    scref = scalRef.discretize(topo)
    ind = topo.mesh.iCompute
    for d in xrange(scal3D.nb_components):
        sc3d.data[d][...] = scref.data[d][...]
        scref.data[d][...] = 0.0
        # reinit ScalRef

    # Read a scalar field, using default configuration for output
    # names and location, with a given iteration number.
    reader = HDF_Reader(variables={scalRef: topo},
                        restart=simu.currentIteration - 1)
    reader.discretize()
    reader.setup()
    for d in xrange(scal3D.nb_components):
        assert not np.allclose(scref.data[d][ind], sc3d.data[d][ind])
    reader.apply()
    reader.finalize()

    for d in xrange(scal3D.nb_components):
        assert np.allclose(scref.data[d][ind], sc3d.data[d][ind])

@postclean(working_dir)
def test_write_read_vectors_3D_defaults():
    dom, topo = init2()
    velo = Field(domain=dom, formula=vec3D, name='velo', is_vector=True)
    vorti = Field(domain=dom, formula=vort3D, name='vorti', is_vector=True)

    # Write a vector field, using default configuration for output
    # names and location
    op = HDF_Writer(variables={velo: topo, vorti: topo})
    simu = Simulation(nbIter=3)
    op.discretize()
    op.setup()
    velo.initialize(simu.time, topo=topo)
    vorti.initialize(simu.time, topo=topo)
    simu.initialize()
    while not simu.isOver:
        op.apply(simu)
        simu.advance()

    op.finalize()
    filename = ''
    names = []
    for var in op.input:
        names.append(var.name)
        names.sort()
    for nn in names:
        filename += nn + '_'
    filename = filename[:-1]
    fullpath = os.path.join(IO.default_path(), filename)

    assert os.path.exists(fullpath + '.xmf')
    assert os.path.exists(fullpath + '_00000.h5')
    assert os.path.exists(fullpath + '_00001.h5')

    v3d = velo.discretize(topo)
    w3d = vorti.discretize(topo)
    ind = topo.mesh.iCompute

    # Copy current values of v3 and w3 into buff1 and buff2, for comparison
    # after reader.apply, below.
    buff1 = Field(domain=dom, name='buff1', is_vector=True)
    buff2 = Field(domain=dom, name='buff2', is_vector=True)
    b1 = buff1.discretize(topo=topo)
    b2 = buff2.discretize(topo=topo)
    for d in xrange(velo.nb_components):
        b1.data[d][...] = v3d.data[d][...]
        b2.data[d][...] = w3d.data[d][...]
        # reset v3 and w3 to zero.
        v3d.data[d][...] = 0.0
        w3d.data[d][...] = 0.0

    # Read vector fields, using default configuration for input
    # names and location, with a given iteration number.
    # If everything works fine, reader must read output from
    # the writer above.
    reader = HDF_Reader(variables={velo: topo, vorti: topo},
                        io_params=IOParams(filename),
                        restart=simu.currentIteration - 1)
    reader.discretize()
    reader.setup()
    for d in xrange(v3d.nb_components):
        assert not np.allclose(b1.data[d][ind], v3d.data[d][ind])
        assert not np.allclose(b2.data[d][ind], w3d.data[d][ind])

    reader.apply()
    reader.finalize()
    # Now, v3 and w3 (just read) must be equal to saved values in b1 and b2.
    for d in xrange(v3d.nb_components):
        assert np.allclose(b1.data[d][ind], v3d.data[d][ind])
        assert np.allclose(b2.data[d][ind], w3d.data[d][ind])


@postclean(working_dir)
def test_write_read_vectors_3D():
    dom, topo = init2()
    velo = Field(domain=dom, formula=vec3D, name='velo', is_vector=True)
    vorti = Field(domain=dom, formula=vort3D, name='vorti', is_vector=True)

    # Write a vector field, using default for output location
    # but with fixed names for datasets
    filename = working_dir + '/testIO_vec'
    iop = IOParams(filename, fileformat=IO.HDF5)
    op = HDF_Writer(variables={velo: topo, vorti: topo},
                    var_names={velo: 'io_1', vorti: 'io_2'}, io_params=iop)
    simu = Simulation(nbIter=3)
    op.discretize()
    op.setup()

    velo.initialize(simu.time, topo=topo)
    vorti.initialize(simu.time, topo=topo)
    simu.initialize()
    while not simu.isOver:
        op.apply(simu)
        simu.advance()

    op.finalize()

    # filename = ''
    # for v in op.input:
    #     filename += v.name
    #     filename += '_'
    fullpath = iop.filename
    assert os.path.exists(fullpath + '.xmf')
    assert os.path.exists(fullpath + '_00000.h5')
    assert os.path.exists(fullpath + '_00001.h5')

    v3d = velo.discretize(topo)
    w3d = vorti.discretize(topo)
    ind = topo.mesh.iCompute

    buff1 = Field(domain=dom, name='buff1', is_vector=True)
    buff2 = Field(domain=dom, name='buff2', is_vector=True)

    # Read vector fields, fixed filename, fixed dataset names.
    iop_read = IOParams(working_dir + '/testIO_vec_00001.h5',
                        fileformat=IO.HDF5)
    reader = HDF_Reader(variables={buff1: topo, buff2: topo},
                        io_params=iop_read,
                        var_names={buff1: 'io_2', buff2: 'io_1'})
    reader.discretize()
    reader.setup()
    reader.apply()
    reader.finalize()
    b1 = buff1.discretize(topo)
    b2 = buff2.discretize(topo)
    for d in xrange(v3d.nb_components):
        assert np.allclose(b2.data[d][ind], v3d.data[d][ind])
        assert np.allclose(b1.data[d][ind], w3d.data[d][ind])


@postclean(working_dir)
def test_write_read_subset_1():
    dom, topo = init2()
    velo = Field(domain=dom, formula=vec3D, name='velo', is_vector=True)

    # A subset of the current domain
    from hysop.domain.subsets import SubBox
    mybox = SubBox(origin=[-0.5, 2.3, 4.1], length=[Lx / 2, Lx / 3, Lx],
                   parent=dom)
    # Write a vector field, using default for output location
    # but with fixed names for datasets
    op = HDF_Writer(variables={velo: topo}, var_names={velo: 'io_1'},
                    subset=mybox)
    simu = Simulation(nbIter=3)
    op.discretize()
    op.setup()
    velo.initialize(simu.time, topo=topo)
    simu.initialize()
    while not simu.isOver:
        op.apply(simu)
        simu.advance()
    op.finalize()

    filename = ''
    for v in op.input:
        filename += v.name
        filename += '_'
    filename = filename[:-1]
    fullpath = os.path.join(IO.default_path(), filename)

    assert os.path.exists(fullpath + '.xmf')
    assert os.path.exists(fullpath + '_00000.h5')
    assert os.path.exists(fullpath + '_00001.h5')

    v3d = velo.discretize(topo)
    ind = topo.mesh.iCompute
    indsubset = mybox.mesh[topo].iCompute

    buff1 = Field(domain=dom, name='buff1', is_vector=True)

    # Read vector fields, fixed filename, fixed dataset names.
    iop = IOParams(filename + '_00000.h5', fileformat=IO.HDF5)
    reader = HDF_Reader(variables={buff1: topo},
                        io_params=iop,
                        var_names={buff1: 'io_1'}, subset=mybox)
    reader.discretize()
    reader.setup()
    reader.apply()
    reader.finalize()
    b1 = buff1.discretize(topo)
    for d in xrange(v3d.nb_components):
        assert not np.allclose(b1.data[d][ind], v3d.data[d][ind])
        assert np.allclose(b1.data[d][indsubset], v3d.data[d][indsubset])


@postclean(working_dir)
def test_write_read_subset_2():
    dom, topo = init2()
    velo = Field(domain=dom, formula=vec3D, name='velo', is_vector=True)

    # A subset of the current domain
    # a plane ...
    mybox = SubBox(origin=[-0.5, 2.3, 4.1], length=[Lx / 2, Lx / 3, 0.0],
                   parent=dom)
    # Write a vector field, using default for output location
    # but with fixed names for datasets
    op = HDF_Writer(variables={velo: topo}, var_names={velo: 'io_1'},
                    subset=mybox)
    simu = Simulation(nbIter=3)
    op.discretize()
    op.setup()
    velo.initialize(simu.time, topo=topo)
    simu.initialize()
    while not simu.isOver:
        op.apply(simu)
        simu.advance()
    op.finalize()

    filename = ''
    for v in op.input:
        filename += v.name
        filename += '_'
    filename = filename[:-1]
    fullpath = os.path.join(IO.default_path(), filename)

    assert os.path.exists(fullpath + '.xmf')
    assert os.path.exists(fullpath + '_00000.h5')
    assert os.path.exists(fullpath + '_00001.h5')

    v3d = velo.discretize(topo)
    ind = topo.mesh.iCompute
    indsubset = mybox.mesh[topo].iCompute

    buff1 = Field(domain=dom, name='buff1', is_vector=True)

    # Read vector fields, fixed filename, fixed dataset names.
    iop = IOParams(filename + '_00000.h5', fileformat=IO.HDF5)
    reader = HDF_Reader(variables={buff1: topo},
                        io_params=iop,
                        var_names={buff1: 'io_1'}, subset=mybox)
    reader.discretize()
    reader.setup()
    reader.apply()
    reader.finalize()
    b1 = buff1.discretize(topo)
    for d in xrange(v3d.nb_components):
        assert not np.allclose(b1.data[d][ind], v3d.data[d][ind])
        assert np.allclose(b1.data[d][indsubset], v3d.data[d][indsubset])

