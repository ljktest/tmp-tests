"""Test fields build and usage (discrete and continuous)
"""
from hysop import Simulation, Box, Field, Discretization, IOParams, IO
import numpy as np
import os
from hysop.fields.tests.func_for_tests import func_scal_1, func_scal_2, \
    func_vec_1, func_vec_2, func_vec_3, func_vec_4, func_vec_5, func_vec_6,\
    v_time_3d, v_time_2d
from numpy import allclose
import shutil
from hysop.mpi import main_rank


d3d = Discretization([33, 33, 33])
d2d = Discretization([33, 33])
nbc = 4


def formula(x, y, z):
    return (x * y * z, 1.0, -x * y * z)


def form2(x, y, z, dt):
    return (x * y * z, 1.0, -x * y * z * dt)


def test_continuous():
    """ Basic continuous field construction """

    dom = Box()
    cf = Field(dom, name='f1')
    assert cf.nb_components == 1
    assert cf.domain is dom
    cf = Field(dom, name='f2', is_vector=True)
    assert cf.nb_components == dom.dimension
    assert cf.domain is dom


def test_analytical():
    """Test formula"""
    dom = Box()
    caf = Field(dom, name='f1', formula=formula, vectorize_formula=True)
    assert caf.value(0., 0., 0.) == (0., 1., 0.)
    assert caf.value(1., 1., 1.) == (1., 1., -1.)
    assert caf.value(1., 2., 3.) == (6., 1., -6.)


def test_analytical_reset():
    """Test formula"""
    dom = Box()
    caf = Field(dom, name='f1', formula=formula, vectorize_formula=True)
    assert caf.value(0., 0., 0.) == (0., 1., 0.)
    assert caf.value(1., 1., 1.) == (1., 1., -1.)
    assert caf.value(1., 2., 3.) == (6., 1., -6.)
    caf.set_formula(form2, vectorize_formula=True)
    assert caf.value(0., 0., 0., 0.) == (0., 1., 0.)
    assert caf.value(1., 1., 1., 1.) == (1., 1., -1.)
    assert caf.value(1., 2., 3., 4.) == (6., 1., -24.)


def test_discretization():
    """Test multiple discretizations"""
    dom = Box()
    csf = Field(dom, name='f1')
    cvf = Field(dom, name='f2', is_vector=True)
    r3d = Discretization([33, 33, 17])
    topo = dom.create_topology(r3d)
    csf.discretize(topo)
    cvf.discretize(topo)
    assert np.equal(csf.discreteFields[topo].resolution,
                    csf.discreteFields[topo].data[0].shape).all()
    assert np.equal(cvf.discreteFields[topo].resolution,
                    cvf.discreteFields[topo].data[0].shape).all()
    assert np.equal(cvf.discreteFields[topo].resolution,
                    cvf.discreteFields[topo].data[1].shape).all()
    assert np.equal(cvf.discreteFields[topo].resolution,
                    cvf.discreteFields[topo].data[2].shape).all()


# Non-Vectorized formula for a scalar
def test_analytical_field_1():
    box = Box()
    topo = box.create_topology(discretization=d3d)
    coords = topo.mesh.coords
    caf = Field(box, name='f1', formula=func_scal_1)
    ref = Field(box, name='f2')
    refd = ref.discretize(topo)
    cafd = caf.discretize(topo)
    refd = ref.discretize(topo)
    ids = id(cafd.data[0])
    caf.initialize()
    refd.data = func_scal_1(refd.data, *(coords + (0.,)))
    assert allclose(cafd[0], refd.data[0])
    assert id(cafd.data[0]) == ids
    time = 3.0
    caf.initialize(time=time)
    refd.data = func_scal_1(refd.data, *(coords + (time,)))
    assert allclose(cafd[0], refd.data[0])
    assert id(cafd.data[0]) == ids


# Vectorized formula
def test_analytical_field_2():
    box = Box()
    topo = box.create_topology(d3d)
    coords = topo.mesh.coords
    caf = Field(box, name='f1', formula=func_scal_2, vectorize_formula=True)
    ref = Field(box, name='f2')
    cafd = caf.discretize(topo)
    ids = id(cafd.data[0])
    refd = ref.discretize(topo)
    caf.initialize()
    refd.data = func_scal_1(refd.data, *(coords + (0.,)))
    assert allclose(cafd[0], refd.data[0])
    assert id(cafd.data[0]) == ids
    time = 3.0
    caf.initialize(time=time)
    refd.data = func_scal_1(refd.data, *(coords + (time,)))
    assert allclose(cafd[0], refd.data[0])
    assert id(cafd.data[0]) == ids


# Non-Vectorized formula for a vector
def test_analytical_field_3():
    box = Box()
    topo = box.create_topology(d3d)
    coords = topo.mesh.coords
    caf = Field(box, name='f1', formula=func_vec_1, is_vector=True)
    ref = Field(box, name='f2', is_vector=True)
    refd = ref.discretize(topo)
    cafd = caf.discretize(topo)
    refd = ref.discretize(topo)
    ids = [0, ] * 3
    for i in xrange(3):
        ids[i] = id(cafd.data[i])
    caf.initialize()
    refd.data = func_vec_1(refd.data, *(coords + (0.,)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]
    time = 3.0
    caf.initialize(time=time)
    refd.data = func_vec_1(refd.data, *(coords + (time,)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]


# Vectorized formula for a vector
def test_analytical_field_4():
    box = Box()
    topo = box.create_topology(d3d)
    coords = topo.mesh.coords
    caf = Field(box, name='f1', formula=func_vec_2, is_vector=True,
                vectorize_formula=True)
    ref = Field(box, name='f2', is_vector=True)
    refd = ref.discretize(topo)
    cafd = caf.discretize(topo)
    refd = ref.discretize(topo)
    ids = [0, ] * 3
    for i in xrange(3):
        ids[i] = id(cafd.data[i])
    caf.initialize()
    refd.data = func_vec_1(refd.data, *(coords + (0.,)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]
    time = 3.0
    caf.initialize(time=time)
    refd.data = func_vec_1(refd.data, *(coords + (time,)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]


# Non-Vectorized formula for a vector, with extra-arguments
def test_analytical_field_5():
    box = Box()
    topo = box.create_topology(d3d)
    coords = topo.mesh.coords
    caf = Field(box, name='f1', formula=func_vec_3, is_vector=True)
    theta = 0.3
    caf.set_formula_parameters(theta)
    ref = Field(box, name='f2', is_vector=True)
    refd = ref.discretize(topo)
    cafd = caf.discretize(topo)
    refd = ref.discretize(topo)
    ids = [0, ] * 3
    for i in xrange(3):
        ids[i] = id(cafd.data[i])
    caf.initialize()
    refd.data = func_vec_3(refd.data, *(coords + (0., theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]
    time = 3.0
    caf.initialize(time=time)
    refd.data = func_vec_3(refd.data, *(coords + (time, theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]


# Vectorized formula for a vector, with extra-arguments
def test_analytical_field_6():
    box = Box()
    topo = box.create_topology(d3d)
    coords = topo.mesh.coords
    caf = Field(box, name='f1', formula=func_vec_4, is_vector=True,
                vectorize_formula=True)
    theta = 0.3
    caf.set_formula_parameters(theta)
    ref = Field(box, name='f2', is_vector=True)
    refd = ref.discretize(topo)
    cafd = caf.discretize(topo)
    refd = ref.discretize(topo)
    ids = [0, ] * 3
    for i in xrange(3):
        ids[i] = id(cafd.data[i])
    caf.initialize()
    refd.data = func_vec_3(refd.data, *(coords + (0., theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]
    time = 3.0
    caf.initialize(time=time)
    refd.data = func_vec_3(refd.data, *(coords + (time, theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]


# Non-Vectorized formula for a field with nb_components
# different from domain dim and  with extra-arguments
def test_analytical_field_7():
    box = Box()
    topo = box.create_topology(d3d)
    coords = topo.mesh.coords
    caf = Field(box, name='f1', formula=func_vec_5, nb_components=nbc)
    theta = 0.3
    caf.set_formula_parameters(theta)
    ref = Field(box, name='f2', nb_components=nbc)
    refd = ref.discretize(topo)
    cafd = caf.discretize(topo)
    refd = ref.discretize(topo)
    ids = [0, ] * nbc
    for i in xrange(nbc):
        ids[i] = id(cafd.data[i])

    caf.initialize()
    refd.data = func_vec_5(refd.data, *(coords + (0., theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]
    time = 3.0
    caf.initialize(time=time)
    refd.data = func_vec_5(refd.data, *(coords + (time, theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]


# Non-Vectorized formula for a 2D field with nb_components
# different from domain dim and  with extra-arguments
def test_analytical_field_8():
    box = Box(dimension=2, length=[1., 1.], origin=[0., 0.])
    topo = box.create_topology(d2d)
    coords = topo.mesh.coords
    caf = Field(box, name='f1', formula=func_vec_6, nb_components=nbc)
    theta = 0.3
    caf.set_formula_parameters(theta)
    ref = Field(box, name='f2', nb_components=nbc)
    refd = ref.discretize(topo)
    cafd = caf.discretize(topo)
    refd = ref.discretize(topo)
    ids = [0, ] * nbc
    for i in xrange(nbc):
        ids[i] = id(cafd.data[i])
    caf.initialize()
    refd.data = func_vec_6(refd.data, *(coords + (0., theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]
    time = 3.0
    caf.initialize(time=time)
    refd.data = func_vec_6(refd.data, *(coords + (time, theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]


# Non-Vectorized formula for a vector, initialization on several
# topologies.
def test_analytical_field_9():
    box = Box()
    topo = box.create_topology(d3d)
    res2 = Discretization([65, 33, 65], [1, 1, 1])
    topo2 = box.create_topology(res2, dim=2)
    coords = topo.mesh.coords
    coords2 = topo2.mesh.coords
    caf = Field(box, name='f1', formula=func_vec_1, is_vector=True)
    ref = Field(box, name='f2', is_vector=True)
    refd = ref.discretize(topo)
    cafd = caf.discretize(topo)
    cafd2 = caf.discretize(topo2)
    refd2 = ref.discretize(topo2)
    ids = [0, ] * 3
    for i in xrange(3):
        ids[i] = id(cafd2.data[i])
        # init on topo2
    caf.initialize(topo=topo2)
    refd2.data = func_vec_1(refd2.data, *(coords2 + (0.,)))
    refd.data = func_vec_1(refd.data, *(coords + (0.,)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd2[i], refd2.data[i])
        assert id(cafd2.data[i]) == ids[i]
        assert not allclose(cafd[i], refd.data[i])
    caf.initialize(topo=topo)
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])


# Non-Vectorized formula for a vector, initialization on several
# topologies.
def test_analytical_field_10():
    box = Box()
    topo = box.create_topology(d3d)
    res2 = Discretization([65, 33, 65], [1, 1, 1])
    topo2 = box.create_topology(res2, dim=2)
    coords = topo.mesh.coords
    coords2 = topo2.mesh.coords
    caf = Field(box, name='f1', formula=func_vec_1, is_vector=True)
    ref = Field(box, name='f2', is_vector=True)
    refd = ref.discretize(topo)
    cafd = caf.discretize(topo)
    cafd2 = caf.discretize(topo2)
    refd2 = ref.discretize(topo2)
    ids = [0, ] * 3
    for i in xrange(3):
        ids[i] = id(cafd2.data[i])
    # init on all topos
    caf.initialize()
    refd2.data = func_vec_1(refd2.data, *(coords2 + (0.,)))
    refd.data = func_vec_1(refd.data, *(coords + (0.,)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd2[i], refd2.data[i])
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd2.data[i]) == ids[i]


def test_copy():
    box = Box()
    topo = box.create_topology(d3d)
    f1 = Field(box, name='source', is_vector=True, formula=func_vec_1)
    f2 = Field(box, name='target', is_vector=True)
    f1d = f1.discretize(topo)
    f1.initialize(topo=topo)
    f2.copy(f1, topo)
    f2d = f2.discretize(topo)
    for d in xrange(f2.nb_components):
        assert np.allclose(f2d[d], f1d[d])
        assert f2d[d].flags.f_contiguous == f1d[d].flags.f_contiguous


def test_randomize():
    box = Box()
    topo = box.create_topology(d3d)
    f2 = Field(box, name='target', is_vector=True)
    f2.randomize(topo)
    f2d = f2.discretize(topo)
    for d in xrange(f2.nb_components):
        assert not np.allclose(f2d[d], 0.)


def hdf_dump_load(discretisation, formula):
    dimension = len(discretisation.resolution)
    box = Box([1., ] * dimension)
    topo = box.create_topology(discretisation)
    fname = 'f1_' + str(dimension)
    ff = Field(box, name=fname, is_vector=True, formula=formula)
    iop = IOParams(ff.name)
    simu = Simulation(nbIter=4)
    simu.initialize()
    ff.initialize(time=simu.time, topo=topo)
    ff.hdf_dump(topo, simu)
    simu.advance()
    ff.initialize(time=simu.time, topo=topo)
    # Write ff for current time and topo
    ff.hdf_dump(topo, simu)
    assert iop.filepath == IO.default_path()
    assert os.path.isfile(iop.filename + '_00000.h5')
    assert os.path.isfile(iop.filename + '_00001.h5')
    assert os.path.isfile(iop.filename + '.xmf')
    fname2 = 'f2_' + str(dimension)
    gg = Field(box, name=fname2, is_vector=True)
    iop_in = IOParams(iop.filename + '_00001.h5')
    dsname = ff.name + '_' + str(topo.get_id())
    # Load gg from ff values at current time, on topo
    for i in xrange(gg.nb_components):
        assert not np.allclose(gg.discretize(topo)[i], ff.discretize(topo)[i])
    gg.hdf_load(topo, iop_in, dataset_name=dsname)
    for i in xrange(gg.nb_components):
        assert np.allclose(gg.discretize(topo)[i], ff.discretize(topo)[i])

    # reset ff with its values at time 'O', on topo
    ff.hdf_load(topo, restart=0)
    for i in xrange(gg.nb_components):
        assert not np.allclose(gg.discretize(topo)[i], ff.discretize(topo)[i])


def test_hdf_dump_load_2d():
    hdf_dump_load(d2d, v_time_2d)


def test_hdf_dump_load_3d():
    hdf_dump_load(d3d, v_time_3d)

