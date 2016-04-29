"""Test initialization of fields with analytic formula
"""
from numpy import allclose
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.analytic import Analytic
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization
from hysop.fields.tests.func_for_tests import func_scal_1, func_scal_2, \
    func_vec_1, func_vec_2, func_vec_3, func_vec_4, func_vec_5, func_vec_6
d3D = Discretization([33, 33, 33])
d2D = Discretization([33, 33])
L2D = [1., 1.]
origin2D = [0., 0.]
nbc = 4
simu = Simulation(tinit=0., tend=0.1, nbIter=1)


# Non-Vectorized and vectorized formulas for a scalar
def test_analytical_op_1():
    box = Box()
    caf = Field(box, formula=func_scal_1, name='caf')
    caf2 = Field(box, name='caf2', formula=func_scal_2, vectorize_formula=True)
    op = Analytic(variables={caf: d3D})
    op2 = Analytic(variables={caf2: d3D})
    op.discretize()
    op2.discretize()
    op.setup()
    op2.setup()
    topo = op.discreteFields[caf].topology
    coords = topo.mesh.coords
    ref = Field(box, name='ref')
    refd = ref.discretize(topo)
    cafd = caf.discreteFields[topo]
    cafd2 = caf2.discreteFields[topo]
    ids = id(cafd.data[0])
    ids2 = id(cafd2.data[0])
    op.apply(simu)
    op2.apply(simu)
    refd.data = func_scal_1(refd.data, *(coords + (simu.time,)))
    assert allclose(cafd[0], refd.data[0])
    assert id(cafd.data[0]) == ids
    assert allclose(cafd2[0], refd.data[0])
    assert id(cafd2.data[0]) == ids2


# Non-Vectorized and vectorized formulas for a vector
def test_analytical_op_3():
    box = Box()
    caf = Field(box, name='caf', formula=func_vec_1, is_vector=True)
    caf2 = Field(box, name='caf', formula=func_vec_2,
                 vectorize_formula=True, is_vector=True)
    op = Analytic(variables={caf: d3D})
    op2 = Analytic(variables={caf2: d3D})
    op.discretize()
    op2.discretize()
    op.setup()
    op2.setup()
    topo = op.discreteFields[caf].topology
    coords = topo.mesh.coords
    ref = Field(box, is_vector=True, name='ref')
    refd = ref.discretize(topo)
    cafd = caf.discreteFields[topo]
    cafd2 = caf2.discreteFields[topo]
    ids = [0, ] * 3
    ids2 = [0, ] * 3
    for i in xrange(3):
        ids[i] = id(cafd.data[i])
        ids2[i] = id(cafd2.data[i])
    op.apply(simu)
    op2.apply(simu)
    refd.data = func_vec_1(refd.data, *(coords + (simu.time,)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]
        assert allclose(cafd2[i], refd.data[i])
        assert id(cafd2.data[i]) == ids2[i]


# Non-Vectorized and vectorized formulas for a vector with extra-args
def test_analytical_op_4():
    box = Box()
    caf = Field(box, formula=func_vec_3, is_vector=True, name='caf')
    caf2 = Field(box, formula=func_vec_4, vectorize_formula=True,
                 name='caf2', is_vector=True)
    op = Analytic(variables={caf: d3D})
    op2 = Analytic(variables={caf2: d3D})
    op.discretize()
    op2.discretize()
    op.setup()
    op2.setup()
    topo = op.discreteFields[caf].topology
    coords = topo.mesh.coords
    ref = Field(box, name='ref', is_vector=True)
    refd = ref.discretize(topo)
    cafd = caf.discreteFields[topo]
    cafd2 = caf2.discreteFields[topo]
    ids = [0, ] * 3
    ids2 = [0, ] * 3
    for i in xrange(3):
        ids[i] = id(cafd.data[i])
        ids2[i] = id(cafd2.data[i])
    theta = 3.
    caf.set_formula_parameters(theta)
    caf2.set_formula_parameters(theta)
    op.apply(simu)
    op2.apply(simu)
    refd.data = func_vec_3(refd.data, *(coords + (simu.time, theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]
        assert allclose(cafd2[i], refd.data[i])
        assert id(cafd2.data[i]) == ids2[i]


# Non-Vectorized formula for a nbc components field with extra-args
def test_analytical_op_5():
    box = Box()
    caf = Field(box, formula=func_vec_5, nb_components=nbc, name='caf')
    op = Analytic(variables={caf: d3D})
    op.discretize()
    op.setup()
    topo = op.discreteFields[caf].topology
    coords = topo.mesh.coords
    ref = Field(box, nb_components=nbc, name='ref')
    refd = ref.discretize(topo)
    cafd = caf.discreteFields[topo]
    ids = [0, ] * nbc
    for i in xrange(nbc):
        ids[i] = id(cafd.data[i])
    theta = 3.
    caf.set_formula_parameters(theta)
    op.apply(simu)
    refd.data = func_vec_5(refd.data, *(coords + (simu.time, theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]


# Non-Vectorized formula for a nbc components field in 2D, with extra-args
def test_analytical_op_6():
    box = Box(dimension=2, length=L2D, origin=origin2D)
    caf = Field(box, formula=func_vec_6, nb_components=nbc, name='caf')
    op = Analytic(variables={caf: d2D})
    op.discretize()
    op.setup()
    topo = op.discreteFields[caf].topology
    coords = topo.mesh.coords
    ref = Field(box, nb_components=nbc, name='ref')
    refd = ref.discretize(topo)
    cafd = caf.discreteFields[topo]
    ids = [0, ] * nbc
    for i in xrange(nbc):
        ids[i] = id(cafd.data[i])
    theta = 3.
    caf.set_formula_parameters(theta)
    op.apply(simu)
    refd.data = func_vec_6(refd.data, *(coords + (simu.time, theta)))
    for i in xrange(caf.nb_components):
        assert allclose(cafd[i], refd.data[i])
        assert id(cafd.data[i]) == ids[i]

