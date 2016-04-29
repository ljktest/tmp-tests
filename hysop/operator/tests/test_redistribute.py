from hysop.operator.redistribute_intra import RedistributeIntra
from hysop.operator.redistribute_inter import RedistributeInter
from hysop.operator.redistribute_overlap import RedistributeOverlap
#from hysop.mpi.main_var import main_size, main_rank, main_comm
from hysop.tools.parameters import Discretization, MPIParams
from hysop import testsenv
import hysop as pp
import numpy as np
import math
from hysop import Simulation
from hysop.operator.analytic import Analytic
from hysop.mpi.main_var import main_comm, main_size
from hysop.mpi.tests.utils import create_inter_topos, CPU, GPU, OTHER,\
    create_multitask_context
from hysop.fields.tests.func_for_tests import v3d, v2d, v3dbis
from hysop.operator.poisson import Poisson
from hysop.mpi.tests.test_bridge import create_subtopos
sin = np.sin
pi = math.pi


dim3 = 3
# Global discretization, no ghosts
r_ng = Discretization([33, ] * dim3)
# Global discretization, with ghosts
r_wg = Discretization([33, ] * dim3, [0, 1, 2])
# Another global discretization, for failed tests.
r_failed = Discretization([27, ] * dim3, [0, 1, 2])
Lx = pi
dom = pp.Box(length=[Lx] * dim3, origin=[0.] * dim3)


def init_3d(domain):
    # Domain and variables
    fields = {}
    fields['velocity'] = pp.Field(domain=domain, name='velocity',
                                  is_vector=True, formula=v3d)
    fields['vorticity'] = pp.Field(domain=domain, name='vorticity',
                                   is_vector=True, formula=v3d)
    fields['scal'] = pp.Field(domain=domain, name='scal')
    simu = Simulation(nbIter=3)
    simu.initialize()
    return fields, simu


def test_distribute_intra_1():
    """
    redistribute data, intra comm, from one
    topology to another
    """
    fields, simu = init_3d(dom)
    velocity = fields['velocity']
    vorticity = fields['vorticity']
    # create the topologies
    # 1D topo, no ghost
    plane_topo = dom.create_topology(discretization=r_ng,
                                     cutdir=(False, False, True))
    # 3D topo, ghosts
    default_topo = dom.create_topology(discretization=r_wg)
    # Redistribute operator
    red = RedistributeIntra(source=plane_topo, target=default_topo,
                            variables=[velocity])
    red.setup()
    # Initialize fields only on plane_topo
    velocity.discretize(topo=plane_topo)
    velocity.initialize(topo=plane_topo)
    # Initialize vorticity on default_topo, used as reference
    wd = vorticity.discretize(topo=default_topo)
    vorticity.initialize(topo=default_topo)
    vnorm = velocity.norm(plane_topo)
    wnorm = vorticity.norm(default_topo)
    assert velocity in red.variables
    assert (vnorm > 0).all()
    assert np.allclose(velocity.norm(default_topo), 0)
    assert np.allclose(wnorm, vnorm)
    red.apply()
    red.wait()
    vd_2 = velocity.discretize(default_topo)
    ind_ng = default_topo.mesh.iCompute
    for d in xrange(velocity.nb_components):
        assert np.allclose(vd_2.data[d][ind_ng], wd.data[d][ind_ng])


@testsenv.hysop_failed
def test_distribute_intra_fail_1():
    """
    redistribute data, intra comm, from one
    topology to another, no variables -> must fail
    """
    # create the topologies
    # 1D topo, no ghost
    plane_topo = dom.create_topology(discretization=r_ng,
                                     cutdir=(False, False, True))
    # 3D topo, ghosts
    default_topo = dom.create_topology(discretization=r_wg)
    # Redistribute operator
    RedistributeIntra(source=plane_topo, target=default_topo)


def test_distribute_intra_2():
    """
    redistribute data, intra comm, from a
    topology to an operator
    """
    fields, simu = init_3d(dom)
    velocity = fields['velocity']
    vorticity = fields['vorticity']
    # create the topologies
    # 1D topo, no ghost
    plane_topo = dom.create_topology(discretization=r_ng,
                                     cutdir=(False, False, True))

    # Create an operator, that will work on a 3D topo.
    op = Analytic(variables={velocity: r_wg})
    op.discretize()
    op.setup()

    # Redistribute operator
    red = RedistributeIntra(source=plane_topo, target=op,
                            variables=[velocity])
    red.setup()
    # Initialize fields only on plane_topo
    velocity.discretize(topo=plane_topo)
    velocity.initialize(topo=plane_topo)

    # Initialize vorticity on default_topo, used as reference
    default_topo = op.variables[velocity]
    vorticity.discretize(topo=default_topo)
    vorticity.initialize(topo=default_topo)
    vnorm = velocity.norm(plane_topo)
    wnorm = vorticity.norm(default_topo)
    assert velocity in red.variables
    assert (vnorm > 0).all()
    assert np.allclose(velocity.norm(default_topo), 0)
    assert np.allclose(wnorm, vnorm)
    red.apply()
    red.wait()
    vd_2 = velocity.discretize(default_topo)
    ind_ng = default_topo.mesh.iCompute
    wd = vorticity.discretize(default_topo)
    for d in xrange(velocity.nb_components):
        assert np.allclose(vd_2.data[d][ind_ng], wd.data[d][ind_ng])


def test_distribute_intra_3():
    """
    redistribute data, intra comm, from an operator
    to a topology.
    """
    fields, simu = init_3d(dom)
    velocity = fields['velocity']
    vorticity = fields['vorticity']

    # Create an operator, that will work on a 3D topo.
    op = Analytic(variables={velocity: r_ng})
    op.discretize()
    op.setup()
    # Initialize velocity on the topo of op
    op.apply(simu)
    source_topo = op.variables[velocity]
    vnorm = velocity.norm(source_topo)
    assert (vnorm > 0).all()

    # create the topologies
    # 1D topo, ghosts
    target_topo = dom.create_topology(discretization=r_wg,
                                      cutdir=(False, False, True))

    # # Redistribute operator
    red = RedistributeIntra(source=op, target=target_topo,
                            variables=[velocity])
    red.setup()
    assert velocity in red.variables
    assert np.allclose(velocity.norm(target_topo), 0)

    # Initialize vorticity on target_topo, used as reference
    wd = vorticity.discretize(topo=target_topo)
    vorticity.initialize(time=simu.time, topo=target_topo)
    wnorm = vorticity.norm(target_topo)

    assert np.allclose(wnorm, vnorm)
    red.apply()
    red.wait()
    vd_2 = velocity.discretize(target_topo)
    ind_ng = target_topo.mesh.iCompute

    for d in xrange(velocity.nb_components):
        assert np.allclose(vd_2.data[d][ind_ng], wd.data[d][ind_ng])


def test_distribute_intra_4():
    """
    redistribute data, intra comm, between two operators.
    """
    fields, simu = init_3d(dom)
    velocity = fields['velocity']
    vorticity = fields['vorticity']

    # Create an operator, that will work on a 3D topo.
    source = Analytic(variables={velocity: r_ng})
    source.discretize()
    source.setup()
    # Initialize velocity on the topo of op
    source.apply(simu)
    source_topo = source.variables[velocity]
    vnorm = velocity.norm(source_topo)
    assert (vnorm > 0).all()

    # create the topologies
    # 1D topo, ghosts
    target_topo = dom.create_topology(discretization=r_wg,
                                      cutdir=(False, False, True))
    # Create an operator from this topo
    target = Analytic(variables={velocity: target_topo}, formula=v3dbis)
    target.discretize()
    target.setup()
    target.apply(simu)
    assert not np.allclose(velocity.norm(target_topo), vnorm)

    # Redistribute operator
    red = RedistributeIntra(source=source, target=target,
                            variables=[velocity])
    red.setup()
    assert velocity in red.variables

    # Initialize vorticity on target_topo, used as reference
    wd = vorticity.discretize(topo=target_topo)
    vorticity.initialize(time=simu.time, topo=target_topo)

    assert np.allclose(vorticity.norm(target_topo), vnorm)
    red.apply()
    red.wait()
    vd_2 = velocity.discretize(target_topo)
    ind_ng = target_topo.mesh.iCompute

    for d in xrange(velocity.nb_components):
        assert np.allclose(vd_2.data[d][ind_ng], wd.data[d][ind_ng])


def test_distribute_intra_5():
    """
    redistribute data, intra comm, between two operators, several variables
    """
    fields, simu = init_3d(dom)
    velocity = fields['velocity']
    vorticity = fields['vorticity']

    # Create an operator, that will work on a 3D topo.
    source = Analytic(variables={velocity: r_ng, vorticity: r_ng})
    source.discretize()
    source.setup()
    # Initialize velocity on the topo of op
    source.apply(simu)
    source_topo = source.variables[velocity]
    vnorm = velocity.norm(source_topo)
    wnorm = vorticity.norm(source_topo)

    assert (vnorm > 0).all() and (wnorm > 0).all()

    # 1D topo, ghosts
    target_topo = dom.create_topology(discretization=r_wg,
                                      cutdir=(False, False, True))
    # Create an operator from this topo
    target = Analytic(variables={velocity: target_topo,
                                 vorticity: target_topo},
                      formula=v3dbis)
    target.discretize()
    target.setup()
    target.apply(simu)
    assert not np.allclose(velocity.norm(target_topo), vnorm)
    assert not np.allclose(vorticity.norm(target_topo), wnorm)

    # Redistribute operator
    red = RedistributeIntra(source=source, target=target)
    red.setup()
    assert velocity in red.variables
    assert vorticity in red.variables

    # Initialize a field of reference
    ref = pp.Field(domain=dom, name='ref', is_vector=True, formula=v3d)
    rd = ref.discretize(topo=target_topo)
    ref.initialize(time=simu.time, topo=target_topo)
    red.apply()
    red.wait()
    vd = velocity.discretize(target_topo)
    wd = vorticity.discretize(target_topo)
    ind_ng = target_topo.mesh.iCompute

    for d in xrange(velocity.nb_components):
        assert np.allclose(vd.data[d][ind_ng], rd.data[d][ind_ng])
        assert np.allclose(wd.data[d][ind_ng], rd.data[d][ind_ng])


@testsenv.hysop_failed
def test_distribute_intra_fail_4():
    """
    redistribute data, intra comm, between two operators.
    """
    fields, simu = init_3d(dom)
    velocity = fields['velocity']

    # Create an operator, that will work on a 3D topo.
    source = Analytic(variables={velocity: r_ng})
    source.discretize()
    source.setup()
    # Initialize velocity on the topo of op
    source.apply(simu)

    # create the topologies
    # 1D topo, ghosts
    target_topo = dom.create_topology(discretization=r_failed,
                                      cutdir=(False, False, True))
    # Create an operator from this topo
    target = Analytic(variables={velocity: target_topo}, formula=v3dbis)
    target.discretize()
    target.setup()
    target.apply(simu)

    # Redistribute operator
    red = RedistributeIntra(source=source, target=target,
                            variables=[velocity])

    red.setup()


@testsenv.hysop_failed
def test_distribute_fail_5():
    """
    Try the pathologic case where source and target do not apply on
    the same group of process but when groups overlap.
    Must failed with standard RedistributeIntra.
    """
    if main_size < 4:
        return
    fields, simu = init_3d(dom)
    velocity = fields['velocity']
    source_topo, target_topo = create_subtopos(dom, r_ng, r_wg)
    # It's important to set mpi_params : main_comm will be used
    # as communicator of reference in red. It works
    # since it handles all the processes of source and all target
    # of target.
    mpi_ref = MPIParams(comm=main_comm)
    red = RedistributeIntra(source=source_topo, target=target_topo,
                            mpi_params=mpi_ref, variables=[velocity])
    red.setup()


def test_distribute_overlap():
    """
    Try the pathologic case where source and target do not apply on
    the same group of process but when groups overlap.
    """
    if main_size < 4:
        return
    fields, simu = init_3d(dom)
    velocity = fields['velocity']
    vorticity = fields['vorticity']
    source_topo, target_topo = create_subtopos(dom, r_ng, r_wg)
    # It's important to set mpi_params : main_comm will be used
    # as communicator of reference in red. It works
    # since it handles all the processes of source and all target
    # of target.
    mpi_ref = MPIParams(comm=main_comm)
    red = RedistributeOverlap(source=source_topo, target=target_topo,
                              mpi_params=mpi_ref, variables=[velocity])
    red.setup()

    if source_topo is not None:
        # Initialize fields only on source_topo
        velocity.discretize(topo=source_topo)
        velocity.initialize(topo=source_topo)
        # Initialize vorticity on default_topo, used as reference
        vnorm = velocity.norm(source_topo)
        assert (vnorm > 0).all()
    if target_topo is not None:
        wd = vorticity.discretize(topo=target_topo)
        vorticity.initialize(topo=target_topo)
        wnorm = vorticity.norm(target_topo)
        assert np.allclose(velocity.norm(target_topo), 0)
        if source_topo is not None:
            assert np.allclose(wnorm, vnorm)
    assert velocity in red.variables

    red.apply()
    red.wait()
    if target_topo is not None:
        vd_2 = velocity.discretize(target_topo)
        ind_ng = target_topo.mesh.iCompute
        for d in xrange(velocity.nb_components):
            assert np.allclose(vd_2.data[d][ind_ng], wd.data[d][ind_ng])


def test_distribute_inter():
    """
    2 tasks, redistribute topo to topo
    """
    if main_size < 4:
        return
    dom_tasks, topo1, topo2 = create_inter_topos(3, r_ng, r_wg)
    fields, simu = init_3d(dom_tasks)
    velocity = fields['velocity']
    # Inititialize velocity on CPU task
    vd = velocity.discretize(topo=topo1)
    if dom_tasks.is_on_task(CPU):
        velocity.initialize(time=simu.time, topo=topo1)

    # A field to compute a reference solution, initialized with an analytic
    # operator, on both tasks.
    reference = fields['vorticity']
    op = Analytic(variables={reference: r_ng})
    op.discretize()
    op.setup()
    op.apply(simu)
    wnorm = reference.norm(topo1)
    vnorm = velocity.norm(topo1)
    if dom_tasks.is_on_task(CPU):
        assert (vnorm > 0).all()
        assert np.allclose(vnorm, wnorm)
    elif dom_tasks.is_on_task(GPU):
        assert (wnorm > 0).all()
        assert np.allclose(vnorm, 0)

    # Redistribute from topo1 on CPU to topo1 on GPU
    red = RedistributeInter(source=topo1, target=topo1, parent=main_comm,
                            variables=[velocity],
                            source_id=CPU, target_id=GPU)
    red.setup()
    red.apply(simu)
    red.wait()
    wd = reference.discretize(topo1)
    if dom.is_on_task(CPU):
        assert (vnorm > 0).all()
        assert np.allclose(vnorm, wnorm)
    elif dom.is_on_task(GPU):
        assert (wnorm > 0).all()
        assert np.allclose(vnorm, wnorm)
        for d in xrange(dom.dimension):
            assert np.allclose(wd.data[d], vd.data[d])
        print wnorm


def test_distribute_inter_2():
    """
    2 tasks, redistribute topo to topo
    """
    if main_size < 4:
        return
    proc_tasks = [CPU, ] * main_size
    if main_size > 2:
        proc_tasks[-1] = GPU
        proc_tasks[0] = GPU
    domtasks = pp.Box(proc_tasks=proc_tasks)
    fields, simu = init_3d(domtasks)
    velocity = fields['velocity']
    # Inititialize velocity on GPU task
    if domtasks.is_on_task(GPU):
        topo_GPU = domtasks.create_topology(r_ng)
        vd = velocity.discretize(topo=topo_GPU)
        velocity.initialize(time=simu.time, topo=topo_GPU)
        vnorm = velocity.norm(topo_GPU)
        assert (vnorm > 0).all()
        topo_CPU = None

    elif domtasks.is_on_task(CPU):
        # A field to compute a reference solution, initialized with an analytic
        # operator, on both tasks.
        reference = fields['vorticity']
        op = Analytic(variables={reference: r_ng})
        op.discretize()
        op.setup()
        op.apply(simu)
        topo_GPU = None
        topo_CPU = op.variables[reference]

    # Redistribute from GPU to CPU
    red = RedistributeInter(source=topo_GPU, target=topo_CPU, parent=main_comm,
                            variables=[velocity],
                            source_id=GPU, target_id=CPU)
    red.setup()
    red.apply(simu)
    red.wait()
    if domtasks.is_on_task(CPU):
        vd = velocity.discretize(topo=topo_CPU)
        wd = reference.discretize(topo=topo_CPU)
        vnorm = velocity.norm(topo_CPU)
        ind = topo_CPU.mesh.iCompute
        wnorm = reference.norm(topo_CPU)
        assert np.allclose(vnorm, wnorm)
        for d in xrange(dom.dimension):
            assert np.allclose(wd.data[d][ind], vd.data[d][ind])


def test_distribute_inter_3():
    """
    2 tasks, redistribute topo to topo
    """
    if main_size < 4:
        return
    dom_tasks, topo1, topo2 = create_inter_topos(3, r_ng, r_wg)
    fields, simu = init_3d(dom_tasks)
    velocity = fields['velocity']
    # Inititialize velocity on GPU task
    if dom_tasks.is_on_task(GPU):
        vd = velocity.discretize(topo=topo1)
        velocity.initialize(time=simu.time, topo=topo1)

    # A field to compute a reference solution, initialized with an analytic
    # operator, on both tasks.
    reference = fields['vorticity']
    op = Analytic(variables={reference: topo2})
    op.discretize()
    op.setup()
    op.apply(simu)
    wnorm = reference.norm(topo2)
    if dom_tasks.is_on_task(GPU):
        vnorm = velocity.norm(topo1)
        assert (vnorm > 0).all()
        assert np.allclose(vnorm, wnorm)
    # Redistribute from topo1 on CPU to topo1 on GPU
    red = RedistributeInter(source=topo1, target=topo2, parent=main_comm,
                            variables=[velocity],
                            source_id=GPU, target_id=CPU)
    red.setup()
    red.apply(simu)
    red.wait()
    if dom_tasks.is_on_task(CPU):
        wd = reference.discretize(topo=topo2)
        vd = velocity.discretize(topo=topo2)
        vnorm = velocity.norm(topo2)
        ind = topo2.mesh.iCompute
        assert np.allclose(vnorm, wnorm)
        for d in xrange(dom.dimension):
            assert np.allclose(wd.data[d][ind], vd.data[d][ind])


def test_distribute_inter_4():
    """
    3 tasks, redistribute topo to topo
    """
    if main_size < 4:
        return
    dom_tasks, topo = create_multitask_context(3, r_ng)
    fields, simu = init_3d(dom_tasks)
    velocity = fields['velocity']
    # Inititialize velocity on GPU task
    if dom_tasks.is_on_task(GPU):
        vd = velocity.discretize(topo=topo)
        velocity.initialize(time=simu.time, topo=topo)

    # A field to compute a reference solution, initialized with an analytic
    # operator, on both tasks.
    reference = fields['vorticity']
    op = Analytic(variables={reference: topo})
    op.discretize()
    op.setup()
    op.apply(simu)

    # Redistribute from topo on CPU to topo on GPU, ignoring OTHER
    if not dom_tasks.is_on_task(OTHER):
        red = RedistributeInter(source=topo, target=topo, parent=main_comm,
                                variables=[velocity],
                                source_id=GPU, target_id=CPU)
        red.setup()
        red.apply(simu)
        red.wait()

    if dom_tasks.is_on_task(CPU):
        wd = reference.discretize(topo=topo)
        vd = velocity.discretize(topo=topo)
        ind = topo.mesh.iCompute
        for d in xrange(dom.dimension):
            assert np.allclose(wd.data[d][ind], vd.data[d][ind])

    if dom_tasks.is_on_task(OTHER):
        assert topo not in velocity.discreteFields


def test_distribute_inter_5():
    """
    2 tasks, redistribute op to op
    """
    if main_size < 4:
        return
    proc_tasks = [CPU, ] * main_size
    proc_tasks[-1] = GPU
    proc_tasks[0] = GPU
    domtasks = pp.Box(proc_tasks=proc_tasks)

    fields, simu = init_3d(domtasks)
    velocity = fields['velocity']
    reference = fields['vorticity']
    if domtasks.is_on_task(CPU):
        # initialize velocity on CPU
        op = Analytic(variables={velocity: r_ng})
        op.discretize()
        op.setup()
        op.apply(simu)
    elif domtasks.is_on_task(GPU):
        # initialize reference on CPU
        op_init = Analytic(variables={reference: r_ng})
        op_init.discretize()
        op_init.setup()
        op_init.apply(simu)
        # An empty operator for velocity
        op = Poisson(output_field=velocity, input_field=reference,
                     discretization=r_ng)
        op.discretize()
        op.setup()

    # Redistribute from CPU to GPU
    red = RedistributeInter(source=op, target=op, parent=main_comm,
                            variables=[velocity],
                            source_id=CPU, target_id=GPU)
    red.setup()
    red.apply(simu)
    red.wait()

    if domtasks.is_on_task(GPU):
        toporef = op.variables[reference]
        vd = velocity.discretize(toporef)
        wd = velocity.discretize(toporef)
        for d in xrange(domtasks.dimension):
            assert np.allclose(wd.data[d], vd.data[d])


def test_distribute_inter_2d():
    """
    2 tasks, redistribute op to op, 2D domain
    """
    if main_size < 4:
        return
    proc_tasks = [CPU, ] * main_size
    proc_tasks[-1] = GPU
    proc_tasks[0] = GPU
    domtasks = pp.Box(dimension=2, proc_tasks=proc_tasks)
    velocity = pp.Field(domain=domtasks, name='velocity',
                        is_vector=True, formula=v2d)
    vort = pp.Field(domain=domtasks, name='vort')
    simu = Simulation(nbIter=3)
    reference = pp.Field(domain=domtasks, name='ref',
                         is_vector=True, formula=v2d)
    r_2d = Discretization([33, ] * 2)
    if domtasks.is_on_task(CPU):
        # initialize velocity on CPU
        op = Analytic(variables={velocity: r_2d})
        op.discretize()
        op.setup()
        op.apply(simu)
    elif domtasks.is_on_task(GPU):
        # initialize reference on CPU
        op_init = Analytic(variables={reference: r_2d})
        op_init.discretize()
        op_init.setup()
        op_init.apply(simu)
        # An empty operator for velocity
        op = Poisson(output_field=velocity, input_field=vort,
                     discretization=r_2d)
        op.discretize()
        op.setup()

    # Redistribute from CPU to GPU
    red = RedistributeInter(source=op, target=op, parent=main_comm,
                            variables=[velocity],
                            source_id=CPU, target_id=GPU)
    red.setup()
    red.apply(simu)
    red.wait()

    if domtasks.is_on_task(GPU):
        toporef = op.variables[velocity]
        vd = velocity.discretize(toporef)
        wd = velocity.discretize(toporef)
        for d in xrange(2):
            assert np.allclose(wd.data[d], vd.data[d])

