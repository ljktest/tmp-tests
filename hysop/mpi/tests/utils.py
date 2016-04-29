"""
Functions used through mpi-related tests.
"""

from hysop.mpi.main_var import main_comm, main_rank, main_size
from hysop.tools.parameters import MPIParams
import hysop as pp
GPU = 4
CPU = 1
OTHER = 12


def create_multitask_context(dim, discr):
    # MPI procs are distributed among three tasks
    # Two tasks:
    # proc 0 and last proc work on 'GPU' task.
    # The others on 'CPU' task.
    proc_tasks = [CPU, ] * main_size
    if main_size > 2:
        proc_tasks[-1] = GPU
        proc_tasks[0] = GPU
        proc_tasks[1] = OTHER
    topodim = max(dim, 2)
    dom = pp.Box(dimension=dim, proc_tasks=proc_tasks)
    # Create a topology, which represents a different context
    # on each task.
    topo = dom.create_topology(discr, dim=topodim)

    return dom, topo


def create_subtopos(domain, discr_source, discr_target):
    # split main comm into two groups
    rk_source = [i for i in xrange(main_size)]
    rk_target = list(rk_source)
    rk_target.pop(-1)
    rk_target.pop(0)
    g_source = main_comm.group.Incl(rk_source)
    g_target = main_comm.group.Incl(rk_target)
    # Create the sub-communicators and the related topologies
    comm_source = main_comm.Create(g_source)
    mpi_source = MPIParams(comm=comm_source)
    if main_rank in rk_source:
        source_topo = domain.create_topology(discretization=discr_source,
                                             mpi_params=mpi_source)
    else:
        source_topo = None

    comm_target = main_comm.Create(g_target)
    mpi_target = MPIParams(comm=comm_target)
    if main_rank in rk_target:
        target_topo = domain.create_topology(discretization=discr_target,
                                             mpi_params=mpi_target)
    else:
        target_topo = None
    return source_topo, target_topo


def create_nonoverlap_topos(domain, discr_source, discr_target):
    # split main comm into two groups
    rk_source = [i for i in xrange(main_size) if i % 2 == 0]
    rk_target = [i for i in xrange(main_size) if i % 2 != 0]
    g_source = main_comm.group.Incl(rk_source)
    g_target = main_comm.group.Incl(rk_target)
    # Create the sub-communicators and the related topologies
    comm_source = main_comm.Create(g_source)
    mpi_source = MPIParams(comm=comm_source)
    if main_rank in rk_source:
        source_topo = domain.create_topology(discretization=discr_source,
                                             mpi_params=mpi_source)
    else:
        source_topo = None

    comm_target = main_comm.Create(g_target)
    mpi_target = MPIParams(comm=comm_target)
    if main_rank in rk_target:
        target_topo = domain.create_topology(discretization=discr_target,
                                             mpi_params=mpi_target)
    else:
        target_topo = None
    return source_topo, target_topo


def create_inter_topos(dim, discr1, discr2):
    # MPI procs are distributed among two tasks

    # proc 0 and last proc work on 'GPU' task.
    # The others on 'CPU' task.
    proc_tasks = [CPU, ] * main_size
    if main_size > 2:
        proc_tasks[-1] = GPU
        proc_tasks[0] = GPU
    topodim = max(dim, 2)
    dom = pp.Box(dimension=dim, proc_tasks=proc_tasks)
    topo1 = dom.create_topology(discr1, dim=topodim)
    topo2 = dom.create_topology(discr2, dim=topodim - 1)

    return dom, topo1, topo2
