"""Abstract interface for physical domains description.

"""
from abc import ABCMeta, abstractmethod
from hysop.constants import debug, DEFAULT_TASK_ID, PERIODIC
from hysop.mpi.topology import Cartesian
from hysop.mpi import main_rank, main_size, main_comm
from hysop.tools.parameters import MPIParams
import numpy as np


class Domain(object):
    """Abstract base class for description of physical domain. """

    __metaclass__ = ABCMeta

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @debug
    @abstractmethod
    def __init__(self, dimension=3, proc_tasks=None, bc=None):
        """

        Parameters
        ----------
        dimension : integer, optional
        proc_tasks : list or array of int, optional
            connectivity between mpi process id and task number.
            See notes below. Default : all procs on task DEFAULT_TASK_ID.
        bc : list or array of int, optional
            type of boundary condtions. See notes below.


        Attributes
        ----------

        dimension : int
        topologies : dictionnary of :class:`~hysop.mpi.topology.Cartesian`
            all topologies already built for this domain.
        comm_task : mpi communicator
            corresponds to the task that owns the current process.
        boundaries : list or array of int
            boundary conditions type.

        Notes
        -----
        About boundary conditions
        ==========================
        At the time, only periodic boundary conditions are implemented.
        Do not use this parameter (i.e. let default values)

        About MPI Tasks
        ================
        proc_tasks[n] = 12 means that task 12 owns proc n.

        """
        # Domain dimension.
        self.dimension = dimension
        # A list of all the topologies defined on this domain.
        # Each topology is unique in the sense defined by
        # the comparison operator in the class hysop.mpi.topology.Cartesian.
        self.topologies = {}

        # Connectivity between mpi tasks and proc numbers :
        # self.task_of_proc[i] returns the task which is bound to proc of
        # rank i in main_comm.
        # Warning FP: rank in main_comm! We consider that each proc has
        # one and only one task.
        # Maybe we should change this and allow proc_tasks in subcomm
        # but at the time it's not necessary.
        if proc_tasks is None:
            self._tasks_on_proc = [DEFAULT_TASK_ID, ] * main_size
            comm_s = main_comm
        else:
            assert len(proc_tasks) == main_size
            self._tasks_on_proc = proc_tasks
            # Split main comm according to the defined tasks.
            comm_s = main_comm.Split(color=proc_tasks[main_rank],
                                     key=main_rank)

        # The sub-communicator corresponding to the task that owns
        # the current process.
        self.comm_task = comm_s
        # Boundary conditions type.
        # At the moment, only periodic conditions are allowed.
        self.boundaries = bc

    def is_on_task(self, params):
        """Test if the current process corresponds to param task.

        :param params: :class:`~hysop.mpi.MPIParams` or int
            description of the mpi setup or task number.
        or an int (task number)
        :return bool:

        Example :
        :code:
        dom.is_on_task(4)
        mpi_params = MPIParams(task_id=4)
        dom.is_on_task(mpi_params)
        :code:

        returns True if the current process runs on task 4.
        """
        if params.__class__ is MPIParams:
            return params.task_id == self._tasks_on_proc[main_rank]
        else:
            return params == self._tasks_on_proc[main_rank]

    def tasks_on_proc(self, index):
        """Get task number for a given mpi process

        :param index: int
            proc number
        :return int:
        """
        return self._tasks_on_proc[index]

    def tasks_list(self):
        """Get task/process number connectivity

        :return list or array of int:
            such that res[proc_number] = task_id
        """
        return self._tasks_on_proc

    def current_task(self):
        """Get task number of the current mpi process.

        :return int:
        """
        return self._tasks_on_proc[main_rank]

    def create_topology(self, discretization, dim=None, mpi_params=None,
                        shape=None, cutdir=None):
        """Create or return an existing :class:`~hysop.mpi.topology.Cartesian`.

        Parameters
        -----------
        discretization : :class:`~hysop.tools.parameters.Discretization`
            description of the global space discretization
            (resolution and ghosts).
        dim : int, optional
            mpi topology dimension.
        mpi_params : :class:`~hysop.tools.parameters.MPIParams`, optional
            mpi setup (comm, task ...).
            If None, comm = main_comm, task = DEFAULT_TASK_ID.
        shape : list or array of int
            mpi grid layout
        cutdir : list or array of bool
            set which directions must (may) be distributed,
            cutdir[dir] = True to distribute data along dir.

        Returns
        -------
        :class:`~hysop.mpi.topology.Cartesian`
            Either it gets the topology corresponding to the input arguments
            if it exists (in the sense of the comparison operator defined in
            :class:`~hysop.mpi.topology.Cartesian`)
            or it creates a new topology and register it in the topology dict.

        """
        # set task number
        tid = self.current_task()
        if mpi_params is None:
            mpi_params = MPIParams(comm=self.comm_task, task_id=tid)
        else:
            msg = 'Try to create a topology on a process that does not'
            msg += ' belong to the current task.'
            assert mpi_params.task_id == tid, msg
        new_topo = Cartesian(self, discretization, dim=dim,
                             mpi_params=mpi_params, shape=shape,
                             cutdir=cutdir)
        newid = new_topo.get_id()
        return self.topologies[newid]

    def create_plane_topology_from_mesh(self, localres, global_start,
                                        cdir=None, **kwds):
        """Create 1D topology from predifined :class:`~hysop.domain.mesh.Mesh`.

        Define a 'plane' (1D) topology for a given mesh resolution.
        This function is to be used when topo/discretization features
        come from an external routine (e.g. scales or fftw)

        Parameters
        ----------
        localres : list or array of int
            local mesh resolution
        global_start : list or array of int
            indices in the global mesh of the lowest point of the local mesh.
        cdir : int, optional
            direction where data must be distributed in priority.
            Default = last if fortran order, first if C order.
        **kwds : other args
            used in :class:`~hysop.mpi.topology.Cartesian` build.

        """
        tid = self.current_task()
        if 'mpi_params' not in kwds:
            kwds['mpi_params'] = MPIParams(comm=self.comm_task, task_id=tid)
        else:
            msg = 'Try to create a topology on a process that does not'
            msg += ' belong to the current task.'
            assert kwds['mpi_params'].task_id == tid, msg
        new_topo = Cartesian.plane_precomputed(localres, global_start, cdir,
                                               domain=self, **kwds)

        newid = new_topo.get_id()
        return self.topologies[newid]

    def _check_topo(self, new_topo):
        """Returns the id of the input topology
        if it exists in the domain list, else return -1.
        """
        otherid = -1
        for top in self.topologies.values():
            if new_topo == top:
                otherid = top.get_id()
                break
        return otherid

    def register(self, new_topo):
        """Add a new topology in the list of this domain.
        Do nothing if an equivalent topology is already
        in the list.

        :Returns the id of the new registered topology
        or of the corresponding 'old one' if it already exists.
        """
        otherid = self._check_topo(new_topo)
        if otherid < 0:
            self.topologies[new_topo.get_id()] = new_topo
            new_topo.isNew = True
            newid = new_topo.get_id()
        else:
            # No registration
            new_topo.isNew = False
            newid = otherid
        return newid

    def remove(self, topo):
        """Remove a topology from the list of this domain.
        Do nothing if the topology does not exist in the list.
        Warning : the object topo is not destroyed, only
        removed from self.topologies.

        Returns either the id of the removed topology
        or -1 if nothing is done.
        """
        otherid = self._check_topo(topo)
        if otherid >= 0:
            self.topologies.pop(otherid)
        return otherid

    def print_topologies(self):
        """
        Print all topologies of the domain.
        """
        if main_rank == 0:
            for topo in self.topologies.values():
                print topo

    @abstractmethod
    def __eq__(self, other):
        """
        Comparison of two domains
        """

    def __ne__(self, other):
        """
        Not equal operator
        """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def i_periodic_boundaries(self):
        """
        Returns the list of directions where
        boundaries are periodic
        """
        return np.where(self.boundaries == PERIODIC)[0]
