"""Tools and definitions for HySoP topologies
(MPI Processes layout + space discretization)

* :class:`~hysop.mpi.topology.Cartesian`
* :class:`~hysop.mpi.topology.TopoTools`

"""

from hysop.constants import debug, ORDER, PERIODIC
from hysop.domain.mesh import Mesh
from itertools import count
from hysop.mpi.main_var import MPI
from hysop.tools.parameters import Discretization, MPIParams
import numpy as np
import hysop.tools.numpywrappers as npw
from hysop.tools.misc import utils


class Cartesian(object):
    """
    In hysop, a topology is defined as the association of
    a mpi process distribution (mpi topology) and of a set of local meshes
    (one per process).

    At the time, only cartesian topologies with cartesian meshes are available.

    Example :
    \code
    >>> from hysop.mpi.topology import Cartesian
    >>> from hysop.tools.parameters import Discretization
    >>> from hysop.domain.box import Box
    >>> dom = Box()
    >>> r = Discretization([33, 33, 33])
    >>> topo = Cartesian(dom, dim=2, discretization=r)
    >>>
    \endcode
    For details about topologies see HySoP User Manual.

    You can also find examples of topologies instanciation in test_topology.py.

    """

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)
#
    # Counter of topology.Cartesian instances to set a unique id for each
    # Cartesian topology instance.
    __ids = count(0)

    @debug
    def __init__(self, domain, discretization, dim=None, mpi_params=None,
                 isperiodic=None, cutdir=None, shape=None, mesh=None):
        """

        Parameters
        -----------
        domain : :class:`~hysop.domain.domain.Domain`
           the geometry on which the topology is defined.
        discretization : :class:`~hysop.tools.parameters.Discretization`
            description of the global space discretization
            (resolution and ghosts).
        dim : int, optional
            mpi topology dimension.
        mpi_params : :class:`~hysop.tools.parameters.MPIParams`, optional
            mpi setup (comm, task ...).
            If None, comm = main_comm, task = DEFAULT_TASK_ID.
        isperiodic : list or array of bool, optional
            mpi grid periodicity
        cutdir : list or array of bool
            set which directions must (may) be distributed,
            cutdir[dir] = True to distribute data along dir.
        shape : list or array of int
            mpi grid layout
        mesh : :class:`~hysop.domain.mesh.Mesh`, optional
            a predifined mesh (it includes local and global grids.)

        Notes:
        ------
        Almost all parameters above are optional.
        Only one must be chosen among dim, cutdir and shape.
        See :ref:`topologies` in the Hysop User Guide for details.

        See hysop.mpi.topology.Cartesian.plane_precomputed
        details to build a plane topology from a given local discretization
        (e.g. from fftw or scales precomputation).
        """
        # ===== 1 - Required parameters : domain and mpi (comm, task) ====
        # An id for the topology
        self.__id = self.__ids.next()

        ## Associated domain
        self.domain = domain
        # - MPI topo params :
        if mpi_params is None:
            mpi_params = MPIParams(comm=domain.comm_task,
                                   task_id=domain.current_task())

        # mpi parameters : Communicator used to build the topology
        # and task_id
        self._mpis = mpi_params
        # Each topo must be associated to one and only one topology
        assert self._mpis.task_id is not None
        # ===== 2 - Prepare MPI processes layout ====

        # 3 methods :
        # (a) - from 'shape' parameter : we choose explicitely the layout.
        # (b) - from 'cutdir' parameter: we choose the directions
        # to split and let MPI fix the number of processes in each dir.
        # (c) - from dimension of the topology ==> let MPI
        # choose the 'best' layout.

        # Layout of the grid of mpi processes.
        self.shape = None
        # Dim of the cartesian grid for MPI processes.
        self.dimension = None
        # MPI grid periodicity
        self._isperiodic = None
        # directions where data are distributed
        self.cutdir = None
        # mpi communicator (cartesian topo)
        self.comm = None

        self._build_mpi_topo(shape, cutdir, dim, isperiodic)

        # ===== 3 - Get features of the mpi processes grid =====
        # Size of the topology (i.e. total number of mpi processes)
        self.size = self.comm.Get_size()
        # Rank of the current process in the topology
        self.rank = self.comm.Get_rank()
        # Coordinates of the current process
        reduced_coords = npw.asdimarray(self.comm.Get_coords(self.rank))

        # Coordinates of the current process
        # What is different between proc_coords and reduced_coords?
        # --> proc_coords has values even for directions that
        # are not distributed. If cutdir = [False, False, True]
        # then reduced_coords = [ nx ] and proc_coords = [0, 0, nx]
        self.proc_coords = npw.dim_zeros(self.domain.dimension)
        self.proc_coords[self.cutdir] = reduced_coords
        # Neighbours : self.neighbours[0,i] (resp. [1,i])
        # previous (resp. next) neighbour in direction i
        # (warning : direction in the grid of process).
        self.neighbours = npw.dim_zeros((2, self.dimension))
        for direction in range(self.dimension):
            self.neighbours[:, direction] = self.comm.Shift(direction, 1)

        # ===== 4 - Computation of the local meshes =====

        # Local mesh on the current mpi process.
        # mesh from external function (e.g. fftw, scales ...)
        self.mesh = mesh
        # If mesh is None, we must compute local resolution and other
        # parameters, using discretization.
        if mesh is None:
            self._compute_mesh(discretization)

        # ===== 5 - Final setup ====

        # The topology is register into domain list.
        # If it already exists (in the sense of the comparison operator
        # of the present class) then isNew is turned to false
        # and the present instance will be destroyed.
        self.isNew = True
        self.__id = self.domain.register(self)
        # If a similar topology (in the sense of operator
        # equal defined below) exists, we link
        # its arg with those of the current topo.
        # It seems to be impossible to delete the present
        # object in its __init__ function.
        if not self.isNew:
            topo = self.domain.topologies[self.__id]
            self.mesh = topo.mesh
            self.comm = topo.comm

    def _build_mpi_topo(self, shape=None, cutdir=None,
                        dim=None, isperiodic=None):
        """ default builder
        See :ref:`topologies` for details.

        """
        # number of process in parent comm
        origin_size = self._mpis.comm.Get_size()

        if shape is not None:
            # method (a)
            msg = ' parameter is useless when shape is provided.'
            assert cutdir is None, 'cutdir ' + msg
            assert dim is None, 'dim ' + msg
            self.shape = npw.asdimarray(shape)
            msg = 'Input shape must be of '
            msg += 'the same size as the domain dimension.'
            assert self.shape.size == self.domain.dimension, msg
            self.cutdir = self.shape != 1

        elif cutdir is not None:
            # method (b)
            msg = ' parameter is useless when cutdir is provided.'
            assert shape is None, 'shape ' + msg
            assert dim is None, 'dim ' + msg
            self.cutdir = npw.asboolarray(cutdir)
            self.dimension = self.cutdir[self.cutdir].size
            shape = npw.asdimarray(MPI.Compute_dims(origin_size,
                                                    self.dimension))
            self._optimizeshape(shape)
            self.shape = npw.dim_ones(self.domain.dimension)
            self.shape[self.cutdir] = shape

        else:
            if dim is not None:
                # method (a)
                msg = ' parameter is useless when dim is provided.'
                assert shape is None, 'shape ' + msg
                assert cutdir is None, 'cutdir ' + msg
                self.dimension = dim
            else:
                # dim, shape and cutdir are None ...
                # ==> default behavior is let MPI compute
                # the best layout for a topology of the
                # same dim as the domain
                self.dimension = self.domain.dimension

            # Since shape is not provided, computation of the
            # "optimal" processes distribution for each direction
            # of the grid topology.
            shape = npw.asdimarray(MPI.Compute_dims(origin_size,
                                                    self.dimension))
            self.shape = npw.dim_ones(self.domain.dimension)
            # Reorder shape according to the data layout
            # if arrays are in "fortran" order (column major)
            # the last dir has priority for distribution.
            # For C-like (row major) arrays, first dir is the
            # first to be distributed, and so on.
            self.shape[:self.dimension] = shape
            self._optimizeshape(self.shape)
            self.cutdir = self.shape != 1

        # MPI processes grid periodicity. Default is true.
        if isperiodic is None:
            self._isperiodic = npw.ones((self.domain.dimension),
                                        dtype=npw.bool)
        else:
            self._isperiodic = npw.asboolarray(isperiodic)
            assert isperiodic.size == self.domain.dimension

        # compute real dim ...
        self.dimension = self.shape[self.cutdir].size

        # Special care for the 1 process case:
        if origin_size == 1:
            self.dimension = 1
            self.cutdir[-1] = True

        # From this point, the following parameters must be properly set:
        # - self.shape
        # - self.cutdir
        # - self._isperiodic
        # Then, we can create the mpi topology.
        self.comm = \
            self._mpis.comm.Create_cart(self.shape[self.cutdir],
                                        periods=self._isperiodic[self.cutdir],
                                        reorder=True)

    @staticmethod
    def _optimizeshape(shape):
        """Reorder 'shape' according to the chosen
        data layout to optimize data distribution.
        """
        shape.sort()
        if ORDER == 'C':
            shape[:] = shape[::-1]

    def parent(self):
        """returns the communicator used to build this topology
        """
        return self._mpis.comm

    def ghosts(self):
        """returns ghost layer width.
        """
        return self.mesh.discretization.ghosts

    def task_id(self):
        """returns id of the task that owns this topology
        """
        return self._mpis.task_id

    @classmethod
    def plane_precomputed(cls, localres, global_start, cdir=None, **kwds):
        """Defines a 'plane' (1D) topology for a given mesh resolution.

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
        """
        msg = 'parameter is not required for plane_precomputed'
        msg += ' topology construction.'
        assert 'dim' not in kwds, 'dim ' + msg
        assert 'shape ' not in kwds, 'shape ' + msg
        assert 'cutdir ' not in kwds, 'cutdir ' + msg
        # Local mesh :
        global_start = npw.asdimarray(global_start)
        localres = npw.asdimarray(localres)
        mesh = Mesh(kwds['domain'], kwds['discretization'],
                    localres, global_start)
        # MPI layout
        domain = kwds['domain']
        cutdir = npw.zeros(domain.dimension, dtype=npw.bool)

        if cdir is not None:
            cutdir[cdir] = True
        else:
            if ORDER == 'C':
                cutdir[0] = True
            else:
                cutdir[-1] = True

        return cls(mesh=mesh, cutdir=cutdir, **kwds)

    def _compute_mesh(self, discretization):
        assert isinstance(discretization, Discretization)
        assert discretization.resolution.size == self.domain.dimension
        assert self.domain.dimension == discretization.resolution.size, \
            'The resolution size differs from the domain dimension.'
        # Number of "computed" points (i.e. excluding ghosts/boundaries).
        pts_noghost = npw.dim_zeros((self.domain.dimension))
        # Warning FP : we should test boundary conditions type here
        # If periodic, resol_calc = (global_resolution - 1)
        # else, resol_calc =  (global_resolution - 2)

        is_periodic = len(np.where(self.domain.boundaries == PERIODIC)[0])\
            == self.domain.dimension
        if is_periodic:
            computational_grid_resolution = discretization.resolution - 1
        else:
            raise AttributeError('Unknwon boundary conditions.')

        pts_noghost[:] = computational_grid_resolution // self.shape

        # If any, remaining points are
        # added on the mesh of the last process.
        remaining_points = npw.dim_zeros(self.domain.dimension)
        remaining_points[:] = computational_grid_resolution % self.shape

        # Total number of points (size of arrays to be allocated)
        nbpoints = pts_noghost.copy()
        for i in range(self.domain.dimension):
            if self.proc_coords[i] == self.shape[i] - 1:
                nbpoints[i] += remaining_points[i]

        local_resolution = computational_grid_resolution.copy()
        local_resolution[:] = nbpoints[:]
        local_resolution[:] += 2 * discretization.ghosts[:]

        # Global indices for the local mesh points
        global_start = npw.dim_zeros((self.domain.dimension))
        global_start[:] = self.proc_coords[:] * pts_noghost[:]

        self.mesh = Mesh(self.domain, discretization,
                         local_resolution, global_start)

    def __eq__(self, other):
        """Comparison of two topologies.

        Two topos are equal if they have the same mesh, shape and domain.
        """
        if self.__class__ != other.__class__:
            return False
        return self.mesh == other.mesh and \
            npw.equal(self.shape, other.shape).all() and \
            self.domain == other.domain

    def __ne__(self, other):
        """Not equal operator.

        Seems to be required in addition to __eq__ to
        avoid 'corner-case' behaviors.
        """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __str__(self):
        s = '======== Topology id : ' + str(self.__id) + ' ========\n'
        s += ' - on task : ' + str(self.task_id()) + '\n'
        s += ' - shape :' + str(self.shape) + '\n'
        s += ' - process of coordinates ' + str(self.proc_coords[:])
        s += ' and of ranks (topo/origin) ' + str(self.rank) + '/'
        s += str(self._mpis.rank) + '.\n'
        s += '- neighbours coordinates : \n'
        s += str(self.neighbours) + '\n'
        s += str(self.mesh)
        s += '\n=================================\n'
        return s

    def has_ghosts(self):
        """
        True if ghost layer length is not zero.
        """
        return not np.all(self.mesh.discretization.ghosts == 0)

    def get_id(self):
        """
        return the id of the present topology.
        This id is unique among all defined topologies.
        """
        return self.__id

    def is_consistent_with(self, target):
        """True if target and current object are equal and
        have the same parent. Equal means same mesh, same shape and
        same domain.
        """
        same_parent = self.parent() == target.parent()
        # Note FP. Is it really required to have the
        # same parent? Inclusion of all proc may be enough?
        return npw.equal(self.shape, target.shape).all() and same_parent

    def can_communicate_with(self, target):
        """True if current topo is complient with target.

        Complient means :
        - all processes in current are in target
        - both topologies belong to the same mpi task
        """
        if self == target:
            return True
        msg = 'You try to connect topologies belonging to'
        msg += ' two different mpi tasks. Set taskids properly or use'
        msg += ' InterBridge.'
        assert self.task_id() == target.task_id(), msg

        # Parent communicator
        # Todo : define some proper conditions for compatibility
        # between topo_from, topo_to and parent:
        # - same size
        # - same domain
        # - common processus ...
        # At the time we check that both topo have
        # the same comm_origin.
        return self.is_consistent_with(target)

    @staticmethod
    def reset_counter():
        Cartesian.__ids = count(0)


class TopoTools(object):

    @staticmethod
    def gather_global_indices(topo, toslice=True, root=None, comm=None):
        """Collect global indices of local meshes on each process of topo

        Parameters
        ----------
        topo : :class:`~ hysop.mpi.topology.Cartesian`
            topology on which indices are collected.
        toslice : boolean, optional
            true to return the result a dict of slices, else
            return a numpy array. See notes below.
        root : int, optional
            rank of the root mpi process. If None, reduce operation
            is done on all processes.
        comm : mpi communicator, optional
            Communicator used to reduce indices. Default = topo.parent

        Returns
        -------
        either :
        * a dictionnary which maps rank number with
        a list of slices such that res[rank][i] = a slice
        defining the indices of the points of the local mesh,
        in direction i, in global notation.
        * or a numpy array where each column corresponds to a rank number,
        with column = [start_x, end_x, start_y, end_y ...]
        Ranks number are the processes numbers in comm.

        """
        if comm is None:
            comm = topo.parent()
        size = comm.size
        start = topo.mesh.start()
        end = topo.mesh.stop() - 1
        # communicator that owns the topology
        rank = comm.Get_rank()
        dimension = topo.domain.dimension
        iglob = npw.int_zeros((dimension * 2, size))
        iglob_res = npw.int_zeros((dimension * 2, size))
        iglob[0::2, rank] = start
        iglob[1::2, rank] = end
        # iglob is saved as a numpy array and then transform into
        # a dict of slices since mpi send operations are much
        # more efficient with numpy arrays.
        if root is None:
            comm.Allgather([iglob[:, rank], MPI.INT], [iglob_res, MPI.INT])
        else:
            comm.Gather([iglob[:, rank], MPI.INT], [iglob_res, MPI.INT],
                        root=root)

        if toslice:
            return utils.arrayToDict(iglob_res)
        else:
            return iglob_res

    @staticmethod
    def gather_global_indices_overlap(topo=None, comm=None, dom=None,
                                      toslice=True, root=None):
        """This functions does the same thing as gather_global_indices but
        may also work when topo is None.

        The function is usefull if you need to collect global indices
        on a topo define only on a subset of comm,
        when for the procs not in this subset, topo will be
        equal to None. In such a case, comm and dom are required.
        This may happen when you want to build a bridge between two topologies
        that do not handle the same number of processes but with an overlap
        between the two groups of processes of the topologies.

        In that case, a call to
        gather_global_indices(topo, comm, dom)
        will work on all processes belonging to comm, topo being None or not.
        The values corresponding to ranks not in topo will be empty slices.

        Parameters
        ----------
        topo : :class:`~ hysop.mpi.topology.Cartesian`, optional
            topology on which indices are collected.
        toslice : boolean, optional
            true to return the result a dict of slices, else
            return a numpy array. See notes below.
        root : int, optional
            rank of the root mpi process. If None, reduce operation
            is done on all processes.
        comm : mpi communicator, optional
            Communicator used to reduce indices. Default = topo.parent
        dom : :class:`~hysop.domain.domain.Domain`
            current domain.

        Returns
        -------
        either :
        * a dictionnary which maps rank number with
        a list of slices such that res[rank][i] = a slice
        defining the indices of the points of the local mesh,
        in direction i, in global notation.
        * or a numpy array where each column corresponds to a rank number,
        with column = [start_x, end_x, start_y, end_y ...]
        Ranks number are the processes numbers in comm.

        """
        if topo is None:
            assert comm is not None and dom is not None
            size = comm.Get_size()
            rank = comm.Get_rank()
            dimension = dom.dimension
            iglob = npw.int_zeros((dimension * 2, size))
            iglob_res = npw.int_zeros((dimension * 2, size))
            iglob[1::2, rank] = -1
            if root is None:
                comm.Allgather([iglob[:, rank], MPI.INT], [iglob_res, MPI.INT])
            else:
                comm.Gather([iglob[:, rank], MPI.INT], [iglob_res, MPI.INT],
                            root=root)
            if toslice:
                return utils.arrayToDict(iglob_res)
            else:
                return iglob_res

        else:
            return TopoTools.gather_global_indices(topo, toslice, root, comm)

    @staticmethod
    def is_parent(child, parent):
        """
        Return true if all mpi processes of child belong to parent
        """
        # Get the list of processes
        assert child is not None
        assert parent is not None
        #child_ranks = [i for i in xrange(child.Get_size())]
        child_group = child.Get_group()
        parent_group = parent.Get_group()
        inter_group = MPI.Group.Intersect(child_group, parent_group)
        return child_group.Get_size() == inter_group.Get_size()

    @staticmethod
    def intersection_size(comm_1, comm_2):
        """Number of processess common to comm_1 and comm_2
        """
        if comm_1 == MPI.COMM_NULL or comm_2 == MPI.COMM_NULL:
            return None
        group_1 = comm_1.Get_group()
        group_2 = comm_2.Get_group()
        inter_group = MPI.Group.Intersect(group_1, group_2)
        return inter_group.Get_size()

    @staticmethod
    def compare_comm(comm_1, comm_2):
        """Compare two mpi communicators.

        Returns true if the two communicators are handles for the same
        group of proc and for the same communication context.

        Warning : if comm_1 or comm_2 is invalid, the
        function will fail.
        """
        assert comm_1 != MPI.COMM_NULL
        assert comm_2 != MPI.COMM_NULL
        result = MPI.Comm.Compare(comm_1, comm_2)
        res = [MPI.IDENT, MPI.CONGRUENT, MPI.SIMILAR, MPI.UNEQUAL]
        return result == res[0]

    @staticmethod
    def compare_groups(comm_1, comm_2):
        """Compare the groups of two mpi communicators.

        Returns true if each comm handles the
        same group of mpi processes.

        Warning : if comm_1 or comm_2 is invalid, the
        function will fail.
        """
        assert comm_1 != MPI.COMM_NULL
        assert comm_2 != MPI.COMM_NULL
        result = MPI.Comm.Compare(comm_1, comm_2)
        res = [MPI.IDENT, MPI.CONGRUENT, MPI.SIMILAR, MPI.UNEQUAL]
        return result in res[:-1]

    @staticmethod
    def convert_ranks(source, target):
        """Find the values of ranks in target from ranks in source.

        Parameters
        ----------
        source, target : mpi communicators

        Returns a list 'ranks' such that ranks[i] = rank in target
        of process of rank i in source.
        """
        assert source != MPI.COMM_NULL and target != MPI.COMM_NULL
        g_source = source.Get_group()
        g_target = target.Get_group()
        size_source = g_source.Get_size()
        r_source = [i for i in xrange(size_source)]
        res = MPI.Group.Translate_ranks(g_source, r_source, g_target)
        return {r_source[i]: res[i] for i in xrange(size_source)}

    @staticmethod
    def create_subarray(sl_dict, data_shape):
        """Create a MPI subarray mask to be used in send/recv operations
        between some topologies.

        Parameters
        ----------
        sl_dict : dictionnary
            indices of the subarray for each rank,
            such that sl_dict[rk] = (slice(...), slice(...), ...)
        data_shape : shape (numpy-like) of the original array

        :Returns : dictionnary of MPI derived types.
        Keys = ranks in parent communicator.
        """
        from hysop.constants import HYSOP_MPI_REAL, ORDERMPI
        subtypes = {}
        dim = len(data_shape)
        for rk in sl_dict.keys():
            subvshape = tuple((sl_dict[rk][i].stop -
                               sl_dict[rk][i].start for i in xrange(dim)))
            substart = tuple((sl_dict[rk][i].start for i in xrange(dim)))
            subtypes[rk] = \
                HYSOP_MPI_REAL.Create_subarray(data_shape, subvshape,
                                               substart, order=ORDERMPI)
            subtypes[rk].Commit()

        return subtypes
