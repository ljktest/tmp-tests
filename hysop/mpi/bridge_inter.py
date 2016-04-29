"""
@file bridge.py
Tools to compute the intersection between
two HySoP topologies.
"""
from hysop.mpi.topology import Cartesian, TopoTools
from hysop.tools.misc import utils
from hysop.mpi import MPI
import hysop.tools.numpywrappers as npw


class BridgeInter(object):
    """
    todo
    """

    def __init__(self, current, parent, source_id, target_id):
        """
        @param source : topology that owns the source mesh
        @param target : topology of the targeted mesh
        """

        # The aim of a bridge if to compute the intersection of mesh grids
        # on source topology with those on target topology, to be able to tell
        # who must send/recv what to which process.
        # This is done in steps:
        # - the indices of grid points of each process are gathered
        # onto the root process, for both source and target --> global_indices.
        # We compute global indices (i.e. relative to the global grid)
        # - an intercommunicator is used to broadcast these indices
        # from source to the processes of target.

        ## source task number
        self.source_id = source_id

        ## target task number
        self.target_id = target_id

        assert isinstance(current, Cartesian)
        assert isinstance(parent, MPI.Intracomm)
        self._topology = current
        # current task id
        current_task = self._topology.domain.current_task()

        # True if current process is in the 'from' group'
        task_is_source = current_task == self.source_id

        # True if current process is in the 'to' group
        task_is_target = current_task == self.target_id

        # Ensure that current process belongs to one and only one task.
        assert task_is_source or task_is_target
        assert not(task_is_source and task_is_target)

        # Create an intercommunicator
        # Create_intercomm attributes are:
        #   - local rank of leader process for current group (always 0)
        #   - parent communicator
        #   - rank of leader process in the remote group

        # rank of the first proc belonging to the remote task
        # (used as remote leader)
        proc_tasks = self._topology.domain.tasks_list()
        if task_is_source:
            remote_leader = proc_tasks.index(self.target_id)
        elif task_is_target:
            remote_leader = proc_tasks.index(self.source_id)
        self.comm = self._topology.comm.Create_intercomm(0, parent,
                                                         remote_leader)

        current_indices, remote_indices = self._swap_indices()

        self._tranfer_indices = {}
        current = current_indices[self._topology.rank]
        for rk in remote_indices:
            inter = utils.intersl(current, remote_indices[rk])
            if inter is not None:
                self._tranfer_indices[rk] = inter

        # Back to local indices
        convert = self._topology.mesh.convert2local
        self._tranfer_indices = {rk: convert(self._tranfer_indices[rk])
                                 for rk in self._tranfer_indices}

        self._transfer_types = None

    def _swap_indices(self):
        # First, we need to collect the global indices, as arrays
        # since we need to broadcast them later.
        current_indices = TopoTools.gather_global_indices(self._topology,
                                                          toslice=False)
        # To allocate remote_indices array, we need the size of
        # the remote communicator.
        remote_size = self.comm.Get_remote_size()
        dimension = self._topology.domain.dimension
        remote_indices = npw.dim_zeros((dimension * 2, remote_size))
        # Then they are broadcasted to the remote communicator
        rank = self._topology.rank
        current_task = self._topology.domain.current_task()
        if current_task is self.source_id:
            # Local 0 broadcast current_indices to remote comm
            if rank == 0:
                self.comm.bcast(current_indices, root=MPI.ROOT)
            else:
                self.comm.bcast(current_indices, root=MPI.PROC_NULL)
            # Get remote indices from remote comm
            remote_indices = self.comm.bcast(remote_indices, root=0)

        elif current_task is self.target_id:
            # Get remote indices from remote comm
            remote_indices = self.comm.bcast(remote_indices, root=0)
            # Local 0 broadcast current_indices to remote comm
            if rank == 0:
                self.comm.bcast(current_indices, root=MPI.ROOT)
            else:
                self.comm.bcast(current_indices, root=MPI.PROC_NULL)

        # Convert numpy arrays to dict of slices ...
        current_indices = utils.arrayToDict(current_indices)
        remote_indices = utils.arrayToDict(remote_indices)

        return current_indices, remote_indices

    def transferTypes(self):
        """
        Return the dictionnary of MPI derived types
        used for send (if on source) or receive (if on target)
        @param data_shape : shape (numpy-like) of the original array
        @return : a dict of MPI types
        """
        if self._transfer_types is None:
            data_shape = self._topology.mesh.resolution
            self._transfer_types = TopoTools.create_subarray(
                self._tranfer_indices, data_shape)
        return self._transfer_types
