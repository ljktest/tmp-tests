"""
@file bridge.py
Tools to compute the intersection between
two HySoP topologies.
"""
from hysop.mpi.topology import Cartesian, TopoTools
from hysop.tools.misc import utils


class Bridge(object):
    """
    todo
    """

    def __init__(self, source, target):
        """
        @param source : topology that owns the source mesh
        @param target : topology of the targeted mesh
        """
        # -- All dictionnaries belows used rank number (in parent comm)
        # as keys. --
        # Dictionnary of indices of grid points to be received on target.
        self._recv_indices = {}
        # Dictionnary of indices of grid points to be sent from sourceId
        self._send_indices = {}
        # Dictionnary of MPI derived types used for MPI receive
        self._recv_types = None
        # Dictionnary of MPI derived types used for MPI send.
        self._send_types = None

        # The communicator that will be used in this bridge.
        self.comm = None
        # current rank in this comm
        self._rank = None

        self._source = source
        self._target = target
        self._check_topologies()
        # nothing to be done ...
        if source == target:
            return

        self._build_send_recv_dict()

    def _check_topologies(self):
        # First check if source and target are complient
        msg = 'Bridge error, one or both topologies are None.'
        msg = 'Bridge error, input source/target must be topologies.'
        assert isinstance(self._source, Cartesian), msg
        assert isinstance(self._target, Cartesian), msg

        msg = 'Bridge error, both source/target topologies'
        msg += ' must have the same parent communicator.'
        assert TopoTools.compare_comm(self._source.parent(),
                                      self._target.parent()), msg
        # The assert above ensure that source and target hold the same
        # group of process in the same communication context.
        self.comm = self._source.parent()
        self._rank = self.comm.Get_rank()

    def _build_send_recv_dict(self):
        # Compute local intersections : i.e. find which grid points
        # are on both source and target mesh.

        # Get global indices of the mesh on source for all mpi processes.
        indices_source = TopoTools.gather_global_indices(self._source)

        # Get global indices of the mesh on target for all mpi processes.
        indices_target = TopoTools.gather_global_indices(self._target)
        # From now on, we have indices_source[rk] = global indices (slice)
        # of grid points of the source on process number rk in parent.
        # And the same thing for indices_target.

        # Compute the intersections of the mesh on source with every mesh on
        # target ---> find which part of the local mesh must be sent to who,
        # which results in the self._send_indices dict.
        # self._send_indices[i] = [slice(...), slice(...), slice(...)]
        # means that the current process must send to process i the grid points
        # defined by the slices above.
        current = indices_source[self._rank]
        for rk in indices_target:
            inter = utils.intersl(current, indices_target[rk])
            if inter is not None:
                self._send_indices[rk] = inter
        # Back to local indices
        convert = self._source.mesh.convert2local
        self._send_indices = {rk: convert(self._send_indices[rk])
                              for rk in self._send_indices}

        # Compute the intersections of the mesh on target with every mesh on
        # source ---> find which part of the local mesh must recv something
        # and from who,
        # which results in the self._recv_indices dict.
        # self._recv_indices[i] = [slice(...), slice(...), slice(...)]
        # means that the current process must recv from process i
        # the grid points defined by the slices above.
        current = indices_target[self._rank]
        for rk in indices_source:
            inter = utils.intersl(current, indices_source[rk])
            if inter is not None:
                self._recv_indices[rk] = inter

        convert = self._target.mesh.convert2local
        self._recv_indices = {rk: convert(self._recv_indices[rk])
                              for rk in self._recv_indices}

    def hasLocalInter(self):
        return self._rank in self._send_indices

    def localSourceInd(self):
        if self._rank in self._send_indices:
            return self._send_indices[self._rank]
        else:
            return {}

    def localTargetInd(self):
        if self._rank in self._recv_indices:
            return self._recv_indices[self._rank]
        else:
            return {}

    def recvTypes(self):
        """
        Return the dictionnary of MPI derived types
        received on targeted topology.
        @param data_shape : shape (numpy-like) of the original array
        @return : a dict of MPI types
        """
        if self._recv_types is None:
            data_shape = self._target.mesh.resolution
            self._recv_types = TopoTools.create_subarray(self._recv_indices,
                                                         data_shape)
        return self._recv_types

    def sendTypes(self):
        """
        Return the dictionnary of MPI derived types
        send from source topology.
        @param data_shape : shape (numpy-like) of the original array
        @return : a dict of MPI types
        """
        if self._send_types is None:
            data_shape = self._source.mesh.resolution
            self._send_types = TopoTools.create_subarray(self._send_indices,
                                                         data_shape)
        return self._send_types
