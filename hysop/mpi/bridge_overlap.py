"""
@file bridge.py
Tools to compute the intersection between
two HySoP topologies defined on the same comm but for a
different number of processes.
"""
from hysop.mpi.topology import Cartesian, TopoTools
from hysop.tools.misc import utils
from hysop.mpi.bridge import Bridge


class BridgeOverlap(Bridge):
    """
    Bridge between two topologies that:
    - have a different number of mpi processes
    - have common mpi processes

    i.e. something in between a standard bridge with intra-comm and
    a bridge dealing with intercommunication. This is probably
    a very pathologic case ...

    The main difference with a standard bridge is that
    this one may be call on processes where either source
    or target does not exist.
    """
    def __init__(self, comm_ref=None, **kwds):
        """
        @param comm_ref : mpi communicator used for all global communications.
        It must include all processes of source and target.
        If None, source.parent() is used.
        """
        self.comm = comm_ref
        self.domain = None
        super(BridgeOverlap, self).__init__(**kwds)

    def _check_topologies(self):
        # First check if source and target are complient
        if self.comm is None:
            if self._source is not None:
                self.comm = self._source.parent()
            else:
                self.comm = self._target.parent()

        # To build a bridge, all process in source/target must be in self.comm
        # and there must be an overlap between source
        # and target processes group. If not, turn to intercommunicator.
        if self._source is not None and self._target is not None:
            msg = 'BridgeOverlap error: mpi group from '
            msg += 'source and topo must overlap. If not '
            msg += 'BridgeInter will probably suits better.'
            assert TopoTools.intersection_size(self._source.comm,
                                               self._target.comm) > 0, msg
            assert self._source.domain == self._target.domain

        if self._source is not None:
            assert isinstance(self._source, Cartesian)
            s_size = self._source.size
            assert TopoTools.intersection_size(self._source.comm,
                                               self.comm) == s_size
            self.domain = self._source.domain

        if self._target is not None:
            assert isinstance(self._target, Cartesian)
            t_size = self._target.size
            assert TopoTools.intersection_size(self._target.comm,
                                               self.comm) == t_size
            self.domain = self._target.domain

        self._rank = self.comm.Get_rank()

    def _build_send_recv_dict(self):
        # Compute local intersections : i.e. find which grid points
        # are on both source and target mesh.
        indices_source = TopoTools.gather_global_indices_overlap(self._source,
                                                                 self.comm,
                                                                 self.domain)
        indices_target = TopoTools.gather_global_indices_overlap(self._target,
                                                                 self.comm,
                                                                 self.domain)

        # From now on, we have indices_source[rk] = global indices (slice)
        # of grid points of the source on process number rk in parent.
        # And the same thing for indices_target.
        dimension = self.domain.dimension
        # Compute the intersections of the mesh on source with every mesh on
        # target (i.e. for each mpi process).
        if self._rank in indices_source:
            current = indices_source[self._rank]
        else:
            current = [slice(None, None, None), ] * dimension

        for rk in indices_target:
            inter = utils.intersl(current, indices_target[rk])
            if inter is not None:
                self._send_indices[rk] = inter

        if self._source is not None:
            # Back to local indices
            convert = self._source.mesh.convert2local
            self._send_indices = {rk: convert(self._send_indices[rk])
                                  for rk in self._send_indices}

        if self._rank in indices_source:
            current = indices_target[self._rank]
        else:
            current = [slice(None, None, None), ] * dimension
        for rk in indices_source:
            inter = utils.intersl(current, indices_source[rk])
            if inter is not None:
                self._recv_indices[rk] = inter

        if self._target is not None:
            convert = self._target.mesh.convert2local
            self._recv_indices = {rk: convert(self._recv_indices[rk])
                                  for rk in self._recv_indices}
