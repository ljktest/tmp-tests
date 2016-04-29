"""
@file redistribute_overlap.py
Setup for data transfer/redistribution between two topologies defined on the
same mpi parent communicator but with a different number of processes.
"""
from hysop.operator.continuous import opsetup
from hysop.operator.redistribute_intra import RedistributeIntra
from hysop.mpi.bridge_overlap import BridgeOverlap


class RedistributeOverlap(RedistributeIntra):
    """
    A specific redistribute where source and target do not work with the same
    group of mpi processes.
    Requirements :
    - work only on topologies, not on operators
    - same global resolution for both topologies
    - group from source topology and target topology MUST overlap.
    """

    @opsetup
    def setup(self, rwork=None, iwork=None):
        """
        Check/set the list of variables to be distributed

        What must be set at setup?
        ---> the list of continuous variables to be distributed
        ---> the bridge (one for all variables, which means
        that all vars must have the same topology in source
        and the same topology in target.
        ---> the list of discrete variables for source and
        for target.
        """
        if self._source is not None:
            self._vsource = self._discrete_fields(self._source)
        if self._target is not None:
            self._vtarget = self._discrete_fields(self._target)

        # We can create the bridge
        self.bridge = BridgeOverlap(source=self._source, target=self._target,
                                    comm_ref=self._mpis.comm)

        # Build mpi derived types for send and receive operations.
        # Shape of reference is the shape of source/target mesh
        if self._source is not None:
            self._send = self.bridge.sendTypes()
        if self._target is not None:
            self._receive = self.bridge.recvTypes()

        self._set_synchro()
        self._is_uptodate = True

    def _discrete_fields(self, topo):
        """
        @param topo : a Cartesian HySoP topology
        Return the dictionnary of discrete fields for topo
        and the variables of this operator.
        """
        from hysop.mpi.topology import Cartesian
        assert isinstance(topo, Cartesian)
        return {v: v.discretize(topo) for v in self.variables}
