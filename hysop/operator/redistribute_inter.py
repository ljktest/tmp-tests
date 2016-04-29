"""
@file redistribute_intercomm.py
Data transfer between two topologies/operators, defined on
different mpi tasks (i.e. intercommunication).
"""
from hysop.constants import debug, S_DIR
from hysop.mpi.bridge_inter import BridgeInter
from hysop.operator.redistribute import Redistribute
from hysop.operator.computational import Computational
from hysop.operator.continuous import opsetup, opapply


class RedistributeInter(Redistribute):
    """
    Operator to redistribute data from one communicator to another.
    Source/target may be either a topology or a computational operator.
    It implies mpi inter-communications.
    """
    @debug
    def __init__(self, parent, source_id=None, target_id=None, **kwds):
        """
        @param parent : parent mpi communicator that must owns all the
        processes involved in source and target.
        @param source_id: mpi task id for the source.
        Required if source is None else infered from source.
        @param target_id:  mpi task id for the target.
        Required if target is None else infered from target.
        See other required parameters in base class.
        """
        super(RedistributeInter, self).__init__(**kwds)

        ## parent communicator, that must contains all processes
        ## involved in source and target tasks.
        self.parent = parent

        # set source and targets ids.
        # They must be known before setup.
        # Either they can be infered from source and target
        # or must be set in argument list, if either source
        # or target is undefined on the current process.
        if self._source is None:
            assert source_id is not None

        if self._target is None:
            assert target_id is not None

        self._source_id = source_id
        self._target_id = target_id

        # Set list of variables and domain.
        self._set_variables()
        # Set mpi related stuff
        self._set_domain_and_tasks()

        # Domain is set, we can check if we are on source or target
        current_task = self.domain.current_task()
        self._is_source = current_task == self._source_id
        self._is_target = current_task == self._target_id
        assert self._is_target or self._is_source
        assert not (self._is_target and self._is_source)

        nbprocs = len(self.domain.tasks_list())
        msg = "Parent communicator size and number of procs "
        msg += "in domain differ."
        assert parent.Get_size() == nbprocs, msg

        # the local topology. May be either source or target
        # depending on the task of the current process.
        self._topology = None

        # dictionnary which maps rank with mpi derived type
        # used for send/recv operations (send on source, recv on target ...)
        self._transfer_types = None

        # dictionnary which maps rank/field name with a
        # send/recv request
        self._requests = {}

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        # First of all, we need to get the current topology:
        if self._is_source:
            assert self._source is not None
            self._topology = self._set_topology(self._source)
        elif self._is_target:
            assert self._target is not None
            self._topology = self._set_topology(self._target)

        # Now we can build the bridge (intercomm)
        self.bridge = BridgeInter(self._topology, self.parent,
                                  self._source_id, self._target_id)

        # And get mpi derived types
        self._transfer_types = self.bridge.transferTypes()

        self._set_synchro()
        self._is_uptodate = True

    def _set_synchro(self):
        """
        Set who must wait for who ...
        """
        if self._is_source and isinstance(self._source, Computational):
            #  redistribute must wait for source if a variable of redistribute
            # is an output from source.
            for v in self.variables:
                vout = v in self._source.output or False
            if vout:
                self.wait_for(self._source)
                # And source must wait for redistribute
                # if a variable of red. is an output from source.
                self._source.wait_for(self)

        if self._is_target and isinstance(self._target, Computational):
            # target operator must wait for
            # the end of this operator to apply.
            self._run_till.append(self._target)

        # Add this operator into wait list of
        # operators listed in run_till
        for op in self._run_till:
            op.wait_for(self)

    def add_run_till_op(self, op):
        """Add an operator to the wait list"""
        if self._is_target:
            self._run_till.append(op)
            op.wait_for(self)

    @debug
    @opapply
    def apply(self, simulation=None):
        """
        Apply this operator to its variables.
        @param simulation : object that describes the simulation
        parameters (time, time step, iteration number ...), see
        hysop.problem.simulation.Simulation for details.
        """
        # --- Standard send/recv ---
        self._requests = {}

        # basetag = self._mpis.rank + 1
        # Comm used for send/receive operations
        # It must contains all proc. of source topo and
        # target topo.
        refcomm = self.bridge.comm
        # Map between rank and mpi types
        # Loop over all required components of each variable
        for v in self.variables:
            rank = self._topology.comm.Get_rank()
            for d in self._range_components(v):
                v_name = v.name + S_DIR[d]
                vtab = v.discreteFields[self._topology].data[d]
                for rk in self._transfer_types:
                    if self._is_target:
                        # Set reception
                        self._requests[v_name + str(rk)] = \
                            refcomm.Irecv([vtab[...], 1,
                                           self._transfer_types[rk]],
                                          source=rk, tag=rk)
                    if self._is_source:
                        self._requests[v_name + str(rk)] = \
                            refcomm.Issend([vtab[...], 1,
                                            self._transfer_types[rk]],
                                           dest=rk, tag=rank)
                    self._has_requests = True

    def wait(self):
        if self._has_requests:
            for rk in self._requests:
                self._requests[rk].Wait()
        for v in self.variables:
            for d in self._range_components(v):
                vtab = v.discreteFields[self._topology].data[d]
        self._has_requests = False

    def test_requests(self):
        res = True
        for rk in self._requests:
            res = self._requests[rk].Test()
            if not res:
                return res
