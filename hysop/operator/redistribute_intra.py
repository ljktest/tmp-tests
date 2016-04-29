"""
@file redistribute_intra.py
Setup for data transfer/redistribution between two topologies
or operators  inside the same mpi communicator.
"""
from hysop.operator.redistribute import Redistribute
from hysop.operator.continuous import opsetup, opapply
from hysop.mpi.bridge import Bridge
from hysop.constants import S_DIR


class RedistributeIntra(Redistribute):
    """
    Data transfer between two operators/topologies.
    Source and target must:
    - be defined on the same communicator
    - work on the same number of mpi process
    - work with the same global resolution
    """
    def __init__(self, **kwds):
        # Base class initialisation
        super(RedistributeIntra, self).__init__(**kwds)

        # Warning : comm from io_params will be used as
        # reference for all mpi communication of this operator.
        # --> rank computed in refcomm
        # --> source and target must work inside refcomm
        # If io_params is None, refcomm will COMM_WORLD.

        # Dictionnary of discrete fields to be sent
        self._vsource = {}
        # Dictionnary of discrete fields to be overwritten
        self._vtarget = {}

        # dictionnary which maps rank with mpi derived type
        # for send operations
        self._send = {}
        # dictionnay which maps rank with mpi derived type
        # for send operations
        self._receive = {}
        # dictionnary which map rank/field name with a
        # receive request
        self._r_request = None
        # dictionnary which map rank/field name with a
        # send request
        self._s_request = None

        # Set list of variables and the domain.
        self._set_variables()
        # Set mpi related stuff
        self._set_domain_and_tasks()

    @opsetup
    def setup(self, rwork=None, iwork=None):
        # At setup, source and topo must be either
        # a hysop.mpi.topology.Cartesian or
        # a computational operator.

        msg = 'Redistribute error : undefined source of target.'
        assert self._source is not None and self._target is not None, msg

        t_source = self._set_topology(self._source)
        t_target = self._set_topology(self._target)

        source_res = t_source.mesh.discretization.resolution
        target_res = t_target.mesh.discretization.resolution
        msg = 'Redistribute error: source and target must '
        msg += 'have the same global resolution.'
        assert (source_res == target_res).all(), msg

        # Set the dictionnaries of source/target variables
        self._vsource = {v: v.discretize(t_source)
                         for v in self.variables}
        self._vtarget = {v: v.discretize(t_target)
                         for v in self.variables}

        # We can create the bridge
        self.bridge = Bridge(t_source, t_target)

        # Shape of reference is the shape of source/target mesh
        self._send = self.bridge.sendTypes()
        self._receive = self.bridge.recvTypes()
        self._set_synchro()
        self._is_uptodate = True

    def _set_synchro(self):
        """
        Set who must wait for who ...
        """
        from hysop.operator.computational import Computational
        # Check input operators
        if isinstance(self._source, Computational):
            #  redistribute must wait for source if a variable of redistribute
            # is an output from source.
            for v in self.variables:
                vout = v in self._source.output or False
            if vout:
                self.wait_for(self._source)
                # And source must wait for redistribute
                # if a variable of red. is an output from source.
                self._source.wait_for(self)

        if isinstance(self._target, Computational):
            # target operator must wait for
            # the end of this operator to apply.
            self._run_till.append(self._target)

        # Add this operator into wait list of
        # operators listed in run_till
        for op in self._run_till:
            op.wait_for(self)

        self._is_uptodate = True

    def add_run_till_op(self, op):
        """Add an operator to the wait list"""
        self._run_till.append(op)
        op.wait_for(self)

    @opapply
    def apply(self, simulation=None):
        # Try different way to send vars?
        # - Buffered : copy all data into a buffer and send/recv
        # - Standard : one send/recv per component
        # --- Standard send/recv ---
        br = self.bridge

        # reset send/recv requests
        self._r_request = {}
        self._s_request = {}

        basetag = self._mpis.rank + 1
        # Comm used for send/receive operations
        # It must contains all proc. of source topo and
        # target topo.
        refcomm = self.bridge.comm
        # Loop over all required components of each variable
        for v in self.variables:
            for d in self._range_components(v):
                v_name = v.name + S_DIR[d]

                # Deal with local copies of data
                if br.hasLocalInter():
                    vTo = self._vtarget[v].data[d]
                    vFrom = self._vsource[v].data[d]
                    vTo[br.localTargetInd()] = vFrom[br.localSourceInd()]

                # Transfers to other mpi processes
                for rk in self._receive:
                    recvtag = basetag * 989 + (rk + 1) * 99 + (d + 1) * 88
                    mpi_type = self._receive[rk]
                    vTo = self._vtarget[v].data[d]
                    self._r_request[v_name + str(rk)] = \
                        refcomm.Irecv([vTo, 1, mpi_type],
                                      source=rk, tag=recvtag)
                    self._has_requests = True
                for rk in self._send:
                    sendtag = (rk + 1) * 989 + basetag * 99 + (d + 1) * 88
                    mpi_type = self._send[rk]
                    vFrom = self._vsource[v].data[d]
                    self._s_request[v_name + str(rk)] = \
                        refcomm.Issend([vFrom, 1, mpi_type],
                                       dest=rk, tag=sendtag)
                    self._has_requests = True

    def wait(self):
        if self._has_requests:
            for rk in self._r_request:
                self._r_request[rk].Wait()
            for rk in self._s_request:
                self._s_request[rk].Wait()
        self._has_requests = False

    def test_requests(self):
        res = True
        for rk in self._r_request.keys():
            res = self._r_request[rk].Test()
            if not res:
                return res
        for rk in self._s_request.keys():
            res = self._s_request[rk].Test()
            if not res:
                return res

    def test_single_request(self, rsend=None, rrecv=None):
        """
        if neither rsend or rrecv is given return
        True if all communication request are complete
        else check for sending to rsend or
        receiving from rrecv. Process ranks
        should be given in parent_comm.
        @param rsend : discrete variable name + S_DIR + rank of the process
        to which a message has been sent
        and for which we want to test
        message completion.
        @param  rrecv : discrete variable name + S_DIR + rank of the process
        from which a message has been receive
        and for which we want to test
        message completion.
        """
        if rsend is not None or rrecv is not None:
            send_res = True
            recv_res = True
            if rsend is not None:
                send_res = self._s_request[rsend].Test()
            if rrecv is not None:
                recv_res = self._r_request[rrecv].Test()
            res = send_res and recv_res
            return res
        else:
            return self.test_requests()
