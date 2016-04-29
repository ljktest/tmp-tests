from hysop.operator.computational import Computational
from hysop.methods_keys import Support
from hysop.operator.continuous import opsetup, opapply
from hysop.numerics.update_ghosts import UpdateGhostsFull

class DataTransfer(Computational):
    """Operator for moving data between CPU and GPU."""

    def __init__(self, source, target, component=None,
                 run_till=None, freq=1, **kwds):
        """
        @param way: HostToDevice or DeviceToHost flag setting
        the data copy direction.
        """
        super(DataTransfer, self).__init__(**kwds)

        self.input = self.variables
        self.output = self.variables

        ## Transfer frequency in iteration number
        self.freq = freq

        # Object (may be an operator or a topology) which handles the
        # fields to be transfered
        self._source = source
        # Object (may an operator or a topology) which handles the fields
        # to be filled in from source.
        self._target = target

        self.component = component
        if component is None:
            # All components are considered
            self._range_components = lambda v: xrange(v.nb_components)
        else:
            # Only the given component is considered
            assert self.component >= 0, 'component value must be positive.'
            self._range_components = lambda v: (self.component)

        # Which operator must wait for this one before
        # any computation
        # Exp : run_till = op1 means that op1 will
        # wait for the end of this operator before
        # op1 starts its apply.
        if run_till is None:
            run_till = []

        assert isinstance(run_till, list)
        self._run_till = run_till

        from hysop.mpi.topology import Cartesian
        if not isinstance(self._target, Cartesian):
            # target operator must wait for
            # the end of this operator to apply.
            self._run_till.append(self._target)

        self._transfer = None
        self._is_discretized = True

    @opsetup
    def setup(self, rwork=None, iwork=None):
        for op in self._run_till:
            op.wait_for(self)
        topo = self.variables.values()[0]
        self._d_var = [v.discreteFields[topo] for v in self.variables]

        from hysop.mpi.topology import Cartesian
        source_is_topo = isinstance(self._source, Cartesian)
        target_is_topo = isinstance(self._target, Cartesian)

        source_is_gpu = False
        try:
            if self._source.method[Support].find('gpu') >= 0:
                source_is_gpu = True
        except:
            pass
        target_is_gpu = False
        try:
            if self._target.method[Support].find('gpu') >= 0:
                target_is_gpu = True
        except:
            pass

        ## Current transfer function
        if source_is_gpu and not target_is_gpu:
            self._transfer = self._apply_toHost
        elif target_is_gpu and not source_is_gpu:
            self._transfer = self._apply_toDevice
        else:
            if source_is_gpu and target_is_gpu:
                raise RuntimeError(
                    "One of source or target must be a GPU operator.")
            if not source_is_gpu and not target_is_gpu:
                if source_is_topo:
                    print self.name, "Assume this is a toHost transfer"
                    self._transfer = self._apply_toHost
                elif target_is_topo:
                    print self.name, "Assume this is a toDevice transfer"
                    self._transfer = self._apply_toDevice
                else:
                    raise RuntimeError(
                        "One of source or target must be a GPU operator " +
                        self.name + ".")
        # Function to synchronize ghosts before send data to device.
        self._ghosts_synchro = None
        # This function is needed only in a toDevice transfer and if the target operator needs ghosts
        if self._transfer == self._apply_toDevice:
            d_target = self._target.discrete_op
            # Test for advection operator
            if self._target._is_discretized and d_target is None:
                d_target = self._target.advec_dir[0].discrete_op
            if  d_target._synchronize is not None and d_target._synchronize:
                self._ghosts_synchro = UpdateGhostsFull(
                    self._d_var[0].topology, 
                    self._d_var[0].nb_components)
                
    @opapply
    def apply(self, simulation=None):
        ite = simulation.currentIteration
        if ite % self.freq == 0:
            self._transfer()

    def _apply_toHost(self):
        for df in self._d_var:
            for c in self._range_components(df):
                df.toHost(component=c)

    def _apply_toDevice(self):
        if self._ghosts_synchro is not None:
            for df in self._d_var:
                # Ghosts Synchronization before sending
                self._ghosts_synchro(df.data)
        for df in self._d_var:
            for c in self._range_components(df):
                df.toDevice(component=c)

    def wait(self):
        for df in self._d_var:
            df.wait()

    def finalize(self):
        pass

    def computation_time(self):
        pass
