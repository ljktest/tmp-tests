# -*- coding: utf-8 -*-
"""
@file reprojection.py
Compute reprojection criterion and divergence maximum
"""
import numpy as np
from hysop.constants import debug, HYSOP_MPI_REAL
from hysop.methods_keys import SpaceDiscretisation
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.numerics.finite_differences import FD_C_4
from hysop.numerics.differential_operations import GradV
import hysop.tools.numpywrappers as npw
from hysop.numerics.update_ghosts import UpdateGhosts
from hysop.mpi import MPI
from hysop.tools.profiler import profile


class Reprojection(DiscreteOperator):
    """
    Update the reprojection frequency, according to the current
    value of the vorticity field.
    """
    def __init__(self, vorticity, threshold, frequency, **kwds):
        """
        Constructor.
        @param vorticity: discretization of the vorticity field
        @param threshold : update frequency when criterion is greater than
        this threshold
        @param frequency : set frequency of execution of the reprojection
        """
        if 'method' in kwds and kwds['method'] is None:
            kwds['method'] = {SpaceDiscretisation: FD_C_4}

        ## vorticity field
        self.vorticity = vorticity
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Reprojection, self).__init__(variables=[vorticity], **kwds)
        ## Frequency for reprojection
        self.frequency = frequency
        ## The initial value will be used as default during
        # simulation
        self._default_frequency = frequency
        # constant defining the reprojection criterion :
        # if the latter is greater than this constant, then a reprojection
        # is needed
        self.threshold = threshold
        # local counter
        self._counter = 0
        ## Numerical methods for space discretization
        assert SpaceDiscretisation in self.method
        self.method = self.method[SpaceDiscretisation]
        self.input = [vorticity]
        self.output = []
        topo = self.vorticity.topology
        # prepare ghost points synchro for vorticity
        self._synchronize = UpdateGhosts(topo, self.vorticity.nb_components)
        # grad function
        self._function = GradV(topo=topo, method=self.method)

    def _set_work_arrays(self, rwork=None, iwork=None):

        memshape = self.vorticity.data[0].shape
        worklength = self.vorticity.nb_components ** 2

        # setup for rwork, iwork is useless.
        if rwork is None:
            # ---  Local allocation ---
            self._rwork = []
            for _ in xrange(worklength):
                self._rwork.append(npw.zeros(memshape))
        else:
            assert isinstance(rwork, list), 'rwork must be a list.'
            # --- External rwork ---
            self._rwork = rwork
            msg = 'Bad shape/length external work. Use get_work_properties'
            msg += ' function to find the right properties for work arrays.'
            assert len(self._rwork) == worklength, msg
            for wk in self._rwork:
                assert wk.shape == memshape

    @debug
    @profile
    def apply(self, simulation=None):
        assert simulation is not None, \
            'Simulation parameter is missing.'
        ite = simulation.currentIteration

        # Reset reprojection frequency to default
        self.frequency = self._default_frequency

        # Synchronize ghost points of vorticity
        self._synchronize(self.vorticity.data)
        # gradU computation
        self._rwork = self._function(self.vorticity.data, self._rwork)
        nb_components = self.vorticity.nb_components
        # maxima of vorticity divergence (abs)
        d1 = np.max(abs(sum([(self._rwork[(nb_components + 1) * i])
                             for i in xrange(nb_components)])))
        # maxima of partial derivatives of vorticity
        d2 = 0.0
        for grad_n in self._rwork:
            d2 = max(d2, np.max(abs(grad_n)))

        # computation of the reprojection criterion and mpi-reduction
        criterion = d1 / d2
        criterion = self.vorticity.topology.comm.allreduce(
            criterion, op=MPI.MAX)
        # is reprojection of vorticity needed for the next time step ?
        if criterion > self.threshold:
            self.frequency = 1

        # update counter
        if self.do_projection(ite):
            self._counter += 1

        # Print results, if required
        # Remark : writer buffer is (pointer) connected to diagnostics
        if self._writer is not None and self._writer.do_write(ite):
            self._writer.buffer[0, 0] = simulation.time
            self._writer.buffer[0, 1] = d1
            self._writer.buffer[0, 2] = d2
            self._writer.buffer[0, 3] = self._counter
            self._writer.write()

    def do_projection(self, ite):
        """
        True if projection must be done
        """
        return ite % self.frequency == 0
