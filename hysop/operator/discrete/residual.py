# -*- coding: utf-8 -*-
"""
@file residual.py
Compute and print the time evolution of the residual
"""
from hysop.constants import debug
from hysop.tools.profiler import profile
import hysop.tools.numpywrappers as npw
import scitools.filetable as ft
import numpy as np
from hysop.operator.discrete.discrete import DiscreteOperator


class Residual(DiscreteOperator):
    """
        Compute and print the residual as a function of time
    """
    def __init__(self, vorticity, **kwds):
        """
        Constructor.
        @param vorticity field
        """
        ## vorticity field
        self.vorticity = vorticity

        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Residual, self).__init__(variables=[vorticity], **kwds)
        topo_w = self.vorticity.topology
        self.shape_w = self.vorticity.data[0][topo_w.mesh.iCompute].shape
        self.space_step = topo_w.mesh.space_step
        self.length = topo_w.domain.length
        self.origin = topo_w.domain.origin
        self.coords = topo_w.mesh.coords
        self.nbIter = 0
        ## Global residual
        self.residual = 0.0
        # Time stem of the previous iteration
        self._old_dt = None
        # Define array to store vorticity field at previous iteration
        self._vortPrev = [npw.zeros_like(d) for d in self.vorticity.data]

    def _set_work_arrays(self, rwork=None, iwork=None):

        w_ind = self.vorticity.topology.mesh.iCompute
        shape_w = self.vorticity.data[0][w_ind].shape
        # setup for rwork, iwork is useless.
        if rwork is None:
            # ---  Local allocation ---
            self._rwork = [npw.zeros(shape_w)]
        else:
            assert isinstance(rwork, list), 'rwork must be a list.'
            # --- External rwork ---
            self._rwork = rwork
            assert len(self._rwork) == 1
            assert self._rwork[0].shape == shape_w

    def get_work_properties(self):

        w_ind = self.vorticity.topology.mesh.iCompute
        shape_w = self.vorticity.data[0][w_ind].shape
        return {'rwork': [shape_w], 'iwork': None}

    def initialize_vortPrev(self):
        
        w_ind = self.vorticity.topology.mesh.iCompute
        for d in xrange(self.vorticity.dimension):
            self._vortPrev[d][w_ind] = self.vorticity[d][w_ind]

    @debug
    @profile
    def apply(self, simulation=None):
        assert simulation is not None, \
            "Missing simulation value for computation."

        time = simulation.time
        ite = simulation.currentIteration
        dt = simulation.timeStep
        if self._old_dt is None:
            self._old_dt = dt
        
        filename = self._writer.filename
        
        # Compute on local proc (w^(n+1)-w^n) ** 2
        local_res = 0.
        # get the list of computation points (no ghosts)
        wd = self.vorticity
        nbc = wd.nb_components
        w_ind = self.vorticity.topology.mesh.iCompute
        for i in xrange(nbc):
            self._rwork[0][...] = (wd[i][w_ind] -
                                   self._vortPrev[i][w_ind]) ** 2
            local_res += npw.real_sum(self._rwork[0])

        # --- Reduce local_res over all proc ---
        sendbuff = npw.zeros((1))
        recvbuff = npw.zeros((1))
        sendbuff[:] = [local_res]
        #
        self.vorticity.topology.comm.Allreduce(sendbuff, recvbuff)
        
        # Update global residual
        self.residual = np.sqrt(recvbuff[0])

        # Print results, if required
        if self._writer is not None and self._writer.do_write(ite) :
            self._writer.buffer[0, 0] = time
            self._writer.buffer[0, 1] = ite
            self._writer.buffer[0, 2] = self.residual
            self._writer.write()

        # update vort(n-1) for next iteration
        for i in xrange(nbc):
            self._vortPrev[i][w_ind] = self.vorticity.data[i][w_ind]
        self._old_dt = dt



