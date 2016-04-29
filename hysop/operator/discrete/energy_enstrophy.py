# -*- coding: utf-8 -*-
"""
@file energy_enstrophy.py
Compute Energy and Enstrophy
"""
from hysop.constants import debug
from hysop.tools.profiler import profile
import hysop.tools.numpywrappers as npw
from hysop.operator.discrete.discrete import DiscreteOperator


class EnergyEnstrophy(DiscreteOperator):
    """
    Discretization of the energy/enstrophy computation process.
    """
    def __init__(self, velocity, vorticity, is_normalized=True, **kwds):
        """
        Constructor.
        @param velocity : discretization of the velocity field
        @param vorticity : discretization of the vorticity field
        @param coeffs : dict of coefficients
        """
        ## velocity field
        self.velocity = velocity
        ## vorticity field
        self.vorticity = vorticity

        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(EnergyEnstrophy, self).__init__(variables=[velocity, vorticity],
                                              **kwds)
        ## Coeffs for integration
        self.coeff = {}
        ## Global energy
        self.energy = 0.0
        ## Global enstrophy
        self.enstrophy = 0.0
        topo_w = self.vorticity.topology
        topo_v = self.velocity.topology
        space_step = topo_w.mesh.space_step
        length = topo_w.domain.length
        # remark topo_w.domain and topo_v.domain
        # must be the same, no need to topo_v...length.
        self.coeff['Enstrophy'] = npw.prod(space_step)
        space_step = topo_v.mesh.space_step
        self.coeff['Energy'] = 0.5 * npw.prod(space_step)
        if is_normalized:
            normalization = 1. / npw.prod(length)
            self.coeff['Enstrophy'] *= normalization
            self.coeff['Energy'] *= normalization

    def _set_work_arrays(self, rwork=None, iwork=None):

        v_ind = self.velocity.topology.mesh.iCompute
        w_ind = self.vorticity.topology.mesh.iCompute
        shape_v = self.velocity.data[0][v_ind].shape
        shape_w = self.velocity.data[0][w_ind].shape
        # setup for rwork, iwork is useless.
        if rwork is None:
            # ---  Local allocation ---
            if shape_v == shape_w:
                self._rwork = [npw.zeros(shape_v)]
            else:
                self._rwork = [npw.zeros(shape_v), npw.zeros(shape_w)]
        else:
            assert isinstance(rwork, list), 'rwork must be a list.'
            # --- External rwork ---
            self._rwork = rwork
            if shape_v == shape_w:
                assert len(self._rwork) == 1
                assert self._rwork[0].shape == shape_v
            else:
                assert len(self._rwork) == 2
                assert self._rwork[0].shape == shape_v
                assert self._rwork[1].shape == shape_w

    def get_work_properties(self):

        v_ind = self.velocity.topology.mesh.iCompute
        w_ind = self.vorticity.topology.mesh.iCompute
        shape_v = self.velocity.data[0][v_ind].shape
        shape_w = self.velocity.data[0][w_ind].shape
        if shape_v == shape_w:
            return {'rwork': [shape_v], 'iwork': None}
        else:
            return {'rwork': [shape_v, shape_w], 'iwork': None}

    @debug
    @profile
    def apply(self, simulation=None):
        assert simulation is not None, \
            "Missing simulation value for computation."

        # --- Kinetic energy computation ---
        vd = self.velocity
        # get the list of computation points (no ghosts)
        nbc = vd.nb_components
        v_ind = self.velocity.topology.mesh.iCompute
        # Integrate (locally) velocity ** 2
        local_energy = 0.
        for i in xrange(nbc):
            self._rwork[0][...] = vd[i][v_ind] ** 2
            local_energy += npw.real_sum(self._rwork[0])

        # --- Enstrophy computation ---
        vortd = self.vorticity
        nbc = vortd.nb_components
        w_ind = self.vorticity.topology.mesh.iCompute
        # Integrate (locally) vorticity ** 2
        work = self._rwork[-1]
        local_enstrophy = 0.
        for i in xrange(nbc):
            work[...] = vortd[i][w_ind] ** 2
            local_enstrophy += npw.real_sum(work)

        # --- Reduce energy and enstrophy values overs all proc ---
        # two ways : numpy or classical. Todo : check perf and comm
        sendbuff = npw.zeros((2))
        recvbuff = npw.zeros((2))
        sendbuff[:] = [local_energy, local_enstrophy]
        #
        self.velocity.topology.comm.Allreduce(sendbuff, recvbuff)
        # the other way :
        #energy = self.velocity.topology.allreduce(local_energy,
        #                                          HYSOP_MPI_REAL,
        #                                          op=MPI.SUM)
        #enstrophy = self.velocity.topology.allreduce(local_enstrophy,
        #                                             HYSOP_MPI_REAL,
        #                                             op=MPI.SUM)

        # Update global values
        self.energy = recvbuff[0] * self.coeff['Energy']
        self.enstrophy = recvbuff[1] * self.coeff['Enstrophy']

        # Print results, if required
        ite = simulation.currentIteration
        if self._writer is not None and self._writer.do_write(ite):
            self._writer.buffer[0, 0] = simulation.time
            self._writer.buffer[0, 1] = self.energy
            self._writer.buffer[0, 2] = self.enstrophy
            self._writer.write()
