# -*- coding: utf-8 -*-
"""
@file monitoringPoints.py
Print time evolution of flow variables (velo, vorti)
at a particular monitoring point in the wake
"""
from hysop.constants import debug
from hysop.tools.profiler import profile
import hysop.tools.numpywrappers as npw
import scitools.filetable as ft
import numpy as np
from hysop.operator.discrete.discrete import DiscreteOperator


class MonitoringPoints(DiscreteOperator):
    """
        Print time evolution of flow variables at a given position in the wake.
    """
    def __init__(self, velocity, vorticity, monitPt_coords, **kwds):
        """
        Constructor.
        @param velocity field
        @param vorticity field
        @param monitPt_coords : list of coordinates corresponding
            to the space location of the different monitoring points in the wake
        """
        ## velocity field
        self.velocity = velocity
        ## vorticity field
        self.vorticity = vorticity
        ## Monitoring point coordinates
        self.monitPt_coords = monitPt_coords

        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(MonitoringPoints, self).__init__(variables=[velocity, vorticity],
                                       **kwds)
        topo_v = self.velocity.topology
        self.shape_v = self.velocity.data[0][topo_v.mesh.iCompute].shape
        self.space_step = topo_v.mesh.space_step
        self.length = topo_v.domain.length
        self.origin = topo_v.domain.origin
        self.coords = topo_v.mesh.coords
        self.nbIter = 0
        ## Normalized flow values at monitoring position (velNorm, vortNorm)
        self.velNorm = 0.0
        self.vortNorm = 0.0

        # Is current processor working ? (Is monitPt_coords(z) in z-coords ?)
        self.is_rk_computing = False
        s = self._dim - 1
        if (self.monitPt_coords[s] in self.coords[s]):
            self.is_rk_computing = True

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

        time = simulation.time
        ite = simulation.currentIteration
        filename = self._writer.filename  #+ '_ite' + format(ite)

        if self.is_rk_computing :
            self.nbIter += 1
            vd = self.velocity.data
            vortd = self.vorticity.data
            nbc = self.velocity.nb_components
            tab = [self.monitPt_coords[0], self.monitPt_coords[1],
                   self.monitPt_coords[2]]

            ind = []
            for d in xrange(nbc):
                cond = np.where(abs(self.coords[d] - tab[d])
                                < (self.space_step[d] * 0.5))
                if cond[0].size > 0:
                    ind.append(cond[d][0])
                else:
                    raise ValueError("Wrong set of coordinates.")

            # Compute velocity and vorticity norm
            # at the monitoring point
            self.velNorm = np.sqrt(vd[0][ind[0],ind[1],ind[2]] ** 2 +
                                   vd[1][ind[0],ind[1],ind[2]] ** 2 +
                                   vd[2][ind[0],ind[1],ind[2]] ** 2)
            self.vortNorm = np.sqrt(vortd[0][ind[0],ind[1],ind[2]] ** 2 +
                                    vortd[1][ind[0],ind[1],ind[2]] ** 2 +
                                    vortd[2][ind[0],ind[1],ind[2]] ** 2)


            if self._writer is not None and self._writer.do_write(ite) :
                self._writer.buffer[0, 0] = time
                self._writer.buffer[0, 1] = self.velNorm
                self._writer.buffer[0, 2] = self.vortNorm
                self._writer.write()



