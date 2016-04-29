# -*- coding: utf-8 -*-
"""
@file profiles.py
Compute and print velo/vorti profiles
"""
from hysop.constants import debug
from hysop.tools.profiler import profile
import hysop.tools.numpywrappers as npw
import scitools.filetable as ft
import numpy as np
from hysop.operator.discrete.discrete import DiscreteOperator


class Profiles(DiscreteOperator):
    """
    Compute and print velo/vorti profiles at a given position.
    """
    def __init__(self, velocity, vorticity, prof_coords, 
                 direction, beginMeanComput, **kwds):
        """
        Constructor.
        @param velocity : discretization of the velocity field
        @param vorticity : discretization of the vorticity field
        @param direction : profile direction (0, 1 or 2)
        @param beginMeanComput : time at which the computation of mean profile must begin
        @param prof_coords : X and Y coordinates of the profile 
        warning : the Z-coordinate is supposed to be 0 for each profile !
        """
        ## velocity field
        self.velocity = velocity
        ## vorticity field
        self.vorticity = vorticity
        ## X and Y coordinates of the profile
        self.prof_coords =  prof_coords
        ## profile direction (0, 1 or 2)
        self.direction = direction
        ## time at which the computation of mean profile must begin
        self.beginMeanComput = beginMeanComput

        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Profiles, self).__init__(variables=[velocity, vorticity],
                                       **kwds)
        topo_v = self.velocity.topology
        self.shape_v = self.velocity.data[0][topo_v.mesh.iCompute].shape
        self.space_step = topo_v.mesh.space_step
        self.length = topo_v.domain.length
        self.origin = topo_v.domain.origin
        self.coords = topo_v.mesh.coords
        self.nbIter = 0
        ## Mean quantities (meanVelNorm, meanVortNorm, meanVelX, 
        ## meanVelY, meanVelY, meanVortX, meanVortY, meanVortZ)
        self.mean_qtities = None
        if direction==0:
            self.mean_qtities = [npw.zeros(self.shape_v[0]) for d in xrange(8)]
        elif direction==1:
            self.mean_qtities = [npw.zeros(self.shape_v[1]) for d in xrange(8)]
        else:
            raise ValueError("Only profiles in the X or Y direction.")

        # Is current processor working ? (Is 0 in z-coords ?)
        self.is_rk_computing = False
        if (0.0 in self.coords[2]):
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
        filename = self._writer.filename  + '_ite' + format(ite)

        if time >= self.beginMeanComput and self.is_rk_computing :
            self.nbIter += 1
            vd = self.velocity.data
            vortd = self.vorticity.data
            nbc = self.velocity.nb_components
            tab = [self.prof_coords[0],  self.prof_coords[1], 0.0]

            ind = []
            for d in xrange(nbc):
                cond = np.where(abs(self.coords[d] - tab[d])
                                < (self.space_step[d] * 0.5))
                if cond[0].size > 0:
                    ind.append(cond[d][0])
                else:
                    raise ValueError("Wrong set of coordinates.")

            if self.direction==0 :
                for i in xrange (self.shape_v[0]):
                    self.mean_qtities[0][i] += np.sqrt(vd[0][i,ind[1],ind[2]] ** 2 +
                                                       vd[1][i,ind[1],ind[2]] ** 2 +
                                                       vd[2][i,ind[1],ind[2]] ** 2)
                    self.mean_qtities[1][i] += np.sqrt(vortd[0][i,ind[1],ind[2]] ** 2 +
                                                       vortd[1][i,ind[1],ind[2]] ** 2 +
                                                       vortd[2][i,ind[1],ind[2]] ** 2)
                    self.mean_qtities[2][i] += vd[0][i,ind[1],ind[2]]
                    self.mean_qtities[3][i] += vd[1][i,ind[1],ind[2]]
                    self.mean_qtities[4][i] += vd[2][i,ind[1],ind[2]]
                    self.mean_qtities[5][i] += vortd[0][i,ind[1],ind[2]]
                    self.mean_qtities[6][i] += vortd[1][i,ind[1],ind[2]]
                    self.mean_qtities[7][i] += vortd[2][i,ind[1],ind[2]]

            elif self.direction==1 :
                for j in xrange (self.shape_v[1]):
                    self.mean_qtities[0][j] += np.sqrt(vd[0][ind[0],j,ind[2]] ** 2 +
                                                       vd[1][ind[0],j,ind[2]] ** 2 +
                                                       vd[2][ind[0],j,ind[2]] ** 2)
                    self.mean_qtities[1][j] += np.sqrt(vortd[0][ind[0],j,ind[2]] ** 2 +
                                                       vortd[1][ind[0],j,ind[2]] ** 2 +
                                                       vortd[2][ind[0],j,ind[2]] ** 2)
                    self.mean_qtities[2][j] += vd[0][ind[0],j,ind[2]]
                    self.mean_qtities[3][j] += vd[1][ind[0],j,ind[2]]
                    self.mean_qtities[4][j] += vd[2][ind[0],j,ind[2]]
                    self.mean_qtities[5][j] += vortd[0][ind[0],j,ind[2]]
                    self.mean_qtities[6][j] += vortd[1][ind[0],j,ind[2]]
                    self.mean_qtities[7][j] += vortd[2][ind[0],j,ind[2]]

            else:
                raise ValueError("Only profiles in the X or Y direction.")

            if self._writer is not None and self._writer.do_write(ite) :
                f = open(filename, 'w')
                if self.direction==0 :
                    for i in xrange (self.shape_v[0]):
                        self._writer.buffer[0, 0] = self.coords[0][i,0,0]
                        self._writer.buffer[0, 1] = self.mean_qtities[0][i] / self.nbIter
                        self._writer.buffer[0, 2] = self.mean_qtities[1][i] / self.nbIter
                        self._writer.buffer[0, 3] = self.mean_qtities[2][i] / self.nbIter
                        self._writer.buffer[0, 4] = self.mean_qtities[3][i] / self.nbIter
                        self._writer.buffer[0, 5] = self.mean_qtities[4][i] / self.nbIter
                        self._writer.buffer[0, 6] = self.mean_qtities[5][i] / self.nbIter
                        self._writer.buffer[0, 7] = self.mean_qtities[6][i] / self.nbIter
                        self._writer.buffer[0, 8] = self.mean_qtities[7][i] / self.nbIter
                        ft.write(f, self._writer.buffer)
                elif self.direction==1 :
                    for j in xrange (self.shape_v[1]):
                        self._writer.buffer[0, 0] = self.coords[1][0,j,0]
                        self._writer.buffer[0, 1] = self.mean_qtities[0][j] / self.nbIter
                        self._writer.buffer[0, 2] = self.mean_qtities[1][j] / self.nbIter
                        self._writer.buffer[0, 3] = self.mean_qtities[2][j] / self.nbIter
                        self._writer.buffer[0, 4] = self.mean_qtities[3][j] / self.nbIter
                        self._writer.buffer[0, 5] = self.mean_qtities[4][j] / self.nbIter
                        self._writer.buffer[0, 6] = self.mean_qtities[5][j] / self.nbIter
                        self._writer.buffer[0, 7] = self.mean_qtities[6][j] / self.nbIter
                        self._writer.buffer[0, 8] = self.mean_qtities[7][j] / self.nbIter
                        ft.write(f, self._writer.buffer)
                else :
                    raise ValueError("Only profiles in the X or Y direction.")
                f.close()



