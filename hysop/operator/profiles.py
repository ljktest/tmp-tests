# -*- coding: utf-8 -*-
"""
@file profiles.py
Compute and print velo/vorti profiles
"""
from hysop.operator.discrete.profiles import Profiles as ProfD
from hysop.operator.computational import Computational
from hysop.operator.continuous import opsetup


class Profiles(Computational):
    """
    Compute and print velo/vorti profiles
    """

    def __init__(self, velocity, vorticity, prof_coords, 
                 direction, beginMeanComput, **kwds):
        """
        Constructor.
        @param velocity field
        @param vorticity field
        @param direction : profile direction (0, 1 or 2)
        @param beginMeanComput : time at which the computation of mean profile must begin
        @param prof_coords : X and Y coordinates of the profile 
        warning : the Z-coordinate is supposed to be 0 for each profile !

        Default file name = 'profile.dat'
        See hysop.tools.io_utils.Writer for details
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Profiles, self).__init__(variables=[velocity, vorticity],
                                       **kwds)
        ## velocity field
        self.velocity = velocity
        ## vorticity field
        self.vorticity = vorticity
        ## X and Y coordinates of the profile
        self.prof_coords = prof_coords
        ## profile direction (0, 1 or 2)
        self.direction = direction
        ## time at which the computation of mean profile must begin
        self.beginMeanComput = beginMeanComput
        self.input = [velocity, vorticity]
        self.output = []

    def get_work_properties(self):
        if not self._is_discretized:
            msg = 'The operator must be discretized '
            msg += 'before any call to this function.'
            raise RuntimeError(msg)
        vd = self.discreteFields[self.velocity]
        wd = self.discreteFields[self.vorticity]
        v_ind = vd.topology.mesh.iCompute
        w_ind = wd.topology.mesh.iCompute
        shape_v = vd[0][v_ind].shape
        shape_w = wd[0][w_ind].shape
        if shape_v == shape_w:
            return {'rwork': [shape_v], 'iwork': None}
        else:
            return {'rwork': [shape_v, shape_w], 'iwork': None}

    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:

            self.discrete_op = ProfD(self.discreteFields[self.velocity],
                                     self.discreteFields[self.vorticity],
                                     self.prof_coords, self.direction,
                                     self.beginMeanComput,
                                     rwork=rwork)
            # Output setup
            self._set_io('profile', (1, 9))
            self.discrete_op.setWriter(self._writer)
            self._is_uptodate = True

