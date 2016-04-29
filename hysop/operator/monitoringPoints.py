# -*- coding: utf-8 -*-
"""
@file monitoringPoints.py
Print time evolution of flow variables (velo, vorti)
at a particular monitoring point in the wake
"""
from hysop.operator.discrete.monitoringPoints import MonitoringPoints as MonitD
from hysop.operator.computational import Computational
from hysop.operator.continuous import opsetup


class MonitoringPoints(Computational):
    """
    Compute and print velo/vorti profiles
    """

    def __init__(self, velocity, vorticity, monitPt_coords, **kwds):
        """
        Constructor.
        @param velocity field
        @param vorticity field
        @param monitPts_coords : coordinates corresponding
            to the space location of the monitoring point in the wake

        Default file name = 'monit.dat'
        See hysop.tools.io_utils.Writer for details
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(MonitoringPoints, self).__init__(variables=
                                               [velocity, vorticity],
                                               **kwds)
        ## velocity field
        self.velocity = velocity
        ## vorticity field
        self.vorticity = vorticity
        ## coordinates of the monitoring point
        self.monitPt_coords = monitPt_coords
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

            self.discrete_op = MonitD(self.discreteFields[self.velocity],
                                      self.discreteFields[self.vorticity],
                                      self.monitPt_coords, rwork=rwork)
            # Output setup
            self._set_io('monit', (1, 3))
            self.discrete_op.setWriter(self._writer)
            self._is_uptodate = True

