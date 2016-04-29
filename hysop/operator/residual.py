# -*- coding: utf-8 -*-
"""
@file residual.py
Compute and print the time evolution of the residual
"""
from hysop.operator.discrete.residual import Residual as ResD
from hysop.operator.computational import Computational
from hysop.operator.continuous import opsetup


class Residual(Computational):
    """
    Compute and print the residual time evolution
    """

    def __init__(self, vorticity, **kwds):
        """
        Constructor.
        @param vorticity field

        Default file name = 'residual.dat'
        See hysop.tools.io_utils.Writer for details
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Residual, self).__init__(variables=[vorticity], **kwds)
        ## vorticity field
        self.vorticity = vorticity
        self.input = [vorticity]
        self.output = []

    def get_work_properties(self):
        if not self._is_discretized:
            msg = 'The operator must be discretized '
            msg += 'before any call to this function.'
            raise RuntimeError(msg)
        wd = self.discreteFields[self.vorticity]
        w_ind = wd.topology.mesh.iCompute
        shape_w = wd[0][w_ind].shape
        return {'rwork': [shape_w], 'iwork': None}

    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:

            self.discrete_op = ResD(self.discreteFields[self.vorticity],
                                    rwork=rwork)
            # Initialization of w^(n-1) vorticity value
            self.discrete_op.initialize_vortPrev()
            # Output setup
            self._set_io('residual', (1, 3))
            self.discrete_op.setWriter(self._writer)
            self._is_uptodate = True

