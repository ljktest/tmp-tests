# -*- coding: utf-8 -*-
"""
@file operator/reprojection.py
Compute reprojection criterion and divergence maximum
"""
from hysop.operator.computational import Computational
from hysop.operator.discrete.reprojection import Reprojection as RD
from hysop.operator.continuous import opsetup


class Reprojection(Computational):
    """
    Computes and prints reprojection criterion.
    See the related PDF called "vorticity_solenoidal_projection.pdf"
    in HySoPDoc for more details.
    """
    def __init__(self, vorticity, threshold, frequency, **kwds):
        """
        Constructor.
        @param vorticity field
        @param threshold : update frequency when criterion is greater than
        this threshold
        @param frequency : set frequency of execution of the reprojection
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Reprojection, self).__init__(variables=[vorticity], **kwds)
        # constant defining the reprojection criterion :
        # if the latter is greater than this constant, then a reprojection
        # is needed
        self.threshold = threshold
        ## Frequency for reprojection
        self.frequency = frequency
        ## vorticity field
        self.vorticity = vorticity
        self.input = [vorticity]
        self.output = []

    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:
            self.discrete_op = RD(self.discreteFields[self.vorticity],
                                       self.threshold,
                                       self.frequency, rwork=rwork,
                                       method=self.method)
        self._set_io('reprojection', (1, 4))
        self.discrete_op.setWriter(self._writer)
        self._is_uptodate = True

    def do_projection(self, ite):
        """
        True if projection must be done
        """
        return self.discrete_op.do_projection(ite)
