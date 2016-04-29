"""
@file custom.py
"""
from hysop.operator.computational import Computational
from hysop.operator.discrete.custom import CustomMonitor as CM
from hysop.operator.discrete.custom import CustomOp as CO
from hysop.operator.continuous import opsetup


class CustomOp(Computational):
    def __init__(self, in_fields, out_fields, function, **kwds):
        super(CustomOp, self).__init__(**kwds)
        self.function = function
        self.input = in_fields
        self.output = out_fields

    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:
            self.discretize()
            self.discrete_op = CO(
                [self.discreteFields[f] for f in self.input],
                [self.discreteFields[f] for f in self.output],
                self.function,
                variables=self.discreteFields.values())
            self._is_uptodate = True


class CustomMonitor(Computational):
    def __init__(self, function, res_shape=1, **kwds):
        super(CustomMonitor, self).__init__(**kwds)
        self.function = function
        self.res_shape = res_shape
        self.input = self.variables

    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:
            self.discretize()
            self.discrete_op = CM(self.function, self.res_shape,
                                  variables=self.discreteFields.values())
            # Output setup
            self._set_io(self.function.__name__, (1, 1 + self.res_shape))
            self.discrete_op.setWriter(self._writer)
            self._is_uptodate = True
