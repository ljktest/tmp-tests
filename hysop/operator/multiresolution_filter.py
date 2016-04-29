"""
@file operator/multiresolution_filter.py
Filter values between grids of different resolution.
"""
from hysop.constants import debug
from hysop.operator.continuous import opsetup
from hysop.operator.computational import Computational
import hysop.default_methods as default
from hysop.methods_keys import Support
from hysop.tools.parameters import Discretization


class MultiresolutionFilter(Computational):
    """
    Apply a filter from a fine grid to a coarser grid.
    """

    @debug
    def __init__(self, d_in, d_out, **kwds):
        """
        Operator to apply Apply a filter from a fine grid to a coarser grid.
        """
        if 'method' not in kwds:
            kwds['method'] = default.MULTIRESOLUTION_FILER
        super(MultiresolutionFilter, self).__init__(**kwds)
        self.d_in, self.d_out = d_in, d_out
        self.output = self.variables

    def discretize(self):
        super(MultiresolutionFilter, self)._standard_discretize()
        if isinstance(self.d_in, Discretization):
            topo_in = self._build_topo(self.d_in, 0)
        else:
            topo_in = self.d_in
        if isinstance(self.d_out, Discretization):
            topo_out = self._build_topo(self.d_out, 0)
        else:
            topo_out = self.d_out
        self._df_in = []
        self._df_out = []
        for v in self.variables:
            if not topo_in in v.discreteFields.keys():
                self._df_in.append(v.discretize(topo_in))
            else:
                self._df_in.append(v.discreteFields[topo_in])
            if not topo_out in v.discreteFields.keys():
                self._df_out.append(v.discretize(topo_out))
            else:
                self._df_out.append(v.discreteFields[topo_out])

    @opsetup
    def setup(self, rwork=None, iwork=None):
        if Support in self.method.keys() and \
           self.method[Support].find('gpu') >= 0:
            from hysop.gpu.gpu_multiresolution_filter \
                import GPUFilterFineToCoarse as discreteFilter
        else:
            from hysop.operator.discrete.multiresolution_filter \
                import FilterFineToCoarse as discreteFilter
        self.discrete_op = discreteFilter(
            field_in=self._df_in, field_out=self._df_out,
            method=self.method, rwork=rwork, iwork=iwork)
        self._is_uptodate = True
