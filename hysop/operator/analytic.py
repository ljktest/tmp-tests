"""
@file operator/analytic.py
Initialize fields on a grid, with a user-defined function
"""
from hysop.constants import debug
from hysop.operator.continuous import opsetup, opapply
from hysop.operator.computational import Computational
from hysop.methods_keys import Support


class Analytic(Computational):
    """
    Applies an analytic formula, given by user, on its fields.
    """

    @debug
    def __init__(self, formula=None, vectorize_formula=False, **kwds):
        """
        Operator to apply a user-defined formula onto a list of fields.
        @param formula : the formula to be applied
        @param vectorize_formula : true if formula must be vectorized (numpy),
        default = false.
        """
        super(Analytic, self).__init__(**kwds)
        isGPU = False
        if 'method' in kwds.keys() and Support in kwds['method'].keys():
            isGPU = kwds['method'][Support].find('gpu') >= 0
        if formula is not None:
            ## A formula applied to all variables of this operator
            self.formula = formula
            for v in self.variables:
                v.set_formula(formula, vectorize_formula)
        elif not isGPU:
            vref = self.variables.keys()[0]
            assert vref.formula is not None
            self.formula = vref.formula
            # Only one formula allowed per operator
            for v in self.variables:
                assert v.formula is self.formula

        self.output = self.variables

    def discretize(self):
        super(Analytic, self)._standard_discretize()

    @opsetup
    def setup(self, rwork=None, iwork=None):
        self._is_uptodate = True

    @debug
    @opapply
    def apply(self, simulation=None):
        assert simulation is not None, \
            "Missing simulation value for computation."
        for v in self.variables:
            topo = self.discreteFields[v].topology
            v.initialize(time=simulation.time, topo=topo)

    def get_profiling_info(self):
        pass
