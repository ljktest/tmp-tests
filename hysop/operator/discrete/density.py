# -*- coding: utf-8 -*-
"""
@file operator/discrete/density.py
Discrete MultiPhase Rot Grad P
"""
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.constants import np, debug
from hysop.tools.profiler import profile


class DensityVisco_d(DiscreteOperator):
    """
    To be documented ...
    """
    @debug
    def __init__(self, density, viscosity,
                 densityVal=None, viscoVal=None, **kwds):
        """
        @param operator.
        """
        if 'variables' in kwds:
            super(DensityVisco_d, self).__init__(**kwds)
            self.density = self.variables[0]
            self.viscosity = self.variables[1]
        else:
            super(DensityVisco_d, self).__init__(variables=[density,
                                                            viscosity],
                                                 **kwds)
            self.density = density
            self.viscosity = viscosity

        self.densityVal = densityVal
        self.viscoVal = viscoVal
        self.input = [self.density, self.viscosity]
        self.output = [self.density, self.viscosity]

        # Note FP : what must be done if densityVal or viscoVal is None???

    @debug
    @profile
    def apply(self, simulation=None):
        assert simulation is not None, \
            "Missing simulation value for computation."

        iCompute = self.density.topology.mesh.iCompute

        # Density reconstruction
        if self.density[0][iCompute].all() <= np.absolute(
                self.densityVal[1] - self.densityVal[0]) / 2.0:
            self.density[0][iCompute] = self.densityVal[1]
        else:
            self.density[0][iCompute] = self.densityVal[0]

        # Viscosity reconstruction :
        # nu = nu1 + (nu2 - nu1) * (density - rho1)/(rho2 - rho1)
        self.viscosity.data[0] = self.viscoVal[0] + \
            (self.viscoVal[1] - self.viscoVal[0]) * \
            ((self.density.data[0] - self.densityVal[0]) /
             (self.densityVal[1] - self.densityVal[0]))
