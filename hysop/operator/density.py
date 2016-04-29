# -*- coding: utf-8 -*-
"""
@file operator/density.py

"""
from hysop.operator.computational import Computational
from hysop.operator.discrete.density import DensityVisco_d
from hysop.operator.continuous import opsetup
from hysop.constants import debug


class DensityVisco(Computational):
    """
    Density and Viscosity reconstruction
    """

    @debug
    def __init__(self, density, viscosity, **kwds):
        """
        @param density : scalar field
        @param viscosity : scalar field
        """
        super(DensityVisco, self).__init__(variables=[density, viscosity],
                                           **kwds)
        self.density = density
        self.viscosity = viscosity
        self.input = [self.density, self.viscosity]
        self.output = [self.density, self.viscosity]

    def discretize(self):
        super(DensityVisco, self)._standard_discretize()

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        """
        Density and Viscosity reconstruction operator discretization method.
        Create a discrete operator from given specifications.
        """

        self.discrete_op = \
            DensityVisco_d(density=self.discreteFields[self.density],
                           viscosity=self.discreteFields[self.viscosity],
                           method=self.method)
        self._is_uptodate = True
