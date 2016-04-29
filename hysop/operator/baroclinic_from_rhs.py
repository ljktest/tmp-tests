# -*- coding: utf-8 -*-
"""
@file operator/baroclinic_from_rhs.py

MultiPhase baroclinic term
"""
from hysop.operator.computational import Computational
from hysop.operator.discrete.baroclinic_from_rhs import BaroclinicFromRHS as BD
from hysop.methods_keys import SpaceDiscretisation
from hysop.numerics.finite_differences import FD_C_4
from hysop.constants import debug
import hysop.default_methods as default
from hysop.operator.continuous import opsetup


class BaroclinicFromRHS(Computational):
    """
    Pressure operator representation
    """

    @debug
    def __init__(self, vorticity, rhs, **kwds):
        """
        Constructor.
        Create a BaroclinicFromRHS operator on a given vorticity and the rhs.

        @param vorticity field
        @param rhs field
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(BaroclinicFromRHS, self).__init__(variables=[vorticity, rhs],
                                                **kwds)
        if self.method is None:
            self.method = default.BAROCLINIC
        self.vorticity = vorticity
        self.rhs = rhs
        self.input = [self.vorticity, self.rhs]
        self.output = [self.vorticity]
        assert SpaceDiscretisation in self.method.keys()

    def discretize(self):
        super(BaroclinicFromRHS, self)._standard_discretize()

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        """
        Baroclinic operator discretization method.
        Create a discrete Baroclinic operator from given specifications.
        """
        self.discrete_op = \
            BD(self.discreteFields[self.vorticity],
               self.discreteFields[self.rhs],
               method=self.method)

        self._is_uptodate = True
