# -*- coding: utf-8 -*-
"""
@file operator/multiphase.py

MultiPhase Rot Grad P
"""
from hysop.operator.computational import Computational
from hysop.operator.discrete.baroclinic import Baroclinic as BD
from hysop.methods_keys import SpaceDiscretisation
from hysop.numerics.finite_differences import FD_C_4
from hysop.constants import debug
import hysop.default_methods as default
from hysop.operator.continuous import opsetup


class Baroclinic(Computational):
    """
    Pressure operator representation
    """

    @debug
    def __init__(self, velocity, vorticity, density, viscosity, **kwds):
        """
        Constructor.
        Create a Pressure operator from given velocity variables.

        @param velocity field
        @param vorticity field
        @param viscosity constant
        @param density field
        @param resolutions : grid resolution of velocity, vorticity, density
        @param method : solving method
        (default = finite differences, 4th order, in space)
        @param topo : a predefined topology to discretize
         velocity/vorticity/density
        @param ghosts : number of ghosts points. Default depends on the method.
        Autom. computed if not set.
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Baroclinic, self).__init__(variables=[velocity,
                                                    vorticity, density],
                                         **kwds)
        if self.method is None:
            self.method = default.BAROCLINIC
        self.velocity = velocity
        self.vorticity = vorticity
        self.density = density
        self.viscosity = viscosity
        self.input = [self.velocity, self.vorticity, self.density]
        self.output = [self.vorticity]
        assert SpaceDiscretisation in self.method.keys()

    def discretize(self):
        if self.method[SpaceDiscretisation] is FD_C_4:
            nbGhosts = 2
        else:
            raise ValueError("Unknown method for space discretization of the\
                baroclinic operator.")

        super(Baroclinic, self)._standard_discretize(nbGhosts)

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        """
        Baroclinic operator discretization method.
        Create a discrete Baroclinic operator from given specifications.
        """
        self.discrete_op = \
            BD(self.discreteFields[self.velocity],
               self.discreteFields[self.vorticity],
               self.discreteFields[self.density],
               self.viscosity,
               method=self.method)

        self._is_uptodate = True


    def initialize_velocity(self):
        self.discrete_op.initialize_velocity()
