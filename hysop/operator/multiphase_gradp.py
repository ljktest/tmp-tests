# -*- coding: utf-8 -*-
"""
@file operator/multiphase_gradp.py

Computation of the pressure gradient in a multiphasic flow:
\f{eqnarray*}
-\frac{\nabla P}{\rho} = \frac{\partial\boldsymbol{u}}{\partial t} + (\boldsymbol{u}\cdot\nabla)\boldsymbol{u}  - \nu\Delta\boldsymbol{u}
\f} with finite differences
"""
from hysop.operator.computational import Computational
from hysop.methods_keys import SpaceDiscretisation
from hysop.numerics.finite_differences import FD_C_4
from hysop.constants import debug, np
import hysop.default_methods as default
from hysop.operator.continuous import opsetup
from hysop.operator.discrete.multiphase_gradp import GradP


class MultiphaseGradP(Computational):
    """
    Pressure operator representation
    """

    @debug
    def __init__(self, velocity, gradp, viscosity, **kwds):
        """
        Constructor.
        Create a Pressure operator from given velocity variables.

        @param velocity field
        @param gradp result
        @param viscosity constant
        @param resolutions : grid resolution of velocity and gradp
        @param method : solving method
        (default = finite differences, 4th order, in space)
        @param topo : a predefined topology to discretize
         velocity/gradp
        @param ghosts : number of ghosts points. Default depends on the method.
        Autom. computed if not set.
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(MultiphaseGradP, self).__init__(
            variables=[velocity, gradp], **kwds)
        if self.method is None:
            self.method = default.MULTIPHASEGRADP
        self.velocity = velocity
        self.gradp = gradp
        self.viscosity = viscosity
        self.input = [self.velocity, ]
        self.output = [self.gradp, ]
        assert SpaceDiscretisation in self.method.keys()

    def discretize(self):
        if self.method[SpaceDiscretisation] is FD_C_4:
            nbGhosts = 2
        else:
            raise ValueError("Unknown method for space discretization of the\
                multiphase gradp operator.")

        super(MultiphaseGradP, self)._standard_discretize(nbGhosts)

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        """
        Baroclinic operator discretization method.
        Create a discrete Baroclinic operator from given specifications.
        """
        self.discrete_op = \
            GradP(self.discreteFields[self.velocity],
                  self.discreteFields[self.gradp],
                  self.viscosity,
                  method=self.method)
        self._is_uptodate = True

    def initialize_velocity(self):
        self.discrete_op.initialize_velocity()
