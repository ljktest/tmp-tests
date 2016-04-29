# -*- coding: utf-8 -*-
"""
@file operator/multiphase_baroclinic_rhs.py

Computation of the baroclinic term in a multiphasic flow:
\f{eqnarray*}
\frac{\partial\vec{\omega}{\partial t} = -\frac{\nabla \rho}{\rho}\times\left(-\frac{\nabla P}{\rho}\right)
\f} with finite differences
"""
from hysop.operator.computational import Computational
from hysop.methods_keys import SpaceDiscretisation, Support
from hysop.constants import debug, np
import hysop.default_methods as default
from hysop.operator.continuous import opsetup
from hysop.gpu.gpu_multiphase_baroclinic_rhs import BaroclinicRHS


class MultiphaseBaroclinicRHS(Computational):
    """
    Baroclinic operator representation
    """

    @debug
    def __init__(self, rhs, rho, gradp, **kwds):
        """
        Constructor.
        Create a Baroclinic operator from given variables.

        @param rhs fiel
        @param rho field
        @param gradp field
        @param method : solving method
        (default = finite differences, 4th order, in space)
        @param ghosts : number of ghosts points. Default depends on the method.
        Autom. computed if not set.
        """
        super(MultiphaseBaroclinicRHS, self).__init__(**kwds)
        if self.method is None:
            self.method = default.MULTIPHASEBAROCLINIC
        self.rhs = rhs
        self.rho = rho
        self.gradp = gradp
        self.input = [self.rho, self.gradp]
        self.output = [self.rhs, ]
        assert SpaceDiscretisation in self.method.keys()
        msg = "This operator is implemented for GPU only"
        assert Support in self.method.keys(), msg
        assert self.method[Support] == 'gpu', msg

    def discretize(self):
        build_topos = self._check_variables()
        assert not self._single_topo, \
            "This operator must have different topologies"
        for v in self.variables:
            if build_topos[v]:
                topo = self.domain.create_topology(
                    discretization=self.variables[v], dim=2)
                self.variables[v] = topo
                build_topos[v] = False
        msg = "Need review for ghosts"
        assert np.all(self.variables[self.rhs].ghosts() == 0), msg
        assert np.all(self.variables[self.gradp].ghosts() > 0), msg
        assert np.all(self.variables[self.rho].ghosts() == 0), msg

        # All topos are built, we can discretize fields.
        self._discretize_vars()
        self._is_discretized = True

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        """
        Baroclinic operator discretization method.
        Create a discrete Baroclinic operator from given specifications.
        """
        self.discrete_op = \
            BaroclinicRHS(self.discreteFields[self.rhs],
                          self.discreteFields[self.rho],
                          self.discreteFields[self.gradp],
                          method=self.method)
        self._is_uptodate = True
