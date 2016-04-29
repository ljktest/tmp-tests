# -*- coding: utf-8 -*-
"""
@file operator/discrete/baroclinic_from_rhs.py
Discrete MultiPhase Rot Grad P
"""
from hysop.operator.discrete.discrete import DiscreteOperator
import hysop.numerics.differential_operations as diff_op
from hysop.constants import debug, XDIR, YDIR, ZDIR, np
from hysop.methods_keys import SpaceDiscretisation
from hysop.numerics.update_ghosts import UpdateGhosts
from hysop.tools.profiler import ftime
import hysop.tools.numpywrappers as npw


class BaroclinicFromRHS(DiscreteOperator):
    """
    TODO : describe this operator ...
    """
    @debug
    def __init__(self, vorticity, rhs, **kwds):
        """
        Constructor.
        Create the baroclinic term in the N.S. equations with a given
        -GradRho/rho x GradP/rho term as the rhs field.
        @param vorticity : discretization of the vorticity field
        @param rhs : right hand side of the term
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        if 'method' not in kwds:
            import hysop.default_methods as default
            kwds['method'] = default.BAROCLINIC

        super(BaroclinicFromRHS, self).__init__(
            variables=[vorticity, rhs], **kwds)
        self.vorticity = vorticity
        self.rhs = rhs
        self.input = [self.vorticity, self.rhs]
        self.output = [self.vorticity]

        # prepare ghost points synchro for velocity (vector)
        # and density (scalar) fields
        self._synchronizeRHS = UpdateGhosts(self.rhs.topology,
                                            self.rhs.nb_components)

    @debug
    def apply(self, simulation=None):
        """Solves dw/dt = -RHS
        """
        if simulation is None:
            raise ValueError("Missing simulation value for computation.")

        dt = simulation.timeStep
        # Synchronize ghost points of velocity and density
        #self._synchronizeRHS(self.rhs.data)

        topo = self.vorticity.topology
        iCompute = topo.mesh.iCompute

        # vorti(n+1) = vorti(n) + dt * baroclinicTerm
        for d in xrange(self.vorticity.dimension):
            self.vorticity[d][iCompute] += self.rhs[d][iCompute] * dt
