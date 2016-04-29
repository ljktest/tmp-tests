# -*- coding: utf-8 -*-
"""
@file operator/discrete/multiphase_gradp.py
Discrer operator of the pressure gradient in a multiphasic flow.
"""
from hysop.operator.discrete.discrete import DiscreteOperator
import hysop.numerics.differential_operations as diff_op
from hysop.constants import debug, XDIR, YDIR, ZDIR, np
from hysop.methods_keys import SpaceDiscretisation
from hysop.numerics.update_ghosts import UpdateGhosts
import hysop.tools.numpywrappers as npw
from hysop.operator.discrete.discrete import get_extra_args_from_method


class GradP(DiscreteOperator):
    """
    TODO : describe this operator ...
    """
    @debug
    def __init__(self, velocity, gradp, viscosity,
                 formula=None, **kwds):
        """
        Constructor.
        Compute the pressure gradient term - GradP/rho
        in N.S equation from velocity.
        @param velocity : discretization of the velocity field
        @param gradp : discretization of the result
        @param viscosity
        @param formula : formula to initialize u^(n-1)
        Note : this should be the formula used to initialize
        the velocity field
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        if 'method' not in kwds:
            import hysop.default_methods as default
            kwds['method'] = default.MULTIPHASEGRADP

        super(GradP, self).__init__(
            variables=[velocity, gradp], **kwds)
        self.velocity = velocity
        self.gradp = gradp
        self.viscosity = viscosity
        self.input = [self.velocity, ]
        self.output = [self.gradp, ]

        # prepare ghost points synchro for velocity (vector)
        # and density (scalar) fields
        self._synchronizeVel = UpdateGhosts(self.velocity.topology,
                                            self.velocity.nb_components)

        self._result = [npw.zeros_like(d) for d in self.velocity.data]
        self._tempGrad = [npw.zeros_like(d) for d in self.velocity.data]
        self._baroclinicTerm = [npw.zeros_like(d) for d in self.velocity.data]

        self._laplacian = diff_op.Laplacian(
            self.velocity.topology,
            indices=self.velocity.topology.mesh.iCompute)
        self._gradOp = diff_op.GradS(
            self.velocity.topology,
            indices=self.velocity.topology.mesh.iCompute,
            method=self.method[SpaceDiscretisation])

        # Gravity vector
        self._gravity = npw.asrealarray(
            get_extra_args_from_method(self, 'gravity', [0., 0., -9.81]))

        # Time stem of the previous iteration
        self._old_dt = None

    def initialize_velocity(self):
        """Initialize the temporary array 'result' with the velocity"""
        topo = self.velocity.topology
        iCompute = topo.mesh.iCompute
        for d in xrange(self.velocity.dimension):
            self._result[d][iCompute] = -self.velocity[d][iCompute]

    @debug
    def apply(self, simulation=None):
        """Computes the pressure gradient term: -grad(P)/rho
        -grad(P)/rho =  du/dt + (u . grad)u - nu laplacien(u) - g
        """
        if simulation is None:
            raise ValueError("Missing simulation value for computation.")

        dt = simulation.timeStep
        if self._old_dt is None:
            self._old_dt = dt
        # Synchronize ghost points of velocity and density
        self._synchronizeVel(self.velocity.data)

        topo = self.velocity.topology
        iCompute = topo.mesh.iCompute

        # result = du/dt = (u^(n)-u^(n-1))/dt
        # result has been initialized with -u^(n-1)
        # and _old_dt equals to the previous dt
        for d in xrange(self.velocity.dimension):
            self._result[d][iCompute] += self.velocity[d][iCompute]
            self._result[d][iCompute] /= self._old_dt

        # result = result + (u . grad)u
        # (u. grad)u = (u.du/dx + v.du/dy + w.du/dz ;
        #               u.dv/dx + v.dv/dy + w.dv/dz ;
        #               u.dw/dx + v.dw/dy + w.dw/dz)
        # Add (u. grad)u components directly in result
        self._tempGrad = self._gradOp(
            self.velocity[XDIR:XDIR + 1], self._tempGrad)
        # result[X] = result[X] + ((u. grad)u)[X]
        #           = result[X] + u.du/dx + v.du/dy + w.du/dz
        for d in xrange(self.velocity.dimension):
            self._result[XDIR][iCompute] += \
                self.velocity[d][iCompute] * self._tempGrad[d][iCompute]

        self._tempGrad = self._gradOp(
            self.velocity[YDIR:YDIR + 1], self._tempGrad)
        # result[Y] = result[Y] + ((u. grad)u)[Y]
        #           = result[Y] + u.dv/dx + v.dv/dy + w.dv/dz
        for d in xrange(self.velocity.dimension):
            self._result[YDIR][iCompute] += \
                self.velocity[d][iCompute] * self._tempGrad[d][iCompute]

        self._tempGrad = self._gradOp(
            self.velocity[ZDIR:ZDIR + 1], self._tempGrad)
        # result[Z] = result[Z] + ((u. grad)u)[Z]
        #           = result[Z] + u.dw/dx + v.dw/dy + w.dw/dz
        for d in xrange(self.velocity.dimension):
            self._result[ZDIR][iCompute] += \
                self.velocity[d][iCompute] * self._tempGrad[d][iCompute]

        # result = result - nu*\Laplacian u (-g) = gradP/rho
        for d in xrange(self.velocity.dimension):
            self._tempGrad[d:d + 1] = self._laplacian(
                self.velocity[d:d + 1], self._tempGrad[d:d + 1])
        for d in xrange(self.velocity.dimension):
            self._tempGrad[d][iCompute] *= self.viscosity
            self._result[d][iCompute] -= self._tempGrad[d][iCompute]

        # gravity term : result = result - g
        for d in xrange(self.velocity.dimension):
            self._result[2][iCompute] -= self._gravity[d]

        for d in xrange(self.velocity.dimension):
            self.gradp[d][...] = self._result[d]

        # reinitialise for next iteration
        # velo(n-1) update
        for d in xrange(self.velocity.dimension):
            self._result[d][iCompute] = -self.velocity.data[d][iCompute]
        self._old_dt = dt
