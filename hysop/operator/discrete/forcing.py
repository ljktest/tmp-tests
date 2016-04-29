# -*- coding: utf-8 -*-
"""DOperator implementing the forcing term in the NS,
    depending on the filtered field
    (--> computation of base flow).
    
.. currentmodule:: hysop.operator.discrete.forcing

"""
from hysop.constants import debug
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.tools.profiler import profile

class Forcing(DiscreteOperator):
    """Discretized forcing operator.
    i.e. solve (with an implicit Euler scheme)
    \f{eqnarray*}
    \frac{\partial \omega}{\partial t} &=& -\chi(\omega - \bar{\omega}))
    \Leftrightarrow \omega^{n+1} &=&
    \frac{\omega^n + \Delta t \chi \bar{\omega^{n+1}}}{1+\Delta t \chi}
    \f}
    """

    @debug
    def __init__(self, strength=None, **kwds):
        """
        Parameters
        ----------
        strength : strength of the forcing, chosen in the order
        of the amplification rate related to the unstable flow.
        **kwds : extra parameters for parent class.

        """
        super(Forcing, self).__init__(**kwds)
        topo = self.variables[0].topology
        for v in self.variables:
            msg = 'Multiresolution not implemented for penalization.'
            assert v.topology == topo, msg
        
        ## variable
        self.var = self.variables[0]
        ## forced variable
        self.varFiltered = self.variables[1]
        ## strength of the forcing
        self.strength = strength

    def _apply(self, dt):
        nbc = self.var.nb_components
        for d in xrange(nbc):
            self.var[d][...] += self.varFiltered[d][...] * \
                                    (dt * self.strength)
            self.var[d][...] *= 1.0 / (1.0 + dt * self.strength)
        print 'forcing: non conservative formulation'

    def apply(self, simulation=None):
        assert simulation is not None, \
            "Simulation parameter is required."
        dt = simulation.timeStep
        self._apply(dt)


class ForcingConserv(Forcing):
    """Discretized forcing operator.
        i.e. solve (with an implicit Euler scheme and a CONSERVATIVE formulation)
        \f{eqnarray*}
        \frac{\partial \omega}{\partial t} &=& -\chi(\omega - \bar{\omega}))
        \Leftrightarrow \omega^{n+1} &=&
        \omega^n + \nabla \times (\frac{\Delta t \chi}{1+\Delta t \chi} (\bar{u^{n+1}}-u^n))
        \f}
        """

    @debug
    def __init__(self, vorticity, velocity, velocityFilt, curl, **kwds):
        """
        Parameters
        ----------
        velocity, vorticity, velocityFilt: :class:`~hysop.fields.continuous.Field`
        curl : :class:`~hysop.operator.differential`
        internal operator to compute the curl of the forced velocity
        **kwds : extra parameters for parent class.
            
        Notes
        -----
        velocity and velocityFilt are not modified by this operator.
        vorticity is an in-out parameter.
        input and ouput variables of the curl are some local buffers.
        """

        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(ForcingConserv, self).__init__(variables=[vorticity,
                                                        velocity,
                                                        velocityFilt],
                                             **kwds)
        # Input vector fields
        self.velocity = velocity
        self.vorticity = vorticity
        self.velocityFilt = velocityFilt
        # warning : a buffer is added for invar variable in curl
        topo = self.velocity.topology
        msg = 'Multiresolution not implemented for vort forcing.'
        assert self.vorticity.topology == topo, msg
        self._curl = curl
            
    def _apply(self, dt):
        # Vorticity forcing
        # warning : the buff0 array ensures "invar" to be 0
        # outside the obstacle for the curl evaluation
        invar = self._curl.invar
        nbc = invar.nb_components
        for d in xrange(nbc):
            invar.data[d][...] = 0.0
        coeff = (dt * self.strength) / (1.0 + dt * self.strength)
        for d in xrange(nbc):
            invar.data[d][...] = \
                (self.velocityFilt[d][...] -
                 self.velocity[d][...]) * coeff
        self._curl.apply()
        for d in xrange(self.vorticity.nb_components):
            self.vorticity[d][...] += self._curl.outvar[d][...]
        print 'forcing: conservative formulation'





