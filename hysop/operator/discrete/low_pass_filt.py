# -*- coding: utf-8 -*-
"""Discrete operator for vorticity low-pass filtering
    (--> computation of base flow).
.. currentmodule:: hysop.operator.discrete.low_pass_filt_vort

"""
from hysop.constants import debug
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.tools.profiler import profile

class LowPassFilt(DiscreteOperator):
    """Discretized vorticity filtering operator.
    i.e. solve (with an explicit Euler scheme)
    \f{eqnarray*}
    \frac{\partial \bar{\omega}}{\partial t} &=& \Omega_c(\omega - \bar{\omega}))
    \Leftrightarrow \bar{\omega^{n+1}} &=& \bar{\omega^{n}} +
    \Delta t \Omega_c (\omega^n - \bar{\omega^n})
    \Leftrightarrow \bar{\omega^{n+1}} &=& 
    \bar{\omega^{n}} (1 - \Delta t \Omega_c) + 
    \Delta t \Omega_c \omega^n
    \f}
    """

    @debug
    def __init__(self, cutFreq=None, **kwds):
        """
        Parameters
        ----------
        cutFreq : cutting circular frequency corresponding to the half of 
        the eigenfrequency of the flow instability
        **kwds : extra parameters for parent class.

        """
        super(LowPassFilt, self).__init__(**kwds)
        topo = self.variables[0].topology
        for v in self.variables:
            msg = 'Multiresolution not implemented for penalization.'
            assert v.topology == topo, msg
        
        ## variable
        self.var = self.variables[0]
        ## filtered variable
        self.varFiltered = self.variables[1]
        ## cutting circular frequency
        self.cutFreq = cutFreq
            
    def _apply(self, dt):
        nbc = self.varFiltered.nb_components
        coeff = dt * self.cutFreq
        for d in xrange(nbc):
            self.varFiltered[d][...] *= (1.0 - coeff)
            self.varFiltered[d][...] += self.var[d][...] * coeff
        print 'filtering: non conservative formulation'

    def apply(self, simulation=None):
        assert simulation is not None, \
            "Simulation parameter is required."
        dt = simulation.timeStep
        self._apply(dt)


class LowPassFiltConserv(LowPassFilt):
    """Discretized vorticity filtering operator.
    i.e. solve (with an explicit Euler scheme and a CONSERVATIVE formulation)
    \f{eqnarray*}
    \frac{\partial \bar{\omega}}{\partial t} &=& \Omega_c(\omega - \bar{\omega}))
    \Leftrightarrow \bar{\omega^{n+1}} &=& \bar{\omega^{n}} +
    \nabla \times (\Delta t \Omega_c (u^n - \bar{u^n}))
    \f}
    """
    
    @debug
    def __init__(self, vorticityFilt, velocity, velocityFilt, curl, **kwds):
        """
        Parameters
        ----------
        vorticityFilt, vorticity, velocityFilt: :class:`~hysop.fields.continuous.Field`
        curl : :class:`~hysop.operator.differential`
        internal operator to compute the curl of the forced velocity
        **kwds : extra parameters for parent class.
            
        Notes
        -----
        velocity and velocityFilt are not modified by this operator.
        vorticityFilt is an in-out parameter.
        input and ouput variables of the curl are some local buffers.
        """

        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(LowPassFiltConserv, self).__init__(variables=[vorticityFilt,
                                                            velocity,
                                                            velocityFilt],
                                                 **kwds)
        # Input vector fields
        self.vorticityFilt = vorticityFilt
        self.velocity = velocity
        self.velocityFilt = velocityFilt
        # warning : a buffer is added for invar variable in curl
        topo = self.velocity.topology
        msg = 'Multiresolution not implemented for vort forcing.'
        assert self.vorticityFilt.topology == topo, msg
        assert self.velocityFilt.topology == topo, msg
        self._curl = curl

    def _apply(self, dt):
        # Vorticity filtering
        # warning : the buff0 array ensures "invar" to be 0
        # outside the obstacle for the curl evaluation
        invar = self._curl.invar
        nbc = invar.nb_components
        for d in xrange(nbc):
            invar.data[d][...] = 0.0
        coeff = dt * self.cutFreq
        for d in xrange(nbc):
            invar.data[d][...] = \
                (self.velocity[d][...] -
                 self.velocityFilt[d][...]) * coeff
        self._curl.apply()
        for d in xrange(self.vorticityFilt.nb_components):
            self.vorticityFilt[d][...] += self._curl.outvar[d][...]
        print 'filtering: conservative formulation'


