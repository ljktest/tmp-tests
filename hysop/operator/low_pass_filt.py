# -*- coding: utf-8 -*-
"""Operator for any field low-pass filtering
   (--> computation of base flow).

.. currentmodule:: hysop.operator.low_pass_filt

"""

from hysop.operator.computational import Computational
from hysop.operator.discrete.low_pass_filt import LowPassFilt as DFilt
from hysop.operator.discrete.low_pass_filt import LowPassFiltConserv as DFiltCsv
from hysop.constants import debug
from hysop.operator.continuous import opsetup
import hysop.default_methods as default
from hysop.methods_keys import SpaceDiscretisation
from hysop.numerics.finite_differences import FD_C_4,\
    FD_C_2
from hysop.operator.differential import Curl
from hysop.fields.continuous import Field


class LowPassFilt(Computational):
    """
    Provides a filtered field by low-pass filtering
    the flow with the frequency of the instability divided by 2.
    i.e. solve
    \f{eqnarray*}
    \frac{\partial \bar{\omega}}{\partial t} &=& \Omega_c(\omega - \bar{\omega}))
    \f}

    """

    @debug
    def __init__(self, cutFreq=None, **kwds):
        """
        Parameters
        ----------
        @param cutFreq : cutting circular frequency corresponding to the half of
        the eigenfrequency of the flow instability
        **kwds : extra parameters for parent class

        """
        super(LowPassFilt, self).__init__(**kwds)

        ## cutting circular frequency
        self.cutFreq = cutFreq
        
        self.input = self.output = self.variables

    def discretize(self):
        super(LowPassFilt, self)._standard_discretize()
        # all variables must have the same resolution
        assert self._single_topo, 'multi-resolution case not allowed.'

    @debug
    def setup(self, rwork=None, iwork=None):
        self.discrete_op = DFilt(
            variables=self.discreteFields.values(),
            cutFreq=self.cutFreq,
            rwork=rwork, iwork=iwork)

        self._is_uptodate = True


class LowPassFiltConserv(LowPassFilt):
    """
        Provides a filtered field by low-pass filtering
        the flow with the frequency of the instability divided by 2.
        i.e. solve
        \f{eqnarray*}
        \frac{\partial \bar{\omega}}{\partial t} &=& \Omega_c(\omega - \bar{\omega}))
        \f}
        The equation is solved using a CONSERVATIVE formulation.
        
        """
    
    @debug
    def __init__(self, velocity, vorticityFilt, velocityFilt, **kwds):
        """
        Parameters
        ----------
        @param[in] velocity, vorticityFilt, velocityFilt fields
        @param[in, out] vorticityFilt field (i.e. filtered vorticity field)
        @param strength : strength of the filter
        **kwds : extra parameters for parent class
            
        Notes
        -----
        velocity and velocityFilt are not modified by this operator.
        vorticityFilt is an in-out parameter.
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(LowPassFiltConserv, self).__init__(
            variables=[velocity, vorticityFilt, velocityFilt], **kwds)

        ## velocity variable
        self.velocity = velocity
        ## filtered vorticity variable
        self.vorticityFilt = vorticityFilt
        ## filtered velocity variable
        self.velocityFilt = velocityFilt
        # A method is required to set how the curl will be computed.
        if self.method is None:
            self.method = default.DIFFERENTIAL
        # operator to compute buffer = curl(penalised velocity)
        self._curl = None

        self.input = [self.velocity, self.vorticityFilt, self.velocityFilt]
        self.output = [self.vorticityFilt]

    def discretize(self):
    
        if self.method[SpaceDiscretisation] is FD_C_4:
            # Finite differences method
            # Minimal number of ghost points
            nb_ghosts = 2
        elif self.method[SpaceDiscretisation] is FD_C_2:
            nb_ghosts = 1
        else:
            raise ValueError("Unknown method for space discretization of the\
                differential operator in penalization.")
        super(LowPassFiltConserv, self)._standard_discretize(nb_ghosts)
        # all variables must have the same resolution
        assert self._single_topo, 'multi-resolution case not allowed.'
        topo = self.variables[self.velocity]
        invar = Field(domain=self.velocity.domain,
                      name='curl_in', is_vector=True)
        dimension = self.domain.dimension
        outvar = Field(domain=self.velocity.domain,
                       name='curl_out',
                       is_vector=dimension == 3)
        self._curl = Curl(invar=invar, outvar=outvar,
                          discretization=topo, method=self.method)
        self._curl.discretize()

    def get_work_properties(self):
        return self._curl.get_work_properties()
    
    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        self._curl.setup(rwork, iwork)
        self.discrete_op = DFiltCsv(
            vorticityFilt=self.discreteFields[self.vorticityFilt],
            velocity=self.discreteFields[self.velocity],
            velocityFilt=self.discreteFields[self.velocityFilt],
            curl=self._curl.discrete_op,
            cutFreq=self.cutFreq,
            rwork=rwork, iwork=iwork)
        self._is_uptodate = True



