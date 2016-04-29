# -*- coding: utf-8 -*-
"""Operator implementing the forcing term in the NS, 
   depending on the filtered field
   (--> computation of base flow).

.. currentmodule:: hysop.operator.forcing

"""

from hysop.operator.computational import Computational
from hysop.operator.discrete.forcing import Forcing as DForcing
from hysop.operator.discrete.forcing import ForcingConserv as DForcingCsv
from hysop.constants import debug
from hysop.operator.continuous import opsetup
import hysop.default_methods as default
from hysop.methods_keys import SpaceDiscretisation
from hysop.numerics.finite_differences import FD_C_4,\
    FD_C_2
from hysop.operator.differential import Curl
from hysop.fields.continuous import Field


class Forcing(Computational):
    """
    Integrate a forcing term in the right hand side 
    of the NS equations, depending on the filtered field.
    i.e. solve
    \f{eqnarray*}
    \frac{\partial \omega}{\partial t} &=& -\chi(\omega - \bar{\omega}))
    \f}
    The strength of the forcing is chosen in the order
    of the amplification rate related to the unstable flow.

    """

    @debug
    def __init__(self, strength=None, **kwds):
        """
        Parameters
        ----------
        @param strength : strength of the filter
        **kwds : extra parameters for parent class

        """
        super(Forcing, self).__init__(**kwds)

        ## strength of the filter
        self.strength = strength
        
        self.input = self.output = self.variables
#        self.output = [self.variables[0]]

    def discretize(self):
        super(Forcing, self)._standard_discretize()
        # all variables must have the same resolution
        assert self._single_topo, 'multi-resolution case not allowed.'

    @debug
    def setup(self, rwork=None, iwork=None):
        self.discrete_op = DForcing(
            variables=self.discreteFields.values(),
            strength=self.strength,
            rwork=rwork, iwork=iwork)

        self._is_uptodate = True


class ForcingConserv(Forcing):
    """
    Integrate a forcing term in the right hand side
    of the NS equations, depending on the filtered vorticity.
    i.e. solve
    \f{eqnarray*}
    \frac{\partial \omega}{\partial t} &=& -\chi(\omega - \bar{\omega}))
    \f}
    The equation is solved using a CONSERVATIVE formulation.
    The strength of the forcing is chosen in the order
    of the amplification rate related to the unstable flow.
        
    """
    @debug
    def __init__(self, velocity, vorticity, velocityFilt, **kwds):
        """
        Parameters
        ----------
        @param[in] velocity, vorticity, velocityFilt fields
        @param[in, out] forced vorticity field
        @param strength : strength of the filter
        **kwds : extra parameters for parent class
            
        Notes
        -----
        velocity and velocityFilt are not modified by this operator.
        vorticity is an in-out parameter.
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(ForcingConserv, self).__init__(
            variables=[velocity, vorticity, velocityFilt], **kwds)
        ## velocity variable
        self.velocity = velocity
        ## vorticity variable
        self.vorticity = vorticity
        ## filtered velocity variable
        self.velocityFilt = velocityFilt
        # A method is required to set how the curl will be computed.
        if self.method is None:
            self.method = default.DIFFERENTIAL
        # operator to compute buffer = curl(penalised velocity)
        self._curl = None

        self.input = [self.velocity, self.vorticity, self.velocityFilt]
        self.output = [self.vorticity]

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
        super(ForcingConserv, self)._standard_discretize(nb_ghosts)
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
        self.discrete_op = DForcingCsv(
            vorticity=self.discreteFields[self.vorticity],
            velocity=self.discreteFields[self.velocity],
            velocityFilt=self.discreteFields[self.velocityFilt],
            curl=self._curl.discrete_op,
            strength=self.strength,
            rwork=rwork, iwork=iwork)
        self._is_uptodate = True


