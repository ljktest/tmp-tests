# -*- coding: utf-8 -*-
"""Operators for penalization problem.

.. currentmodule:: hysop.operator.penalization

* :class:`Penalization` : standard penalisation
* :class:`PenalizeVorticity`  : vorticity formulation

See details in :ref:`penalisation` section of HySoP user guide.

"""

from hysop.operator.computational import Computational
from hysop.operator.discrete.penalization import Penalization as DPenalV
from hysop.operator.discrete.penalization import PenalizeVorticity as DPenalW
from hysop.constants import debug
from hysop.operator.continuous import opsetup
from hysop.domain.subsets import Subset
import hysop.default_methods as default
from hysop.methods_keys import SpaceDiscretisation
from hysop.numerics.finite_differences import FD_C_4,\
    FD_C_2
from hysop.operator.differential import Curl
from hysop.fields.continuous import Field


class Penalization(Computational):
    """
    Solves
    \f{eqnarray*}
    v = Op(\v)
    \f} with :
    \f{eqnarray*}
    \frac{\partial v}{\partial t} &=& \lambda\chi_s(v_D - v)
    \f}

    """

    @debug
    def __init__(self, obstacles, coeff=None, **kwds):
        """
        Parameters
        ----------
        obstacles : dict or list of :class:`~hysop.domain.subsets.Subset`
            sets of geometries on which penalization must be applied
        coeff : double, optional
            penalization factor applied to all geometries.
        **kwds : extra parameters for parent class

        Notes
        -----
        Set::

        obstacles = {obs1: coeff1, obs2: coeff2, ...}
        coeff = None

        to apply a different coefficient on each subset.
        Set::

        obstacles = [obs1, obs2, ...]
        coeff = some_value

        to apply the same penalization on all subsets.
        obs1, ob2 ... must be some :class:`~hysop.domain.subsets.Subset`
        and some_value must be either a real scalar or a function of the
        coordinates like::

            def coeff(*args):
                return 3 * args[0]

        with args[0,1,...] = x,y,...

        Warning : coeff as a function is not yet implemented!!
        """
        super(Penalization, self).__init__(**kwds)

        # The list of subset on which penalization must be applied
        self.obstacles = obstacles

        # Penalization functions or coef
        self.coeff = coeff
        self.input = self.output = self.variables

    def discretize(self):
        super(Penalization, self)._standard_discretize()
        # all variables must have the same resolution
        assert self._single_topo, 'multi-resolution case not allowed.'
        topo = self.variables.values()[0]
        for obs in self.obstacles:
            assert isinstance(obs, Subset)
            obs.discretize(topo)

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        self.discrete_op = DPenalV(
            variables=self.discreteFields.values(), obstacles=self.obstacles,
            coeff=self.coeff, rwork=rwork, iwork=iwork)

        self._is_uptodate = True


class PenalizeVorticity(Penalization):
    """
    Solve
    \f{eqnarray*}
    \frac{\partial w}{\partial t} &=& \lambda\chi_s\nabla\times(v_D - v)
    \f}
    using penalization.
    """

    @debug
    def __init__(self, velocity, vorticity, **kwds):
        """
        Parameters
        ----------
        velocity, vorticity: :class:`~hysop.fields.continuous.Field`
        **kwds : extra parameters for parent class.

        Notes
        -----
        velocity is not modified by this operator.
        vorticity is an in-out parameter.
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(PenalizeVorticity, self).__init__(
            variables=[velocity, vorticity], **kwds)
        # velocity field
        self.velocity = velocity
        # vorticity field
        self.vorticity = vorticity
        # A method is required to set how the curl will be computed.
        if self.method is None:
            self.method = default.DIFFERENTIAL
        # operator to compute buffer = curl(penalised velocity)
        self._curl = None
        self.input = self.variables
        self.output = self.vorticity

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
        super(PenalizeVorticity, self)._standard_discretize(nb_ghosts)
        # all variables must have the same resolution
        assert self._single_topo, 'multi-resolution case not allowed.'
        topo = self.variables[self.velocity]
        for obs in self.obstacles:
            assert isinstance(obs, Subset)
            obs.discretize(topo)
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
        self.discrete_op = DPenalW(
            vorticity=self.discreteFields[self.vorticity],
            velocity=self.discreteFields[self.velocity],
            curl=self._curl.discrete_op,
            obstacles=self.obstacles,
            coeff=self.coeff, rwork=rwork, iwork=iwork)
        self._is_uptodate = True
