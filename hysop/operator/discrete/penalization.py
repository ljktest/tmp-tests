# -*- coding: utf-8 -*-
"""Discrete operators for penalization problem.
.. currentmodule:: hysop.operator.discrete.penalization
* :class:`Penalization` : standard penalisation
* :class:`PenalizeVorticity`  : vorticity formulation

"""
from hysop.constants import debug
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.tools.profiler import profile
from hysop.domain.subsets import Subset


class Penalization(DiscreteOperator):
    """Discretized penalisation operator.
    """

    @debug
    def __init__(self, obstacles, coeff=None, **kwds):
        """
        Parameters
        ----------
        obstacles : dictionnary or list of `~hysop.domain.subsets.Subset`
            sets of geometries on which penalization must be applied
        coeff : double, optional
            penalization factor applied to all geometries.

        """
        super(Penalization, self).__init__(**kwds)

        topo = self.variables[0].topology
        # indices of points of grid on which penalization is applied.
        # It may be a single condition (one penal coeff for all subsets)
        # or a list of conditions (one different coeff for each subset).
        self._cond = None
        if isinstance(obstacles, list):
            msg = 'A penalization factor is required for the obstacles.'
            assert coeff is not None, msg
            self._coeff = coeff
            self._cond = self._init_single_coeff(obstacles, topo)
            self._apply = self._apply_single_coeff

        elif isinstance(obstacles, dict):
            # cond is a dictionnary, key = list of indices,
            # value = penalization coeff
            self._cond, self._coeff = self._init_multi_coeff(obstacles, topo)
            self._apply = self._apply_multi_coeff

        for v in self.variables:
            msg = 'Multiresolution not implemented for penalization.'
            assert v.topology == topo, msg

        # list of numpy arrays to penalize
        self._varlist = []
        for v in self.variables:
            for d in xrange(v.nb_components):
                self._varlist.append(v[d])

    def _init_single_coeff(self, obstacles, topo):
        """
        Compute a condition which represents the union
        of all obstacles.
        """
        msg = 'Warning : you use a porous obstacle but apply the same'
        msg += ' penalisation factor everywhere.'
        for _ in [obs for obs in obstacles if obs.is_porous]:
            print msg
        assert isinstance(obstacles, list)
        return Subset.union(obstacles, topo)

    def _init_multi_coeff(self, obstacles, topo):
        """
        Compute a condition which represents the union
        of all obstacles.
        """
        cond = []
        coeff = []
        for obs in obstacles:
            if obs.is_porous:
                assert isinstance(obstacles[obs], list)
                current = obs.ind[topo]
                nb_layers = len(current)
                assert len(current) == nb_layers
                for i in xrange(nb_layers):
                    # append the list of indices
                    cond.append(current[i])
                    # and its corresponding coeff
                    coeff.append(obstacles[obs][i])
            else:
                cond.append(obs.ind[topo][0])
                coeff.append(obstacles[obs])
        return cond, coeff

    @debug
    @profile
    def _apply_single_coeff(self, dt):
        coef = 1.0 / (1.0 + dt * self._coeff)
        for v in self._varlist:
            v[self._cond] *= coef

    def _apply_multi_coeff(self, dt):
        for i in xrange(len(self._cond)):
            coef = 1.0 / (1.0 + dt * self._coeff[i])
            cond = self._cond[i]
            for v in self._varlist:
                v[cond] *= coef

    def apply(self, simulation=None):
        assert simulation is not None, \
            "Simulation parameter is required."
        dt = simulation.timeStep
        self._apply(dt)


class PenalizeVorticity(Penalization):
    """
    Discretized penalisation operator.
    See details in hysop.operator.penalization
    """

    @debug
    def __init__(self, vorticity, velocity, curl, **kwds):
        """
        Parameters
        ----------
        velocity, vorticity: :class:`~hysop.fields.continuous.Field`
        curl : :class:`~hysop..operator.differential`
            internal operator to compute the curl of the penalised velocity
        **kwds : extra parameters for parent class.

        Notes
        -----
        velocity is not modified by this operator.
        vorticity is an in-out parameter.
        input and ouput variables of the curl are some local buffers.
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(PenalizeVorticity, self).__init__(variables=[vorticity,
                                                           velocity],
                                                **kwds)
        self.velocity = velocity
        self.vorticity = vorticity
        # warning : a buffer is added for invar variable in curl
        topo = self.velocity.topology
        msg = 'Multiresolution not implemented for penalization.'
        assert self.vorticity.topology == topo, msg
        self._curl = curl

    def _apply_single_coeff(self, dt):
        # Vorticity penalization
        # warning : the buff0 array ensures "invar" to be 0
        # outside the obstacle for the curl evaluation
        invar = self._curl.invar
        nbc = invar.nb_components
        for d in xrange(nbc):
            invar.data[d][...] = 0.0
        coeff = -dt * self._coeff / (1.0 + dt * self._coeff)
        for d in xrange(nbc):
            invar.data[d][self._cond] = \
                self.velocity[d][self._cond] * coeff
        self._curl.apply()
        for d in xrange(self.vorticity.nb_components):
            self.vorticity[d][...] += self._curl.outvar[d][...]

    def _apply_multi_coeff(self, dt):
        invar = self._curl.invar
        nbc = invar.nb_components

        for d in xrange(nbc):
            invar.data[d][...] = 0.0

        for i in xrange(len(self._cond)):
            coeff = -dt * self._coeff[i] / (1.0 + dt * self._coeff[i])
            cond = self._cond[i]
            for d in xrange(nbc):
                invar.data[d][cond] = self.velocity[d][cond] * coeff

        self._curl.apply()

        for d in xrange(self.vorticity.nb_components):
            self.vorticity[d][...] += self._curl.outvar[d][...]
