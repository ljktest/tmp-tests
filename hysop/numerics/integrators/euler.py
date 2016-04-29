# -*- coding: utf-8 -*-
"""
@file euler.py

Forward Euler method.
"""
from hysop.numerics.integrators.odesolver import ODESolver
import numpy as np


class Euler(ODESolver):
    """
    ODESolver implementation for solving an equation system
    with forward Euler method.
    y'(t) = f(t,y)

    y(t_{n+1}) = y(t_n) + dt*f(y(t_n))

    """
    def __init__(self, nb_components, work, topo, f=None, optim=None):
        """
        @param f function f(t,y,f_work) : Right hand side of the equation to
        solve.
        @param nb_Components : number of components of the input field
        @param optim : to choose the level of optimization (memory management).
        Default = None. See hysop.numerics.integrators for details.
        """
        ODESolver.__init__(self, nb_components, work, topo, f=f, optim=optim)

    @staticmethod
    def getWorkLengths(nb_components, domain_dim=None):
        """
        Compute the number of required work arrays for this method.
        @param nb_components : number of components of the
        @param domain_dim : dimension of the domain
        fields on which this method operates.
        @return length of list of work arrays of reals.
        """
        return nb_components

    def _core(self, t, y, dt, result):
        """
        Computational core for Euler
        Highest level of optimization : result and self.work
        must be provided and work must contain a first evaluation of
        f(t,y)
        optim = WITH_GUESS
        """
        assert len(result) == self._nb_components
        # result = f(t, y)
        # result = y + dt * work0

        # result = y + work0
        [np.add(y[i], self.work[i] * dt, result[i])
         for i in xrange(self._nb_components)]

        return result
