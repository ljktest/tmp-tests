# -*- coding: utf-8 -*-
"""
@file runge_kutta4.py

RK4 method interface.
"""
from hysop.numerics.integrators.odesolver import ODESolver
import numpy as np


class RK4(ODESolver):
    """
    ODESolver implementation for solving an equation system with RK4 method.
    """
    def __init__(self, nb_components, work, topo, f=None, optim=None):
        """
        @param f function f(t,y,f_work) :
        Right hand side of the equation to solve.
        @param nb_components : dimensions
        @param optim : to choose the level of optimization (memory management).
        Default = None. See hysop.numerics.integrators for details.
        """
        ODESolver.__init__(self, nb_components, work, topo,
                           f=f, optim=optim)

    @staticmethod
    def getWorkLengths(nb_components, domain_dim=None):
        """
        Compute the number of required work arrays for this method.
        @param nb_components : number of components of the
        @param domain_dim : dimension of the domain
        fields on which this method operates.
        @return length of list of work arrays of reals.
        """
        return 3 * nb_components

    def _core(self, t, y, dt, result):
        """
        Computational core for RK4.
        """
        for i in xrange(self._nb_components):
            cond = [y[i] is self.work[j] for j in xrange(len(self.work))]
            assert cond.count(True) is 0
        assert len(result) == self._nb_components

        work0 = self.work[:self._nb_components]
        yn = self.work[self._nb_components:2 * self._nb_components]
        kn = self.work[2 * self._nb_components:3 * self._nb_components]

        # k1 = f(t,y)
        # k2 = f(t + dt/2, y + dt/2 * k1)
        # k3 = f(t + dt/2, y + dt/2 * k2)
        # k4 = f(t + dt, y + dt * k3)
        # result = y + dt/6( k1 + 2 * k2 + 2 * k3 + k4)

        # yn
        [np.add(y[i], work0[i] * dt / 2, yn[i])
         for i in xrange(self._nb_components)]
        # Update ghosts
        self._synchronize.apply(yn)
        # k2 in kn
        kn = self.f(t + dt / 2, yn, kn)

        # k1 + 2 * k2 in work0
        [np.add(work0[i],  2 * kn[i], work0[i])
         for i in xrange(self._nb_components)]

        # yn
        [np.add(y[i],  dt / 2 * kn[i], yn[i])
         for i in xrange(self._nb_components)]
        # Update ghosts
        self._synchronize.apply(yn)
        # k3 in kn
        kn = self.f(t + dt / 2, yn, kn)

        # k1 + 2 * k2 + 2 * k3 in work0
        [np.add(work0[i],  2 * kn[i], work0[i])
         for i in xrange(self._nb_components)]

        # yn
        [np.add(y[i],  dt * kn[i], yn[i])
         for i in xrange(self._nb_components)]
        # Update ghosts
        self._synchronize.apply(yn)
        # K4 in kn
        kn = self.f(t + dt, yn, kn)

        # k1 + 2 * k2 + 2 * k3 + k4
        [np.add(work0[i],  kn[i], work0[i])
         for i in xrange(self._nb_components)]
        [np.multiply(work0[i],  dt / 6, work0[i])
         for i in xrange(self._nb_components)]
        # result = y + work0
        [np.add(work0[i],  y[i], result[i])
         for i in xrange(self._nb_components)]

        return result
