"""
@file runge_kutta2.py

RK2 method interface.
"""
from hysop.numerics.integrators.odesolver import ODESolver
import numpy as np


class RK2(ODESolver):
    """
    ODESolver implementation for solving an equation system with RK2 method.
    y'(t) = f(t,y)

    y(t_n+1)= y(t_n) + dt*y'[y(t_n)+dt/2*y'(y(t_n))].

    """
    def __init__(self, dim, work, topo, f=None, optim=None):
        """
        @param dim, vector field (f) dimension (number of component)
        @param f function f(t, y) : Right hand side of the equation to
        solve.
        @param optim : to choose the level of optimization (memory management).
        Default = None. See hysop.numerics.integrators for details.
        """
        ODESolver.__init__(self, dim, work, topo, f=f, optim=optim)

    @staticmethod
    def getWorkLengths(nb_components, domain_dim=None):
        """
        Compute the number of required work arrays for this method.
        @param nb_components : number of components of the
        @param domain_dim : dimension of the domain
        fields on which this method operates.
        @return length of list of work arrays of reals.
        """
        return 2 * nb_components

    def _core(self, t, y, dt, result):
        """
        Computational core for RK2.
        self.work[:nb_components] must contain a first evaluation of
        f(t, y)
        optim = WITH_GUESS

        Note : since result may be equal to y, it can not
        be used as a temporary workspace.
        """
        for i in xrange(self._nb_components):
            cond = [y[i] is self.work[j] for j in xrange(len(self.work))]
            assert cond.count(True) is 0
            assert len(result) == self._nb_components

        work0 = self.work[:self._nb_components]
        yn = self.work[self._nb_components:2 * self._nb_components]

        # k1 = f(t,y) = work0
        # k2 = f(t + dt/2, y + dt/2 * k1)
        # result = y + dt * k2

        # yn
        [np.add(y[i], work0[i] * 0.5 * dt, yn[i])
         for i in xrange(self._nb_components)]
        # Update ghosts
        self._synchronize.apply(yn)
        # k2 in work0
        work0 = self.f(t + 0.5 * dt, yn, work0)
        # *= dt
        [np.multiply(work0[i],  dt, work0[i])
         for i in xrange(self._nb_components)]
        # result = y + work0
        [np.add(work0[i],  y[i], result[i])
         for i in xrange(self._nb_components)]

        return result
