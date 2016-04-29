"""
@file runge_kutta3.py

RK3 method interface.
"""
from hysop.numerics.integrators.odesolver import ODESolver
import numpy as np


class RK3(ODESolver):
    """
    ODESolver implementation for solving an equation system with RK3 method.
    """
    def __init__(self, nb_components, work, topo, f=None, optim=None):
        """
        @param f function f(t,y): Right hand side of the equation to
        solve.
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
        Highest level of optimization : work and result
        must be provided and
        work[:nb_components] must contain a first evaluation of
        f(t,y,f_work)
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
        kn = self.work[2 * self._nb_components:3 * self._nb_components]

        # k1 = f(t,y) = work0
        # k2 = f(t+dt/3, y + dt/3 *k1))
        # k3 = f(t + 2/3 *dt , y + 2/3 dt * k2))
        # result = y + 0.25 * dt * (k1 + 3 * k3)
        # yn
        [np.add(y[i], work0[i] * dt / 3, yn[i])
         for i in xrange(self._nb_components)]
        # Update ghosts
        self._synchronize.apply(yn)
        # k2 in kn
        kn = self.f(t + dt / 3, yn, kn)
        # yn
        [np.add(y[i],  2 * dt / 3 * kn[i], yn[i])
         for i in xrange(self._nb_components)]
        # Update ghosts
        self._synchronize.apply(yn)
        # k3 in kn
        kn = self.f(t + 2 * dt / 3, yn, kn)
        # k1 + 3 * k3 in work0
        [np.add(work0[i],  3 * kn[i], work0[i])
         for i in xrange(self._nb_components)]
        # *= dt / 4
        [np.multiply(work0[i],  dt / 4, work0[i])
         for i in xrange(self._nb_components)]
        # result = y + work0
        [np.add(work0[i],  y[i], result[i])
         for i in xrange(self._nb_components)]
        #self._synchronize.apply(result)

        return result
