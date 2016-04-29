"""
@file odesolver.py

Abstract class for time integrators.
"""

from abc import ABCMeta, abstractmethod
from hysop.numerics.method import NumMethod
from hysop.constants import WITH_GUESS
from hysop.numerics.update_ghosts import UpdateGhosts
import hysop.tools.numpywrappers as npw


class ODESolver(NumMethod):
    """
    Abstract description for ODE solvers.
    Solve the system :
    \f[ \frac{dy(t)}{dt} = f(t,y) \f]

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, nb_components, work, topo, f=None, optim=None):
        """
        @param nb_components: number of components of the right-hand side.
        @param work : a list of numpy arrays used as work space.
        @param f : function f(t, y, f_work), right hand side of the equation
        to solve. f must have this signature.
        @remark - y argument in function f must be a list of numpy arrays
        @remark - work argument in function f is to pass a list working
        numpy arrays to the function. user-defined f function must have this
        argument even if it unused in f.
        @remark - f function must return a list of numpy arrays.
        @param optim : to choose the level of optimization (memory management).
        Default = None. See hysop.numerics.integrators for details.
        """
        ## RHS.
        self.f = f
        if f is None:
            self.f = lambda t, y, work: [npw.zeros_like(y[i])
                                         for i in xrange(nb_components)]
        if optim is None:
            self._fcall = self._basic
        elif optim is WITH_GUESS:
            self._fcall = self._core
        self._nb_components = nb_components

        self.work = work
        # Length of work arrays (float and int)
        lwork = self.getWorkLengths(nb_components)
        assert len(self.work) == lwork, 'Wrong length for work arrays list.'
        self._memshape = tuple(topo.mesh.resolution)
        print topo.mesh
        for wk in self.work:
            assert wk.shape == tuple(self._memshape)
        # Allocate buffers for ghost points synchronization.
        self._synchronize = UpdateGhosts(topo, self._nb_components)

    def __call__(self, t, y, dt, result):
        """
        @param t : current time.
        @param y : position at time t.
        \remark - y arg must be a list of numpy arrays.
        @param dt : time step.
        @param result : a predefined list of numpy arrays to solve the result
        @return y at t+dt.
        """
        return self._fcall(t, y, dt, result)

    def _basic(self, t, y, dt, result):
        """
        self.work is used as temp. array to compute the rhs.
        result provided as input arg.
        result may be equal to y.
        """
        self.work[:self._nb_components] = \
            self.f(t, y, self.work[:self._nb_components])
        return self._core(t, y, dt, result)

