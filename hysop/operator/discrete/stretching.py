# -*- coding: utf-8 -*-
"""Discretisation of the stretching operator.

Formulations:

* :class:`~hysop.operator.discrete.stretching.Conservative`
* :class:`~hysop.operator.discrete.stretching.GradUW`
* :class:`~hysop.operator.discrete.stretching.Symmetric`

"""

from hysop.constants import debug, WITH_GUESS
from hysop.methods_keys import TimeIntegrator, SpaceDiscretisation
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.numerics.integrators.euler import Euler
from hysop.numerics.integrators.runge_kutta2 import RK2
from hysop.numerics.integrators.runge_kutta3 import RK3
from hysop.numerics.integrators.runge_kutta4 import RK4
import hysop.numerics.differential_operations as diff_op
import hysop.tools.numpywrappers as npw
from hysop.numerics.update_ghosts import UpdateGhosts
from hysop.tools.profiler import profile
from abc import ABCMeta, abstractmethod
import math
ceil = math.ceil


class Stretching(DiscreteOperator):
    """
    Abstract interface to stretching discrete operators.
    Three formulations are available:
    - conservative, see operator.discrete.stretching.Conservative
    - 'Grad(UxW), see operator.discrete.stretching.GradUW
    - Symmetric, see operator.discrete.stretching.Symmetric
    """

    __metaclass__ = ABCMeta

    @debug
    def __init__(self, velocity, vorticity, formulation, rhs, **kwds):
        """Abstract interface for stretching operator

        Parameters
        ----------
        velocity, vorticity : :class:`~hysop.fields.discrete.DiscreteField`
        formulation : one of the discrete stretching classes.
        **kwds : extra parameters for base class

        """
        # velocity discrete field
        self.velocity = velocity
        # vorticity discrete field
        self.vorticity = vorticity
        # Formulation for stretching (divWV or GradVxW)
        self.formulation = formulation

        if 'method' not in kwds:
            import hysop.default_methods as default
            kwds['method'] = default.STRETCHING
        # Work vector used by time-integrator
        self._ti_work = None
        # Work vector used by numerical diff operator.
        self._str_work = None
        super(Stretching, self).__init__(variables=[self.velocity,
                                                    self.vorticity], **kwds)

        self.input = self.variables
        self.output = [self.vorticity]
        # \todo multiresolution case
        assert self.velocity.topology.mesh == self.vorticity.topology.mesh,\
            'Multiresolution case not yet implemented.'

        # Number of components of the operator (result)
        self.nb_components = 3  # Stretching only in 3D and for vector fields.

        # prepare ghost points synchro for velocity and vorticity
        self._synchronize = UpdateGhosts(self.velocity.topology,
                                         self.velocity.nb_components
                                         + self.vorticity.nb_components)

        # A function to compute the gradient of a vector field
        # Work vector is provided in input.
        self.strFunc = \
            self.formulation(topo=self.velocity.topology,
                             work=self._str_work,
                             method=self.method[SpaceDiscretisation])

        # Time integrator
        self.timeIntegrator = \
            self.method[TimeIntegrator](self.nb_components,
                                        self._ti_work,
                                        self.velocity.topology,
                                        f=rhs,
                                        optim=WITH_GUESS)

    def _set_work_arrays(self, rwork=None, iwork=None):

        shape_v = self.velocity.data[0][...].shape
        ti = self.method[TimeIntegrator]
        # work list length for time-integrator
        work_length_ti = ti.getWorkLengths(3)
        rwork_length = work_length_ti + self.formulation.get_work_length()
        # setup for rwork, iwork is useless.
        if rwork is None:
            # ---  Local allocation ---
            self._rwork = []
            for _ in xrange(rwork_length):
                self._rwork.append(npw.zeros(shape_v))
        else:
            assert isinstance(rwork, list), 'rwork must be a list.'
            # --- External rwork ---
            self._rwork = rwork
            assert len(self._rwork) == rwork_length
            for wk in rwork:
                assert wk.shape == shape_v
        self._ti_work = self._rwork[:work_length_ti]
        self._str_work = self._rwork[work_length_ti:]

    @profile
    def update_ghosts(self):
        """
        Update ghost points values
        """
        self._synchronize(self.velocity.data + self.vorticity.data)
    
    def _apply(self, t, dt):
        # Synchronize ghost points
        self._synchronize(self.velocity.data + self.vorticity.data)
        # Compute stretching term (rhs) and update vorticity
        self._compute(dt, t)

    def apply(self, simulation=None):
        """
            """
        assert simulation is not None, \
            "Missing simulation value for computation."
        # current time
        t = simulation.time
        # time step
        dt = simulation.timeStep
        self._apply(t, dt)

#    def apply(self, simulation=None):
#        """
#        """
#        assert simulation is not None, \
#            "Missing simulation value for computation."
#
#        # time step
#        dt = simulation.timeStep
#        # current time
#        t = simulation.time
#
#        # Synchronize ghost points of velocity
#        self._synchronize(self.velocity.data + self.vorticity.data)
#        self._compute(dt, t)

    @abstractmethod
    def _compute(self, dt, t):
        """
        """

    @profile
    def _integrate(self, dt, t):
        # - Call time integrator -
        # Init workspace with a first evaluation of the
        # rhs of the integrator
        self._ti_work[:self.nb_components] = \
            self.timeIntegrator.f(t, self.vorticity.data,
                                  self._ti_work[:self.nb_components])
        # perform integration and save result in-place
        self.vorticity.data = self.timeIntegrator(t, self.vorticity.data, dt,
                                                  result=self.vorticity.data)


class Conservative(Stretching):
    """Conservative formulation
    """

    @profile
    def __init__(self, **kwds):

        # Right-hand side for time integration
        def rhs(t, y, result):
            result = self.strFunc(y, self.velocity.data, result)
            return result

        super(Conservative, self).__init__(formulation=diff_op.DivWV,
                                           rhs=rhs, **kwds)

    @profile
    def _compute(self, dt, t):
        # No subcycling for this formulation
        self._integrate(dt, t)


class GradUW(Stretching):
    """GradUW formulation
    """

    def __init__(self, **kwds):
        # a vector to save diagnostics computed from GradVxW (max div ...)
        self.diagnostics = npw.ones(2)

        def rhs(t, y, result):
            result, self.diagnostics =\
                self.strFunc(self.velocity.data, y, result, self.diagnostics)
            return result

        super(GradUW, self).__init__(formulation=diff_op.GradVxW,
                                     rhs=rhs, **kwds)

        # stability constant
        self.cststretch = 0.
        # Depends on time integration method
        timeint = self.method[TimeIntegrator]
        classtype = timeint.mro()[0]
        if classtype is Euler:
            self.cststretch = 2.0
        elif classtype is RK2:
            self.cststretch = 2.0
        elif classtype is RK3:
            self.cststretch = 2.5127
        elif classtype is RK4:
            self.cststretch = 2.7853

    @debug
    @profile
    def _compute(self, t, dt):
        # Compute the number of required subcycles
        ndt, subdt = self._check_stability(dt)
        assert sum(subdt) == dt

        for i in xrange(ndt):
            self._integrate(t, subdt[i])

    def _check_stability(self, dt):
        """Computes a stability condition depending on some
        diagnostics (from GradVxW)

        :param dt: current time step

        Returns
        --------
        nb_cylces : int
            the number of required subcycles and
        subdt : array of float
            the subcycles time-step.
        """
        dt_stab = min(dt, self.cststretch / self.diagnostics[1])
        nb_cycles = int(ceil(dt / dt_stab))
        subdt = npw.zeros((nb_cycles))
        subdt[:] = dt_stab
        subdt[-1] = dt - (nb_cycles - 1) * dt_stab
        return nb_cycles, subdt


class StretchingLinearized(Stretching):
    """By default: Conservative formulation
    """
    
    def __init__(self, vorticity_BF, usual_op, **kwds):
        
        # vorticity of the base flow (steady solution)
        self.vorticity_BF = vorticity_BF
        # prepare ghost points synchro for vorticity_BF
        self._synchronize_vort_BF = \
            UpdateGhosts(self.vorticity_BF.topology,
                         self.vorticity_BF.nb_components)
        self.usual_op = usual_op

        # Right-hand side for time integration
        def rhs(t, y, result, form):
            if form == "div(w:u)":
                result = self.strFunc(y, self.velocity.data, result)
            else:
                result = self.strFunc(y, self.vorticity_BF.data, result)
            return result
        
        super(StretchingLinearized, self).__init__(formulation=diff_op.DivWV,
                                                   rhs=rhs, **kwds)
    
    def _integrate(self, dt, t):
        # - Call time integrator (1st term over 3) -
        # Init workspace with a first evaluation of the div(wb:u') term in the
        # rhs of the integrator
        self._ti_work[:self.nb_components] = \
            self.timeIntegrator.f(t, self.vorticity_BF.data,
                                  self._ti_work[:self.nb_components], "div(w:u)")
        # perform integration and save result in-place
        self.vorticity.data = self.timeIntegrator(t, self.vorticity.data, dt,
                                                  result=self.vorticity.data)
        # - Call time integrator (2nd term over 3) -
        # Init workspace with a first evaluation of the div(u':wb) term in the
        # rhs of the integrator
        self._ti_work[:self.nb_components] = \
            self.timeIntegrator.f(t, self.velocity.data,
                                  self._ti_work[:self.nb_components], "div(u:w)")
        # perform integration and save result in-place
        self.vorticity.data = self.timeIntegrator(t, self.vorticity.data, dt,
                                                  result=self.vorticity.data)

    def _compute(self, dt, t):
        # No subcycling for this formulation
        self._integrate(dt, t)

    def _apply(self, t, dt):
        # Synchronize ghost points
        self._synchronize(self.velocity.data + self.vorticity.data)
        self._synchronize_vort_BF(self.vorticity_BF.data)
        # Compute the 2 first "stretching" terms (div(wb:u') and div(u':wb))
        # and update vorticity for each of them
        self._compute(dt, t)
        # Compute the 3rd stretching term (div(w':ub)) and update vorticity
        self.usual_op._apply(t, dt)

