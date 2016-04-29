# -*- coding: utf-8 -*-
"""
Evaluation of the adaptative time step according to the flow fields.
"""

from hysop.constants import debug
from hysop.methods_keys import TimeIntegrator, SpaceDiscretisation,\
    dtCrit
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.numerics.differential_operations import GradV
import hysop.tools.numpywrappers as npw
from hysop.numerics.update_ghosts import UpdateGhosts
from hysop.mpi import MPI
from hysop.constants import np, HYSOP_MPI_REAL
from hysop.tools.profiler import profile


class AdaptTimeStep_D(DiscreteOperator):
    """
    The adaptative Time Step is computed according
    to the following expression :
    dt_adapt = min (dt_advection, dt_stretching, dt_cfl)
    """

    @debug
    def __init__(self, velocity, vorticity, simulation,
                 lcfl=0.125, cfl=0.5, time_range=None, maxdt=9999., **kwds):
        """
        @param velocity : discretization of the velocity field
        @param vorticity : discretization of the vorticity field
        @param dt_adapt : adaptative timestep
        (a hysop.variable_parameter.VariableParameter)
        @param lcfl : the lagrangian CFL coefficient used
        for advection stability
        @param cfl : the CFL coefficient.
        @param time_range : [start, end] use to define a 'window' in which
        the current operator is applied. Outside start-end, this operator
        has no effect. Start/end are iteration numbers.
        Default = [2, endofsimu]
        """
        ## velocity discrete field
        self.velocity = velocity
        ## vorticity discrete field
        self.vorticity = vorticity
        ## adaptative time step variable
        from hysop.problem.simulation import Simulation
        assert isinstance(simulation, Simulation)
        self.simulation = simulation
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(AdaptTimeStep_D, self).__init__(variables=[velocity, vorticity],
                                              **kwds)

        self.input = self.variables
        self.output = []
        ## Courant Fredrich Levy coefficient
        self.cfl = cfl
        ## Lagrangian CFL coefficient
        self.lcfl = lcfl
        ## Max. timestep
        self.maxdt = maxdt

        # Definition of criterion for dt_advec computation
        self.dtCrit = self.method[dtCrit]
        if not isinstance(self.dtCrit, list):
            self.dtCrit = [self.dtCrit]
        ## Time range
        if time_range is None:
            time_range = [2, np.infty]
        self.time_range = time_range

        # local buffer :
        # [time, dt, d1, d2, d3, d4, d5]
        # for d1...d5 see computation details in apply.
        self.diagnostics = npw.zeros((7))
        self._t_diagnostics = npw.zeros_like(self.diagnostics)

        # All diagnostcs function  definition:
        # (Index in self.diagnostics, function, is gradU needed)
        self._all_functions = {
            'gradU': (2, self._compute_gradU, True),
            'stretch': (3, self._compute_stretch, True),
            'cfl': (4, self._compute_cfl, False),
            'vort': (5, self._compute_vort, False),
            'deform': (6, self._compute_deform, True),
            }

        # Build the user required function list
        self._used_functions = []
        self._is_gradU_needed = False
        for crit in self.dtCrit:
            self._is_gradU_needed = \
                self._is_gradU_needed or self._all_functions[crit][-1]
            self._used_functions.append(self._all_functions[crit])
        assert len(self._used_functions) >= 1, "You must specify at least " + \
            "one criterion among: " + str(self._all_functions.keys())

        # Definition of dt:
        self.get_all_dt = []
        self._prepare_dt_list()

        # prepare ghost points synchro for velocity
        self._synchronize = UpdateGhosts(self.velocity.topology,
                                         self.velocity.nb_components)
        # gradU function
        self._function = GradV(topo=self.velocity.topology,
                               method=self.method[SpaceDiscretisation])

    def _set_work_arrays(self, rwork=None, iwork=None):
        memshape = self.velocity.data[0].shape
        worklength = self.velocity.nb_components ** 2
        # rwork is used to save gradU
        if rwork is None:
            self._rwork = [npw.zeros(memshape) for _ in xrange(worklength)]

        else:
            assert isinstance(rwork, list), 'rwork must be a list.'
            self._rwork = rwork
            assert len(self._rwork) == worklength
            for wk in self._rwork:
                assert wk.shape == memshape

    @staticmethod
    def _compute_stability_coeff(timeint):
        from hysop.numerics.integrators.euler import Euler
        from hysop.numerics.integrators.runge_kutta2 import RK2
        from hysop.numerics.integrators.runge_kutta3 import RK3
        from hysop.numerics.integrators.runge_kutta4 import RK4
        # Definition of stability coefficient for stretching operator
        coef_stretch = 0.0
        classtype = timeint.mro()[0]
        if classtype is Euler:
            coef_stretch = 2.0
        elif classtype is RK2:
            coef_stretch = 2.0
        elif classtype is RK3:
            coef_stretch = 2.5127
        elif classtype is RK4:
            coef_stretch = 2.7853
        return coef_stretch

    def _prepare_dt_list(self):
        # definition of dt_advection
        if 'gradU' in self.dtCrit:
            # => based on gradU
            self.get_all_dt.append(
                lambda diagnostics: self.lcfl / diagnostics[2])
        if 'deform' in self.dtCrit:
            # => based on the deformations
            self.get_all_dt.append(
                lambda diagnostics: self.lcfl / diagnostics[6])
        if 'vort' in self.dtCrit:
            # => based on the vorticity
            self.get_all_dt.append(
                lambda diagnostics: self.lcfl / diagnostics[5])
        if 'stretch' in self.dtCrit:
            coeff_stretch = self._compute_stability_coeff(
                self.method[TimeIntegrator])
            self.get_all_dt.append(
                lambda diagnostics: coeff_stretch / diagnostics[3])
        if 'cfl' in self.dtCrit:
            h = self.velocity.topology.mesh.space_step[0]
            self.get_all_dt.append(
                lambda diagnostics: (self.cfl * h) / diagnostics[4])

    def _gradU(self):
        # Synchronize ghost points of velocity
        self._synchronize(self.velocity.data)
        # gradU computation
        self._rwork = self._function(self.velocity.data, self._rwork)

    def _compute_gradU(self):
        res = 0.
        nb_components = self.velocity.nb_components
        for direc in xrange(nb_components):
            # maxima of partial derivatives of velocity :
            # needed for advection stability condition (1st option)
            res = max(res, np.max(abs(self._rwork[(nb_components + 1)
                                                  * direc])))
        return res

    def _compute_stretch(self):
        res = 0.
        nb_components = self.velocity.nb_components
        for direc in xrange(nb_components):
            # maxima of partial derivatives of velocity:
            # needed for stretching stability condition
            tmp = np.max(sum([abs(self._rwork[i])
                              for i in xrange(nb_components * direc,
                                              nb_components * (direc + 1))]))
            res = max(res, tmp)
        return res

    def _compute_cfl(self):
        # maxima of velocity : needed for CFL based time step
        return np.max([np.max(np.abs(v_c)) for v_c in self.velocity.data])

    def _compute_vort(self):
        # maxima of vorticity :
        # needed for advection stability condition (2nd option)
        return np.max([np.max(np.abs(w_c)) for w_c in self.vorticity.data])

    def _compute_deform(self):
        # 1/2(gradU + gradU^T) computation
        self._rwork[1] += self._rwork[3]
        self._rwork[2] += self._rwork[6]
        self._rwork[5] += self._rwork[7]
        self._rwork[1] *= 0.5
        self._rwork[2] *= 0.5
        self._rwork[5] *= 0.5
        self._rwork[3][...] = self._rwork[1][...]
        self._rwork[6][...] = self._rwork[2][...]
        self._rwork[7][...] = self._rwork[5][...]
        # maxima of deformation tensor:
        # needed for advection stability condition (3rd option)
        res = 0.
        nb_components = self.velocity.nb_components
        for direc in xrange(nb_components):
            tmp = np.max(sum([abs(self._rwork[i])
                              for i in xrange(nb_components * direc,
                                              nb_components * (direc + 1))
                              ]))
            res = max(res, tmp)
        return res

    @debug
    @profile
    def apply(self, simulation=None):
#        if simulation is not None:
#            assert self.simulation is simulation

        # current time
        time = self.simulation.time
        iteration = self.simulation.currentIteration
        Nmax = min(self.simulation.iterMax, self.time_range[1])
        self.diagnostics[0] = time
        if iteration >= self.time_range[0] and iteration <= Nmax:
            if self._is_gradU_needed:
                self._gradU()
            for func in self._used_functions:
                # func: (Index in self.diagnostics, function, is gradU needed)
                self.diagnostics[func[0]] = func[1]()

            self.velocity.topology.comm.Allreduce(
                sendbuf=[self.diagnostics, 7, HYSOP_MPI_REAL],
                recvbuf=[self._t_diagnostics, 7, HYSOP_MPI_REAL],
                op=MPI.MAX)
            self.diagnostics[...] = self._t_diagnostics

            dt = np.min([dt(self.diagnostics) for dt in self.get_all_dt] +
                        [self.maxdt])
            self.diagnostics[1] = dt
            if self._writer is not None and self._writer.do_write(iteration):
                self._writer.buffer[0, :] = self.diagnostics
                self._writer.write()

            # Update simulation time step with the new dt
            self.simulation.updateTimeStep(dt)
            # Warning this update is done only for the current MPI task!
            # See wait function in base class.
