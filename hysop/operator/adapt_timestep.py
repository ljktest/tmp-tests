# -*- coding: utf-8 -*-
"""
@file operator/adapt_timestep.py

Definition of the adaptative time step according to the flow fields.

"""
from hysop.constants import debug
from hysop.methods_keys import TimeIntegrator, SpaceDiscretisation,\
    dtCrit
from hysop.numerics.finite_differences import FD_C_4
from hysop.operator.discrete.adapt_timestep import AdaptTimeStep_D
from hysop.operator.continuous import opsetup
from hysop.operator.computational import Computational
import hysop.default_methods as default
from hysop.mpi import main_comm, MPI


class AdaptTimeStep(Computational):
    """
    The adaptative Time Step is computed according
    to the following expression :
    dt_adapt = min (dt_advection, dt_stretching, dt_cfl)
    """

    @debug
    def __init__(self, velocity, vorticity, simulation,
                 time_range=None, lcfl=0.125, cfl=0.5,
                 maxdt=9999., **kwds):
        """
        Create a timeStep-evaluation operator from given
        velocity and vorticity variables.

        Note : If cfl is None, the computation of the adaptative time step
        is only based on dt_advection and dt_stretching, taking the minimum
        value of the two.

        @param velocity field
        @param vorticity field
        @param dt_adapt : adaptative timestep variable
        @param time_range : [start, end] use to define a 'window' in which
        the current operator is applied. Outside start-end, this operator
        has no effect. Start/end are iteration numbers.
        Default = [2, endofsimu]
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(AdaptTimeStep, self).__init__(variables=[velocity, vorticity],
                                            **kwds)
        if self.method is None:
            self.method = default.ADAPT_TIME_STEP
        assert SpaceDiscretisation in self.method.keys()
        assert TimeIntegrator in self.method.keys()
        if dtCrit not in self.method.keys():
            self.method[dtCrit] = 'vort'

        ## velocity variable (vector)
        self.velocity = velocity
        ## vorticity variable (vector)
        self.vorticity = vorticity
        ## adaptative time step variable ("variable" object)
        self.simulation = simulation
        #assert isinstance(self.dt_adapt, VariableParameter)
        # Check if 'dt' key is present in dt_adapt dict
        #assert 'dt' in self.dt_adapt.data

        self.input = self.variables
        self.output = []
        self.time_range = time_range
        self.lcfl, self.cfl, self.maxdt = lcfl, cfl, maxdt
        self._intercomms = {}
        self._set_inter_comm()

    def _set_inter_comm(self):
        """
        Create intercommunicators, if required (i.e. if there are several
        tasks defined in the domain).
        """
        task_is_source = self._mpis.task_id == self.domain.current_task()
        tasks_list = self.domain.tasks_list()
        others = (v for v in tasks_list if v != self._mpis.task_id)
        if task_is_source:
            remote_leader = set([tasks_list.index(i) for i in others])
        else:
            remote_leader = set([tasks_list.index(self._mpis.task_id)])

        for rk in remote_leader:
            self._intercomms[rk] = self.domain.comm_task.Create_intercomm(
                0, main_comm, rk)

    def get_work_properties(self):
        if not self._is_discretized:
            msg = 'The operator must be discretized '
            msg += 'before any call to this function.'
            raise RuntimeError(msg)

        vd = self.discreteFields[self.velocity]
        shape_v = vd[0].shape
        rwork_length = self.velocity.nb_components ** 2
        res = {'rwork': [], 'iwork': None}
        for _ in xrange(rwork_length):
            res['rwork'].append(shape_v)
        return res

    def discretize(self):
        if self.method[SpaceDiscretisation] is FD_C_4:
            nbGhosts = 2
        else:
            raise ValueError("Unknown method for space discretization of the\
                time step-evaluation operator.")
        super(AdaptTimeStep, self)._standard_discretize(nbGhosts)

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        self.discrete_op =\
            AdaptTimeStep_D(self.discreteFields[self.velocity],
                            self.discreteFields[self.vorticity],
                            self.simulation, method=self.method,
                            time_range=self.time_range,
                            lcfl=self.lcfl,
                            cfl=self.cfl,
                            maxdt=self.maxdt,
                            rwork=rwork, iwork=iwork)
        # Output setup
        self._set_io('dt_adapt', (1, 7))
        self.discrete_op.setWriter(self._writer)
        self._is_uptodate = True

    def wait(self):
        task_is_source = self._mpis.task_id == self.domain.current_task()
        rank = self._mpis.rank
        dt = self.simulation.timeStep
        for rk in self._intercomms:
            if task_is_source:
                # Local 0 broadcast current_indices to remote comm
                if rank == 0:
                    self._intercomms[rk].bcast(dt, root=MPI.ROOT)
                else:
                    self._intercomms[rk].bcast(dt, root=MPI.PROC_NULL)
            else:
                dt = self._intercomms[rk].bcast(dt, root=0)
                self.simulation.updateTimeStep(dt)
