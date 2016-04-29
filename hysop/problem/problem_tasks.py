"""
@file problem_tasks.py

Extending problem description to handle tasks parallelism.
Each operator owns a task id that define a process group that are sharing the
same tasks.
"""
from hysop.constants import debug
from hysop import __VERBOSE__
from hysop.problem.problem import Problem
from hysop.operator.redistribute_inter import RedistributeInter
from hysop.operator.redistribute_intra import RedistributeIntra
from hysop.operator.redistribute import Redistribute
from hysop.gpu.gpu_transfer import DataTransfer
from hysop.tools.profiler import profile


class ProblemTasks(Problem):
    """
    As in Problem, it contains several operators that apply
    on variables. The operators are labeled by task_id that defines
    a identifier of a task.
    Tasks are subset of operators and are assigned to a subset of the MPI
    process by means of the task_list parameter.
    """
    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @debug
    def __init__(self, operators, simulation, domain, tasks_list,
                 dumpFreq=100, name=None, main_comm=None):
        """
        Creates the problem.
        @param operators : list of operators.
        @param simulation : a hysop.simulation.Simulation object
        to describe simulation parameters.
        @param tasks_list : list of task identifiers for each process rank
        @param name : an id for the problem
        @param dumpFreq : frequency of dump (i.e. saving to a file)
        for the problem; set dumpFreq = -1 for no dumps. Default = 100.
        @param main_comm : MPI communicator that contains all process
        involved in this problem.

        @remark : process number in communicator main_comm must equal the
        length of tasks_list.
        """
        Problem.__init__(self, operators, simulation,
                         domain=domain, dumpFreq=dumpFreq, name=name)
        self.tasks_list = tasks_list
        if main_comm is None:
            from hysop.mpi.main_var import main_comm
        self.main_comm = main_comm
        self._main_rank = self.main_comm.Get_rank()
        assert self.main_comm.Get_size() == len(self.tasks_list), \
            "The given task list length (" + str(self.tasks_list) + ") " \
            "does not match the communicator size" \
            " ({0})".format(self.main_comm.Get_size())
        self.my_task = self.tasks_list[self._main_rank]
        self.operators_on_task = []

    def pre_setup(self):
        """
        - Removes operators that not have the same task identifier
        as the current process
        - Keep the Redistribute_intercomm in both 'from' and 'to' task_id
        - Partial setup : only for 'computational' operators
        (i.e. excluding rendering, data distribution ...)
        - Initialize variables.
        """
        if self._isReady:
            pass

        ## Remove operators with a tasks not handled by this process.
        for op in self.operators:
            if op.task_id() == self.my_task:
                self.operators_on_task.append(op)

        # Discretize and setup computational operators
        for op in self.operators_on_task:
            if not isinstance(op, Redistribute) and \
               not isinstance(op, DataTransfer):
                op.discretize()
        for op in self.operators_on_task:
            if not isinstance(op, Redistribute) and \
               not isinstance(op, DataTransfer):
                op.setup()

        # Build variables list to initialize
        # These are operators input variables that are not output of
        # previous operators in the operator stack.
        # Set the variables input topology as the the topology of the fist
        # operator that uses this variable as input.
        self.input = []
        for op in self.operators_on_task:
            for v in op.input:
                if v not in self.input:
                    self.input.append(v)
        for op in self.operators_on_task[::-1]:
            for v in op.output:
                if v in self.input:
                    if isinstance(op, RedistributeInter):
                        if op._target_id == self.my_task:
                            self.input.remove(v)
                    else:
                        self.input.remove(v)
            for v in op.input:
                if v not in self.input:
                    if isinstance(op, RedistributeInter):
                        if op._source_id == self.my_task:
                            self.input.append(v)
                    else:
                        self.input.append(v)

        self._isReady = True

    @debug
    @profile
    def setup(self):
        """
        Prepare operators (create topologies, allocate memories ...)
        """
        # Set up for 'computational' operators
        if not self._isReady:
            self.pre_setup()

        # for v in self.input:
        #     v.initialize()

        # other operators
        for op in self.operators_on_task:
            if isinstance(op, RedistributeIntra) or \
               isinstance(op, DataTransfer):
                op.setup()

        for op in self.operators_on_task:
            if isinstance(op, RedistributeInter):
                op.setup()

        if __VERBOSE__ and self._main_rank == 0:
            print("====")

    @debug
    @profile
    def solve(self):
        """
        Solve problem.

        Performs simulations iterations by calling each
        operators of the list until timer ends.\n
        At end of time step, call an io step.\n
        Displays timings at simulation end.
        """
        self.simulation.initialize()
        self.main_comm.Barrier()
        if self._main_rank == 0:
            print ("\n\n Start solving ...")
        while not self.simulation.isOver:
            if self._main_rank == 0:
                self.simulation.printState()
            for op in self.operators:
                if op.task_id() == self.my_task:
                    op.apply(self.simulation)
                    if isinstance(op, RedistributeInter):
                        if op._source_id == self.my_task:
                            op.wait()
            testdump = \
                self.simulation.currentIteration % self.dumpFreq is 0
            self.simulation.advance()
            if self._doDump and testdump:
                self.dump()

    @debug
    def finalize(self):
        """
        Finalize method
        """
        if self._main_rank == 0:
            print ("\n\n==== End ====")
        for op in self.operators_on_task:
            op.finalize()
        var = []
        for op in self.operators_on_task:
            for v in op.variables:
                if v not in var:
                    var.append(v)
        for v in var:
            v.finalize()
        self.profiler.summarize()
        if self._main_rank == 0:
            print ("===\n")
