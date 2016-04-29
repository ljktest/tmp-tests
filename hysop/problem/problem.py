"""
@file problem.py

Complete problem description.
"""
from hysop.constants import debug
import cPickle
from hysop import __VERBOSE__
from hysop.operator.redistribute import Redistribute
from hysop.operator.redistribute_intra import RedistributeIntra
from hysop.tools.profiler import profile, Profiler
from hysop.mpi import main_rank
from hysop.gpu.gpu_transfer import DataTransfer


class Problem(object):
    """
    Problem representation.

    Contains several operators that apply on variables.
    Variables are defined on different domains.\n
    Each operator is set up and variables are initialized in a set up step.\n
    To solve the problem, a loop over time-steps is launched. A step consists
    in calling the apply method of each operators.\n
    To finish, a finalize method is called.\
    """

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @debug
    def __init__(self, operators, simulation,
                 dumpFreq=100, name=None):
        """
        Create a transport problem instance.

        @param operators : list of operators.
        @param simulation : a hysop.simulation.Simulation object
        to describe simulation parameters.
        @param name : an id for the problem
        @param dumpFreq : frequency of dump (i.e. saving to a file)
        for the problem; set dumpFreq = -1 for no dumps. Default = 100.
        """
        ## Problem name
        self.name = name
        ## Problem operators
        self.operators = operators
        ## Computes time step and manage iterations
        self.simulation = simulation
        vref = self.operators[0].variables.keys()[0]
        self.domain = vref.domain
        for op in self.operators:
            for v in (v for v in op.variables if v is not vref):
                print id(v.domain), id(self.domain)
                print v.domain, self.domain
                if self.domain is not v.domain:
                    raise ValueError("Problem must have only one " +
                                     "domain for variables.")
        ## A list of variables that must be initialized before
        ## any call to op.apply()
        self.input = []
        ## call to problem.dump frequency during apply.
        if dumpFreq >= 0:
            ## dump problem every self.dumpFreq iter
            self.dumpFreq = dumpFreq
            self._doDump = True
        else:
            self._doDump = False
            self.dumpFreq = 100000

        ## Id for the problem. Used for dump file name.
        if name is None:
            self.name = 'HySoPPb'
        else:
            self.name = name
        ## Object to store computational times of lower level functions
        self.profiler = Profiler(self, self.domain.comm_task)
        ## Default file name prefix for dump.
        self.filename = str(self.name)
        self._filedump = self.filename + '_rk_' + str(main_rank)

        # Flag : true when operators for computation are up
        # and when variables are initialized (i.e. after a call to pre_setup)
        # Note : 3 categories of op : computation (stretching, poisson ...),
        # and data distribution (Redistribute)
        self._isReady = False

    @debug
    @profile
    def setup(self):
        """
        Prepare operators (create topologies, allocate memories ...)
        """
        # Set up for 'computational' operators
        if not self._isReady:
            self.pre_setup()
        print "Fin setup op"
        # for v in self.input:
        #     v.initialize()

        # other operators
        for op in self.operators:
            if isinstance(op, RedistributeIntra) or \
               isinstance(op, DataTransfer):
                op.setup()

        for op in self.operators:
            if isinstance(op, Redistribute):
                op.setup()

        if __VERBOSE__ and main_rank == 0:
            print ("====")

    def pre_setup(self):
        """
        - Partial setup : only for 'computational' operators
        (i.e. excluding rendering, data distribution ...)
        - Initialize variables.
        """
        if self._isReady:
            pass

        for op in self.operators:
            if not isinstance(op, Redistribute) and \
               not isinstance(op, DataTransfer):
                op.discretize()

        for op in self.operators:
            if not isinstance(op, Redistribute) and \
               not isinstance(op, DataTransfer):
                op.setup()

        if __VERBOSE__ and main_rank == 0:
            print ("==== Variables initialization ====")

        # Build variables list to initialize
        # These are operators input variables that are not output of
        # previous operators in the operator stack.
        # Set the variables input topology as the the topology of the fist
        # operator that uses this variable as input.
        self.input = []
        for op in self.operators:
            for v in op.input:
                if v not in self.input:
                    self.input.append(v)
        for op in self.operators[::-1]:
            for v in op.output:
                if v in self.input:
                    self.input.remove(v)
            for v in op.input:
                if not v in self.input:
                    self.input.append(v)

        self._isReady = True

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
        if main_rank == 0:
            print ("\n\n Start solving ...")
        while not self.simulation.isOver:
            if main_rank == 0:
                self.simulation.printState()

            for op in self.operators:
                if __VERBOSE__:
                    print (main_rank, op.name)
                op.apply(self.simulation)

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
        if main_rank == 0:
            print ("\n\n==== End ====")
        for op in self.operators:
            op.finalize()

        var = []
        for op in self.operators:
            for v in op.variables:
                if not v in var:
                    var.append(v)
        for v in var:
            v.finalize()
        self.profiler.summarize()
        if main_rank == 0:
            print ("===\n")

    def get_profiling_info(self):
        for op in self.operators:
            self.profiler += op.profiler
        for op in self.operators:
            for v in op.variables:
                self.profiler += v.profiler

    def __str__(self):
        """ToString method"""
        s = "Problem based on\n"
        s += str(self.domain.topologies)
        s += "with following operators : \n"
        for op in self.operators:
            s += str(op)
        return s

    def dump(self, filename=None):
        """
        Serialize some data of the problem to file
        (only data required for a proper restart, namely fields in self.input
        and simulation).
        @param filename : prefix for output file. Real name = filename_rk_N,
        N being current process number. If None use default value from problem
        parameters (self.filename)
        """
        if filename is not None:
            self.filename = filename
            self._filedump = filename + '_rk_' + str(main_rank)
        db = open(self._filedump, 'wb')
        cPickle.dump(self.simulation, db)
        # TODO : review dump process using hdf files instead of pickle.
        # for v in self.input:
        #     v.hdf_dump(self.filename)

    def restart(self, filename=None):
        """
        Load serialized data to restart from a previous state.
        self.input variables and simulation are loaded.
        @param  filename : prefix for downloaded file.
        Real name = filename_rk_N, N being current process number.
        If None use default value from problem
        parameters (self.filename)
        """
        if filename is not None:
            self.filename = filename
            self._filedump = filename + '_rk_' + str(main_rank)
        db = open(self._filedump, 'r')
        self.simulation = cPickle.load(db)
        self.simulation.reset()
        for v in self.input:
            print ("load ...", self.filename)
            v.load(self.filename)

        for op in self.operators:
            if isinstance(op, Redistribute):
                op.setup()

    def setDumpFreq(self, freq):
        """
        set rate of problem.dump call (every 'rate' iteration)
        @param freq : the frequency of output
        """
        self.dumpFreq = freq
        if freq < 0:
            self._doDump = False
