""" Abstract interface for computational operators

"""
from abc import ABCMeta, abstractmethod
from hysop.constants import debug
from hysop.operator.continuous import Operator, opapply
from hysop.mpi.topology import Cartesian
from hysop.tools.parameters import Discretization
from hysop.tools.profiler import profile
import hysop.tools.numpywrappers as npw


class Computational(Operator):
    """
    Abstract base class for computational operators.

    An operator is composed of :
    - a set of continuous variables (at least one)
    - a method which defined how it would be discretized/processed
    - a discrete operator : object build using the method
    and the discretized variables.

    To each variable a 'resolution' is associated, used
    to create a topology and a discrete field.
    See details in 'real' operators (derived classes) description.
    """
    __metaclass__ = ABCMeta

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @debug
    @abstractmethod
    def __init__(self, discretization=None, method=None, **kwds):
        """

        Parameters
        ----------
        discretization : :class:`hysop.mpi.topology.Cartesian`
         or :class:`tools.parameters.Discretization`
            Defined the data mpi distribution. See Notes below.
        method : :class:`hysop.method`
            specific parameters for the operator
            (space discretisation ...). See methods.py for authorized values.
        **kwds : arguments passed to base class.

        Notes:
        ------
        """
        # Base class init
        super(Computational, self).__init__(**kwds)
        # Set mpi related stuff
        self._set_domain_and_tasks()

        # A dictionnary of parameters, to define numerical methods used
        # used to discretize the operator.
        # When method is None, each operator must provide a default
        # behavior.
        self.method = method

        # The discretization of this operator.
        self.discrete_op = None

        # A dictionnary of discreteFields associated with this operator
        # key = continuous variable \n
        # Example : discrete_variable = discreteFields[velocity]
        # returns the discrete fields that corresponds to the continuous
        # variable velocity.
        self.discreteFields = {}

        if not self._varsfromList:
            msg = 'discretization parameter is useless when variables are set'
            msg += ' from a dict.'
            assert discretization is None, msg

        self._discretization = discretization
        # Remark FP: discretization may be None in two cases:
        # - not required when variables is set from a dict
        # - the current task does not need to discretize this operator.

        # False if fields have different discretizations.
        # Set during discretize call.
        self._single_topo = True

        # If true, ready for setup ...
        # Turn to true after self._discretize_vars call.
        self._is_discretized = False

    def get_work_properties(self):
        """Get properties of internal work arrays. Must be call after discretize
        but before setup.

        Returns
        -------
        dictionnary
           keys = 'rwork' and 'iwork', values = list of shapes of internal
           arrays required by this operator (real arrays for rwork, integer
           arrays for iwork).

        Example
        -------

        >> works_prop = op.get_work_properties()
        >> print works_prop
        {'rwork': [(12, 12), (45, 12, 33)], 'iwork': None}

        means that the operator requires two real arrays of
        shape (12,12) and (45, 12, 33) and no integer arrays

        """
        return {'rwork': None, 'iwork': None}

    def discretize(self):
        """
        For each variable, check if a proper topology has been defined,
        if not, build one according to 'discretization' parameters set
        during initialization of the class.
        Then, discretize each variable on this topology.
        """
        self._standard_discretize()

    def _discretize_vars(self):
        """
        Discretize all variables of the current operator.
        """
        for v in self.variables:
            msg = 'Missing topology to discretize ' + v.name
            msg += ' in operator ' + self.name
            assert isinstance(self.variables[v], Cartesian), msg

            self.discreteFields[v] = v.discretize(self.variables[v])
        self._is_discretized = True

    def _check_variables(self):
        """
        Check variables and discretization parameters
        Set single_topo: if true all fields are discretized with the
        same topo
        @return build_topos : a dict (key == field), if build_topos[v] is true,
        a topology must be built for v. In that case, the discretization has
        been saved in self.variables[v] during init. In the other case
        self.variables[v] is the required topology.

        Remark : since operators belong to one and only one task, this function
        must not be called by all tasks. So it can not be called at init.
        """
        if self._varsfromList:
            # In that case, self._single_topo is True
            # but we need to check if discretization param
            # was a topology or a Discretization.
            msg = 'required parameter discretization has not been'
            msg += ' set during operator construction.'
            assert self._discretization is not None
            # Fill variables dictionnary
            for v in self.variables:
                self.variables[v] = self._discretization
            self._single_topo = True
            if isinstance(self._discretization, Cartesian):
                # No need to build topologies
                build_topos = False
            elif isinstance(self._discretization, Discretization):
                build_topos = True
            else:
                msg = 'Wrong type for parameter discretization in'
                msg += ' operator construction.'
                raise ValueError(msg)
        else:
            msg = 'discretization parameter in operator construction is '
            msg += 'useless when variables are set from a dict.'
            assert self._discretization is None, msg
            self._single_topo = False
            build_topos = {}
            for v in self.variables:
                disc = self.variables[v]
                if isinstance(disc, Cartesian):
                    build_topos[v] = False
                elif isinstance(disc, Discretization):
                    build_topos[v] = True
                else:
                    msg = 'Wrong type for values in variables dictionnary '
                    msg += '(parameter in operator construction).'
                    raise ValueError(msg)

            ref = self.variables.values()[0]
            self._single_topo = True
            for disc in self.variables.values():
                self._single_topo = ref == disc and self._single_topo

            if self._single_topo:
                build_topos = build_topos.values()[0]
                self._discretization = self.variables.values()[0]

        return build_topos

    def _standard_discretize(self, min_ghosts=0):
        """
        This functions provides a standard way to discretize the operator,
        but some operators may need a specific discretization process.
        """
        if self._is_discretized:
            return

        build_topos = self._check_variables()
        if self._single_topo:
            # One topo for all fields ...
            if build_topos:
                topo = self._build_topo(self._discretization, min_ghosts)
                for v in self.variables:
                    self.variables[v] = topo
            else:
                # Topo is already built, just check its ghosts
                topo = self.variables.values()[0].mesh.discretization.ghosts
                assert (topo >= min_ghosts).all()

        else:
            # ... or one topo for each field.
            for v in self.variables:
                if build_topos[v]:
                    self.variables[v] = self._build_topo(self.variables[v],
                                                         min_ghosts)
                else:
                    assert (self.variables[v].ghosts >= min_ghosts).all()

        # All topos are built, we can discretize fields.
        self._discretize_vars()

    def _build_topo(self, discretization, min_ghosts):
        # Reset ghosts if necessary
        ghosts = discretization.ghosts
        ghosts[ghosts < min_ghosts] = min_ghosts
        # build a topology from the given discretization
        return self.domain.create_topology(discretization)

    def _fftw_discretize(self):
        """
        fftw specific way to discretize variables for a given
        'reference' resolution.
        We assume that in fft case, only one topology must be used
        for all variables.
        """
        if self._is_discretized:
            return
        build_topos = self._check_variables()
        assert self._single_topo, 'All fields must use the same topology.'
        # Get local mesh parameters from fftw
        comm = self._mpis.comm
        from hysop.f2hysop import fftw2py
        if build_topos:
            # In that case, self._discretization must be
            # a Discretization object, used for all fields.
            # We use it to initialize scales solver
            msg = 'Wrong type for parameter discretization (at init).'
            assert isinstance(self._discretization, Discretization), msg
            resolution = npw.asintarray(self._discretization.resolution)
            localres, global_start = fftw2py.init_fftw_solver(
                resolution, self.domain.length, comm=comm.py2f())
            # Create the topo (plane, cut through ZDIR)
            topo = self.domain.create_plane_topology_from_mesh(
                global_start=global_start, localres=localres,
                discretization=self._discretization)
            for v in self.variables:
                self.variables[v] = topo
        else:
            # In that case, self._discretization must be
            # a Cartesian object, used for all fields.
            # We use it to initialize fftw solver
            assert isinstance(self._discretization, Cartesian)
            topo = self._discretization
            msg = 'input topology is not compliant with fftw.'
            assert topo.dimension == 1, msg

            from hysop.constants import ORDER
            if ORDER == 'C':
                assert topo.shape[0] == self._mpis.comm.Get_size(), msg
            else:
                assert topo.shape[-1] == self._mpis.comm.Get_size(), msg

            resolution = npw.asintarray(topo.mesh.discretization.resolution)

            localres, global_start = fftw2py.init_fftw_solver(
                resolution, self.domain.length, comm=comm.py2f())

        assert (topo.mesh.resolution == localres).all()
        assert (topo.mesh.start() == global_start).all()
        msg = 'Ghosts points not yet implemented for fftw-type operators.'
        assert (topo.ghosts() == 0).all(), msg

        # All topos are built, we can discretize fields.
        self._discretize_vars()

    @abstractmethod
    def setup(self, rwork=None, iwork=None):
        """
        Last step of initialization. After this, the operator must be
        ready for apply call.

        Main step : setup for discrete operators.
        """
        assert self._is_discretized
        super(Computational, self).setup()

    @debug
    def finalize(self):
        """
        Memory cleaning.
        """
        if self.discrete_op is not None:
            self.discrete_op.finalize()

    @debug
    @opapply
    def apply(self, simulation=None):
        """Apply this operator to its variables.

        Parameters
        ----------
        simulation : `:class::~hysop.problem.simulation.Simulation`

        """
        if self.discrete_op is not None:
            self.discrete_op.apply(simulation)

    def computation_time(self):
        """ Time monitoring."""
        if self.discrete_op is not None:
            self.discrete_op.computation_time()
            self.time_info = self.discrete_op.time_info
        else:
            from hysop.mpi.main_var import main_rank
            shortName = str(self.__class__).rpartition('.')[-1][0:-2]
            s = '[' + str(main_rank) + '] ' + shortName
            s += " : operator not discretized --> no computation, time = 0."
            print s

    def update_ghosts(self):
        """
        Update ghost points values, if any.
        """
        assert self._is_discretized
        self.discrete_op.update_ghosts()

    def __str__(self):
        """
        Common printings for operators
        """
        shortName = str(self.__class__).rpartition('.')[-1][0:-2]
        if self.discrete_op is not None:
            s = str(self.discrete_op)
        else:
            s = shortName + " operator. Not discretised."
        return s + "\n"

    def get_profiling_info(self):
        if self.discrete_op is not None:
            self.profiler += self.discrete_op.profiler
