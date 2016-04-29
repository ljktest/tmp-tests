"""Abstract interface for discrete operators.
"""
from abc import ABCMeta, abstractmethod
from hysop.constants import debug
from hysop.methods_keys import GhostUpdate
from hysop.tools.profiler import Profiler


class DiscreteOperator(object):
    """Common interface to all discrete
    operators.

    """

    __metaclass__ = ABCMeta

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @debug
    @abstractmethod
    def __init__(self, variables, rwork=None, iwork=None, method=None,
                 mpi_params=None):
        """
        Parameters
        -----------
        variables : list of :class:`hysop.fields.discrete.DiscreteField`
            the fields on which this operator works.
        rwork : list of numpy real arrays, optional
            internal work arrays. See notes below.
        iwork : list of numpy integer arrays, optional
            internal work arrays. See notes below
        method : dictionnary, optional
            internal solver parameters (discretisation ...).
            If None, use default method from hysop.default_method.py.
        mpi_params : :class:`hysop.tools.parameters.MPIParams
            parameters to set mpi context. See Notes below.


        Attributes
        ----------

        variables : list of discrete fields on which the operator works.
        domain : physical domain
        input : fields used as input (i.e. read-only)
        output : fields used as in/out (i.e. modified during apply call)

        Notes
        -----

        Methods : to be done ...

        MPIParams : to be done ...

        """
        if isinstance(variables, list):
            # variables
            self.variables = variables
        else:
            self.variables = [variables]

        self.domain = self.variables[0].domain
        self._dim = self.domain.dimension

        # Input variables
        self.input = []
        # Output variables
        self.output = []
        # Operator numerical method.
        if method is None:
            method = {}
        self.method = method
        if GhostUpdate not in method:
            method[GhostUpdate] = True
        # Operator name
        self.name = self.__class__.__name__
        # Object to store computational times of lower level functions
        self.profiler = Profiler(self, self.domain.comm_task)

        # Allocate or check work arrays.
        # Their shapes, number ... strongly depends
        # on the type of discrete operator.
        # A _set_work_arrays function must be implemented
        # in all derived classes where work are required.
        self._rwork = None
        self._iwork = None
        self._set_work_arrays(rwork, iwork)

        # Function to synchronize ghosts if needed
        self._synchronize = None

        # Object that deals with output file writing.
        # Optional.
        self._writer = None
        # Check topologies consistency
        if self.variables is not None:
            toporef = self.variables[0].topology
            for v in self.variables:
                assert v.topology.is_consistent_with(toporef)

    def get_work_properties(self):
        """
        Get properties of internal work arrays.
        Returns
        -------
        dictionnary
           keys = 'rwork' and 'iwork', values = list of shapes of internal
           arrays required by this operator (real arrays for rwork, integer
           arrays for iwork).

        Example
        -------
        >>> works_prop = op.get_work_properties()
        >>> print works_prop
        {'rwork': [(12, 12), (45, 12, 33)], 'iwork': None}

        means that the operator requires two real arrays of
        shape (12,12) and (45, 12, 33) and no integer arrays

        """
        return {'rwork': None, 'iwork': None}

    def _set_work_arrays(self, rwork, iwork):
        """
        To set the internal work arrays used by this operator.
        Parameters
        ----------
        rwork : list of numpy real arrays
            real buffers for internal work
        iwork : list of numpy integer arrays
            integer buffers for internal work

        """
        pass

    def setWriter(self, writer):
        """
        Assign a writer to the current operator
        """
        self._writer = writer

    @debug
    @abstractmethod
    def apply(self, simulation=None):
        """Execute the operator for the current simulation state

        Parameters
        ----------
        simulation : :class:`~hysop.problem.simulation.Simulation`
        """

    @debug
    def finalize(self):
        """
        Cleaning, if required.
        """
        pass

    def __str__(self):
        """Common printings for discrete operators."""
        shortName = str(self.__class__).rpartition('.')[-1][0:-2]
        s = shortName + " discrete operator. \n"
        if self.input is not None:
            s += "Input fields : \n"
            for f in self.input:
                s += str(f) + "\n"
        if self.output is not None:
            s += "Output fields : \n"
            for f in self.output:
                s += str(f) + "\n"
        return s

    def update_ghosts(self):
        """
        Update ghost points values, if any.
        This function must be implemented in the discrete
        operator if it may be useful to ask for ghost
        points update without a call to apply.
        For example for monitoring purpose before
        operator apply.
        """
        pass

    def get_profiling_info(self):
        """Get the manual profiler informations into the default profiler"""
        pass


def get_extra_args_from_method(op, key, default_value):
    """
    Returns the given extra arguments dictionary from method attribute.

    Parameters
    -----------
    op : operator
        extract method attribute from
    key : string
        key to extract
    default_value :
        default value when ExtraArgs is not in op.method or
        key is not in op.method[ExtraArgs]

    """
    from hysop.methods_keys import ExtraArgs
    try:
        return op.method[ExtraArgs][key]
    except KeyError:
        return default_value
