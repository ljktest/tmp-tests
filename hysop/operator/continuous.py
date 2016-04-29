"""Common interface for all continuous operators.

"""
from abc import ABCMeta, abstractmethod
from hysop.constants import debug
from hysop.tools.profiler import Profiler, profile
from hysop.tools.io_utils import IOParams, IO
from hysop.tools.parameters import MPIParams
import hysop.tools.io_utils as io
import inspect


class Operator(object):
    """Abstract interface to continuous operators.
    """
    __metaclass__ = ABCMeta

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @debug
    @abstractmethod
    def __init__(self, variables=None, mpi_params=None,
                 io_params=None):
        """
        Parameters
        ----------
        variables : list or dictionnary. See Notes for details.
        mpi_params : :class:`hysop.tools.parameters.MPIParams`
            mpi config for the operator (comm, task ...)
        io_params : :class:`hysop.tools.io_utils.IOParams`
            file i/o config (filename, format ...)

        Notes
        -----
        Variables arg may be either a list or a dictionnary, depending
        on the type of operator.
        The elements of the list or the keys of the dict
        are :class:`hysop.fields.continuous.Fields`.

        The values of the dict can be either
        :class:`hysop.mpi.topology.Cartesian`
        or :class:`hysop.tools.parameters.Discretization`::

        op = Analytic(variables = [velo, vorti], ...)

        or::

        op = Analytic(variables = {velo: topo, vorti: discr3D}, ...)

        Attributes
        ----------
        variables : dict or list
            :class:`~hysop.fields.continuous.Continuous`
            with their discretisation
        domain : :class:`~hysop.domain.domain.Domain`,
        the geometry on which this operator applies

        """
        # 1 ---- Variables setup ----
        # List of hysop.continuous.Fields involved in the operator.
        if isinstance(variables, list):
            self.variables = {}
            for v in variables:
                self.variables[v] = None
            self._varsfromList = True
            # Details on descretization process must be provided
            # in derived class (extra args like resolution, topo ...)
        elif isinstance(variables, dict):
            self._varsfromList = False
            self.variables = variables
        elif variables is not None:
            # Note that some operators may not have variables (redistribute for
            # example).
            msg = 'Wrong type for variables arg.'
            msg += 'It must be a list or a dictionnary.'
            raise AttributeError(msg)
        else:
            # this last case corresponds with redistribute operators
            # that may have variables implicitely defined from input
            # source and target operators
            #(see hysop.operator.redistribute.Redistribute for details).
            self.variables = {}

        # Domain of definition.
        # Must be the same for all variables
        # Set in derived class.
        self.domain = None
        """Physical domain of definition for the operator """
        # mpi context
        self._mpis = mpi_params
        # tools for profiling
        self.profiler = None

        # Remark : domain, _mpis and profiler will be set properly in
        # _set_domain_and_tasks, called in derived class, since it may
        # require some specific initialization (check domain ...)

        # Input variables.
        self.input = []
        # Output variables.
        self.output = []
        # bool to check if the setup function has been called for
        # this operator
        self._is_uptodate = False

        self.name = self.__class__.__name__
        # List of operators that must be waited for.
        self._wait_list = []
        # time monitoring
        self.time_info = None
        # Dictionnary of optional parameters for output
        self.io_params = io_params
        # Object that deals with output file writing.
        # Optional.
        self._writer = None
        self.ontask = False

    def _set_domain_and_tasks(self):
        """
        Initialize the mpi context, depending on local variables, domain
        and so on.
        """
        # When this function is called, the operator must at least
        # have one variable.
        assert len(self.variables) > 0
        if isinstance(self.variables, list):
            self.domain = self.variables[0].domain
        elif isinstance(self.variables, dict):
            self.domain = self.variables.keys()[0].domain

        # Check if all variables have the same domain
        for v in self.variables:
            assert v.domain is self.domain, 'All variables of the operator\
            must be defined on the same domain.'
        # Set/check mpi context
        if self._mpis is None:
            self._mpis = MPIParams(comm=self.domain.comm_task,
                                   task_id=self.domain.current_task())

        # Set profiler
        self.profiler = Profiler(self, self.domain.comm_task)

    @staticmethod
    def _error_():
        """internal error message
        """
        raise RuntimeError("This operator is not defined for the current task")

    def wait_for(self, op):
        """MPI synchronisation

        :param op:  :class:`~hysop.operator.continuous.Continuous`
            Add an operator into 'wait' list of the present object.
            It means that before any apply of this operator, all
            (mpi) operations in op must be fulfilled, which implies
            a call to op.wait().

        """
        self._wait_list.append(op)

    def wait_list(self):
        """Get MPI running comm. list

        Returns
        -------
        python list of all operators that must be fulfilled
        before any attempt to apply the present operator.
        """
        return self._wait_list

    def wait(self):
        """
        MPI wait for synchronisation: when this function is called,
        the programm wait for the fulfillment of all the running
        operations of this operator (mpi requests for example).
        This is a blocking call.
        """
        pass

    def test_requests(self):
        """Checks for unfinished mpi communications.

        Returns
        -------
        bool : MPI send/recv test for synchronisation, when this function is
        called, the programm checks if this operator handles some uncomplete
        mpi requests (if so return true, else false).
        This is a non-blocking call.
        """
        pass

    @abstractmethod
    def setup(self, rwork=None, iwork=None):
        """
        Last step of initialization. After this, the operator must be
        ready to apply.
        In derived classes, called through @opsetup decorator.
        """
        if not self.domain.current_task() == self._mpis.task_id:
            self.ontask = False
            self._error_()

    @abstractmethod
    @debug
    def apply(self, simulation=None):
        """
        Apply this operator to its variables.

        Parameters
        ----------
        simulation : hysop.problem.simulation.Simulation
            describes the simulation parameters
            (time, time step, iteration number ...)

        In derived classes, called through @opapply decorator.
        """
        for op in self.wait_list():
            op.wait()

    def finalize(self):
        """
        Memory cleaning.
        """
        # wait for all remaining communications, if any
        self.wait()

    @abstractmethod
    def computation_time(self):
        """
        Time monitoring.
        """

    def is_up(self):
        """Returns True if ready to be applied
        (--> setup function has been called succesfully)
        """
        return self._is_uptodate

    def _set_io(self, filename, buffshape):
        """
        Parameters
        -----------
        filename : string
            name of the output file used by this operator
        buffshape : tuple
            shape of the numpy buffer used to save data to
            be printed out into filename. Must be 2D.
            Example : shape (2,4)

        Notes
        -----
        This function is private and must not be called by
        external object. It is usually called by operator
        during construction (__init__).

        """
        iopar = self.io_params
        # if iopar is not None (i.e. set in operator init)
        # and True or set as an IOParams , then
        # build a writer
        if iopar:
            if isinstance(iopar, bool):
                # default values for iop
                self.io_params = IOParams(filename, fileformat=IO.ASCII)
            else:
                if self.io_params.fileformat is not IO.ASCII:
                    self.io_params.fileformat = IO.ASCII
                    msg = 'Warning, wrong file format for operator output.'
                    msg += 'This will be reset to ASCII.'
                    print msg
            self._writer = io.Writer(io_params=self.io_params,
                                     mpi_params=self._mpis,
                                     buffshape=buffshape)

    def task_id(self):
        """
        Returns the id of the task on which this operator works.
        """
        return self._mpis.task_id


def opsetup(f):
    """
    Setup decorator: what must be done by all operators
    at setup.
    Usage : add @opsetup before setup class method
    """

    def decorator(*args, **kwargs):
        """Decorate 'setup' method
        """
        # Job before setup of the function ...
        # nothing for the moment
        name = inspect.getmro(args[0].setup.im_class)
        # call the setup function
        retval = f(*args, **kwargs)
        # Warning : we cannot call super(...) since
        # it leads to infinite cycling when setup
        # is not defined in the class but in its
        # base class and when this base class is derived
        # from Computational ...
        # So we directly call Computational.setup()
        # It's ugly but it seems to work.
        # Job after setup of the function ...
        name[-3].setup(args[0])
        #super(args[0].__class__, args[0]).setup()
        return retval

    return decorator

from hysop.tools.profiler import ftime


def opapply(f):
    """
    What must be done by all operators
    before apply.
    Usage : add @opapply before apply class method
    """
    def decorator(*args, **kwargs):
        """decorate 'apply' method"""
        # get 'continuous' base class and run its apply function
        # --> call wait function of ops in wait_list
        t0 = ftime()
        name = inspect.getmro(args[0].apply.im_class)
        name[-2].apply(args[0])
        #t0 = ftime()
        res = f(*args, **kwargs)
        args[0].profiler[f.func_name] += ftime() - t0
        return res

    return decorator


class Tools(object):
    """
    Static class with utilities related to operators
    """

    @staticmethod
    def check_device(op):
        """
        Returns true if op operates on a GPU
        """
        from hysop.methods_keys import Support

        try:
            is_device = \
                op.method[Support].find('gpu') >= 0
        except KeyError:  # op.method is a dict not containing Support in keys
            is_device = False
        except IndexError:  # op.method is a string
            is_device = False
        except TypeError:  # op.method is None
            is_device = False
        return is_device
