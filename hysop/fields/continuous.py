"""Continuous fields description.

Examples
--------

>>> from hysop import Box
>>> dom = Box()
>>> from hysop.fields.continuous import Field
>>> scal = Field(domain=dom, name='Scalar')
>>> vec = Field(domain=dom, name='Velocity', is_vector=True)


.. currentmodule:: hysop

"""
from hysop.constants import debug
from hysop.fields.discrete import DiscreteField
from hysop.mpi import main_rank
from hysop.tools.profiler import Profiler, profile
from hysop.tools.io_utils import IOParams, IO
from hysop.operator.hdf_io import HDF_Writer, HDF_Reader
from hysop.mpi.topology import Cartesian


class Field(object):
    """
    Continuous field defined on a physical domain.

    This object handles a dictionnary of discrete fields
    (from 0 to any number).
    Each discrete field is uniquely defined by the topology used to
    discretize it.

    Example :
    if topo1 and topo2 are two hysop.mpi.topology.Cartesian objects::

    scal = Field(domain=dom, name='Scalar')
    # Discretize scal on two different topologies
    scal.discretize(topo1)
    scal.discretize(topo2)
    # Access to the discrete fields :
    scal_discr1 = scal.discreteFields[topo1]
    scal_discr2 = scal.discreteFields[topo2]
    # or in a more  straightforward way:
    scal_discr2 = scal.discretize(topo2)

    """

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @debug
    @profile
    def __init__(self, domain, name, formula=None, is_vector=False,
                 nb_components=None, vectorize_formula=False):
        """Defines a continuous field (scalar or vector) on a domain.

        Parameters
        ----------

        domain : domain.Domain
            physical domain where this field is defined.
        name : string
            a name for the variable. Required because of h5py i/o.
         formula : python function, optional
            a user-defined python function. See notes for details.
        is_vector : bool, optional
            true if the field is a vector.
        nb_components : int, optional
            number of components for the field.
        vectorize_formula : bool, optional
            true if formula must be vectorized
            (i.e. is of type 'user_func_2')

        Notes
        ------

        By default, fields are initialized to zero everywhere in the domain.

        To compute space-dependant values, define a function like:

        >>> def myfunc(res, x, y, z, t):
        ...     res[0][...] = sin(x)
        ...     res[1][...] = cos(z)
        ...     res[2][...] = sin(y * t)
        ...     return res

        """

        # Physical domain of definition of the variable.
        self.domain = domain
        # dimension (1, 2 or 3D), equal to domain dimension.
        self.dimension = self.domain.dimension
        # Id
        self.name = name
        # Dictionnary of all the discretizations of this field.
        # Key = hysop.mpi.topology.Cartesian,
        # value =hysop.fields.discrete.DiscreteField.
        # Example : vel = discreteFields[topo]
        self.discreteFields = {}
        # Analytic formula.
        self.formula = formula
        # A list of extra parameters, used in _formula
        self.extraParameters = tuple([])
        # Number of components of the field
        if nb_components is None:
            self.nb_components = self.dimension if is_vector else 1
        else:
            self.nb_components = nb_components

        # True if the given formula and must be vectorized
        self._vectorize = vectorize_formula

        # Time profiling
        self.profiler = Profiler(self, self.domain.comm_task)

        # i/o : a dictionnary of hdf writers.
        # None by default, if needed must be
        # initialized with init_writer function.
        self.writer = {}

    def get_profiling_info(self):
        """Update profiler"""
        for d in self.discreteFields.values():
            self.profiler += d.profiler

    @debug
    def discretize(self, topo):
        """Discretization of the field on a given topology.

        Parameters
        ----------
        topo : mpi.topology.Cartesian

        Returns
        --------
        fields.discrete.DiscreteField

        Note
        ----
        In the case the discretization already exists,
        simply returns the discretized field.

        """
        assert isinstance(topo, Cartesian)
        if topo in self.discreteFields:
            return self.discreteFields[topo]
        else:
            self.discreteFields[topo] = DiscreteField(
                topo,
                is_vector=self.nb_components > 1,
                name=self.name + '_' + str(topo.get_id()),
                nb_components=self.nb_components)

        return self.discreteFields[topo]

    def set_formula(self, formula, vectorize_formula):
        """Set a user-defined function to compute field values on the domain.

        Parameters
        ----------
        formula : python function
        vectorize_formula : bool
            true if formula must be vectorized

        Warning: if set, this formula will overwrite any previous setting.
        """
        self.formula = formula
        self._vectorize = vectorize_formula

    @debug
    @profile
    def initialize(self, time=0., topo=None):
        """Initialize one or all the discrete fields associated with
        this continuous field using the current formula.

        Parameters
        -----------
        time : double, optional
            time to be used in formula
        topo : :class:`mpi.topology.Cartesian`, optional
            the topology on which initialization is performed.

        Notes:
        ------
        - if topo is not set, all pre-defined discrete
         fields will be initialized.
        - if no formula has been set, the field will be set to zero everywhere
        on the space mesh. This is the default behavior.

        """
        if topo is None:
            for df in self.discreteFields.values():
                df.initialize(self.formula, self._vectorize, time,
                              *self.extraParameters)
        else:
            df = self.discretize(topo)
            df.initialize(self.formula, self._vectorize, time,
                          *self.extraParameters)

    @profile
    def randomize(self, topo):
        """Initialize a field on topo with random values.

        Parameters
        -----------
        topo : :class:`mpi.topology.Cartesian`
            the topology on which initialization is performed.
        """
        df = self.discretize(topo)
        df.randomize()
        return self.discreteFields[topo]

    def copy(self, field_in, topo):
        """Initialize a field on topo with values from another field.

        Parameters
        -----------
        field_in : :class:`fields.continuous.Continuous`
            field to be copied
        topo : :class:`mpi.topology.Cartesian`
            the topology on which initialization is performed.
        """
        df = self.discretize(topo)
        field_in_d = field_in.discretize(topo)
        df.copy(field_in_d)

    def value(self, *pos):
        """Evaluation of the field at a given position, according
        to the analytic formula given during construction.

        Parameters
        ----------
        pos :
            arguments to pass to formula to evaluate the field
            --> depends on formula's definition ...

        Returns
        -------
        formula(pos) : array-like
            evaluation of the field at the given position.

        """

        if self.formula is not None:
            return self.formula(*pos)
        else:
            return 0.0

    def __str__(self):
        """Field info display """

        s = '[' + str(main_rank) + '] " ' + self.name + ' ", '
        s += str(self.dimension) + 'D '
        if self.formula is not None:
            s += 'an analytic formula is associated to this field:'
            s += str(self.formula.func_name)
        if self.nb_components > 1:
            s += 'vector field '
        else:
            s += 'scalar field '
        s += 'with the following (local) discretizations :\n'
        for f in self.discreteFields.values():
            s += f.__str__()
        return s

    def norm(self, topo):
        """Compute the norm-2 of the discretisation of the
        current field corresponding to topology topo (input arg).

        Parameters
        -----------
        topo : mpi.topology.Cartesian, optional
            the topology on which initialization is performed.

        Returns
        -------
        array-like
            norm of the field.

        Note
        -----
        May return None if the topo is not defined for the current
        MPI task.

        See Also
        --------
        Field.normh

       """
        if topo is None:
            return None
        return self.discreteFields[topo].norm()

    def normh(self, topo):
        """Compute a 'grid-norm' for the discrete field
        norm = ( hd * sum data[d](i,...)**p)**1/p for d = 1..dim
        and hd the grid step size in direction d.
        Sum on all grid points excluding ghosts.

        Parameters
        -----------
        topo : mpi.topology.Cartesian, optional
            the topology on which initialization is performed.

        Returns
        -------
        array-like
            norm of the field.

        Note
        -----
        May return None if the topo is not defined for the current
        MPI task.

        See Also
        --------
        Field.norm

        """
        if topo is None:
            return None
        return self.discreteFields[topo].normh()

    def set_formula_parameters(self, *args):
        """Set values for (optional) list of extra parameters
        that may be required in formula.

        :params args: tuple
            the extra parameters to set.
        """
        self.extraParameters = args

    def init_writer(self, discretization, io_params=None):
        """Create an hdf writer for a given discretization

        Parameters
        ----------
        discretization : :class:`~hysop.tools.params.Discretization` or
         :class:`~hysop.mpi.topology.Cartesian`
            set a data distribution description for the output, i.e.
            choose the space discretisation used for hdf file.
        io_params : :class:`~hysop.tools.io_utils.IOParams`, optional
            output file description
        """
        if io_params is None:
            io_params = IOParams(self.name, fileformat=IO.HDF5)
        wr = HDF_Writer(variables={self: discretization}, io_params=io_params)
        wr.discretize()
        wr.setup()
        self.writer[self.discreteFields[wr.topology]] = wr

    def hdf_dump(self, topo, simu):
        """Save field data into hdf file

        Parameters
        ----------
        topo : :class:`~hysop.mpi.topology.Cartesian`
            data distribution description
        simu : :class:`~hysop.problem.simulation.Simulation`
            time discretisation
        """
        df = self.discretize(topo)
        if df not in self.writer:
            self.init_writer(topo)
        self.writer[df].apply(simu)
        self.writer[df].finalize()

    def hdf_load(self, topo, io_params=None,
                 restart=None, dataset_name=None):
        """Load field from hdf fileformat

        Parameters
        ----------
        topo : :class:`~hysop.mpi.topology.Cartesian`
            data distribution description
        io_params : :class:`~hysop.tools.io_utils.IOParams`, optional
            output file description
        restart : int, optional
            iteration number. Default=None, i.e. read first iteration.
        dataset_name : string, optional
            name of the dataset in h5 file used to fill this field in.
            If None, look for self.name as dataset name.

        Notes
        -----

        * a call to hdf_load(topo, ...) will discretize the
        field on topo, if not already done
        * to read the dataset 'field1' from file ff_00001.h5::

            myfield = Field(box, name='f1')
            myfield.hdf_load(topo, IOParams('ff', restart=1,
                             dataset_name='field1')

        """
        if dataset_name is None:
            var_names = None
        else:
            var_names = {self: dataset_name}
        read = HDF_Reader(variables={self: topo},
                          io_params=io_params, restart=restart,
                          var_names=var_names)
        read.discretize()
        read.setup()
        read.apply()
        read.finalize()

    def zero(self, topo=None):
        """Reset to 0.0 all components of this field.

        :param topo : :class:`~hysop.mpi.topology.Cartesian`, optional
            if given, only discreteFields[topo] is set to zero
         """
        if topo is not None:
            self.discreteFields[topo].zero()
        else:
            for dfield in self.discreteFields.values():
                dfield.zero()
