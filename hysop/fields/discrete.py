"""Discrete fields (scalars or vectors) descriptions.

.. currentmodule:: hysop

Documentation and examples can be found in :ref:`fields`.

"""
from hysop.constants import debug, ORDER
from itertools import count
from hysop.tools.profiler import Profiler, profile
import numpy as np
import hysop.tools.numpywrappers as npw
import numpy.linalg as la


class DiscreteField(object):
    """
    Discrete representation of scalar or vector fields,
    handling distributed (mpi) data (numpy arrays).

    """
    # Counter for discrete field instanciations.
    # Used to set default name and for serialization ids.
    __field_counter = count(0)

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @debug
    def __init__(self, topology, is_vector=False, name=None,
                 nb_components=None):
        """Creates a discrete field for a given topology.

        Parameters
        ----------

        topology :  :class:`~hysop.mpi.topology.Cartesian`
        is_vector : bool
            True to create a vector field
        name : string
            an id for the field (required for proper hdf i/o)
        nb_components : int
            number of components of the field

        Attributes
        ----------
        topology : :class:`~hysop.mpi.topology.Cartesian`
            description of the mpi distribution for data
        domain : :class:`~hysop.mpi.domain.box.Box`
            where the field is defined

        Note:
        -----

        is_vector = True is equivalent to nb_components = domain.dimension

        Set name parameter is always a good idea, since it allows
        a proper management of hdf i/o.

        """
        # Topology used to distribute/discretize the present field.
        self.topology = topology

        # Id (unique) for the field
        self.__id = self.__field_counter.next()
        # Field name.
        if name is not None:
            self.name = name
        else:
            self.name = 'unamed'
        #: Field dimension.
        self.dimension = topology.domain.dimension
        # Field resolution.
        self.resolution = self.topology.mesh.resolution
        # Application domain of the variable.
        self.domain = self.topology.domain
        # Object to store computational times
        self.profiler = Profiler(self, self.domain.comm_task)
        # Number of components of the field
        if nb_components is None:
            self.nb_components = self.dimension if is_vector else 1
        else:
            self.nb_components = nb_components
        # The memory space for data ...
        self.data = [npw.zeros(self.resolution)
                     for _ in xrange(self.nb_components)]

    def __getitem__(self, i):
        """Access to the content of the field.
        :param i : int
            requested index.
        :returns npw.array:
           component i of the field.
        """
        return self.data[i]

    def __setitem__(self, i, value):
        """Set all values of a component

        """
        self.data[i][...] = value

    def initialize(self, formula=None, vectorize_formula=False,
                   time=0., *args):
        """Initialize the field components

        Parameters
        ----------
        formula : python function, optional
            user-defined function of the space coordinates (at least) and time
            used to compute field values for each grid point.
        vectorize_formula : bool, optional
            true if formula must be vectorized
            (i.e. is of type 'user_func_2', see :ref:`fields` for details)
        time : real, optional
            time value used to call formula
        args : extra argument, optional, depends on formula

        """
        if formula is not None:
            # Argument of formula. Usually : x, y, z, t, extras
            arg_list = self.topology.mesh.coords + (time,) + args
            if vectorize_formula:
                # input formula is not defined for numpy arrays
                if isinstance(formula, np.lib.function_base.vectorize):
                    v_formula = formula
                else:
                    v_formula = np.vectorize(formula)
                if len(self.data) == 1:
                    self.data[0][...] = v_formula(*arg_list)
                elif len(self.data) == 2:
                    self.data[0][...], self.data[1][...] = v_formula(*arg_list)
                elif len(self.data) == 3:
                    self.data[0][...], self.data[1][...], self.data[2][...] = \
                        v_formula(*arg_list)
                else:
                    # Warning : in this case, self.data[i] memory will
                    # be reallocated.
                    print ("Warning : reallocation of memory for fields data\
                        during initialisation. See hysop.fields\
                        documentation for details.")
                    self.data = v_formula(*arg_list)
                    # Ensure that data is of the right type,
                    # in the right order.
                    for i in xrange(self.nb_components):
                        self.data[i][...] = npw.asrealarray(self.data[i])

            else:
                # In that cas, we assume that formula has been properly
                # defined as a function of (res, x, y, ...),
                # res[i], x, y ... being numpy arrays.
                self.data = formula(self.data, *arg_list)
        else:
            # No formula, set all components to zero"
            for d in xrange(self.nb_components):
                self.data[d][...] = 0.0
        assert all([(s == self.resolution).all()
                    for s in [dat.shape for dat in self.data]]),\
            "WARNING: The shape of " + self.name + " has changed during"\
            " field initialisation."

    @profile
    def randomize(self):
        """Initialize a the with random values.

        """
        for d in xrange(self.nb_components):
            self.data[d][...] = np.random.random(self.topology.mesh.resolution)

    def copy(self, field_in):
        """Initialize a field on topo with values from another field.

        Parameters
        -----------
        field_in : :class:`fields.discrete.DiscreteField`
            field to be copied
        """
        for d in xrange(self.nb_components):
            self.data[d][...] = field_in.data[d].copy(order=ORDER)

    def norm(self):
        """Compute the euclidian norm of the discrete field
        p-norm = (sum data[d](i,...)**p)**1/p for d = 1..dim
        sum on all grid points excluding ghosts.
        Warning : this is probably not a proper way
        to compute the 'real' norm of the field but
        just a way to check field distribution. To be improved.

        :return array:
            values of the norm for each component
        """
        result = npw.zeros((self.nb_components))
        gresult = npw.zeros((self.nb_components))
        ind_ng = self.topology.mesh.iCompute
        for d in range(self.nb_components):
            result[d] = la.norm(self.data[d][ind_ng]) ** 2
        self.topology.comm.Allreduce(result, gresult)
        return gresult ** 0.5

    def normh(self):
        """
        Compute a 'grid-norm' for the discrete field
        norm = ( hd * sum data[d](i,...)**2)**0.5 for d = 1..dim
        and hd the grid step size in direction d.
        Sum on all grid points excluding ghosts.
        """
        result = npw.zeros((self.nb_components))
        gresult = npw.zeros((self.nb_components))
        ind_ng = self.topology.mesh.iCompute
        step = self.topology.mesh.space_step
        for d in range(self.nb_components):
            result[d] = (step[d] * la.norm(self.data[d][ind_ng])) ** 2
        self.topology.comm.Allreduce(result, gresult)
        return gresult ** 0.5

    def zero(self):
        """ set all components to zero"""
        for dim in xrange(self.nb_components):
            self.data[dim][...] = 0.0

    def get_profiling_info(self):
        """update profiler"""
        pass
