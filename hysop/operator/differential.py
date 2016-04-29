"""Differential operators

.. currentmodule hysop.operator.differential

* :class:`~Curl`,
* :class:`~Grad`,
* :class:`~DivAdvection`,
* :class:`~Differential` (abstract base class).


"""
from hysop.constants import debug
from hysop.operator.computational import Computational
from hysop.operator.discrete.differential import CurlFFT, CurlFD,\
    GradFD, DivAdvectionFD
from hysop.methods_keys import SpaceDiscretisation
from hysop.operator.continuous import opsetup
from hysop.numerics.finite_differences import FiniteDifference
import hysop.default_methods as default
from abc import ABCMeta, abstractmethod
from hysop.numerics.differential_operations import Curl as NumCurl


class Differential(Computational):
    """Abstract base class for differential operators
    """
    __metaclass__ = ABCMeta

    # @debug
    # def __new__(cls, *args, **kw):
    #     return object.__new__(cls, *args, **kw)

    @debug
    def __init__(self, invar, outvar, **kwds):
        """
        Parameters
        ----------
        invar, outvar : :class:`~hysop.fields.continuous.Field`
           input/output scalar or vector fields
            such that outvar = op(invar).
        **kwds : base class parameters

       """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Differential, self).__init__(variables=[invar, outvar], **kwds)
        if self.method is None:
            self.method = default.DIFFERENTIAL

        # input variable
        self.invar = invar
        # result of the operator
        self.outvar = outvar

        # Remark : at the time, all variables must have the same topology.
        # This is implicitely checked with the assert on kwds['variables']:
        # the only construction allowed is :
        # (invar= ..., outvar=..., discretization=...)
        self.output = [outvar]
        self.input = [invar]
        # Discrete operator type. Set in derived class.
        self._discrete_op_class = self._init_space_discr_method()

    @abstractmethod
    def _init_space_discr_method(self):
        """Select the proper discretisation for the operator
        """

    def discretize(self):
        """Build topologies for all variables.

        At the time, two space-discretization methods : based on
        FFT or on Finite Differences.
        """
        space_d = self.method[SpaceDiscretisation]
        if space_d is 'fftw':
            super(Differential, self)._fftw_discretize()

        elif space_d.mro()[1] is FiniteDifference:
            number_of_ghosts = space_d.ghosts_layer_size
            super(Differential, self)._standard_discretize(number_of_ghosts)
        else:
            raise ValueError("Unknown method for space discretization of the\
                differential operator.")

        msg = 'Operator not yet implemented for multiple resolutions.'
        assert self._single_topo, msg

    @opsetup
    def setup(self, rwork=None, iwork=None):
        """
        Last step of initialization. After this, the operator must be
        ready for apply call.

        Main step : setup for discrete operators.
        """
        self.discrete_op = self._discrete_op_class(
            invar=self.discreteFields[self.invar],
            outvar=self.discreteFields[self.outvar],
            method=self.method, rwork=rwork)
        self._is_uptodate = True


class Curl(Differential):
    """Computes outVar = nabla X inVar
    """

    def _init_space_discr_method(self):
        if self.method[SpaceDiscretisation] is 'fftw':
            op_class = CurlFFT
        elif self.method[SpaceDiscretisation].mro()[1] is FiniteDifference:
            op_class = CurlFD
        else:
            raise ValueError("The required Space Discretisation is\
                not available for Curl.")
        return op_class

    def get_work_properties(self):
        """Get properties of internal work arrays. Must be call after discretize
        but before setup.

        Returns
        -------
        dictionnary
           keys = 'rwork' and 'iwork', values = list of shapes of internal
           arrays required by this operator (real arrays for rwork, integer
           arrays for iwork).
        """
        if not self._is_discretized:
            msg = 'The operator must be discretized '
            msg += 'before any call to this function.'
            raise RuntimeError(msg)
        res = {'rwork': None, 'iwork': None}
        # Only FD methods need internal work space
        if self.method[SpaceDiscretisation].mro()[1] is FiniteDifference:
            work_length = NumCurl.get_work_length()
            shape = self.discreteFields[self.invar].data[0].shape
            res['rwork'] = []
            for _ in xrange(work_length):
                res['rwork'].append(shape)
        return res


class Grad(Differential):
    """Computes outVar = nabla(inVa)
    """

    def _init_space_discr_method(self):
        if self.method[SpaceDiscretisation].mro()[1] is FiniteDifference:
            op_class = GradFD
        else:
            raise ValueError("The required Space Discretisation is\
                not available for Grad.")
        return op_class


class DivAdvection(Differential):
    """Computes outVar = -nabla .(invar . nabla(nvar))
    """
    def _init_space_discr_method(self):
        if self.method[SpaceDiscretisation].mro()[1] is FiniteDifference:
            op_class = DivAdvectionFD
        else:
            raise ValueError("The required Space Discretisation is\
                not available for DivAdvection.")
        return op_class
