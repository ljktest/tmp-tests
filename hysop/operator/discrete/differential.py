# -*- coding: utf-8 -*-
"""Discretisation of the differential operators (curl, grad ...)

..currentmodule hysop.operator.discrete.differential
* :class:`~CurlFFT`,
* :class:`~CurlFD`,
* :class:`~GradFD`,
* :class:`~DivAdvectionFD`,
* :class:`~Differential` (abstract base class).

"""
from hysop.constants import debug
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.numerics.differential_operations import Curl, GradV,\
    DivAdvection
import hysop.tools.numpywrappers as npw
from abc import ABCMeta, abstractmethod
from hysop.numerics.update_ghosts import UpdateGhosts
from hysop.methods_keys import SpaceDiscretisation
try:
    from hysop.f2hysop import fftw2py
except ImportError:
    from hysop.fakef2py import fftw2py
import hysop.default_methods as default
from hysop.tools.profiler import profile


class Differential(DiscreteOperator):
    """Abstract base class for discrete differential operators
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
        invar, outvar : :class:`~hysop.fields.discrete.DiscreteField`
           input/output scalar or vector fields
            such that outvar = op(invar).
        **kwds : base class parameters

        """
        self.invar = invar
        self.outvar = outvar
        if 'method' not in kwds:
            kwds['method'] = default.DIFFERENTIAL
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Differential, self).__init__(variables=[invar, outvar],
                                           **kwds)
        self.input = [self.invar]
        self.output = [self.outvar]
        self._synchronize = None
        # connexion to a numerical method
        self._function = None

    @abstractmethod
    def apply(self, simulation=None):
        """
        Abstract interface
        """


class CurlFFT(Differential):
    """Computes the curl of a discrete field, using Fourier fftw
    """

    def __init__(self, **kwds):
        super(CurlFFT, self).__init__(**kwds)
        if self.domain.dimension == 3:
            self._apply = self._apply_3d
        elif self.domain.dimension == 2:
            self._apply = self._apply_2d

    def apply(self, simulation=None):
        self._apply()

    @debug
    @profile
    def _apply_3d(self):
        ghosts_in = self.invar.topology.ghosts()
        ghosts_out = self.outvar.topology.ghosts()
        self.outvar.data[0], self.outvar.data[1], self.outvar.data[2] = \
            fftw2py.solve_curl_3d(self.invar.data[0], self.invar.data[1],
                                  self.invar.data[2], self.outvar.data[0],
                                  self.outvar.data[1], self.outvar.data[2],
                                  ghosts_in, ghosts_out)

    def _apply_2d(self):
        ghosts_in = self.invar.topology.ghosts()
        ghosts_out = self.outvar.topology.ghosts()
        self.outvar.data[0] = \
            fftw2py.solve_curl_2d(self.invar.data[0], self.invar.data[1],
                                  self.outvar.data[0],
                                  ghosts_in, ghosts_out)

    def finalize(self):
        """Clean memory (fftw plans and so on)
        """
        fftw2py.clean_fftw_solver(self.outvar.dimension)


class CurlFD(Differential):
    """Computes the curl of a discrete field, using finite differences.
    """

    def __init__(self, **kwds):

        super(CurlFD, self).__init__(**kwds)

        # prepare ghost points synchro for velocity
        self._synchronize = UpdateGhosts(self.invar.topology,
                                         self.invar.nb_components)
        self._function = Curl(topo=self.invar.topology, work=self._rwork,
                              method=self.method[SpaceDiscretisation])

    def _set_work_arrays(self, rwork, iwork):
        worklength = Curl.get_work_length()
        memshape = self.invar.data[0].shape
        if rwork is None:
            self._rwork = [npw.zeros(memshape) for _ in xrange(worklength)]
        else:
            assert isinstance(rwork, list), 'rwork must be a list.'
            # --- External rwork ---
            self._rwork = rwork
            assert len(self._rwork) == worklength
            for wk in self._rwork:
                assert wk.shape == memshape

    @debug
    @profile
    def apply(self, simulation=None):
        self._synchronize(self.invar.data)
        self.outvar.data = self._function(self.invar.data, self.outvar.data)


class GradFD(Differential):
    """Computes the grad of a discrete field, using finite differences.
    """

    def __init__(self, **kwds):

        super(GradFD, self).__init__(**kwds)
        # prepare ghost points synchro for velocity
        self._synchronize = UpdateGhosts(self.invar.topology,
                                         self.invar.nb_components)
        dim = self.domain.dimension
        assert self.outvar.nb_components == dim * self.invar.nb_components
        self._function = GradV(topo=self.invar.topology,
                               method=self.method[SpaceDiscretisation])

    @debug
    @profile
    def apply(self, simulation=None):
        self._synchronize(self.invar.data)
        self.outvar.data = self._function(self.invar.data, self.outvar.data)


class DivAdvectionFD(Differential):
    """Computes  outVar = -nabla .(invar . nabla(nvar))
    """

    def __init__(self, **kwds):

        super(DivAdvectionFD, self).__init__(**kwds)
        # prepare ghost points synchro for velocity
        self._synchronize = UpdateGhosts(self.invar.topology,
                                         self.invar.nb_components)
        assert self.outvar.nb_components == 1
        self._function = DivAdvection(topo=self.invar.topology,
                                      method=self.method[SpaceDiscretisation],
                                      work=self._rwork)

    def _set_work_arrays(self, rwork, iwork):
        worklength = DivAdvection.get_work_length()
        memshape = self.invar.data[0].shape
        if rwork is None:
            self._rwork = [npw.zeros(memshape) for _ in xrange(worklength)]
        else:
            assert isinstance(rwork, list), 'rwork must be a list.'
            # --- External rwork ---
            self._rwork = rwork
            assert len(self._rwork) == worklength
            for wk in self._rwork:
                assert wk.shape == memshape

    @debug
    @profile
    def apply(self, simulation=None):
        self._synchronize(self.invar.data)
        self.outvar.data = self._function(self.invar.data, self.outvar.data)
