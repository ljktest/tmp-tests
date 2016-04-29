"""Finite difference schemes
"""
from abc import ABCMeta, abstractmethod
from hysop.constants import debug
import hysop.tools.numpywrappers as npw
import numpy as np


class FiniteDifference(object):
    """Describe and apply a finite difference scheme to compute
    1st or second derivative of a variable saved in a numpy array.

    Usage :

    >> step = topo.mesh_space_step
    >> scheme = FD_C_4(step, topo.mesh.iCompute)

    For a given numpy array (obviously discretized on the topo
    used to compute indices), to compute result as the derivative
    of tab according to dir:

    >> scheme.compute(tab, dir, result)

    or result += the derivative of tab according to dir:

    >> scheme.compute_and_add(tab, dir, result)

    Notes FP :
    Compute method is much more performant than compute and add
    and needs less memory. But in some cases (when fd results need
    to be accumulate into a field) compute_and_add may be useful
    to limit memory usage, if required.
    See Global_tests/testPerfAndMemForFD_and_div.py for perf results.
    """

    __metaclass__ = ABCMeta

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    def __init__(self, step, indices, reduce_output_shape=False):
        """

        Parameters
        ----------
        step : list or array of int
            resolution of the mesh
        indices : list of slices
            Represents the local mesh on which finite-differences
            will be applied, like iCompute in :class:`~hysop.domain.mesh.Mesh`.
        reduce_output_shape : boolean, optional
            True to return the result in a reduced array. See notes below.

        Attributes
        ----------
        minimal_ghost_layer : int
            minimal number of points required inside the ghost layer
            for this scheme.

        Notes
        -----
        Two ways to compute result = scheme(input)
        * 1 - either input and result are arrays of the same shape, and
        result[indices] = scheme(input[indices]) will be computed
        * 2 - or result is a smaller array than input, and
        result[...] = scheme(input(indices))
        To use case 2, set reduce_output_shape = True.
        """
        
        step = np.asarray(step)
        #  dim of the field on which scheme will be applied
        # (i.e dim of the domain)
        self._dim = step.size
        self._m1 = None
        self._a1 = None
        self._m2 = None
        self._a2 = None
        self._coeff = None
        # List of slices representing the mesh on which fd scheme is applied
        self._indices = indices
        if not reduce_output_shape:
            self.result_shape = None
            self.output_indices = self._indices
        else:
            self.result_shape = tuple([indices[i].stop - indices[i].start
                                       for i in xrange(len(indices))])
            self.output_indices = [slice(0, self.result_shape[i])
                                   for i in xrange(len(indices))]

        # Minimal size of the ghost layer.
        self.minimal_ghost_layer = 1
        self._step = step
        self._compute_indices(step)

    @abstractmethod
    def _compute_indices(self, step):
        """Internal index lists and fd coeff computation

        """

    @abstractmethod
    def __call__(self, tab, cdir, result):
        """Apply FD scheme.
        """

    def compute(self, tab, cdir, result):
        """Apply FD scheme. Result is overwritten.

        Parameters
        ----------
        tab : numpy array
            input field
        cdir : int
            direction of differentiation
        result : numpy array
            in/out, derivative of tab
        """
        assert result is not tab
        assert result.__class__ is np.ndarray
        assert result.shape == self.result_shape or result.shape == tab.shape
        assert tab.__class__ is np.ndarray
        self.__call__(tab, cdir, result)

    @abstractmethod
    def compute_and_add(self, tab, cdir, result):
        """Apply FD scheme and add the result inplace.

        Parameters
        ----------
        tab : numpy array
            input field
        cdir : int
            direction of differentiation
        result : numpy array
            in/out, derivative of tab + result
        """


class FD_C_2(FiniteDifference):
    """
    1st derivative, centered scheme, 2nd order.

    """

    ghosts_layer_size = 1

    def _compute_indices(self, step):

        self.minimal_ghost_layer = 1
        self._coeff = npw.asarray(1. / (2. * step))
        self._m1 = []
        self._a1 = []
        for dim in xrange(self._dim):
            self._m1.append(list(self._indices))
            self._m1[dim][dim] = slice(self._indices[dim].start - 1,
                                       self._indices[dim].stop - 1,
                                       self._indices[dim].step)
            self._a1.append(list(self._indices))
            self._a1[dim][dim] = slice(self._indices[dim].start + 1,
                                       self._indices[dim].stop + 1,
                                       self._indices[dim].step)

    def __call__(self, tab, cdir, result):
        result[self.output_indices] = tab[self._a1[cdir]]
        result[self.output_indices] -= tab[self._m1[cdir]]
        result[self.output_indices] *= self._coeff[cdir]
        return result

    def compute_and_add(self, tab, cdir, result):
        """Apply FD scheme and add the result inplace.

        Parameters
        ----------
        tab : numpy array
            input field
        cdir : int
            direction of differentiation
        result : numpy array
            in/out, derivative of tab + result
        """
        assert result.__class__ is np.ndarray
        assert tab.__class__ is np.ndarray
        result[self.output_indices] += \
            self._coeff[cdir] * (tab[self._a1[cdir]] - tab[self._m1[cdir]])


class FD2_C_2(FiniteDifference):
    """
    Second derivative, centered scheme, 2nd order.
    """

    ghosts_layer_size = 1

    def _compute_indices(self, step):

        self.minimal_ghost_layer = 1
        self._m1 = []
        self._a1 = []
        self._coeff = npw.asarray(1. / (step * step))
        for dim in xrange(self._dim):
            self._m1.append(list(self._indices))
            self._m1[dim][dim] = slice(self._indices[dim].start - 1,
                                       self._indices[dim].stop - 1,
                                       self._indices[dim].step)
            self._a1.append(list(self._indices))
            self._a1[dim][dim] = slice(self._indices[dim].start + 1,
                                       self._indices[dim].stop + 1,
                                       self._indices[dim].step)

    def __call__(self, tab, cdir, result):
        result[self.output_indices] = tab[self._indices]
        result[self.output_indices] *= -2
        result[self.output_indices] += tab[self._a1[cdir]]
        result[self.output_indices] += tab[self._m1[cdir]]
        result[self.output_indices] *= self._coeff[cdir]
        return result

    def compute_and_add(self, tab, cdir, result):
        """Apply FD scheme and add the result inplace.

        Parameters
        ----------
        tab : numpy array
            input field
        cdir : int
            direction of differentiation
        result : numpy array
            in/out, derivative of tab + result
        """
        assert result.__class__ is np.ndarray
        assert tab.__class__ is np.ndarray
        result[self.output_indices] += self._coeff[cdir] * (
            -2 * tab[self._indices] + tab[self._m1[cdir]] + tab[self._a1[cdir]]
        )


class FD_C_4(FiniteDifference):
    """
    1st derivative, centered scheme, 4th order.
    """

    ghosts_layer_size = 2

    def _compute_indices(self, step):

        self.minimal_ghost_layer = 2
        self._m1 = []
        self._m2 = []
        self._a1 = []
        self._a2 = []
        # FD scheme coefficients
        self._coeff = npw.asarray(1. / (12. * step))
        for dim in xrange(self._dim):
            self._m1.append(list(self._indices))
            self._m1[dim][dim] = slice(self._indices[dim].start - 1,
                                       self._indices[dim].stop - 1,
                                       self._indices[dim].step)
            self._m2.append(list(self._indices))
            self._m2[dim][dim] = slice(self._indices[dim].start - 2,
                                       self._indices[dim].stop - 2,
                                       self._indices[dim].step)
            self._a1.append(list(self._indices))
            self._a1[dim][dim] = slice(self._indices[dim].start + 1,
                                       self._indices[dim].stop + 1,
                                       self._indices[dim].step)
            self._a2.append(list(self._indices))
            self._a2[dim][dim] = slice(self._indices[dim].start + 2,
                                       self._indices[dim].stop + 2,
                                       self._indices[dim].step)

    def __call__(self, tab, cdir, result):
        result[self.output_indices] = tab[self._a1[cdir]]
        result[self.output_indices] -= tab[self._m1[cdir]]
        result[self.output_indices] *= 8
        result[self.output_indices] += tab[self._m2[cdir]]
        result[self.output_indices] -= tab[self._a2[cdir]]
        result[self.output_indices] *= self._coeff[cdir]
        return result

    def compute_and_add(self, tab, cdir, result):
        """Apply FD scheme and add the result inplace.

        Parameters
        ----------
        tab : numpy array
            input field
        cdir : int
            direction of differentiation
        result : numpy array
            in/out, derivative of tab + result
        """
        assert result.__class__ is np.ndarray
        assert tab.__class__ is np.ndarray
        result[self.output_indices] += self._coeff[cdir] * (
            8 * (tab[self._a1[cdir]] - tab[self._m1[cdir]]) +
            tab[self._m2[cdir]] - tab[self._a2[cdir]])
