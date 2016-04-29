# -*- coding: utf-8 -*-
"""Library of functions used to perform classical vector calculus
(diff operations like grad, curl ...)

* :class:`~hysop.numerics.differential_operations.Curl`,
* :class:`~hysop.numerics.differential_operations.DivRhoV`,
* :class:`~hysop.numerics.differential_operations.DivWV`,
* :class:`~hysop.numerics.differential_operations.GradS`,
* :class:`~hysop.numerics.differential_operations.GradV`,
* :class:`~hysop.numerics.differential_operations.GradVxW`,
* :class:`~hysop.numerics.differential_operations.DivAdvection`,
* :class:`~hysop.numerics.differential_operations.Laplacian`,
* :class:`~hysop.numerics.differential_operations.DifferentialOperation`,
 (abstract base class).

Notes
-----

For coherence sake, all input/output variables used in differential operations
are list of numpy arrays, even when only scalar fields are required.

For example :

* a = curl(b) in 3d, a and b vector fields and so lists of 3 arrays.
* a = DivRhoV(c,d), d a vector field, a and c scalar fields. d is a list of
3 arrays and a and c lists of 1 array.

"""
from hysop.constants import debug, XDIR, YDIR, ZDIR
from abc import ABCMeta
from hysop.numerics.finite_differences import FD_C_4, FD_C_2, FD2_C_2
import numpy as np
import hysop.tools.numpywrappers as npw


class DifferentialOperation(object):
    __metaclass__ = ABCMeta
    _authorized_methods = []

    # @debug
    # def __new__(cls, *args, **kw):
    #     return object.__new__(cls, *args, **kw)

    @debug
    def __init__(self, topo, indices=None, reduce_output_shape=None,
                 method=None, work=None):
        """
        Parameters
        ----------
        topo : :class:`hysop.mpi.topology.Cartesian`
        indices : list of slices, optional
            Represents the local mesh on which the operation
            will be applied,
            like iCompute in :class:`~hysop.domain.mesh.Mesh`.
            See details in notes.
        reduce_output_shape : boolean, optional
            True to return the result in a reduced array. See notes below.
        method : :class:`hysop.numerics.finite_differences.FiniteDifference`
        , optional
            the chosen FD scheme. If None, a default method is set, equal
            to self._authorized_methods[0].
        work : list of numpy arrays, optional
            internal buffers

        Notes
        -----
        * Two ways to compute outvar = operation(invar, ...)
        ** 1 - either invar and outvar are arrays of the same shape, and
        outvar[...] = op(invar[...]) will be computed. In that case
        indices and reduce_output_shape are not required.
        Indices will be set to topo.mesh.iCompute.
        ** 2 - or outvar is a smaller array than invar, and
        outvar[...] = op(invar(indices))
        To use case 2, set reduce_output_shape = True and provide indices.
        * See ClassName._authorized_methods[0] to find which default method
        is set for each Operation.
        * If work = None, some work arrays will be allocated internally.
        Else, you must provide a list of lwk arrays of shape
        topo.mesh.resolution,
        with lwk = ClassName.get_work_length(), ClassName being Curl, DivV ...
        or any other DifferentialOperation.

        """
        self._dim = topo.domain.dimension
        # Set default method
        if method is None:
            method = self._authorized_methods[0]

        self.method = method
        if indices is None:
            indices = topo.mesh.iCompute
            reduce_output_shape = False
        self._indices = indices
        self.fd_scheme = self._init_fd_method(topo, reduce_output_shape)
        self.output_indices = self.fd_scheme.output_indices
        self._work = self._set_work_arrays(work, topo, reduce_output_shape)

    def _set_work_arrays(self, work, topo, reduce_output_shape):
        """Check and allocate internal work buffers.
        """
        lwk = self.get_work_length()
        if reduce_output_shape:
            shape = self.fd_scheme.result_shape
        else:
            shape = tuple(topo.mesh.resolution)
        if work is None:
            work = []
            for _ in xrange(lwk):
                work.append(npw.asrealarray(shape))
        else:
            msg = 'Wrong input for work arrays.'
            assert isinstance(work, list), msg
            assert len(work) == lwk, msg
            for wk in work:
                assert wk.shape == shape
            return work

    @staticmethod
    def get_work_length():
        """
        Compute the number of required work arrays for this method.
        """
        return 0

    def _init_fd_method(self, topo, reduce_output_shape):
        """Build the finite difference scheme
        """
        msg = 'FD scheme Not yet implemented for this operation.'
        assert self.method.__mro__[0] in self._authorized_methods, msg
        fd_scheme = self.method(topo.mesh.space_step,
                                self._indices,
                                reduce_output_shape)
        msg = 'Ghost layer is too small for the chosen FD scheme.'
        required_ghost_layer = fd_scheme.minimal_ghost_layer
        assert (topo.ghosts() >= required_ghost_layer).all(), msg
        return fd_scheme


class Curl(DifferentialOperation):
    """
    Computes nabla X V, V being a vector field.
    """
    _authorized_methods = [FD_C_2, FD_C_4]

    def __init__(self, **kwds):
        """Curl of a vector field
        """
        super(Curl, self).__init__(**kwds)
        assert len(self._indices) > 1
        # connect to fd function call
        self.fcall = self._central
        if len(self._indices) == 2:
            # a 'fake' curl for 2D case
            self.fcall = self._central_2d

    @staticmethod
    def get_work_length():
        return 1

    def __call__(self, variable, result):
        return self.fcall(variable, result)

    def _central(self, variable, result):
        """ 3D Curl

        Parameters
        ----------
        variable : list of numpy arrays
            the input vector field
        result : list of numpy arrays
            in/out result
        """
        assert len(result) == len(variable)
        #--  d/dy vz -- in result[XDIR]
        self.fd_scheme.compute(variable[ZDIR], YDIR, result[XDIR])
        # -- -d/dz vy -- in work
        self.fd_scheme.compute(variable[YDIR], ZDIR, self._work[0])
        # result_x = d/dy vz - d/dz vy
        result[XDIR][self.output_indices] -= self._work[0][self.output_indices]

        #--  d/dz vx -- in result[YDIR]
        self.fd_scheme.compute(variable[XDIR], ZDIR, result[YDIR])
        # -- -d/dx vz -- in work
        self.fd_scheme.compute(variable[ZDIR], XDIR, self._work[0])
        # result_y = d/dz vx - d/dx vz
        result[YDIR][self.output_indices] -= self._work[0][self.output_indices]

        #-- d/dx vy in result[ZDIR]
        self.fd_scheme.compute(variable[YDIR], XDIR, result[ZDIR])
        # result_z = d/dx vy - d/dy vx
        self.fd_scheme.compute(variable[XDIR], YDIR, self._work[0])
        result[ZDIR][self.output_indices] -= self._work[0][self.output_indices]

        return result

    def _central_2d(self, variable, result):
        """ 2D Curl
        Parameters
        ----------
        variable : list of numpy arrays
            the input vector field
        result : list of numpy arrays
            in/out result
        """
        assert len(result) == 1
        #-- d/dx vy in result[ZDIR]
        self.fd_scheme.compute(variable[YDIR], XDIR, result[0])
        # result_z = d/dx vy - d/dy vx
        self.fd_scheme.compute(variable[XDIR], YDIR, self._work[0])
        result[0][self.output_indices] -= self._work[0][self.output_indices]
        return result


class DivRhoV(DifferentialOperation):
    """
    Computes \f$ \nabla.(\rho V) \f$, \f$ \rho\f$ a scalar field and
    V a vector field.
    Works for any dimension.

    Methods : FD_C_4
    1 - default : based on fd.compute.
    2 - opt : based on fd.compute_and_add.
    To use this one, call self._central4_caa.
    shape is a tuple (like np.ndarray.shape), e.g. (nx,ny,nz).

    Note FP:
    - 1 seems to be faster than 2 for large systems (>200*3) and
    less memory consumming. For smaller systems, well, it depends ...
    """

    _authorized_methods = [FD_C_4]

    def __init__(self, **kwds):
        """Divergence of rho V, rho a scalar field, V a vector field

        Parameters
        ----------
        fd_optim : bool, optional
            if 'CAA', compute result += div(rhoV) else
            result = div(rhoV), which is the default.
        **kwds : parameters for base class

        Default method : FD_C_4
        """
        super(DivRhoV, self).__init__(**kwds)
        #self.fcall = self._central4_caa
        self.fcall = self._central4

    @staticmethod
    def get_work_length():
        return 2

    def __call__(self, var1, scal, result):
        """Apply operation

        Parameters
        ----------
        var1 : list of numpy arrays
            the vector field 'V'
        scal : list of numpy array
            the scalar field
        result : list of numpy arrays
            in/out buffer
        Returns
        -------
        numpy array

        """
        assert scal is not result
        assert len(result) == len(scal)
        for i in xrange(len(var1)):
            assert var1[i] is not result
        return self.fcall(var1, scal, result)

    def _central4(self, var1, scal, result):
        """
        Compute central finite difference scheme, order 4.
        No work vector provided by user --> self._work will be
        used. It must be created at init and thus memshape is required.
        """

        # _work[0:1] are used as temporary space
        # for computation
        # div computations are accumulated into result.
        # Result does not need initialisation to zero.

        # d/dx (scal * var1x), saved into result
        np.multiply(scal[0], var1[XDIR], self._work[0])
        self.fd_scheme.compute(self._work[0], XDIR, result[0])
        # other components (if any) saved into _work[0] and added into result
        # d/dy (scal * var1y), saved in work and added into result
        # d/dz(scal * var1z), saved in work and added into result (if 3D)
        for cdir in xrange(1, len(var1)):
            np.multiply(scal[0], var1[cdir], self._work[0])
            self.fd_scheme.compute(self._work[0], cdir, self._work[1])
            result[0][self.output_indices] += \
                self._work[1][self.output_indices]

        return result

    def _central4_caa(self, var1, scal, result):
        """
        Compute central finite difference scheme, order 4.
        Use fd_scheme.compute_and_add.
        """
        # _work[0] is used as temporary space
        # for computation
        # div computations are accumulate into result.
        # result does not need initialisation to zero.

        # d/dx (scal * var1x), saved into result
        np.multiply(scal[0][self._indices], var1[XDIR][self._indices],
                    self._work[0][self.output_indices])
        self.fd_scheme.compute(self._work[0], XDIR, result[0])
        # other components (if any) saved into _work[0] and added into result
        # d/dy (scal * var1y), saved in work and added into result
        # d/dz(scal * var1z), saved in work and added into result (if 3D)
        for cdir in xrange(1, len(var1)):
            np.multiply(scal[0][self._indices], var1[cdir][self._indices],
                        self._work[0][self.output_indices])
            self.fd_scheme.compute_and_add(self._work[0], cdir, result[0])

        return result


class DivWV(DifferentialOperation):
    """
    Computes nabla.(W.Vx, W.Vy, W.Vz), W and V some vector fields.

    """

    _authorized_methods = [FD_C_4]

    def __init__(self, **kwds):
        """Divergence of (W.Vx, W.Vy, W.Vz), W, V two vector fields

        Parameters
        ----------
        fd_optim : bool, optional
            if 'CAA', compute result += div(rhoV) else
            result = div(rhoV), which is the default.
        **kwds : parameters for base class

        Default method : FD_C_4
        """
        super(DivWV, self).__init__(**kwds)
        self.fcall = self._central4
        # set div(scal.var1) operator
        self.div = DivRhoV(**kwds)

    @staticmethod
    def get_work_length():
        return DivRhoV.get_work_length()

    def __call__(self, var1, var2, result):
        """Apply operation

        Parameters
        ----------
        var1 : list of numpy arrays
            the vector field 'W'
        var2 : list of numpy arrays
            the vector field 'V'
        result : list of numpy arrays
            in/out buffers
        Returns
        -------
        list of numpy arrays

        """
        assert len(result) == len(var1)
        return self.fcall(var1, var2, result)

    def _central4(self, var1, var2, result):
        # Note FP var1[dir] and var2[dir] must be different from result.
        # This is checked inside divV call.

        for cdir in xrange(len(var1)):
            result[cdir:cdir + 1] = self.div(var1, var2[cdir:cdir + 1],
                                             result=result[cdir:cdir + 1])

        return result


class Laplacian(DifferentialOperation):
    """Computes  the laplacian of a field.
    """
    _authorized_methods = [FD2_C_2]

    @staticmethod
    def get_work_length():
        return 0

    def __call__(self, var, result):
        """"Apply laplacian

        Parameters
        ----------
        var : list of numpy arrays
            the scalar field
        result : list of numpy arrays
            in/out buffers
        Returns
        -------
        numpy array
        """
        assert len(var) == len(result)
        for d in xrange(len(var)):
            self.fd_scheme.compute(var[d], 0, result[d])
            for cdir in xrange(1, self._dim):
                self.fd_scheme.compute_and_add(var[d], cdir, result[d])
        return result


class GradS(DifferentialOperation):
    """Gradient of a scalar field
    """
    _authorized_methods = [FD_C_4, FD_C_2]

    def __call__(self, scal, result):
        """Apply gradient, with central finite difference scheme.

        Parameters
        ----------
        scal : list of numpy arrays
            the input scalar field
        result : list of numpy arrays
            in/out result
        """
        assert len(result) == self._dim
        for cdir in xrange(self._dim):
            #  d/dcdir (scal), saved in data[cdir]
            self.fd_scheme.compute(scal[0], cdir, result[cdir])

        return result


class GradV(DifferentialOperation):
    """Gradient of a vector field
    """
    _authorized_methods = [FD_C_4, FD_C_2]

    def __init__(self, **kwds):
        super(GradV, self).__init__(**kwds)
        self.grad = GradS(**kwds)

    def __call__(self, var1, result):
        """Apply gradient

        Parameters
        ----------
        var1 : list of numpy arrays
            the input vector field
        result : list of numpy arrays
            in/out result
        """
        nbc = len(var1)
        assert len(result) == nbc * self._dim
        for cdir in xrange(self._dim):
            i1 = cdir * nbc
            i2 = i1 + nbc
            result[i1:i2] = self.grad(var1[cdir:cdir + 1], result[i1:i2])

        return result


class GradVxW(DifferentialOperation):
    """Computes [nabla(V)][W] with
    V and W some vector fields.
    """
    _authorized_methods = [FD_C_4, FD_C_2]

    @staticmethod
    def get_work_length():
        return 2

    def __call__(self, var1, var2, result, diagnostics):
        """Apply gradient, with central finite difference scheme.

        Parameters
        ----------
        var1, var2 : list of numpy arrays
           the input vector fields
        result : list of numpy arrays
            in/out result. Overwritten.
        diagnostics : numpy array
            some internal diagnostics (like max of div(v) ...).
            In/out param, overwritten
        """
        assert len(result) == len(var1)
        nbc = len(var1)
        diagnostics[:] = 0.0
        for comp in xrange(nbc):
            result[comp][...] = 0.0
            self._work[1][...] = 0.0
            for cdir in xrange(self._dim):
                # self._work = d/dcdir (var1_comp)
                self.fd_scheme.compute(var1[comp], cdir, self._work[0])
                # some diagnostics ...
                if cdir == comp:
                    tmp = np.max(abs(self._work[0]))
                    diagnostics[0] = max(diagnostics[0], tmp)
                np.add(abs(self._work[0]), self._work[1], self._work[1])

                # compute self._work = self._work.var2[cdir]
                np.multiply(self._work[0][self.output_indices],
                            var2[cdir][self._indices],
                            self._work[0][self.output_indices])
                # sum to obtain nabla(var_comp) . var2, saved into result[comp]
                npw.add(result[comp][self.output_indices],
                        self._work[0][self.output_indices],
                        result[comp][self.output_indices])
            diagnostics[1] = max(diagnostics[1], np.max(self._work[1]))
        return result, diagnostics


class DivAdvection(DifferentialOperation):
    """ Computes -nabla .(V . nabla V) with V a vector field.
    """
    _authorized_methods = [FD_C_4, FD_C_2]

    @staticmethod
    def get_work_length():
        return 3

    def __call__(self, var1, result):
        """Apply divergence, with central finite difference scheme.

        Parameters
        ----------
        var1 : list of numpy arrays
            the input vector field
        result : list of numpy array
            in/out result, overwritten.
        """
        assert len(result) == 1
        nbc = len(var1)
        assert nbc == 3, 'Only 3D case is implemented.'
        # Compute diff(var[dir], dir), saved in work[dir]
        for d in xrange(nbc):
            self.fd_scheme.compute(var1[d], d, self._work[d])

        # result = Vx,x * Vy,y
        np.multiply(self._work[XDIR][self.output_indices],
                    self._work[YDIR][self.output_indices],
                    result[0][self.output_indices])
        # wk[0] = Vx,x * Vz,z
        np.multiply(self._work[XDIR][self.output_indices],
                    self._work[ZDIR][self.output_indices],
                    self._work[XDIR][self.output_indices])
        # result = result + Vx,x * Vz,z
        np.add(self._work[XDIR][self.output_indices],
               result[0][self.output_indices],
               result[0][self.output_indices])
        # wk[1] = Vy,y * Vz,z
        np.multiply(self._work[YDIR][self.output_indices],
                    self._work[ZDIR][self.output_indices],
                    self._work[YDIR][self.output_indices])
        # result = result + Vy,y * Vz,z
        np.add(self._work[YDIR][self.output_indices],
               result[0][self.output_indices],
               result[0][self.output_indices])

        self.fd_scheme.compute(var1[XDIR], YDIR, self._work[0])
        self.fd_scheme.compute(var1[YDIR], XDIR, self._work[1])
        # wk[0] = Vx,y * Vy,x
        np.multiply(self._work[0][self.output_indices],
                    self._work[1][self.output_indices],
                    self._work[0][self.output_indices])
        # result = result - Vx,y * Vy,x
        np.subtract(result[0], self._work[0], result[0])

        self.fd_scheme.compute(var1[XDIR], ZDIR, self._work[0])
        self.fd_scheme.compute(var1[ZDIR], XDIR, self._work[1])
        # wk[0] = Vx,z * Vz,x
        np.multiply(self._work[0], self._work[1], self._work[0])
        # result = result - Vx,z * Vz,x
        np.subtract(result[0][self.output_indices],
                    self._work[0][self.output_indices],
                    result[0][self.output_indices])
        self.fd_scheme.compute(var1[YDIR], ZDIR, self._work[0])
        self.fd_scheme.compute(var1[ZDIR], YDIR, self._work[1])
        # wk[0] = Vy,z * Vz,y
        np.multiply(self._work[0][self.output_indices],
                    self._work[1][self.output_indices],
                    self._work[0][self.output_indices])
        # result = result - Vy,z * Vz,y
        np.subtract(result[0][self.output_indices],
                    self._work[0][self.output_indices],
                    result[0][self.output_indices])

        result[0][self.output_indices] *= 2.0
        return result
