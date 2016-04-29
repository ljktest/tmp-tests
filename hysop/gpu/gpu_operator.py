"""
@file gpu_operator.py

Discrete operator for GPU architecture.
"""
from abc import ABCMeta
from hysop.constants import HYSOP_REAL, S_DIR
from hysop.methods_keys import Precision
from hysop.gpu.tools import get_opencl_environment


class GPUOperator(object):
    """
    Abstract class for GPU operators.

    In practice, discrete operators must inherit from the classic
    version and this abstract layer.
    """
    __metaclass__ = ABCMeta

    def __init__(self, platform_id=None, device_id=None, device_type=None,
                 user_src=None, **kwds):
        """
        Create the common attributes of all GPU discrete operators.
        All in this interface is independant of a discrete operator.

        @param platform_id : OpenCL platform id (default = 0).
        @param device_id : OpenCL device id (default = 0).
        @param device_type : OpenCL device type (default = 'gpu').
        @param user_src : User OpenCL sources.
        """
        ## real type precision on GPU
        self.gpu_precision = HYSOP_REAL
        if 'method' in kwds and Precision in kwds['method'].keys():
            self.gpu_precision = kwds['method'][Precision]

        # Initialize opencl environment
        comm_ref = self.variables[0].topology.comm
        # use mpi_param comm instead?
        self.cl_env = get_opencl_environment(
            platform_id=platform_id, device_id=device_id,
            device_type=device_type, precision=self.gpu_precision,
            comm=comm_ref)

        # Functions to get the appropriate vectors for the current direction
        self.dim = self.domain.dimension
        self._reorderVect = lambda v: v
        if self.dim == 2 and self.direction == 1:
            self._reorderVect = lambda v: (v[1], v[0])
        if self.dim == 3 and self.direction == 1:
            self._reorderVect = lambda v: (v[1], v[0], v[2])
        if self.dim == 3 and self.direction == 2:
            self._reorderVect = lambda v: (v[2], v[0], v[1])

        # Size constants for local mesh size
        self._size_constants = ""

        self._kernel_cfg = \
            self.cl_env.kernels_config[self.dim][self.gpu_precision]

        self._num_locMem = None
        ## Global memory allocated on gpu by this operator
        self.size_global_alloc = 0
        ## Local memory allocated on gpu by this operator
        self.size_local_alloc = 0

    def _append_size_constants(self, values, prefix='NB', suffix=None):
        """
        Append to the string containing the constants for building kernels.
        @param values : values to add
        @param prefix : prefix of variables names
        @param suffix : suffix of variables names
        """
        if suffix is None:
            suffix = S_DIR
        assert len(values) <= len(suffix), str(values) + str(suffix)
        for v, s in zip(values, suffix):
            self._size_constants += " -D " + prefix + s + "=" + str(v)
