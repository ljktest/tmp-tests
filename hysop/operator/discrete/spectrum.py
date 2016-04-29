# -*- coding: utf-8 -*-
"""
@file spectrum.py
Discrete Spectrum operator using FFTW (fortran)
"""
try:
    from hysop.f2hysop import fftw2py
except ImportError:
    from hysop.fakef2py import fftw2py
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.constants import debug, np, HYSOP_MPI_REAL
from hysop.tools.profiler import profile
from hysop.tools.io_utils import IO
import hysop.tools.numpywrappers as npw
from hysop.mpi import MPI
import os


class FFTSpectrum(DiscreteOperator):
    """
    Discretized Spectrum operator based on FFTW.

    """
    @debug
    def __init__(self, field, prefix=None, **kwds):
        """
        Constructor.
        @param[in] vorticity : field to compute.
        """
        # Discretisation of the input field
        self.field = field

        if self.field.nb_components > 1:
            raise AttributeError("Vector case not yet implemented.")
        # Base class initialisation
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(FFTSpectrum, self).__init__(variables=[field],
                                          **kwds)
        self.input = [self.field]
        l = np.min(self.field.topology.mesh.discretization.resolution)
        self._tmp = npw.zeros(((l - 1) / 2, ))
        self._kx = npw.zeros(((l - 1) / 2, ))
        self.res = npw.zeros(((l - 1) / 2, ))
        self._prefix = "spectrum" if prefix is None else prefix
        IO.check_dir(os.path.join(IO.default_path(),
                                  self._prefix + "_00000.dat"))

    @debug
    @profile
    def apply(self, simulation=None):
        assert simulation is not None, \
            "Missing dt value for diffusion computation."
        ite = simulation.currentIteration
        ghosts = self.field.topology.ghosts()
        if self.field.dimension == 3:
            fftw2py.spectrum_3d(self.field.data[0],
                                self._tmp, self._kx,
                                ghosts, np.min(self.domain.length))
            if self.field.topology.size == 1:
                self.res[...] = self._tmp
            else:
                self.field.topology.comm.Reduce(
                    [self._tmp, self.res.shape[0], HYSOP_MPI_REAL],
                    [self.res, self.res.shape[0], HYSOP_MPI_REAL],
                    op=MPI.SUM, root=0)
            if self.field.topology.rank == 0:
                _file = open(os.path.join(
                    IO.default_path(),
                    self._prefix + "_{0:05d}.dat".format(ite)), 'w')
                for i in xrange(self.res.shape[0]):
                    _file.write("{0} {1}\n".format(
                        self._kx[i], self.res[i]))
                _file.close()
        else:
            raise ValueError("invalid problem dimension")

    def finalize(self):
        """
        Clean memory (fftw plans and so on)
        """
        pass
        # TODO : fix bug that occurs when several finalize
        # of fft operators are called.
        # fftw2py.clean_fftw_solver(self.vorticity.dimension)
