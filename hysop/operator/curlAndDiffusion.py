# -*- coding: utf-8 -*-
"""
@file diffusion.py

Operator for diffusion problem.

"""
from hysop.operator.continuous import Operator
try:
    from hysop.f2hysop import fftw2py
except ImportError:
    from hysop.fakef2py import fftw2py
from hysop.operator.discrete.diffusion_fft import DiffusionFFT
from hysop.constants import debug
from hysop.operator.continuous import opsetup


class CurlDiffusion(Operator):
    """
    Diffusion operator
    \f{eqnarray*}
    \omega = Op(\omega)
    \f} with :
    \f{eqnarray*}
    \frac{\partial \omega}{\partial t} &=& \nu\Delta\omega
    \f}
    """

    @debug
    def __init__(self, velocity, vorticity, **kwds):
        """
        Constructor.
        Create a Diffusion operator using FFT.

        @param velocity ContinuousVectorField : velocity variable.
        @param vorticity ContinuousVectorField : vorticity variable.
        @param viscosity : viscosity of the considered medium.
        """
        super(CurlDiffusion, self).__init__(variables=[velocity, vorticity], **kwds)
        self.velocity = velocity
        self.vorticity = vorticity
        raise ValueError("This operator is obsolete and must be reviewed.\
                          Do not use it.")

    @debug
    @opsetup
    def setup(self):
        """
        Diffusion operator discretization method.
        Create a discrete Diffusion operator from given specifications.
        """
        if self._comm is None:
            from hysop.mpi.main_var import main_comm as comm
        else:
            comm = self._comm

        localres, localoffset = fftw2py.init_fftw_solver(
            self.resolutions[self.vorticity],
            self.domain.length, comm=comm.py2f())

        topodims = self.resolutions[self.vorticity] / localres
        print ('topodims DIFFUSION', topodims)
        # variables discretization

        for v in self.variables:
            topo = self.domain.getOrCreateTopology(self.domain.dimension,
                                                   self.resolutions[v],
                                                   topodims,
                                                   comm=comm)
            vd = v.discretize(topo)
            self.discreteFields[v] = vd

        self.discrete_op =\
            DiffusionFFT(self.discreteFields[self.velocity],
                         self.discreteFields[self.vorticity],
                         self.method, **self.config)

        self.discrete_op.setup()
        self._is_uptodate = True
