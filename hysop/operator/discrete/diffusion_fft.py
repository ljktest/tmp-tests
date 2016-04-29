# -*- coding: utf-8 -*-
"""
@file diffusion_fft.py
Discrete Diffusion operator using FFTW (fortran)
"""
try:
    from hysop.f2hysop import fftw2py
except ImportError:
    from hysop.fakef2py import fftw2py
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.constants import debug
from hysop.tools.profiler import profile


class DiffusionFFT(DiscreteOperator):
    """
    Discretized Poisson operator based on FFTW.
    See details in hysop.operator.diffusion.

    """
    @debug
    def __init__(self, vorticity, viscosity, **kwds):
        """
        Constructor.
        @param[in,out] vorticity :  discretisation of the field \f$ \omega \f$.
        @param[in] viscosity : \f$\nu\f$, viscosity of the considered medium.
        """
        ## Discretisation of the solution field
        self.vorticity = vorticity
        ## Viscosity.
        self.viscosity = viscosity

        if self.vorticity.dimension == 1:
            raise AttributeError("1D case not yet implemented.")
        # Base class initialisation
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(DiffusionFFT, self).__init__(variables=[vorticity],
                                           **kwds)
        self.input = [self.vorticity]
        self.output = [self.vorticity]

    @debug
    @profile
    def apply(self, simulation=None):
        assert simulation is not None, \
            "Missing dt value for diffusion computation."
        dt = simulation.timeStep
        ghosts = self.vorticity.topology.ghosts()

        if self.vorticity.dimension == 2:
            self.vorticity.data = fftw2py.solve_diffusion_2d(
                self.viscosity * dt, self.vorticity.data, ghosts)

        elif self.vorticity.dimension == 3:
            self.vorticity.data[0], self.vorticity.data[1],\
                self.vorticity.data[2] = \
                fftw2py.solve_diffusion_3d(self.viscosity * dt,
                                           self.vorticity.data[0],
                                           self.vorticity.data[1],
                                           self.vorticity.data[2],
                                           ghosts)

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
