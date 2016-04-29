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
    def __init__(self, vorticity, viscosity, method=None):
        """
        Constructor.
        @param[in,out] vorticity :  discretisation of the field \f$ \omega \f$.
        @param[in] viscosity : \f$\nu\f$, viscosity of the considered medium.
        """

        DiscreteOperator.__init__(self, [vorticity], method,
                                  name="Diffusion FFT")
        ## Velocity.
        self.velocity = velocity
        ## Vorticity.
        self.vorticity = vorticity
        ## Viscosity.
        self.viscosity = viscosity
        ## Boolean : pure vort diffusion or curl(u) + vort diffusion.
        self.with_curl = with_curl

    @debug
    @profile
    def apply(self, simulation):

        if (self.vorticity.dimension == 2):

            if self.with_curl:
                raise ValueError("Not yet implemented")

            else:
                self.vorticity.data = \
                    fftw2py.solve_diffusion_2d(self.viscosity * dt,
                                               self.vorticity.data)

        elif (self.vorticity.dimension == 3):

            if self.with_curl:
                self.vorticity.data[0], self.vorticity.data[1],\
                    self.vorticity.data[2] = \
                    fftw2py.solve_curl_diffusion_3d(self.viscosity * dt,
                                                    self.velocity.data[0],
                                                    self.velocity.data[1],
                                                    self.velocity.data[2],
                                                    self.vorticity.data[0],
                                                    self.vorticity.data[1],
                                                    self.vorticity.data[2])

            else:
                self.vorticity.data[0], self.vorticity.data[1],\
                    self.vorticity.data[2] = \
                    fftw2py.solve_diffusion_3d(self.viscosity * dt,
                                               self.vorticity.data[0],
                                               self.vorticity.data[1],
                                               self.vorticity.data[2])

        else:
            raise ValueError("invalid problem dimension")
#        ind0a = self.topology.mesh.local_start[0]
#        ind0b = self.topology.mesh.local_end[0] + 1
#        ind1a = self.topology.mesh.local_start[1]
#        ind1b = self.topology.mesh.local_end[1] + 1
#        ind2a = self.topology.mesh.local_start[2]
#        ind2b = self.topology.mesh.local_end[2] + 1

#        vorticityNoG = [npw.zeros((self.resolution - 2 * self.ghosts))
#                        for d in xrange(self.dim)]
#        velocityNoG = [nwp.zeros((self.resolution - 2 * self.ghosts))
#                       for d in xrange(self.dim)]
#        for i in xrange(self.dim):
#            vorticityNoG[i][...] = self.vorticity[i][ind0a:ind0b,
#                                                     ind1a:ind1b, ind2a:ind2b]
#            velocityNoG[i][...] = self.velocity[i][ind0a:ind0b,
#                                                   ind1a:ind1b, ind2a:ind2b]

#        # Curl + Vorticity diffusion
##        vorticityNoG[0][...], vorticityNoG[1][...], vorticityNoG[2][...] = \
##            fftw2py.solve_curl_diffusion_3d(self.viscosity * dt,
##                                       velocityNoG[0][...],
##                                       velocityNoG[1][...],
##                                       velocityNoG[2][...],
##                                       vorticityNoG[0][...],
##                                       vorticityNoG[1][...],
##                                       vorticityNoG[2][...])

#        # Pure Vorticity diffusion
#        vorticityNoG[0][...], vorticityNoG[1][...], vorticityNoG[2][...] = \
#            fftw2py.solve_diffusion_3d(self.viscosity * dt,
#                                                  vorticityNoG[0][...],
#                                                  vorticityNoG[1][...],
#                                                  vorticityNoG[2][...])

#        for i in xrange(self.dim):
#            self.vorticity[i][ind0a:ind0b, ind1a:ind1b, ind2a:ind2b] = \
#                vorticityNoG[i][...]

    def __str__(self):
        s = "Diffusion_d (DiscreteOperator). " + DiscreteOperator.__str__(self)
        return s
