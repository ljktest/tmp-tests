# -*- coding: utf-8 -*-
"""
@file diffusion.py

Operator for diffusion problem.

"""
from hysop.operator.computational import Computational
from hysop.operator.discrete.diffusion_fft import DiffusionFFT
from hysop.constants import debug
from hysop.operator.continuous import opsetup
from hysop.methods_keys import SpaceDiscretisation
import hysop.default_methods as default


class Diffusion(Computational):
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
    def __init__(self, viscosity, vorticity=None, **kwds):
        """
        Constructor for the diffusion operator.
        @param[in,out] vorticity : vorticity field. If None, it must be passed
        through variables argument
        @param[in] viscosity : viscosity of the considered medium.
        """
        if vorticity is not None:
            super(Diffusion, self).__init__(variables=[vorticity], **kwds)
        else:
            super(Diffusion, self).__init__(**kwds)

        # The only available method at the time is fftw
        if self.method is None:
            self.method = default.DIFFUSION
        ## input/output field, solution of the problem
        if vorticity is not None:
            self.vorticity = vorticity
        else:
            self.vorticity = self.variables.keys()[0]
        ## viscosity
        self.viscosity = viscosity

        self.kwds = kwds

        self.input = [self.vorticity]
        self.output = [self.vorticity]

    def discretize(self):
        if self.method[SpaceDiscretisation] is 'fftw':
            super(Diffusion, self)._fftw_discretize()
        elif self.method[SpaceDiscretisation] is 'fd':
            super(Diffusion, self)._standard_discretize()
        else:
            raise AttributeError("Method not yet implemented.")

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        if self.method[SpaceDiscretisation] is 'fftw':
            self.discrete_op = DiffusionFFT(
                self.discreteFields[self.vorticity], self.viscosity,
                method=self.method)
        elif self.method[SpaceDiscretisation] is 'fd':
            from hysop.gpu.gpu_diffusion import GPUDiffusion
            kw = self.kwds.copy()
            if 'discretization' in kw.keys():
                kw.pop('discretization')
            self.discrete_op = GPUDiffusion(
                self.discreteFields[self.vorticity],
                viscosity=self.viscosity,
                **kw)
        self._is_uptodate = True
