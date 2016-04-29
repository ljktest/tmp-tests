# -*- coding: utf-8 -*-
"""Poisson problem.
"""
from hysop.operator.computational import Computational
from hysop.operator.discrete.poisson_fft import PoissonFFT
from hysop.constants import debug
from hysop.operator.velocity_correction import VelocityCorrection
from hysop.operator.reprojection import Reprojection
from hysop.methods_keys import SpaceDiscretisation, Formulation
from hysop.operator.continuous import opsetup
import hysop.default_methods as default


class Poisson(Computational):
    """
    \f{eqnarray*}
    v = Op(\omega)
    \f} with :
    \f{eqnarray*}
    \Delta \phi &=& -\omega \\
    v &=& \nabla \times \phi
    \f}
    """

    @debug
    def __init__(self, output_field, input_field, flowrate=None,
                 projection=None, **kwds):
        """
        Constructor for the Poisson problem.

        @param[out] output_field : solution field
        @param[in] input_field : rhs field
        @param[in] flowrate : a flow rate value (through input_field surf,
        normal to xdir) used to compute a correction of the solution field.
        Default = 0 (no correction). See hysop.operator.output_field_correction.
        @param projection : if None, no projection. Else:
        - either the value of the frequency of reprojection, never update.
        - or a tuple = (frequency, threshold).
        In that case, a criterion
        depending on the input_field will be computed at each time step, if
        criterion > threshold, then frequency projection is active.

        Note about method:
        - SpaceDiscretisation == fftw
        - Formulation = 'velocity' or 'pressure'
        velocity : laplacian(phi) = -w and v = nabla X psi, in = vorticity, out = velo
        pressure : laplacian(p) = -nabla.(u.nabla u, in = velo, out = pressure
        """
        # Warning : for fftw all variables must have
        # the same resolution.
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Poisson, self).__init__(variables=[output_field, input_field],
                                      **kwds)
        ## solution of the problem
        self.output_field = output_field
        ## -(right-hand side)
        self.input_field = input_field
        if self.method is None:
            self.method = default.POISSON

        if self.method[SpaceDiscretisation] is not 'fftw':
            raise AttributeError("Method not yet implemented.")

        # Deterlination of the Poisson equation formulation :
        # Velo Poisson eq or Pressure Poisson eq
        self.formulation = None
        if self.method[Formulation] is not 'velocity':
            self.formulation = self.method[Formulation]

        self.input = [self.input_field]
        self.output = [self.output_field]
        if flowrate is not None:
            self.withCorrection = True
            self._flowrate = flowrate
        else:
            self.withCorrection = False
        self.correction = None
        self.projection = projection
        self._config = kwds

        if projection is not None:
            self.output.append(self.input_field)

    def discretize(self):
        # Poisson solver based on fftw
        if self.method[SpaceDiscretisation] is 'fftw':
            super(Poisson, self)._fftw_discretize()
            if self.withCorrection:
                toporef = self.discreteFields[self.output_field].topology
                if 'discretization' in self._config:
                    self._config['discretization'] = toporef
                self.correction = VelocityCorrection(
                    self.output_field, self.input_field,
                    req_flowrate=self._flowrate, **self._config)
                self.correction.discretize()

                if isinstance(self.projection, tuple):
                    freq = self.projection[0]
                    threshold = self.projection[1]
                    self.projection = Reprojection(self.input_field,
                                                   threshold, freq,
                                                   **self._config)
                    self.projection.discretize()
        else:
            raise AttributeError("Method not yet implemented.")

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        # Activate correction, if required
        if self.withCorrection:
            self.correction.setup()
            cd = self.correction.discrete_op
        else:
            cd = None

        # Activate projection, if required
        if isinstance(self.projection, Reprojection):
            # Projection frequency is updated at each
            # time step, and depends on the input_field
            self.projection.setup(rwork=rwork)
            projection_discr = self.projection.discrete_op
        else:
            projection_discr = self.projection

        self.discrete_op = PoissonFFT(self.discreteFields[self.output_field],
                                      self.discreteFields[self.input_field],
                                      correction=cd,
                                      rwork=rwork, iwork=iwork,
                                      projection=projection_discr,
                                      formulation=self.formulation)

        self._is_uptodate = True
