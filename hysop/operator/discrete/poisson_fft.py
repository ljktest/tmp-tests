# -*- coding: utf-8 -*-
"""
@file poisson_fft.py
Discrete operator for Poisson problem (fftw based)
"""
import hysop.tools.numpywrappers as npw
try:
    from hysop.f2hysop import fftw2py
except ImportError:
    from hysop.fakef2py import fftw2py

from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.operator.discrete.reprojection import Reprojection
from hysop.constants import debug
from hysop.tools.profiler import profile


class PoissonFFT(DiscreteOperator):
    """
    Discretized Poisson operator based on FFTW.
    See details in hysop.operator.poisson
    """

    @debug
    def __init__(self, output_field, input_field, projection=None,
                 filterSize=None, correction=None, formulation=None, **kwds):
        """
        Constructor.
        @param[out] output_field : discretization of the solution field
        @param[in] input_field : discretization of the RHS (mind the minus rhs!)
        @param projection : if None, no projection. Else:
        - either the value of the frequency of reprojection, never updated.
        - or Reprojection discrete operator. In that case, a criterion
        depending on the input_field will be computed at each time step, if
        criterion > threshold, then frequency projection is active.
        @param filterSize :
        @param correction : operator used to shift output_field according
        to a given input (fixed) flowrate.
        See hysop.operator.velocity_correction.
        Default = None.
        """
        # Base class initialisation
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(PoissonFFT, self).__init__(variables=[output_field, input_field],
                                         **kwds)
        ## Solution field
        self.output_field = output_field
        ## RHS field
        self.input_field = input_field
        ## Solenoidal projection of input_field ?
        self.projection = projection
        ## Filter size array = domainLength/(CoarseRes-1)
        self.filterSize = filterSize
        # If 2D problem, input_field must be a scalar
        self.dim = self.output_field.domain.dimension
        if self.dim == 2:
            assert self.input_field.nb_components == 1
        self.correction = correction
        self.formulation = formulation
        self.input = [self.input_field]
        self.output = [self.output_field]

        ## The function called during apply
        self.solve = None
        # a sub function ...
        self._solve = None
        self.do_projection = None
        self._select_solve()

    def _select_solve(self):

        """
        TODO : add pressure solver selection
        f(output.nbComponents) + pb type 'pressure-poisson'
        """
        
        ## Multiresolution ?
        multires = self.output_field.topology.mesh != self.input_field.topology.mesh

        # connexion to the required apply function
        if self.dim == 2:
            self._solve = self._solve2D
        elif self.dim == 3:
            # If there is a projection, input_field is also an output
            if self.projection is not None:
                self.output.append(self.input_field)
                if multires:
                    self._solve = self._solve3D_proj_multires
                else:
                    self._solve = self._solve3D_proj

                if isinstance(self.projection, Reprojection):
                    self.do_projection = self.do_projection_with_op
                else:
                    self.do_projection = self.do_projection_no_op

            else:
                if multires:
                    self._solve = self._solve3D_multires
                elif self.formulation is not None:
                    self._solve = self._solve_3d_scalar_fd
                else:
                    self._solve = self._solve3D
        else:
            raise AttributeError('Not implemented for 1D problems.')

        # Operator to shift output_field according to an input required flowrate
        if self.correction is not None:
            self.solve = self._solve_and_correct
        else:
            self.solve = self._solve

    def do_projection_with_op(self, simu):
        self.projection.apply(simu)
        ite = simu.currentIteration
        return self.projection.do_projection(ite)

    def do_projection_no_op(self, simu):
        ite = simu.currentIteration
        return ite % self.projection == 0

    def _solve2D(self, simu=None):
        """
        Solve 2D poisson problem
        """
        ghosts_v = self.output_field.topology.ghosts()
        ghosts_w = self.input_field.topology.ghosts()
        self.output_field.data[0], self.output_field.data[1] =\
            fftw2py.solve_poisson_2d(self.input_field.data[0],
                                     self.output_field.data[0],
                                     self.output_field.data[1],
                                     ghosts_w, ghosts_v)

    def _project(self):
        """
        apply projection onto input_field
        """
        ghosts_w = self.input_field.topology.ghosts()
        self.input_field.data[0], self.input_field.data[1], \
            self.input_field.data[2] = \
               fftw2py.projection_om_3d(self.input_field.data[0],
                                        self.input_field.data[1],
                                        self.input_field.data[2], ghosts_w)

    def _solve3D_multires(self, simu=None):
        """
        3D, multiresolution
        """
        # Projects input_field values from fine to coarse grid
        # in frequencies space by nullifying the smallest modes
        vortFilter = npw.copy(self.input_field.data)
        vortFilter[0], vortFilter[1], vortFilter[2] = \
           fftw2py.multires_om_3d(self.filterSize[0], self.filterSize[1],
                                  self.filterSize[2], self.input_field.data[0],
                                  self.input_field.data[1],
                                  self.input_field.data[2])

        # Solves Poisson equation using filter input_field
        ghosts_v = self.output_field.topology.ghosts()
        ghosts_w = self.input_field.topology.ghosts()
        self.output_field.data[0], self.output_field.data[1], self.output_field.data[2] = \
            fftw2py.solve_poisson_3d(vortFilter[0], vortFilter[1],
                                     vortFilter[2], self.output_field.data[0],
                                     self.output_field.data[1],
                                     self.output_field.data[2],
                                     ghosts_w, ghosts_v)

    def _solve3D_proj_multires(self, simu):
        """
        3D, multiresolution, with projection
        """
        if self.do_projection(simu):
            self._project()
        self._solve3D_multires()

    def _solve3D_proj(self, simu):
        """
        3D, with projection
        """
        if self.do_projection(simu):
            self._project()
        self._solve3D()

    @profile
    def _solve3D(self,simu=None):
        """
        Basic solve
        """
        # Solves Poisson equation using usual input_field
        ghosts_v = self.output_field.topology.ghosts()
        ghosts_w = self.input_field.topology.ghosts()
        self.output_field.data[0], self.output_field.data[1], self.output_field.data[2] =\
            fftw2py.solve_poisson_3d(self.input_field.data[0],
                                     self.input_field.data[1],
                                     self.input_field.data[2],
                                     self.output_field.data[0],
                                     self.output_field.data[1],
                                     self.output_field.data[2], ghosts_w, ghosts_v)

    def _solve_and_correct(self, simu):
        self._solve(simu)
        self.correction.apply(simu)


    def _solve_3d_scalar_fd(self, simu=None):
        """solve poisson-pressure like problem
        input = 3D vector field
        output = 3D scalar field
        """
        # Compute rhs = f(input) inplace
        # --> output == rhs
        # Call fftw filter
        # !!! pressure3d use the same arg for input and output
        # ---> input_field will be overwritten
        ghosts = self.output_field.topology.ghosts()
        self.output_field.data[0] = fftw2py.pressure_3d(
            self.input_field.data[0], ghosts)
        
    def _solve_3d_scalar(self, simu=None):
        """solve poisson-pressure like problem
        input = 3D vector field
        output = 3D scalar field
        """
        # # Call fftw filter
        # self._output_field.data[0] = fftw2py.solve_poisson_3d_pressure(
        #     self._input_field.data[0],
        #     self._input_field.data[1],
        #     self._input_field.data[2])
        pass

    @debug
    @profile
    def apply(self, simulation=None):
        self.solve(simulation)

    def finalize(self):
        """
        Clean memory (fftw plans and so on)
        """
        pass
        #fftw2py.clean_fftw_solver(self.output_field.dimension)
