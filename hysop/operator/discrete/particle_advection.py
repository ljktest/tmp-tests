"""
@file particle_advection.py

Advection solver, particular method, pure-python version.

"""
from hysop.constants import debug, WITH_GUESS, HYSOP_REAL, HYSOP_DIM
from hysop.methods_keys import TimeIntegrator, Interpolation, Remesh, Support
from hysop.operator.discrete.discrete import DiscreteOperator
import hysop.tools.numpywrappers as npw
import hysop.default_methods as default
import numpy as np
from hysop.numerics.remeshing import Remeshing
from hysop.tools.profiler import profile


class ParticleAdvection(DiscreteOperator):
    """
    Particular method for advection of a list of fields in a given direction.
    """

    @debug
    def __init__(self, velocity, fields_on_grid, direction, **kwds):
        """
        Constructor.
        @param velocity: discretization of the velocity field
        @param fields_on_grid : list of discretized fields to be advected.
        @param direction : direction of advection
        """
        ## Advection velocity
        self.velocity = velocity

        # set variables list ...
        variables = [self.velocity]
        if not isinstance(fields_on_grid, list):
            self.fields_on_grid = [fields_on_grid]
        else:
            self.fields_on_grid = fields_on_grid
        for f in self.fields_on_grid:
            variables.append(f)

        if 'method' not in kwds:
            kwds['method'] = default.ADVECTION

        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(ParticleAdvection, self).__init__(variables=variables, **kwds)

        self.input = self.variables
        self.output = [df for df in self.variables if df is not self.velocity]
        self.direction = direction

        self._configure_numerical_methods()

    def _configure_numerical_methods(self):
        """
        Function to set the numerical method for python operator and link them
        to the proper working arrays.
        """
        # Use first field topology as reference
        topo = self.fields_on_grid[0].topology

        # --- Initialize time integrator for advection ---

        w_interp, iw_interp =\
            self.method[Interpolation].getWorkLengths(
                domain_dim=self.domain.dimension)
        self._rw_interp = self._rwork[:w_interp]
        self._iw_interp = self._iwork[:iw_interp]

        vd = self.velocity.data[self.direction]
        num_interpolate = \
            self.method[Interpolation](vd, self.direction, topo,
                                       work=self._rw_interp,
                                       iwork=self._iw_interp)

        w_rk = self.method[TimeIntegrator].getWorkLengths(nb_components=1)
        self._rw_integ = self._rwork[w_interp:w_interp + w_rk]
        self.num_advec = self.method[TimeIntegrator](1, work=self._rw_integ,
                                                     f=num_interpolate,
                                                     topo=topo,
                                                     optim=WITH_GUESS)
        # --- Initialize remesh ---
        w_remesh, iw_remesh =\
            Remeshing.getWorkLengths(
                domain_dim=self.domain.dimension)
        self._rw_remesh = self._rwork[:w_remesh]
        self._iw_remesh = self._iwork[:iw_remesh]

        self.num_remesh = Remeshing(self.method[Remesh],
                                    self.domain.dimension,
                                    topo, self.direction,
                                    work=self._rw_remesh,
                                    iwork=self._iw_remesh)

        ## Particles positions
        start = max(w_interp + w_rk, w_remesh)
        self.part_position = [self._rwork[start]]

        ## Fields on particles
        self.fields_on_part = {}
        start += 1
        for f in self.fields_on_grid:
            self.fields_on_part[f] = self._rwork[start: start + f.nb_components]
            start += f.nb_components

    def _set_work_arrays(self, rwork=None, iwork=None):
        memshape = self.fields_on_grid[0].data[0].shape
        # Get work lengths
        dimension = self.domain.dimension
        if self.method[Support].find('gpu') < 0:
            tiw = self.method[TimeIntegrator].getWorkLengths(1)
            iw, iiw = \
                      self.method[Interpolation].getWorkLengths(domain_dim=dimension)
            rw, riw = Remeshing.getWorkLengths(domain_dim=dimension)
            iwl = max(iiw, riw)
            rw = max(tiw + iw, rw)
        else:
            # For GPU version, no need of numerics works
            iwl, rw = 0, 0
        # Shape of reference comes from fields, not from velocity
        fd = self.fields_on_grid[0]
        memshape = fd.topology.mesh.resolution
        rw += np.sum([f.nb_components for f in self.fields_on_grid])
        if self.method[Support].find('gpu') < 0 or \
           self.method[Support].find('gpu_2k') >= 0:
            rw += 1  # work array for positions

        if rwork is None:
            self._rwork = []
            for i in xrange(rw):
                self._rwork.append(npw.zeros(memshape))
        else:
            assert len(rwork) == rw
            try:
                for wk in rwork:
                    assert wk.shape == tuple(memshape)
            except AttributeError:
                # Work array has been replaced by an OpenCL Buffer
                # Testing the buffer size instead of shape
                for wk in rwork:
                    s = wk.size / np.prod(memshape)
                    assert (HYSOP_REAL is np.float32 and s == 4) or \
                        (HYSOP_REAL is np.float64 and s == 8)
            self._rwork = rwork

        if iwork is None:
            self._iwork = []
            for i in xrange(iwl):
                self._iwork.append(npw.dim_zeros(memshape))
        else:
            assert len(iwork) == iwl
            try:
                for wk in iwork:
                    assert wk.shape == tuple(memshape)
            except AttributeError:
                # Work array has been replaced by an OpenCL Buffer
                # Testing the buffer size instead of shape
                for wk in iwork:
                    s = wk.size / np.prod(memshape)
                    assert (HYSOP_DIM is np.int16 and s == 2) or \
                        (HYSOP_DIM is np.int32 and s == 4) or \
                        (HYSOP_DIM is np.int64 and s == 8)
            self._iwork = iwork

    @debug
    @profile
    def apply(self, simulation=None, dt_coeff=1., split_id=0, old_dir=0):
        """
        Advection algorithm:
        - initialize particles and fields with their values on the grid.
        - compute particle positions in splitting direction,
        (time integrator), resolution of dx_p/dt = a_p.
        - remesh fields from particles to grid
        """
        assert simulation is not None, \
            'Simulation parameter is missing.'

        t, dt = simulation.time, simulation.timeStep * dt_coeff
        # Initialize fields on particles with fields on grid values.
        for fg in self.fields_on_grid:
            for d in xrange(fg.nb_components):
                self.fields_on_part[fg][d][...] = fg[d][...]

        # Initialize particles on the grid
        toporef = self.fields_on_grid[0].topology
        self.part_position[0][...] = toporef.mesh.coords[self.direction]

        # Advect particles
        # RK use the first 2 (or 3) works and leave others to interpolation
        # First work contains fist evaluation of ode right hand side.
        self._rw_integ[0][...] = self.velocity.data[self.direction][...]
        self.part_position = self.num_advec(
            t, self.part_position, dt, result=self.part_position)

        # Remesh particles
        # It uses the last dim + 2 workspaces (same as interpolation)
        for fg in self.fields_on_grid:
            fp = self.fields_on_part[fg]
            for d in xrange(fg.nb_components):
                fg[d][...] = self.num_remesh(
                    self.part_position, fp[d], result=fg[d])
