"""
@file advection.py

Advection of a field.
"""
from hysop.constants import debug, S_DIR, ZDIR
from hysop.operator.computational import Computational
from hysop.methods_keys import Scales, TimeIntegrator, Interpolation,\
    Remesh, Support, Splitting, MultiScale
from hysop.numerics.remeshing import L2_1
from hysop.operator.continuous import opsetup, opapply
from hysop.operator.advection_dir import AdvectionDir
import hysop.default_methods as default
from hysop.tools.parameters import Discretization
from hysop.mpi.topology import Cartesian
import hysop.tools.numpywrappers as npw


class Advection(Computational):
    """
    Advection of a field,
    \f{eqnarray*}
    X = Op(X,velocity)
    \f} for :
    \f{eqnarray*}
    \frac{\partial{X}}{\partial{t}} + velocity.\nabla X = 0
    \f}
    Note : we assume incompressible flow.

    Computations are performed within a dimensional splitting as folows:
      - 2nd order:
        - X-dir, half time step
        - Y-dir, half time step
        - Z-dir, full time step
        - Y-dir, half time step
        - X-dir, half time step
      - 2nd order full half-steps:
        - X-dir, half time step
        - Y-dir, half time step
        - Z-dir, half time step
        - Z-dir, half time step
        - Y-dir, half time step
        - X-dir, half time step
      - 1st order g:
        - X-dir, half time step
        - Y-dir, half time step
        - Z-dir, half time step

    """

    @debug
    def __init__(self, velocity, advected_fields=None, **kwds):
        """
        Advection of a set of fields for a given velocity.

        @param velocity : velocity field used for advection
        @param advectedFields : the list of fields to be advected.
        It may be a single field (no list).
        """
        ## Transport velocity
        self.velocity = velocity
        if 'variables' in kwds:
            kw = kwds.copy()
            kw['variables'] = kwds['variables'].copy()
            # In that case, variables must contains only the advected fields
            # with their discretization param.
            # Velocity must always be given outside variables, with its
            # own discretization.
            assert advected_fields is None, 'too many input arguments.'
            self.advected_fields = kwds['variables'].keys()
            kw['variables'][self.velocity] = kwds['discretization']
            kw.pop('discretization')
            super(Advection, self).__init__(**kw)

        else:
            v = [self.velocity]
            if isinstance(advected_fields, list):
                self.advected_fields = advected_fields
            else:
                self.advected_fields = [advected_fields]
            v += self.advected_fields
            super(Advection, self).__init__(variables=v, **kwds)

        # Set default method, if required
        if self.method is None:
            self.method = default.ADVECTION

        self.output = self.advected_fields
        self.input = [var for var in self.variables]

        vars_str = "_("
        for vv in self.advected_fields:
            vars_str += vv.name + ","
        self.name += vars_str[0:-1] + ')'

        self.config = {}

        # Find which solver is used for advection,
        # among Scales, pure-python and GPU-like.
        # Check also operator-splitting type.

        # ---- Scales advection ----
        if Scales in self.method.keys():
            self._is_scales = True
            if not self.domain.dimension == 3:
                raise ValueError("Scales Advection not implemented in 2D.")
            # Default splitting = Strang
            if Splitting not in self.method.keys():
                self.method[Splitting] = 'strang'

            self._my_setup = self._setup_scales
            self.advec_dir = None

        else:
            # ---- Python or GPU advection ----
            self._is_scales = False
            assert TimeIntegrator in self.method.keys()
            assert Interpolation in self.method.keys()
            assert Remesh in self.method.keys()
            assert Support in self.method.keys()
            if Splitting not in self.method.keys():
                self.method[Splitting] = 'o2'
            dimension = self.domain.dimension
            self.advec_dir = [None] * dimension
            name = vars_str[0:-1] + ')'
            if 'variables' in kwds:
                for i in xrange(self.domain.dimension):
                    self.advec_dir[i] = AdvectionDir(
                        self.velocity, direction=i,
                        name_suffix=name, **kwds)
            else:
                for i in xrange(self.domain.dimension):
                    self.advec_dir[i] = AdvectionDir(
                        self.velocity, direction=i,
                        advected_fields=self.advected_fields,
                        name_suffix=name, **kwds)

            self._my_setup = self._setup_python
            self.apply = self._apply_python

        self._old_dir = 0
        self.splitting = []

    def scales_parameters(self):
        """
        Return the name of the particular method used in scales
        and the type of splitting.
        """
        order = None
        for o in ['p_O2', 'p_O4', 'p_L2',
                  'p_M4', 'p_M6', 'p_M8',
                  'p_44', 'p_64', 'p_66', 'p_84']:
            if self.method[Scales].find(o) >= 0:
                order = o
        if order is None:
            print ('Unknown advection method, turn to default (p_M6).')
            order = 'p_M6'

        # - Extract splitting form self.method (default strang) -
        splitting = 'strang'
        for s in ['classic', 'strang', 'particle']:
            if self.method[Splitting].find(s) >= 0:
                splitting = s

        return order, splitting

    def discretize(self):
        """
        Discretisation (create topologies and discretize fields)
        Available methods :
        - 'scales' : SCALES fortran routines (3d only, list of vector
        and/or scalar)
        - 'gpu' : OpenCL kernels (2d and 3d, single field, scalar or vector)
        - other : Pure python (2d and 3d, list of vector and/or scalar)
        """
        if self._is_discretized:
            return

        if self._is_scales:
            self._scales_discretize()
        else:
            self._no_scales_discretize()
        advected_discrete_fields = [self.discreteFields[f]
                                    for f in self.advected_fields]
        toporef = advected_discrete_fields[0].topology
        msg = 'All advected fields must have the same topology.'
        for f in advected_discrete_fields:
            assert f.topology == toporef, msg

        if self._single_topo:
            self.method[MultiScale] = None

    @debug
    def _scales_discretize(self):
        """
        Discretization (create topologies and discretize fields)
        when using SCALES fortran routines (3d only, list of vector
        and/or scalar)
        - 'p_O2' : order 4 method, corrected to allow large CFL number,
        untagged particles
        - 'p_O4' : order 4 method, corrected to allow large CFL number,
        untagged particles
        - 'p_L2' : limited and corrected lambda 2
        - 'p_M4' : Lambda_2,1 (=M'4) 4 point formula
        - 'p_M6' (default) : Lambda_4,2 (=M'6) 6 point formula
        - 'p_M8' : M8prime formula
        - 'p_44' : Lambda_4,4 formula
        - 'p_64' : Lambda_6,4 formula
        - 'p_66' : Lambda_6,6 formula
        - 'p_84' : Lambda_8,4 formula
        """
        assert self._is_scales
        # - Extract order form self.method (default p_M6) -
        order, splitting = self.scales_parameters()

        # Check if topos need to be created
        build_topos = self._check_variables()
        from hysop.f2hysop import scales2py as scales

        # Scales, single resolution
        if self._single_topo:
            if build_topos:
                # In that case, self._discretization must be
                # a Discretization object, used for all fields.
                # We use it to initialize scales solver
                topo = self._create_scales_topo(self._discretization,
                                                order, splitting)
                for v in self.variables:
                    self.variables[v] = topo
            else:
                # In that case, self._discretization must be
                # a Cartesian object, used for all fields.
                # We use it to initialize scales solver
                assert isinstance(self._discretization, Cartesian)
                topo = self._discretization
                msg = 'input topology is not compliant with scales.'
                #assert topo.dimension == 1, msg
                msg = 'Ghosts points not yet implemented for scales operators.'
                assert (topo.mesh.discretization.ghosts == 0).all(), msg

                nbcells = topo.mesh.discretization.resolution - 1
                topodims = topo.shape
                scalesres, global_start = \
                    scales.init_advection_solver(nbcells,
                                                 self.domain.length,
                                                 npw.asintarray(topodims),
                                                 self._mpis.comm.py2f(),
                                                 order=order,
                                                 dim_split=splitting)

                assert (topo.shape == topodims).all()
                assert (topo.mesh.resolution == scalesres).all()
                assert (topo.mesh.start() == global_start).all()

            msg = 'Scales Advection not yet implemented with ghosts points.'
            assert (topo.ghosts() == 0).all(), msg

        # Scales, multi-resolution
        else:
            if build_topos[self.velocity]:
                # Resolution used for velocity
                v_resol = self.variables[self.velocity].resolution - 1

            else:
                topo = self.variables[self.velocity]
                v_resol = topo.mesh.discretization.resolution

            vbuild = [v for v in self.variables if build_topos[v]]
            for v in vbuild:
                self.variables[v] = self._create_scales_topo(
                    self.variables[v], order, splitting)

            topo = self.variables.values()[0]
            self._check_scales_topo(topo, order, splitting)

            # Init multiscale in scales
            scales.init_multiscale(v_resol[0], v_resol[1], v_resol[2],
                                   self.method[MultiScale])

        # All topos are built, we can discretize fields.
        self._discretize_vars()

    def _create_scales_topo(self, d3d, order, splitting):
        from hysop.f2hysop import scales2py as scales
        comm = self._mpis.comm
        topodims = [1, 1, comm.Get_size()]
        msg = 'Wrong type for parameter discretization (at init).' + str(self._discretization)
        assert isinstance(d3d, Discretization), msg
        nbcells = d3d.resolution - 1
        scalesres, global_start = \
            scales.init_advection_solver(nbcells,
                                         self.domain.length,
                                         npw.asintarray(topodims),
                                         comm.py2f(),
                                         order=order,
                                         dim_split=splitting)
        # Create the topo (plane, cut through ZDIR)
        return self.domain.create_plane_topology_from_mesh(
            global_start=global_start, localres=scalesres,
            discretization=d3d, cdir=ZDIR)

    def _check_scales_topo(self, toporef, order, splitting):
        from hysop.f2hysop import scales2py as scales
        # In that case, self._discretization must be
        # a Cartesian object, used for all fields.
        # We use it to initialize scales solver
        comm = self._mpis.comm
        #topodims = [1, 1, comm.Get_size()]
        nbcells = toporef.mesh.discretization.resolution - 1

        scalesres, global_start = \
            scales.init_advection_solver(nbcells, self.domain.length,
                                         npw.asintarray(toporef.shape),
                                         comm.py2f(),
                                         order=order, dim_split=splitting)
        for v in self.variables:
            topo = self.variables[v]
            assert isinstance(topo, Cartesian), str(topo)
            #assert (topo.shape == topodims).all(), \
            #    str(topo.shape) + ' != ' + str(topodims)
            assert not self._single_topo or (topo.mesh.resolution == scalesres).all(), \
                str(topo.mesh.resolution) + ' != ' + str(scalesres)
            assert not self._single_topo or (topo.mesh.start() == global_start).all(), \
                str(topo.mesh.start()) + ' != ' + str(global_start)

    def _no_scales_discretize(self):
        """
        GPU or pure-python advection
        """
        if not self._is_discretized:
            for i in xrange(self.domain.dimension):
                self.advec_dir[i].discretize()
            self.discreteFields = self.advec_dir[0].discreteFields
            self._single_topo = self.advec_dir[0]._single_topo
            self._is_discretized = True

    def get_work_properties(self):
        """
        Return the length of working arrays lists required
        for the discrete operator.
        @return shapes, shape of the arrays:
        shapes['rwork'] == list of shapes for real arrays,
        shapes['iwork'] == list of shapes for int arrays.
        len(shapes['...'] gives the number of required arrays.
        """
        if self._is_scales:
            return {'rwork': None, 'iwork': None}
        else:
            if not self.advec_dir[0]._is_discretized:
                msg = 'The operator must be discretized '
                msg += 'before any call to this function.'
                raise RuntimeError(msg)
            return self.advec_dir[0].get_work_properties()

    @opsetup
    def setup(self, rwork=None, iwork=None):
        # Check resolutions to set multiscale case, if required.
        if not self._single_topo and MultiScale not in self.method:
            self.method[MultiScale] = L2_1
        if not self._is_uptodate:
            self._my_setup(rwork, iwork)

    def _setup_scales(self, rwork=None, iwork=None):

        advected_discrete_fields = [self.discreteFields[f]
                                    for f in self.advected_fields]
        # - Create the discrete_op from the
        # list of discrete fields -
        from hysop.operator.discrete.scales_advection import \
            ScalesAdvection
        self.discrete_op = ScalesAdvection(
            self.discreteFields[self.velocity],
            advected_discrete_fields, method=self.method,
            rwork=rwork, iwork=iwork,
            **self.config)
        self._is_uptodate = True

    def _setup_advec_dir(self, rwork=None, iwork=None):
        """
        Local allocation of work arrays,
        common to advec_dir operators and setup for those
        operators
        """
        wk_p = self.advec_dir[0].get_work_properties()
        wk_length = len(wk_p['rwork'])
        if rwork is None:
            rwork = []
            for i in xrange(wk_length):
                memshape = wk_p['rwork'][i]
                rwork.append(npw.zeros(memshape))
        else:
            assert len(rwork) == wk_length
            for wk, refshape in zip(rwork, wk_p['rwork']):
                assert wk.shape == refshape
        wk_length = len(wk_p['iwork'])
        if iwork is None:
            iwork = []
            for i in xrange(wk_length):
                memshape = wk_p['iwork'][i]
                iwork.append(npw.int_zeros(memshape))
        else:
            assert len(iwork) == wk_length
            for wk, refshape in zip(iwork, wk_p['iwork']):
                assert wk.shape == refshape
        # Work arrays are common between all directions
        # of advection.
        for i in xrange(self.domain.dimension):
            self.advec_dir[i].setup(rwork, iwork)

    def _setup_python(self, rwork=None, iwork=None):

        # setup for advection in each direction
        self._setup_advec_dir(rwork, iwork)
        # set splitting parameters (depends on method)
        self._configure_splitting()

        # configure gpu
        if self.method[Support].find('gpu') >= 0:
            self._configure_gpu()

        self._is_uptodate = True

    def _configure_splitting(self):
        dimension = self.domain.dimension
        if self.method[Splitting] == 'o2_FullHalf':
            ## Half timestep in all directions
            [self.splitting.append((i, 0.5))
             for i in xrange(dimension)]
            [self.splitting.append((dimension - 1 - i, 0.5))
             for i in xrange(dimension)]
        elif self.method[Splitting] == 'o1':
            [self.splitting.append((i, 1.)) for i in xrange(dimension)]
        elif self.method[Splitting] == 'x_only':
            self.splitting.append((0, 1.))
        elif self.method[Splitting] == 'y_only':
            self.splitting.append((1, 1.))
        elif self.method[Splitting] == 'z_only':
            self.splitting.append((2, 1.))
        elif self.method[Splitting] == 'o2':
            ## Half timestep in all directions but last
            [self.splitting.append((i, 0.5))
             for i in xrange(dimension - 1)]
            self.splitting.append((dimension - 1, 1.))
            [self.splitting.append((dimension - 2 - i, 0.5))
             for i in xrange(dimension - 1)]
        else:
            raise ValueError('Unknown splitting configuration:' +
                             self.method[Splitting])

    def _configure_gpu(self):
        splitting_nbSteps = len(self.splitting)
        for d in xrange(self.domain.dimension):
            dOp = self.advec_dir[d].discrete_op
            assert len(dOp.exec_list) == splitting_nbSteps, \
                "Discrete operator execution " + \
                "list and splitting steps sizes must be equal " + \
                str(len(dOp.exec_list)) + " != " + \
                str(splitting_nbSteps)
        s = ""
        device_id = self.advec_dir[0].discrete_op.cl_env._device_id
        gpu_comm = self.advec_dir[0].discrete_op.cl_env.gpu_comm
        gpu_rank = gpu_comm.Get_rank()
        if gpu_rank == 0:
            s += "=== OpenCL buffers allocated"
            s += " on Device:{0} ===\n".format(device_id)
            s += "Global memory used:\n"
        total_gmem = 0
        for d in xrange(self.domain.dimension):
            g_mem_d = 0
            # allocate all variables in advec_dir
            for df in self.advec_dir[d].discrete_op.variables:
                if not df.gpu_allocated:
                    df.allocate()
                    g_mem_df = gpu_comm.allreduce(df.mem_size)
                    g_mem_d += g_mem_df
            if gpu_rank == 0:
                s += " Advection" + S_DIR[d] + ": {0:9d}".format(g_mem_d)
                s += "Bytes ({0:5d} MB)\n".format(g_mem_d / (1024 ** 2))
            total_gmem += g_mem_d
        if gpu_rank == 0:
            s += " Total      : {0:9d}".format(total_gmem)
            s += "Bytes ({0:5d} MB)\n".format(total_gmem / (1024 ** 2))
            s += "Local memory used:\n"
        total_lmem = 0
        for d in xrange(self.domain.dimension):
            l_mem_d = gpu_comm.allreduce(
                self.advec_dir[d].discrete_op.size_local_alloc)
            if gpu_rank == 0:
                s += " Advection" + S_DIR[d] + ": {0:9d}".format(l_mem_d)
                s += "Bytes ({0:5d} MB)\n".format(l_mem_d / (1024 ** 2))
            total_lmem += l_mem_d
        if gpu_rank == 0:
            s += " Total      : {0:9d}".format(total_lmem) + "Bytes"
            print s

    @debug
    @opapply
    def _apply_python(self, simulation=None):
        """
        Apply this operator to its variables.
        @param simulation : object that describes the simulation
        parameters (time, time step, iteration number ...), see
        hysop.problem.simulation.Simulation for details.

        Redefinition for advection. Applying a dimensional splitting.
        """
        assert simulation is not None
        for split_id, split in enumerate(self.splitting):
            self.advec_dir[split[0]].apply(
                simulation, split[1], split_id, self._old_dir)
            self._old_dir = split[0]

    @debug
    def finalize(self):
        """
        Memory cleaning.
        """
        if self._is_scales:
            Computational.finalize(self)
        else:
            for dop in self.advec_dir:
                dop.finalize()

    def get_profiling_info(self):
        if self._is_uptodate:
            if self._is_scales:
                self.profiler += self.discrete_op.profiler
            else:
                for dop in self.advec_dir:
                    self.profiler += dop.profiler

    def __str__(self):
        """
        Common printings for operators
        """
        shortName = str(self.__class__).rpartition('.')[-1][0:-2]
        if self._is_scales:
            super(Advection, self).__str__()
        else:
            for i in xrange(self.domain.dimension):
                if self.advec_dir[i].discrete_op is not None:
                    s = str(self.advec_dir[i].discrete_op)
                else:
                    s = shortName + " operator. Not discretised."
        return s + "\n"
