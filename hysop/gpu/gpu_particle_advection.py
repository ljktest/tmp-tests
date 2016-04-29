"""
@file gpu_particle_advection.py

Discrete advection representation
"""
from abc import ABCMeta, abstractmethod
from hysop import __VERBOSE__
from hysop.constants import np, debug, S_DIR
from hysop.methods_keys import TimeIntegrator, Remesh, \
    Support, Splitting, MultiScale
from hysop.numerics.integrators.euler import Euler
from hysop.operator.discrete.particle_advection import ParticleAdvection
from hysop.operator.discrete.discrete import get_extra_args_from_method
from hysop.gpu import cl
from hysop.gpu.gpu_kernel import KernelLauncher
import hysop.default_methods as default
import hysop.tools.numpywrappers as npw
from hysop.gpu.gpu_discrete import GPUDiscreteField
from hysop.gpu.gpu_operator import GPUOperator
from hysop.tools.profiler import profile
from hysop.numerics.update_ghosts import UpdateGhostsFull


class GPUParticleAdvection(ParticleAdvection, GPUOperator):
    """
    Particle advection operator representation on GPU.

    """
    __metaclass__ = ABCMeta

    @debug
    def __init__(self, **kwds):
        """
        Create a Advection operator.
        Work on a given field (scalar or vector) at a given velocity to compute
        advected values.
        OpenCL kernels are build once per dimension in order to handle
        directional splitting with resolution non uniform in directions.

        @param velocity : Velocity field
        @param fields_on_grid : Advected fields
        @param d : Direction to advect
        @param device_type : OpenCL device type (default = 'gpu').
        @param method : the method to use. {'m4prime', 'm6prime', 'm8prime',
        'l6star'}
        @param user_src : User OpenCL sources.
        @param splittingConfig : Directional splitting configuration
        (hysop.numerics.splitting.Splitting.__init__)
        """
        # Set default method if unknown
        if 'method' not in kwds:
            kwds['method'] = default.ADVECTION
            kwds['method'][Support] = 'gpu_2k'

        # init base class
        super(GPUParticleAdvection, self).__init__(**kwds)
        self.fields_topo = self.fields_on_grid[0].topology
        self.velocity_topo = self.velocity.topology
        self._comm = self.fields_topo.comm
        self._comm_size = self._comm.Get_size()
        self._comm_rank = self._comm.Get_rank()

        user_src = get_extra_args_from_method(self, 'user_src', None)

        # init the second base class (the previous super only call
        # the fist __init__ method found in the order of
        # [ParticleAdvection, GPUOperator], i.e. ParticleAdvection.__init__)
        # the GPUOperator.__init__ must be explicitely called.
        # see http://stackoverflow.com/questions/3277367/how-does-pythons-super-work-with-multiple-inheritance
        GPUOperator.__init__(
            self,
            platform_id=get_extra_args_from_method(self, 'platform_id', None),
            device_id=get_extra_args_from_method(self, 'device_id', None),
            device_type=get_extra_args_from_method(self, 'device_type', None),
            user_src=user_src,
            **kwds)

        ## Work arrays for fields on particles (cpu)
        self.fields_on_part = None

        # The default is one kernel for all operations
        self._is2kernel = False
        if self.method[Support].find('gpu_2k') >= 0:
            # different kernels for advection and remesh
            self._is2kernel = True

        self._isMultiScale = False
        if MultiScale in self.method:
            if self.method[MultiScale] is not None:
                self._isMultiScale = True

        self._synchronize = None
        if self._isMultiScale:
            self._synchronize = UpdateGhostsFull(
                self.velocity.topology, self.velocity.nb_components)

        # Compute resolutions for kernels for each direction.
        ## Resolution of the local mesh but reoganized redarding
        ## splitting direction:
        ## direction X : XYZ
        ## direction Y : YXZ
        ## direction Z : ZYX in parallel, ZXY in sequentiel.
        self.resol_dir = npw.dim_ones((self.dim,))
        self.v_resol_dir = npw.dim_ones((self.dim,))
        shape = self.fields_topo.mesh.resolution
        v_shape = self.velocity_topo.mesh.resolution
        # Local mesh resolution
        resol = shape.copy()
        self.resol_dir[:self.dim] = self._reorderVect(shape)
        v_resol = v_shape.copy()
        self.v_resol_dir[:self.dim] = self._reorderVect(v_shape)

        self._append_size_constants(resol)
        self._append_size_constants(v_resol, prefix='V_NB')
        self._append_size_constants(
            [self.velocity_topo.ghosts()[self.direction]],
            prefix='V_GHOSTS_NB', suffix=[''])
        enum = ['I', 'II', 'III']
        self._append_size_constants(
            self._reorderVect(['NB' + d for d in S_DIR[:self.dim]]),
            prefix='NB_', suffix=enum[:self.dim])
        self._append_size_constants(
            self._reorderVect(['V_NB' + d for d in S_DIR[:self.dim]]),
            prefix='V_NB_', suffix=enum[:self.dim])

        fields_topo = self.fields_topo
        # Coordinates of the local origin
        self._coord_min = npw.ones(4, dtype=self.gpu_precision)
        self._coord_min[:self.dim] = fields_topo.mesh.origin

        # Space step for fields
        self._mesh_size = npw.ones(4, dtype=self.gpu_precision)
        self._mesh_size[:self.dim] = self._reorderVect(
            self.fields_topo.mesh.space_step)

        # Space step for velocity
        self._v_mesh_size = npw.ones(4, dtype=self.gpu_precision)
        self._v_mesh_size[:self.dim] = self._reorderVect(
            self.velocity_topo.mesh.space_step)

        self._mesh_info = npw.ones((12, ))
        self._mesh_info[:4] = self._mesh_size
        self._mesh_info[4:8] = self._v_mesh_size
        self._mesh_info[8] = self._coord_min[self.direction]
        self._mesh_info[9] = 1. / self._mesh_size[0]
        self._mesh_info[10] = 1. / self._v_mesh_size[0]
        self._cl_mesh_info = cl.Buffer(self.cl_env.ctx, cl.mem_flags.READ_ONLY,
                                       size=self._mesh_info.nbytes)
        cl.enqueue_write_buffer(self.cl_env.queue,
                                self._cl_mesh_info, self._mesh_info).wait()

        assert self._coord_min.dtype == self.gpu_precision
        assert self._mesh_size.dtype == self.gpu_precision
        assert self._v_mesh_size.dtype == self.gpu_precision

        ## opencl kernels build options
        self.build_options = ""

        # user defined opencl sources
        self.prg = None
        self._collect_usr_cl_src(user_src)

        # Set copy kernel
        self.copy = None
        self._collect_kernels_cl_src_copy()

        # Set transposition kernels
        self.transpose_xy, self.transpose_xy_r = None, None
        self.transpose_xz, self.transpose_xz_r = None, None
        self._collect_kernels_cl_src_transpositions_xy()
        if self.dim == 3:
            self._collect_kernels_cl_src_transpositions_xz()

        # Set advection and remesh kernels
        self.num_advec, self.num_remesh = None, None
        self.num_advec_and_remesh = None
        if self._is2kernel:
            self._collect_kernels_cl_src_2k()
            self._compute = self._compute_2k
        else:
            self._collect_kernels_cl_src_1k()
            if self._isMultiScale:
                self._compute = self._compute_1k_multiechelle
            else:
                if self.method[TimeIntegrator] is Euler:
                    self._compute = self._compute_1k_euler_simpleechelle
                else:
                    self._compute = self._compute_1k_simpleechelle

        self._buffer_allocations()
        if self.direction == 0:
            self._buffer_initialisations()

        ## List of executions
        self.exec_list = None
        self._build_exec_list()

        ## Particle initialisation OpenCL events for each field:
        self._init_events = {self.fields_on_grid[0]: []}

    @abstractmethod
    def globalMemoryUsagePreview(self, v_shape, shape):
        """
        @param[in] v_shape: shape of the discretization of the velocity
        @param[in] shape: shape of the discretization of the advected fields
        @return size of the required memory
        """
        pass

    def _build_exec_list(self):
        # Build execution list regarding splitting:
        # Splitting Strang 2nd order:
        #   3D: X(dt/2), Y(dt/2), Z(dt), Y(dt/2), X(dt/2)
        #   2D: X(dt/2), Y(dt), X(dt/2)
        if self.method[Splitting] == 'o2':
            if self.dim == 2:
                self.exec_list = [
                    [self._init_copy, self._compute],  # X(dt/2)
                    [self._init_transpose_xy, self._compute],  # Y(dt)
                    [self._init_transpose_xy, self._compute]  # X(dt/2)
                ]
            elif self.dim == 3:
                self.exec_list = [
                    [self._init_copy, self._compute],  # X(dt/2)
                    [self._init_transpose_xy, self._compute],  # Y(dt/2)
                    [self._init_transpose_xz, self._compute],  # Z(dt)
                    [self._init_transpose_xz, self._compute],  # Y(dt/2)
                    [self._init_transpose_xy, self._compute]  # X(dt/2)
                ]

        # Splitting Strang 2nd order (fullHalf):
        #   X(dt/2), Y(dt/2), Z(dt/2), Z(dt/2), Y(dt/2), X(dt/2)
        elif self.method[Splitting] == 'o2_FullHalf':
            if self.dim == 2:
                self.exec_list = [
                    [self._init_copy, self._compute],  # X(dt/2)
                    [self._init_transpose_xy, self._compute],  # Y(dt)
                    [self._init_copy, self._compute],  # Y(dt)
                    [self._init_transpose_xy, self._compute]  # X(dt/2)
                ]
            elif self.dim == 3:
                self.exec_list = [
                    [self._init_copy, self._compute],  # X(dt/2)
                    [self._init_transpose_xy, self._compute],  # Y(dt/2)
                    [self._init_transpose_xz, self._compute],  # Z(dt/2)
                    [self._init_copy, self._compute],  # Z(dt/2)
                    [self._init_transpose_xz, self._compute],  # Y(dt/2)
                    [self._init_transpose_xy, self._compute]  # X(dt/2)
                ]
        elif self.method[Splitting] == 'x_only':
            self.exec_list = [
                [self._init_copy, self._compute],  # X(dt)
                #[self._init_copy, self._init_copy_r],  # X(dt)
                ]
        else:
            raise ValueError('Not yet implemeted Splitting on GPU : ' +
                             self.method[Splitting])

    def globalMemoryUsagePreview(self, v_shape, shape):
        if self._is2kernel:
            r = (self.velocity.nb_components * v_shape.prod() +
                 (2 * self.fields_on_grid[0].nb_components + 1) * shape.prod())
        else:
            r = (self.velocity.nb_components * v_shape.prod() +
                 2 * self.fields_on_grid[0].nb_components * shape.prod())
        return r * self.cl_env.prec_size

    def _configure_numerical_methods(self):
        pass

    def _buffer_allocations(self):
        """
        Allocate OpenCL buffers for velocity and advected field.
        """
        ## Velocity.
        alloc = not isinstance(self.velocity, GPUDiscreteField)
        GPUDiscreteField.fromField(self.cl_env, self.velocity,
                                   self.gpu_precision, simple_layout=False)
        if alloc:
            self.size_global_alloc += self.velocity.mem_size

        ## Transported field.
        alloc = not isinstance(self.fields_on_grid[0], GPUDiscreteField)
        GPUDiscreteField.fromField(self.cl_env,
                                   self.fields_on_grid[0],
                                   self.gpu_precision,
                                   layout=False)
        if alloc:
            self.size_global_alloc += self.fields_on_grid[0].mem_size

        ## Fields on particles
        self.fields_on_part = {}
        start = 0
        for f in self.fields_on_grid:
            for i in xrange(start, start + f.nb_components):
                if type(self._rwork[i]) is np.ndarray:
                    self._rwork[i] = \
                        self.cl_env.global_allocation(self._rwork[i])
            self.fields_on_part[f] = self._rwork[start: start + f.nb_components]
            start += f.nb_components

        if self._is2kernel:
            ## Particles position
            if type(self._rwork[start]) is np.ndarray:
                self._rwork[start] = \
                    self.cl_env.global_allocation(self._rwork[start])
            self.part_position = self._rwork[start:start + 1]

        self._work = self.fields_on_part.values()

    def _buffer_initialisations(self):
        """
        OpenCL buffer initializations from user OpenCL kernels.
        Looking for kernels named <code>init<FieldName></code>.
        """
        for gpudf in self.variables:
            match = 'init' + '_'.join(gpudf.name.split('_')[:-1])
            # Looking for initKernel
            if self.prg is not None:
                for k in self.prg.all_kernels():
                    k_name = k.get_info(cl.kernel_info.FUNCTION_NAME)
                    if match.find(k_name) >= 0:
                        if __VERBOSE__:
                            print gpudf.name, '-> OpenCL Kernel', k_name
                        if gpudf == self.velocity:
                            workItemNumber, gwi, lwi = \
                                self.cl_env.get_WorkItems(self.v_resol_dir)
                        else:
                            workItemNumber, gwi, lwi = \
                                self.cl_env.get_WorkItems(self.resol_dir)
                        gpudf.setInitializationKernel(KernelLauncher(
                            cl.Kernel(self.prg, k_name), self.cl_env.queue,
                            gwi, lwi))

    def _collect_kernels_cl_src_copy(self):
        """
        Compile OpenCL sources for copy kernel.
        """
        # build_options = self.build_options
        # # copy settings
        # src, t_dim, b_rows, vec, f_space = self._kernel_cfg['copy']
        # while t_dim > self.resol_dir[0] or (self.resol_dir[0] % t_dim) > 0:
        #     t_dim /= 2
        # gwi, lwi = f_space(self.resol_dir, t_dim, b_rows, vec)

        # # Build code
        # build_options += " -D TILE_DIM_COPY={0}".format(t_dim)
        # build_options += " -D BLOCK_ROWS_COPY={0}".format(b_rows)
        # build_options += self._size_constants
        # prg = self.cl_env.build_src(
        #     src,
        #     build_options,
        #     vec)
        # self.copy = KernelLauncher(prg.copy,
        #                            self.cl_env.queue, gwi, lwi)
        self.copy = KernelLauncher(cl.enqueue_copy,
                                   self.cl_env.queue)

    def _collect_kernels_cl_src_transpositions_xy(self):
        """Compile OpenCL sources for transpositions kernel.
        @remark : Transpositions kernels are launched as initialization.
        Arrays are taken to their destination layout (for initialize in Y
        directions, either we came from X or Z but shapes are the Y ones).

        This routine sets transpose_xy and transpose_xy_r.
        """
        resol = self.fields_topo.mesh.resolution
        resol_tmp = npw.zeros_like(resol)

        # XY transposition settings
        is_XY_needed = self.direction == 1 or self.direction == 0
        if is_XY_needed:
            resol_tmp[...] = resol[...]
            if self.direction == 1:  # (XY -> YX)
                resol_tmp[0] = resol[1]
                resol_tmp[1] = resol[0]
                ocl_cte = " -D NB_I=NB_Y -D NB_II=NB_X -D NB_III=NB_Z"
            elif self.direction == 0:  # (YX -> XY) only for sequential
                ocl_cte = " -D NB_I=NB_X -D NB_II=NB_Y -D NB_III=NB_Z"

            self.transpose_xy = self._make_transpose_xy(resol_tmp, ocl_cte)

        # is_XY_r_needed = self.direction == 1 and self._comm_size > 1
        # if is_XY_r_needed:
        #     # Reversed XY transposition settings (YX -> XY), only in parallel
        #     resol_tmp[...] = resol[...]
        #     ocl_cte = " -D NB_I=NB_X -D NB_II=NB_Y -D NB_III=NB_Z"

        #     self.transpose_xy_r = self._make_transpose_xy(resol_tmp, ocl_cte)

    def _make_transpose_xy(self, resol_tmp, ocl_cte):

        build_options = self.build_options + self._size_constants
        src, t_dim, b_rows, is_padding, vec, f_space = \
            self._kernel_cfg['transpose_xy']
        while t_dim > resol_tmp[0] or t_dim > resol_tmp[1] or \
                (resol_tmp[0] % t_dim) > 0 or (resol_tmp[1] % t_dim) > 0:
            t_dim /= 2
        gwi, lwi, blocs_nb = f_space(resol_tmp, t_dim, b_rows, vec)

        if is_padding:
            build_options += " -D PADDING_XY=1"
        else:
            build_options += " -D PADDING_XY=0"
        build_options += " -D TILE_DIM_XY={0}".format(t_dim)
        build_options += " -D BLOCK_ROWS_XY={0}".format(b_rows)
        build_options += " -D NB_GROUPS_I={0}".format(blocs_nb[0])
        build_options += " -D NB_GROUPS_II={0}".format(blocs_nb[1])
        build_options += ocl_cte
        prg = self.cl_env.build_src(src, build_options, vec)
        return KernelLauncher(prg.transpose_xy, self.cl_env.queue, gwi, lwi)

    def _collect_kernels_cl_src_transpositions_xz(self):
        resol = self.fields_topo.mesh.resolution
        resol_tmp = npw.zeros_like(resol)

        is_XZ_needed = self.direction == 2 or self.direction == 1
        # XZ transposition settings
        if is_XZ_needed:
            resol_tmp[...] = resol[...]
            if self.direction == 1:  # ZXY -> YXZ (only for seqential)
                resol_tmp[0] = resol[1]
                resol_tmp[1] = resol[0]
                resol_tmp[2] = resol[2]
                ocl_cte = " -D NB_I=NB_Y -D NB_II=NB_X -D NB_III=NB_Z"
            elif self.direction == 2:
                # YXZ -> ZXY
                resol_tmp[0] = resol[2]
                resol_tmp[1] = resol[0]
                resol_tmp[2] = resol[1]
                ocl_cte = " -D NB_I=NB_Z -D NB_II=NB_X -D NB_III=NB_Y"
                # else:  # XYZ -> ZYX
                #     resol_tmp[0] = resol[2]
                #     resol_tmp[1] = resol[1]
                #     resol_tmp[2] = resol[0]
                #     ocl_cte = " -D NB_I=NB_Z -D NB_II=NB_Y -D NB_III=NB_X"
            self.transpose_xz = self._make_transpose_xz(resol_tmp, ocl_cte)

        # is_XZ_r_needed = self.direction == 2 and self._comm_size > 1
        # if is_XZ_r_needed:
        #     # Reversed XZ transposition settings (ZYX -> XYZ)
        #     resol_tmp[...] = resol[...]
        #     ocl_cte = " -D NB_I=NB_X -D NB_II=NB_Y -D NB_III=NB_Z"
        #     self.transpose_xz_r = self._make_transpose_xz(resol_tmp, ocl_cte)

    def _make_transpose_xz(self, resol_tmp, ocl_cte):

        build_options = self.build_options + self._size_constants
        src, t_dim, b_rows, b_deph, is_padding, vec, f_space = \
            self._kernel_cfg['transpose_xz']

        while t_dim > resol_tmp[0] or t_dim > resol_tmp[2] or \
                (resol_tmp[0] % t_dim) > 0 or (resol_tmp[2] % t_dim) > 0:
            t_dim /= 2
        gwi, lwi, blocs_nb = f_space(resol_tmp, t_dim, b_rows, b_deph, vec)
        if is_padding:
            build_options += " -D PADDING_XZ=1"
        else:
            build_options += " -D PADDING_XZ=0"
        build_options += " -D TILE_DIM_XZ={0}".format(t_dim)
        build_options += " -D BLOCK_ROWS_XZ={0}".format(b_rows)
        build_options += " -D BLOCK_DEPH_XZ={0}".format(b_deph)
        build_options += " -D NB_GROUPS_I={0}".format(blocs_nb[0])
        build_options += " -D NB_GROUPS_III={0}".format(blocs_nb[2])
        build_options += ocl_cte
        prg = self.cl_env.build_src(
            src,
            build_options,
            vec)
        return KernelLauncher(prg.transpose_xz, self.cl_env.queue, gwi, lwi)

    def _collect_usr_cl_src(self, usr_src):
        """
        Build user sources.

        """
        if usr_src is not None:
            build_options = self.build_options + self._size_constants
            workItemNb, gwi, lwi = self.cl_env.get_WorkItems(self.resol_dir)
            v_workItemNb, gwi, lwi = self.cl_env.get_WorkItems(self.v_resol_dir)
            build_options += " -D WI_NB=" + str(workItemNb)
            build_options += " -D V_WI_NB=" + str(v_workItemNb)
            self.prg = self.cl_env.build_src(usr_src, build_options, 1)


    def _collect_kernels_cl_src_1k(self):
        """
        Compile OpenCL sources for advection and remeshing kernel.
        """
        build_options = self.build_options + self._size_constants
        src, is_noBC, vec, f_space = self._kernel_cfg['advec_and_remesh']
        gwi, lwi = f_space(self.resol_dir, vec)
        WINb = lwi[0]
        build_options += " -D FORMULA=" + self.method[Remesh].__name__.upper()
        if self._isMultiScale:
            build_options += " -D MS_FORMULA="
            build_options += self.method[MultiScale].__name__.upper()
        if is_noBC:
            build_options += " -D WITH_NOBC=1"
        build_options += " -D WI_NB=" + str(WINb)
        build_options += " -D PART_NB_PER_WI="
        build_options += str(self.resol_dir[0] / WINb)
        ## Build code
        src = [s.replace('RKN', self.method[TimeIntegrator].__name__.lower())
               for s in src]
        ## Euler integrator
        if self.method[TimeIntegrator] is Euler:
            if not self._isMultiScale:
                src = [s for s in src if s.find(Euler.__name__.lower()) < 0]
                src[-1] = src[-1].replace('advection', 'advection_euler')
        prg = self.cl_env.build_src(
            src, build_options, vec,
            nb_remesh_components=self.fields_on_grid[0].nb_components)

        self.num_advec_and_remesh = KernelLauncher(
            prg.advection_and_remeshing, self.cl_env.queue, gwi, lwi)

    def _collect_kernels_cl_src_2k(self):
        """
        Compile OpenCL sources for advection and remeshing kernel.
        """
        # Advection
        build_options = self.build_options + self._size_constants
        src, is_noBC, vec, f_space = self._kernel_cfg['advec']
        gwi, lwi = f_space(self.resol_dir, vec)
        WINb = lwi[0]
        if self._isMultiScale:
            build_options += " -D MS_FORMULA="
            build_options += self.method[MultiScale].__name__.upper()
            self._compute_advec = self._compute_advec_multiechelle
        else:
            self._compute_advec = self._compute_advec_simpleechelle

        if is_noBC:
            build_options += " -D WITH_NOBC=1"
        build_options += " -D WI_NB=" + str(WINb)
        build_options += " -D PART_NB_PER_WI="
        build_options += str(self.resol_dir[0] / WINb)
        # Build code
        src = [s.replace('RKN', self.method[TimeIntegrator].__name__.lower())
               for s in src]
        ## Adding remeshing weights for the multiscale advection
        if self._isMultiScale:
            src.insert(1, self._kernel_cfg['remesh'][0][1])
        ## Euler integrator
        if self.method[TimeIntegrator] is Euler:
            if not self._isMultiScale:
                src = [s for s in src if s.find(Euler.__name__.lower()) < 0]
                src[-1] = src[-1].replace('advection', 'advection_euler')
                self._compute_advec = self._compute_advec_euler_simpleechelle
        prg = self.cl_env.build_src(
            src,
            build_options,
            vec,
            nb_remesh_components=self.fields_on_grid[0].nb_components)

        self.num_advec = KernelLauncher(
            prg.advection_kernel, self.cl_env.queue, gwi, lwi)

        # remeshing
        build_options = self.build_options + self._size_constants
        src, is_noBC, vec, f_space = self._kernel_cfg['remesh']
        gwi, lwi = f_space(self.resol_dir, vec)
        WINb = lwi[0]

        build_options += " -D FORMULA=" + self.method[Remesh].__name__.upper()
        if is_noBC:
            build_options += " -D WITH_NOBC=1"
        build_options += " -D WI_NB=" + str(WINb)
        build_options += " -D PART_NB_PER_WI="
        build_options += str(self.resol_dir[0] / WINb)
        ## Build code
        prg = self.cl_env.build_src(
            src, build_options, vec,
            nb_remesh_components=self.fields_on_grid[0].nb_components)
        self.num_remesh = KernelLauncher(
            prg.remeshing_kernel, self.cl_env.queue, gwi, lwi)

    @debug
    @profile
    def apply(self, simulation, dtCoeff, split_id, old_dir):
        """
        Apply operator along specified splitting direction.
        @param t : Current time
        @param dt : Time step
        @param d : Splitting direction
        @param split_id : Splitting step id
        """
        # If first direction of advection, wait for work gpu fields
        # It avoid wait_for lists to increase indelinitely
        # In practice, all events are terminated so wait() resets events list
        if split_id == 0:
            for v in self.fields_on_grid + [self.velocity]:
                v.clean_events()
        for exe in self.exec_list[split_id]:
            exe(simulation, dtCoeff, split_id, old_dir)

    def _init_copy(self, simulation, dtCoeff, split_id, old_dir):
        wait_evt = self.fields_on_grid[0].events
        for g, p in zip(self.fields_on_grid[0].gpu_data,
                        self.fields_on_part[self.fields_on_grid[0]]):
            evt = self.copy.launch_sizes_in_args(p, g, wait_for=wait_evt)
            #evt = self.copy(g, p, wait_for=wait_evt)
            self._init_events[self.fields_on_grid[0]].append(evt)

    # def _init_copy_r(self, simulation, dtCoeff, split_id, old_dir):
    #     wait_evt = self.fields_on_grid[0].events
    #     for g, p in zip(self.fields_on_grid[0].gpu_data,
    #                  self.fields_on_part[self.fields_on_grid[0]]):
    #         evt = self.copy.launch_sizes_in_args(g, p, wait_for=wait_evt)
    #         #evt = self.copy(p, g, wait_for=wait_evt)
    #         self._init_events[self.fields_on_grid[0]].append(evt)

    def _init_transpose_xy(self, simulation, dtCoeff, split_id, old_dir):
        wait_evt = self.fields_on_grid[0].events
        for g, p in zip(self.fields_on_grid[0].gpu_data,
                        self.fields_on_part[self.fields_on_grid[0]]):
            evt = self.transpose_xy(g, p, wait_for=wait_evt)
            self._init_events[self.fields_on_grid[0]].append(evt)

    # def _init_transpose_xy_r(self, simulation, dtCoeff, split_id, old_dir):
    #     wait_evt = self.fields_on_grid[0].events
    #     for g, p in zip(self.fields_on_grid[0].gpu_data,
    #                     self.fields_on_part[self.fields_on_grid[0]]):
    #         evt = self.transpose_xy_r(p, g, wait_for=wait_evt)
    #         self._init_events[self.fields_on_grid[0]].append(evt)

    def _init_transpose_xz(self, simulation, dtCoeff, split_id, old_dir):
        wait_evt = self.fields_on_grid[0].events
        for g, p in zip(self.fields_on_grid[0].gpu_data,
                        self.fields_on_part[self.fields_on_grid[0]]):
            evt = self.transpose_xz(g, p, wait_for=wait_evt)
            self._init_events[self.fields_on_grid[0]].append(evt)

    # def _init_transpose_xz_r(self, simulation, dtCoeff, split_id, old_dir):
    #     wait_evt = self.fields_on_grid[0].events
    #     for g, p in zip(self.fields_on_grid[0].gpu_data,
    #                     self.fields_on_part[self.fields_on_grid[0]]):
    #         evt = self.transpose_xz_r(p, g, wait_for=wait_evt)
    #         self._init_events[self.fields_on_grid[0]].append(evt)

    def _compute_advec_euler_simpleechelle(self, simulation, dtCoeff, split_id, old_dir):
        dt = simulation.timeStep * dtCoeff
        wait_evts = self.velocity.events + \
            self._init_events[self.fields_on_grid[0]]
        # Advection
        evt = self.num_advec(
            self.velocity.gpu_data[self.direction],
            self.part_position[0],
            self.gpu_precision(dt),
            self._cl_mesh_info,
            wait_for=wait_evts)
        self._init_events[self.fields_on_grid[0]].append(evt)

    def _compute_advec_simpleechelle(self, simulation, dtCoeff, split_id, old_dir):
        dt = simulation.timeStep * dtCoeff
        wait_evts = self.velocity.events + \
            self._init_events[self.fields_on_grid[0]]
        # Advection
        evt = self.num_advec(
            self.velocity.gpu_data[self.direction],
            self.part_position[0],
            self.gpu_precision(dt),
            self._cl_mesh_info,
            wait_for=wait_evts)
        self._init_events[self.fields_on_grid[0]].append(evt)

    def _compute_advec_multiechelle(self, simulation, dtCoeff, split_id, old_dir):
        dt = simulation.timeStep * dtCoeff
        wait_evts = self.velocity.events + \
            self._init_events[self.fields_on_grid[0]]
        # Advection
        evt = self.num_advec(
            self.velocity.gpu_data[self.direction],
            self.part_position[0],
            self.gpu_precision(dt),
            self.gpu_precision(1. / self._v_mesh_size[1]),
            self.gpu_precision(1. / self._v_mesh_size[2]),
            self._cl_mesh_info,
            wait_for=wait_evts)
        self._init_events[self.fields_on_grid[0]].append(evt)

    def _compute_2k(self, simulation, dtCoeff, split_id, old_dir):
        self._compute_advec(simulation, dtCoeff, split_id, old_dir)
        wait_evts = self._init_events[self.fields_on_grid[0]] + \
            self.fields_on_grid[0].events
        nbc = self.fields_on_grid[0].nb_components
        evt = self.num_remesh(*tuple(
            [self.part_position[0], ] +
            [self.fields_on_part[self.fields_on_grid[0]][i]
             for i in xrange(nbc)] +
            [self.fields_on_grid[0].gpu_data[i] for i in xrange(nbc)] +
            [self._cl_mesh_info, ]),
                              wait_for=wait_evts)
        self.fields_on_grid[0].events.append(evt)
        self._init_events[self.fields_on_grid[0]] = []

    def _compute_1k_multiechelle(self, simulation, dtCoeff, split_id, old_dir):
        if split_id==0 and self._synchronize is not None:
            self._synchronize(self.velocity.data)
            self.velocity.toDevice()
        dt = simulation.timeStep * dtCoeff
        wait_evts = self.velocity.events + \
            self._init_events[self.fields_on_grid[0]] + \
            self.fields_on_grid[0].events
        nbc = self.fields_on_grid[0].nb_components
        evt = self.num_advec_and_remesh(*tuple(
            [self.velocity.gpu_data[self.direction], ] +
            [self.fields_on_part[self.fields_on_grid[0]][i]
             for i in xrange(nbc)] +
            [self.fields_on_grid[0].gpu_data[i] for i in xrange(nbc)] +
            [self.gpu_precision(dt),
             self.gpu_precision(1. / self._v_mesh_size[1]),
             self.gpu_precision(1. / self._v_mesh_size[2]),
             self._cl_mesh_info]),
                                        wait_for=wait_evts)
        self.fields_on_grid[0].events.append(evt)
        self._init_events[self.fields_on_grid[0]] = []

    def _compute_1k_simpleechelle(self, simulation, dtCoeff, split_id, old_dir):
        dt = simulation.timeStep * dtCoeff
        wait_evts = self.velocity.events + \
            self._init_events[self.fields_on_grid[0]] + \
            self.fields_on_grid[0].events
        nbc = self.fields_on_grid[0].nb_components
        evt = self.num_advec_and_remesh(*tuple(
            [self.velocity.gpu_data[self.direction], ] +
            [self.fields_on_part[self.fields_on_grid[0]][i]
             for i in xrange(nbc)] +
            [self.fields_on_grid[0].gpu_data[i] for i in xrange(nbc)] +
            [self.gpu_precision(dt), self._cl_mesh_info]),
                                        wait_for=wait_evts)
        self.fields_on_grid[0].events.append(evt)
        self._init_events[self.fields_on_grid[0]] = []


    def _compute_1k_euler_simpleechelle(self, simulation, dtCoeff, split_id, old_dir):
        dt = simulation.timeStep * dtCoeff
        wait_evts = self.velocity.events + \
            self._init_events[self.fields_on_grid[0]] + \
            self.fields_on_grid[0].events
        nbc = self.fields_on_grid[0].nb_components
        evt = self.num_advec_and_remesh(*tuple(
            [self.velocity.gpu_data[self.direction], ] +
            [self.fields_on_part[self.fields_on_grid[0]][i]
             for i in xrange(nbc)] +
            [self.fields_on_grid[0].gpu_data[i] for i in xrange(nbc)] +
            [self.gpu_precision(dt), self._cl_mesh_info]),
                                        wait_for=wait_evts)
        self.fields_on_grid[0].events.append(evt)
        self._init_events[self.fields_on_grid[0]] = []

    def get_profiling_info(self):
        for k in [self.copy, self.transpose_xy, self.transpose_xy_r,
                  self.transpose_xz, self.transpose_xz_r,
                  self.num_advec_and_remesh,
                  self.num_advec, self.num_remesh]:
            if k is not None:
                for p in k.profile:
                    self.profiler += p

    @debug
    def finalize(self):
        """
        Cleaning, if required.
        """
        pass
        # for w in self._rwork:
        #     self.cl_env.global_deallocation(w)
        # self.cl_env.global_deallocation(self._cl_mesh_info)
