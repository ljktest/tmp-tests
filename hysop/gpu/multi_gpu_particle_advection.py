"""
@file multi_gpu_particle_advection.py

Discrete advection representation for Multi-GPU architecture.
"""
from abc import ABCMeta
from hysop.constants import np, debug, HYSOP_INTEGER, HYSOP_REAL, ORDER,\
    HYSOP_MPI_REAL, SIZEOF_HYSOP_REAL
from hysop.gpu.gpu_particle_advection import GPUParticleAdvection
from hysop.operator.discrete.discrete import get_extra_args_from_method
from hysop.methods_keys import Support, TimeIntegrator, MultiScale, Remesh
from hysop.numerics.integrators.runge_kutta2 import RK2
from hysop.numerics.remeshing import Linear as Linear_rmsh
from hysop.gpu.gpu_kernel import KernelLauncher
from hysop.tools.profiler import FProfiler, profile
from hysop.gpu import cl, CL_PROFILE
from hysop.mpi.main_var import MPI
import hysop.tools.numpywrappers as npw


class MultiGPUParticleAdvection(GPUParticleAdvection):
    """
    Particle advection operator representation on multi-GPU.

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

        @param max_velocity : maximum velocity estimation for computing communications buffer sizes.
        The estimation may be global or by components.
        @param max_dt : maimum time step estimation.
        @remark : Buffer widths are computed from max_velocity, max_dt and the mesh sizes:
          - velocity buffers : |max_velocity * max_dt / v_dx| + 1
          - scalar buffers : |max_velocity * max_dt / s_dx| + 1 + remesh_stencil/2
        @remark : by default, velocity data are supposed to be on the host. If not, user
        should set the arttribute velocity_only_on_device to True.
        """
        # fields_topo = kwds['fields_on_grid'][0].topology
        # direction = kwds['direction']
        # self._cut_dir = fields_topo.cutdir
        # self._is_cut_dir = self._cut_dir[direction]

        super(MultiGPUParticleAdvection, self).__init__(**kwds)
        max_velocity = get_extra_args_from_method(self, 'max_velocity', None)
        max_dt = get_extra_args_from_method(self, 'max_dt', None)
        max_cfl = get_extra_args_from_method(self, 'max_cfl', None)
        self._velocity_only_on_device = get_extra_args_from_method(
            self, 'velocity_only_on_device', False)
        if self._velocity_only_on_device:
            self._get_velocity_buffers = self._get_velocity_buffers_from_device
        else:
            self._get_velocity_buffers = self._get_velocity_buffers_from_host

        msg = "The Multi-GPU works only with the RK2 TimeIntegrator"
        assert self.method[TimeIntegrator] == RK2, msg
        assert self._comm_size > 1, 'Parallel only'
        assert self.dim == 3, 'A 2D multi-GPU version is not yet available'

        msg = "Either max_dt and _max_velocity or max_cfl must be given to advection "
        msg += "for computing communication buffer sizes."
        assert (max_dt is not None and max_velocity is not None) or max_cfl is not None

        assert self.fields_topo.cutdir[self.direction]
        assert self.fields_topo.shape[self.direction] > 1

        ## Neighbours process in the current direction
        first_cut_dir = self.fields_topo.cutdir.tolist().index(True)
        msh = self.fields_topo.mesh
        v_msh = self.velocity_topo.mesh
        # Global start index (lowest computed point, excluding ghosts)
        self._start_index = HYSOP_INTEGER(
            msh.start()[self.direction])
        # Velocity lobal start index (lowest computed point, excluding ghosts)
        self._v_start_index = HYSOP_INTEGER(
            v_msh.start()[self.direction])
        # Global end  index (highest computed point, excluding ghosts)
        self._stop_index = HYSOP_INTEGER(
            self._start_index + msh.resolution[self.direction] - 1
            - 2 * self.fields_topo.ghosts()[self.direction])
        # Velocity global end  index (highest computed point, excluding ghosts)
        self._v_stop_index = HYSOP_INTEGER(
            self._v_start_index + v_msh.resolution[self.direction] - 1
            - 2 * self.velocity_topo.ghosts()[self.direction])
        if self.fields_topo.cutdir[self.direction]:
            self._L_rk = self.fields_topo.neighbours[
                0, self.direction - first_cut_dir]
            self._R_rk = self.fields_topo.neighbours[
                1, self.direction - first_cut_dir]

        # Global resolution
        self.t_nb = \
            msh.discretization.resolution[self.direction] - 1
        # Velocity global resolution
        self.v_t_nb = \
            v_msh.discretization.resolution[self.direction] - 1
        i_s = [0] * self.dim
        v_i_s = [0] * self.dim
        i_s[self.direction] = self.fields_topo.ghosts()[self.direction]
        self._start_coord = msh.coords[self.direction][tuple(i_s)]
        v_i_s[self.direction] = self.velocity_topo.ghosts()[self.direction]
        self._v_start_coord = v_msh.coords[self.direction][tuple(v_i_s)]
        i_s[self.direction] = -1 - self.fields_topo.ghosts()[self.direction]
        self._stop_coord = msh.coords[self.direction][tuple(i_s)]
        v_i_s[self.direction] = -1 - \
            self.velocity_topo.ghosts()[self.direction]
        self._v_stop_coord = v_msh.coords[self.direction][tuple(v_i_s)]
        # mesh step
        self._space_step = msh.space_step
        self._v_space_step = v_msh.space_step

        # Maximum cfl for velocity and scalar
        if max_cfl is not None:
            scale_factor = self._v_space_step[self.direction]/self._space_step[self.direction]
            try:
                self.max_cfl_s = int(max_cfl[self.direction] * scale_factor) + 1
                self.max_cfl_v = int(max_cfl[self.direction]) + 1
            except TypeError:
                self.max_cfl_s = int(max_cfl * scale_factor) + 1
                self.max_cfl_v = int(max_cfl) + 1
        else:
            try:
                self.max_cfl_s = int(max_velocity[self.direction] * max_dt /
                                     self._space_step[self.direction]) + 1
                self.max_cfl_v = int(max_velocity[self.direction] * max_dt /
                                     self._v_space_step[self.direction]) + 1
            except TypeError:
                self.max_cfl_s = int(max_velocity * max_dt /
                                     self._space_step[self.direction]) + 1
                self.max_cfl_v = int(max_velocity * max_dt /
                                     self._v_space_step[self.direction]) + 1

        # Slice
        self._sl_dim = slice(self.dim, 2 * self.dim)
        self._cl_work_size = 0

        #Advection variables
        self._v_buff_width = self.max_cfl_v
        _v_r_buff = npw.zeros((self._v_buff_width,
                               self.v_resol_dir[1],
                               self.v_resol_dir[2]))
        _v_l_buff = npw.zeros_like(_v_r_buff)
        self._v_r_buff_loc = npw.zeros_like(_v_r_buff)
        self._v_l_buff_loc = npw.zeros_like(_v_r_buff)
        self._cl_v_r_buff = self.cl_env.global_allocation(_v_l_buff)
        self._cl_v_l_buff = self.cl_env.global_allocation(_v_r_buff)
        cl.enqueue_copy(self.cl_env.queue,
                        self._cl_v_r_buff, _v_r_buff).wait()
        cl.enqueue_copy(self.cl_env.queue,
                        self._cl_v_l_buff, _v_l_buff).wait()
        self._cl_work_size += 2 * _v_l_buff.nbytes
        self._v_r_buff, evt = cl.enqueue_map_buffer(
            self.cl_env.queue,
            self._cl_v_r_buff,
            offset=0,
            shape=_v_r_buff.shape,
            dtype=HYSOP_REAL,
            flags=cl.map_flags.READ | cl.map_flags.WRITE,
            is_blocking=False,
            order=ORDER)
        evt.wait()
        self._v_l_buff, evt = cl.enqueue_map_buffer(
            self.cl_env.queue,
            self._cl_v_l_buff,
            offset=0,
            shape=_v_l_buff.shape,
            dtype=HYSOP_REAL,
            flags=cl.map_flags.READ | cl.map_flags.WRITE,
            is_blocking=False,
            order=ORDER)
        evt.wait()
        self._v_l_buff_flat = self._v_l_buff.ravel(order='F')
        self._v_r_buff_flat = self._v_r_buff.ravel(order='F')
        self._v_l_buff_loc_flat = self._v_l_buff_loc.ravel(order='F')
        self._v_r_buff_loc_flat = self._v_r_buff_loc.ravel(order='F')

        self._v_buff_size = self._v_buff_width * \
            self.v_resol_dir[1] * self.v_resol_dir[2]
        self._v_pitches_host = (int(_v_l_buff[:, 0, 0].nbytes),
                                int(_v_l_buff[:, :, 0].nbytes))
        self._v_buffer_region = (
            int(self._v_buff_width * SIZEOF_HYSOP_REAL),
            int(self.v_resol_dir[1]),
            int(self.v_resol_dir[2]))
        self._v_block_size = 1024 * 1024  # 1MByte
        while self._v_l_buff.nbytes % self._v_block_size != 0:
            self._v_block_size /= 2
        w = "WARNING: block size for pipelined GPU-to-GPU transfer is small, "
        if self._v_block_size < 256 * 1024:
            self._v_block_size = self._v_l_buff.nbytes / 4
            print w + "use blocks of {0} MB (4 blocks velocity)".format(
                self._v_block_size / (1024. * 1024.))
        self._v_n_blocks = self._v_l_buff.nbytes / self._v_block_size
        self._v_elem_block = np.prod(self._v_l_buff.shape) / self._v_n_blocks
        self._l_recv_v = [None, ] * self._v_n_blocks
        self._r_recv_v = [None, ] * self._v_n_blocks
        self._send_to_l_v = [None, ] * self._v_n_blocks
        self._send_to_r_v = [None, ] * self._v_n_blocks
        self._evt_l_v = [None, ] * self._v_n_blocks
        self._evt_r_v = [None, ] * self._v_n_blocks
        self._v_block_slice = [None, ] * self._v_n_blocks
        for b in xrange(self._v_n_blocks):
            self._v_block_slice[b] = slice(
                b * self._v_elem_block, (b + 1) * self._v_elem_block)

        ## Python remeshing formula for the multiscale interpolation
        self._py_ms_formula = self.method[MultiScale]
        if self._isMultiScale:
            if not self._py_ms_formula is Linear_rmsh:
                raise ValueError('Not yet implemented' +
                                 str(self.method[MultiScale]))
        ## Python remeshing formula
        self._py_remesh = self.method[Remesh]()
        ## Number of weights
        self._nb_w = len(self._py_remesh.weights)
        self._s_buff_width = self.max_cfl_s + self._nb_w / 2
        _s_l_buff = npw.zeros(
            (self._s_buff_width * self.resol_dir[1] * self.resol_dir[2], ))
        _s_r_buff = npw.zeros(
            (self._s_buff_width * self.resol_dir[1] * self.resol_dir[2], ))
        self._s_froml_buff_max = npw.zeros((self._s_buff_width,
                                            self.resol_dir[1],
                                            self.resol_dir[2]))
        self._s_fromr_buff_max = npw.zeros_like(self._s_froml_buff_max)
        self._cl_s_r_buff = self.cl_env.global_allocation(_s_l_buff)
        self._cl_s_l_buff = self.cl_env.global_allocation(_s_r_buff)
        cl.enqueue_copy(self.cl_env.queue,
                        self._cl_s_r_buff, _s_r_buff).wait()
        cl.enqueue_copy(self.cl_env.queue,
                        self._cl_s_l_buff, _s_l_buff).wait()
        self._cl_work_size += 2 * self._s_froml_buff_max.nbytes
        self._s_l_buff, evt = cl.enqueue_map_buffer(
            self.cl_env.queue,
            self._cl_s_l_buff,
            offset=0,
            shape=_s_l_buff.shape,
            dtype=HYSOP_REAL,
            flags=cl.map_flags.READ | cl.map_flags.WRITE,
            is_blocking=False,
            order=ORDER)
        evt.wait()
        self._s_r_buff, evt = cl.enqueue_map_buffer(
            self.cl_env.queue,
            self._cl_s_r_buff,
            offset=0,
            shape=_s_r_buff.shape,
            dtype=HYSOP_REAL,
            flags=cl.map_flags.READ | cl.map_flags.WRITE,
            is_blocking=False,
            order=ORDER)
        evt.wait()
        self._s_froml_buff_flat = self._s_froml_buff_max.ravel(order='F')
        self._s_fromr_buff_flat = self._s_fromr_buff_max.ravel(order='F')
        # attributes declarations, values are recomputed at each time
        self._s_buff_width_loc_p, self._s_buff_width_loc_m = 0, 0
        self._s_buff_width_from_l, self._s_buff_width_from_r = 0, 0
        self._s_froml_buff, self._s_locl_buff = None, None
        self._s_fromr_buff, self._s_locr_buff = None, None
        self._s_buffer_region_on_l, self._s_buffer_region_on_r = None, None
        self._origin_locl, self._origin_locr = None, None
        self._s_block_size_to_r, self._s_block_size_to_l = None, None
        self._s_block_size_from_r, self._s_block_size_from_l = None, None
        self._s_n_blocks_to_r, self._s_n_blocks_to_l = None, None
        self._s_n_blocks_from_r, self._s_n_blocks_from_l = None, None
        self._s_elem_block_to_r, self._s_elem_block_to_l = None, None
        self._s_elem_block_from_r, self._s_elem_block_from_l = None, None
        self._s_block_slice_to_r, self._s_block_slice_to_l = None, None
        self._s_block_slice_from_r, self._s_block_slice_from_l = None, None
        self._r_recv, self._l_recv = None, None
        self._evt_get_l, self._evt_get_r = None, None
        self._l_send, self._r_send = None, None

        self._queue_comm_m = self.cl_env.create_other_queue()
        self._queue_comm_p = self.cl_env.create_other_queue()

        self.profiler += FProfiler('comm_gpu_advec_set')
        self.profiler += FProfiler('comm_cpu_advec_get')
        self.profiler += FProfiler('comm_cpu_advec')
        self.profiler += FProfiler('comm_gpu_remesh_get')
        self.profiler += FProfiler('comm_gpu_remesh_get_loc')
        self.profiler += FProfiler('comm_gpu_remesh_set_loc')
        self.profiler += FProfiler('comm_cpu_remesh')
        self.profiler += FProfiler('comm_calc_remesh')

        # Collect sources for communication
        self._compute = self._compute_1c_comm
        if self._is2kernel:
            self._collect_kernels_cl_src_2k_comm()
            self._num_comm_l = self._num_2k_comm_l
            self._num_comm_r = self._num_2k_comm_r
            self._num_comm = self._num_2k_comm
        else:
            self._collect_kernels_cl_src_1k_comm()
            if self._isMultiScale:
                self._num_comm_l = self._num_1k_ms_comm_l
                self._num_comm_r = self._num_1k_ms_comm_r
                self._num_comm = self._num_1k_ms_comm
            else:
                self._num_comm_l = self._num_1k_comm_l
                self._num_comm_r = self._num_1k_comm_r
                self._num_comm = self._num_1k_comm

        if self.direction == 2:
            # Device is in ZXY layout
            self._pitches_dev = (
                int(self.fields_on_grid[0].data[0][0, 0, :].nbytes),
                int(self.fields_on_grid[0].data[0][:, 0, :].nbytes))
            self._v_pitches_dev = (
                int(self.velocity.data[0][0, 0, :].nbytes),
                int(self.velocity.data[0][:, 0, :].nbytes))
        elif self.direction == 1:
            # Device is in YXZ layout
            self._pitches_dev = (
                int(self.fields_on_grid[0].data[0][0, :, 0].nbytes),
                int(self.fields_on_grid[0].data[0][:, :, 0].nbytes))
            self._v_pitches_dev = (
                int(self.velocity.data[0][0, :, 0].nbytes),
                int(self.velocity.data[0][:, :, 0].nbytes))
        elif self.direction == 0:
            # Device is in XYZ layout
            self._pitches_dev = (
                int(self.fields_on_grid[0].data[0][:, 0, 0].nbytes),
                int(self.fields_on_grid[0].data[0][:, :, 0].nbytes))
            self._v_pitches_dev = (
                int(self.velocity.data[0][:, 0, 0].nbytes),
                int(self.velocity.data[0][:, :, 0].nbytes))
        # Beanching the proper _compute function
        if self.fields_on_grid[0].nb_components > 1:
            raise ValueError("Not yet implemented")

        self._build_exec_list()

    def _collect_kernels_cl_src_2k(self):
        pass

    def _collect_kernels_cl_src_1k(self):
        pass

    def _collect_kernels_cl_src_1k_comm(self):
        """
        Compile OpenCL sources for advection and remeshing kernel when
        communications needed.
        """
        build_options = self.build_options + self._size_constants
        if self._isMultiScale:
            src, is_noBC, vec, f_space = \
                self._kernel_cfg['advec_MS_and_remesh_comm']
        else:
            src, is_noBC, vec, f_space = \
                self._kernel_cfg['advec_and_remesh_comm']
        gwi, lwi = f_space(self.resol_dir, vec)
        WINb = lwi[0]

        build_options += " -D WI_NB=" + str(WINb)
        if self._isMultiScale:
            build_options += " -D MS_FORMULA="
            build_options += self.method[MultiScale].__name__.upper()
        build_options += " -D V_START_INDEX=" + str(self._v_start_index)
        build_options += " -D V_STOP_INDEX=" + str(self._v_stop_index)
        build_options += " -D START_INDEX=" + str(self._start_index)
        build_options += " -D STOP_INDEX=" + str(self._stop_index)
        build_options += " -D V_BUFF_WIDTH=" + str(self._v_buff_width)
        build_options += " -D FORMULA=" + self.method[Remesh].__name__.upper()
        build_options += " -D PART_NB_PER_WI="
        build_options += str(self.resol_dir[0] / WINb)
        build_options += " -D BUFF_WIDTH=" + str(self._s_buff_width)
        prg = self.cl_env.build_src(
            src, build_options, 1)
        self.num_advec_and_remesh_comm_l = KernelLauncher(
            prg.buff_advec_and_remesh_l, self.cl_env.queue,
            (gwi[1], gwi[2]), (32, 1))
        self.num_advec_and_remesh_comm_r = KernelLauncher(
            prg.buff_advec_and_remesh_r, self.cl_env.queue,
            (gwi[1], gwi[2]), (32, 1))
        self.num_advec_and_remesh = KernelLauncher(
            prg.buff_advec_and_remesh, self.cl_env.queue,
            gwi, lwi)

    def _collect_kernels_cl_src_2k_comm(self):
        """
        Compile OpenCL sources for advection and remeshing kernel when
        communications needed.
        """
        build_options = self.build_options + self._size_constants
        if self._isMultiScale:
            src, is_noBC, vec, f_space = self._kernel_cfg['advec_MS_comm']
        else:
            src, is_noBC, vec, f_space = self._kernel_cfg['advec_comm']
        gwi, lwi = f_space(self.resol_dir, vec)
        WINb = lwi[0]

        build_options += " -D WI_NB=" + str(WINb)
        if self._isMultiScale:
            build_options += " -D MS_FORMULA="
            build_options += self.method[MultiScale].__name__.upper()
        build_options += " -D V_START_INDEX=" + str(self._v_start_index)
        build_options += " -D V_STOP_INDEX=" + str(self._v_stop_index)
        build_options += " -D START_INDEX=" + str(self._start_index)
        build_options += " -D STOP_INDEX=" + str(self._stop_index)
        build_options += " -D V_BUFF_WIDTH=" + str(self._v_buff_width)
        prg = self.cl_env.build_src(
            src, build_options, 1)
        self.num_advec = KernelLauncher(
            prg.buff_advec, self.cl_env.queue,
            gwi, lwi)

        ## remeshing
        build_options = self.build_options + self._size_constants
        src, is_noBC, vec, f_space = self._kernel_cfg['remesh_comm']
        gwi, lwi = f_space(self.resol_dir, vec)
        WINb = lwi[0]

        build_options += " -D FORMULA=" + self.method[Remesh].__name__.upper()
        build_options += " -D WI_NB=" + str(WINb)
        build_options += " -D PART_NB_PER_WI="
        build_options += str(self.resol_dir[0] / WINb)
        #Build code
        build_options += " -D START_INDEX=" + str(self._start_index)
        build_options += " -D STOP_INDEX=" + str(self._stop_index)
        build_options += " -D BUFF_WIDTH=" + str(self._s_buff_width)
        prg = self.cl_env.build_src(
            src, build_options, 1)
        self.num_remesh_comm_l = KernelLauncher(
            prg.buff_remesh_l, self._queue_comm_m,
            (gwi[1], gwi[2]), (32, 1))
        self.num_remesh_comm_r = KernelLauncher(
            prg.buff_remesh_r, self._queue_comm_p,
            (gwi[1], gwi[2]), (32, 1))
        self.num_remesh = KernelLauncher(
            prg.remesh, self.cl_env.queue,
            gwi, lwi)

    def _recompute_scal_buffers(self, max_velo_m, max_velo_p, dt):
        dx = self._space_step[self.direction]
        max_cfl_p_s = int(max_velo_p * dt / dx) + 1
        max_cfl_m_s = int(max_velo_m * dt / dx) + 1
        self._s_buff_width_loc_p = max_cfl_p_s + self._nb_w / 2
        self._s_buff_width_loc_m = max_cfl_m_s + self._nb_w / 2
        assert self._s_froml_buff_max.shape[0] >= self._s_buff_width_loc_p, \
            "Multi-GPU Comm R-Buffer too small: {0} >= {1}".format(
                self._s_froml_buff_max.shape[0] >= self._s_buff_width_loc_p)
        assert self._s_froml_buff_max.shape[0] >= self._s_buff_width_loc_m, \
            "Multi-GPU Comm L-Buffer too small: {0} >= {1}".format(
                self._s_froml_buff_max.shape[0] >= self._s_buff_width_loc_m)
        self._s_buff_width_from_l = self._comm.sendrecv(
            sendobj=self._s_buff_width_loc_p, dest=self._R_rk,
            sendtag=1 + 7 * self._R_rk,
            source=self._L_rk,
            recvtag=1 + 7 * self._comm_rank)
        self._s_buff_width_from_r = self._comm.sendrecv(
            sendobj=self._s_buff_width_loc_m, dest=self._L_rk,
            sendtag=10000 + 9 * self._L_rk,
            source=self._R_rk,
            recvtag=10000 + 9 * self._comm_rank)

        s = self._s_buff_width_from_l * \
            self.resol_dir[1] * self.resol_dir[2]
        self._s_froml_buff = self._s_froml_buff_flat[:s].reshape(
            (self._s_buff_width_from_l,
             self.resol_dir[1],
             self.resol_dir[2]), order=ORDER)
        self._s_locl_buff = \
            self.fields_on_grid[0].host_data_pinned[0].reshape(
                self.resol_dir, order=ORDER)[:self._s_buff_width_from_l, :, :]
        s = self._s_buff_width_from_r * \
            self.resol_dir[1] * self.resol_dir[2]
        self._s_fromr_buff = self._s_fromr_buff_flat[:s].reshape(
            (self._s_buff_width_from_r,
             self.resol_dir[1],
             self.resol_dir[2]), order=ORDER)
        self._s_locr_buff = \
            self.fields_on_grid[0].host_data_pinned[0].reshape(
                self.resol_dir, order=ORDER)[-self._s_buff_width_from_r:, :, :]

        self._s_buffer_region_on_l = (
            int(SIZEOF_HYSOP_REAL * self._s_buff_width_from_l),
            int(self.resol_dir[1]),
            int(self.resol_dir[2]))
        self._origin_locl = (0, 0, 0)
        self._s_buffer_region_on_r = (
            int(SIZEOF_HYSOP_REAL * self._s_buff_width_from_r),
            int(self.resol_dir[1]),
            int(self.resol_dir[2]))
        self._origin_locr = (
            int((self.resol_dir[0] - self._s_buff_width_from_r)
                * SIZEOF_HYSOP_REAL), 0, 0)

        # Recompute blocks number and block size
        self._s_block_size_to_r, self._s_n_blocks_to_r, \
            self._s_elem_block_to_r, self._s_block_slice_to_r = \
            self._compute_block_number_and_size(
                SIZEOF_HYSOP_REAL * self._s_buff_width_loc_p *
                self.resol_dir[1] * self.resol_dir[2])
        self._s_block_size_to_l, self._s_n_blocks_to_l, \
            self._s_elem_block_to_l, self._s_block_slice_to_l = \
            self._compute_block_number_and_size(
                SIZEOF_HYSOP_REAL * self._s_buff_width_loc_m *
                self.resol_dir[1] * self.resol_dir[2])
        self._s_block_size_from_r, self._s_n_blocks_from_r, \
            self._s_elem_block_from_r, self._s_block_slice_from_r = \
            self._compute_block_number_and_size(self._s_fromr_buff.nbytes)
        self._s_block_size_from_l, self._s_n_blocks_from_l, \
            self._s_elem_block_from_l, self._s_block_slice_from_l = \
            self._compute_block_number_and_size(self._s_froml_buff.nbytes)
        # print "[" + str(self._comm_rank) + \
        #     "] Multi-GPU comm: send to L=({0} MB, {1} bloc, {2}, {3}),".format(
        #         self._s_block_size_to_l * self._s_n_blocks_to_l /
        #         (1024. * 1024),
        #         self._s_n_blocks_to_l,
        #         self._s_buff_width_loc_m, self._s_froml_buff_max.shape[0]) + \
        #     "  R=({0} MB, {1} bloc, {2}, {3})".format(
        #         self._s_block_size_to_r * self._s_n_blocks_to_r /
        #         (1024. * 1024),
        #         self._s_n_blocks_to_r,
        #         self._s_buff_width_loc_p, self._s_froml_buff_max.shape[0]) + \
        #     "; recv from L=({0} MB, {1} bloc),".format(
        #         self._s_block_size_from_l * self._s_n_blocks_from_l /
        #         (1024. * 1024),
        #        self._s_n_blocks_from_l) + \
        #     "  R=({0} MB, {1} bloc)".format(
        #         self._s_block_size_from_r * self._s_n_blocks_from_r /
        #         (1024. * 1024),
        #         self._s_n_blocks_from_r)

        # Events lists
        self._r_recv = [None, ] * self._s_n_blocks_from_r
        self._l_recv = [None, ] * self._s_n_blocks_from_l
        self._evt_get_l = [None, ] * self._s_n_blocks_to_l
        self._evt_get_r = [None, ] * self._s_n_blocks_to_r
        self._l_send = [None, ] * self._s_n_blocks_to_l
        self._r_send = [None, ] * self._s_n_blocks_to_r

    def _compute_block_number_and_size(self, buff_size):
        block = 1024 * 1024  # 1MByte
        while buff_size % block != 0:
            block /= 2
        if block < 256 * 1024:
            block = buff_size / 4
        n_b = buff_size / block
        n_elem = block / SIZEOF_HYSOP_REAL
        slices = [None, ] * n_b
        for b in xrange(n_b):
            slices[b] = slice(b * n_elem, (b + 1) * n_elem)
        return int(block), int(n_b), n_elem, slices

    def _get_velocity_buffers_from_host(self, ghosts):
        if self.direction == 0:
            velo_sl = (slice(self.v_resol_dir[0] - self._v_buff_width - ghosts,
                             self.v_resol_dir[0] - ghosts),
                       slice(None), slice(None),)
            self._v_r_buff_loc[...] = self.velocity.data[0][velo_sl]
            velo_sl = (slice(0 + ghosts, self._v_buff_width + ghosts),
                       slice(None), slice(None))
            self._v_l_buff_loc[...] = self.velocity.data[0][velo_sl]
        if self.direction == 1:
            velo_sl = (slice(None),
                       slice(self.v_resol_dir[0] - self._v_buff_width - ghosts,
                             self.v_resol_dir[0] - ghosts),
                       slice(None))
            self._v_r_buff_loc[...] = \
                self.velocity.data[1][velo_sl].swapaxes(0, 1)
            velo_sl = (slice(None),
                       slice(0 + ghosts, self._v_buff_width + ghosts),
                       slice(None))
            self._v_l_buff_loc[...] = \
                self.velocity.data[1][velo_sl].swapaxes(0, 1)
        if self.direction == 2:
            velo_sl = (slice(None), slice(None),
                       slice(self.v_resol_dir[0] - self._v_buff_width - ghosts,
                             self.v_resol_dir[0] - ghosts))
            self._v_r_buff_loc[...] = \
                self.velocity.data[2][velo_sl].swapaxes(0, 1).swapaxes(0, 2)
            velo_sl = (slice(None), slice(None),
                       slice(0 + ghosts, self._v_buff_width + ghosts))
            self._v_l_buff_loc[...] = \
                self.velocity.data[2][velo_sl].swapaxes(0, 1).swapaxes(0, 2)

    def _get_velocity_buffers_from_device(self, ghosts):
        _evt_l_v = cl.enqueue_copy(
            self._queue_comm_m,
            self._v_l_buff_loc,
            self.velocity.gpu_data[self.direction],
            host_origin=(0, 0, 0),
            host_pitches=self._v_pitches_host,
            buffer_origin=(int(SIZEOF_HYSOP_REAL * ghosts), 0, 0),
            buffer_pitches=self._v_pitches_dev,
            region=self._v_buffer_region,
            is_blocking=False)
        _evt_r_v = cl.enqueue_copy(
            self._queue_comm_m,
            self._v_r_buff_loc,
            self.velocity.gpu_data[self.direction],
            host_origin=(0, 0, 0),
            host_pitches=self._v_pitches_host,
            buffer_origin=(int(SIZEOF_HYSOP_REAL * (
                self.v_resol_dir[0] - self._v_buff_width - ghosts)), 0, 0),
            buffer_pitches=self._v_pitches_dev,
            region=self._v_buffer_region,
            is_blocking=False)
        _evt_l_v.wait()
        _evt_r_v.wait()

    def _exchange_velocity_buffers(self, dt):
        ctime = MPI.Wtime()
        ghosts = self.velocity_topo.ghosts()[self.direction]
        if self.direction == 0:
            max_velo_p = np.max(self.velocity.data[0][-ghosts - 1:, :, :])
            max_velo_m = np.max(self.velocity.data[0][:ghosts + 1, :, :])
        if self.direction == 1:
            max_velo_p = np.max(self.velocity.data[1][:, -ghosts - 1:, :])
            max_velo_m = np.max(self.velocity.data[1][:, :ghosts + 1, :])
        if self.direction == 2:
            max_velo_p = np.max(self.velocity.data[2][:, :, -ghosts - 1:])
            max_velo_m = np.max(self.velocity.data[2][:, :, :ghosts + 1])
        self._recompute_scal_buffers(max_velo_m, max_velo_p, dt)
        self._get_velocity_buffers(ghosts)
        self.profiler['comm_cpu_advec_get'] += MPI.Wtime() - ctime

        ctime = MPI.Wtime()
        for b in xrange(self._v_n_blocks):
            self._l_recv_v[b] = self._comm.Irecv(
                [self._v_l_buff_flat[self._v_block_slice[b]],
                 self._v_elem_block, HYSOP_MPI_REAL],
                source=self._L_rk, tag=17 + 19 * self._L_rk + 59 * b)
            self._r_recv_v[b] = self._comm.Irecv(
                [self._v_r_buff_flat[self._v_block_slice[b]],
                 self._v_elem_block, HYSOP_MPI_REAL],
                source=self._R_rk, tag=29 + 23 * self._R_rk + 57 * b)
        for b in xrange(self._v_n_blocks):
            self._send_to_r_v[b] = self._comm.Issend(
                [self._v_r_buff_loc_flat[self._v_block_slice[b]],
                 self._v_elem_block, HYSOP_MPI_REAL],
                dest=self._R_rk, tag=17 + 19 * self._comm_rank + 59 * b)
            self._send_to_l_v[b] = self._comm.Issend(
                [self._v_l_buff_loc_flat[self._v_block_slice[b]],
                 self._v_elem_block, HYSOP_MPI_REAL],
                dest=self._L_rk, tag=29 + 23 * self._comm_rank + 57 * b)
        if CL_PROFILE:
            for b in xrange(self._v_n_blocks):
                self._l_recv_v[b].Wait()
                self._r_recv_v[b].Wait()
        self.profiler['comm_cpu_advec'] += MPI.Wtime() - ctime

    def _todevice_velocity_buffers(self):
        for b in xrange(self._v_n_blocks):
            self._l_recv_v[b].Wait()
            self._evt_l_v[b] = cl.enqueue_copy(
                self._queue_comm_m,
                self._cl_v_l_buff, self._v_l_buff,
                host_origin=(b * self._v_block_size, 0, 0),
                host_pitches=(self._v_l_buff.nbytes, 0),
                buffer_origin=(b * self._v_block_size, 0, 0),
                buffer_pitches=(self._v_l_buff.nbytes, 0),
                region=(self._v_block_size, 1, 1),
                is_blocking=False)
        for b in xrange(self._v_n_blocks):
            self._r_recv_v[b].Wait()
            self._evt_r_v[b] = cl.enqueue_copy(
                self._queue_comm_p,
                self._cl_v_r_buff, self._v_r_buff,
                host_origin=(b * self._v_block_size, 0, 0),
                host_pitches=(self._v_r_buff.nbytes, 0),
                buffer_origin=(b * self._v_block_size, 0, 0),
                buffer_pitches=(self._v_r_buff.nbytes, 0),
                region=(self._v_block_size, 1, 1),
                is_blocking=False)

        if CL_PROFILE:
            advec_gpu_time = 0.
            for evt in self._evt_l_v + self._evt_r_v:
                evt.wait()
                advec_gpu_time += (evt.profile.end - evt.profile.start) * 1e-9
            self.profiler['comm_gpu_advec_set'] += advec_gpu_time

    def _init_copy(self, simulation, dtCoeff, split_id, old_dir):
        self._exchange_velocity_buffers(simulation.timeStep * dtCoeff)
        wait_evt = self.fields_on_grid[0].events
        for g, p in zip(self.fields_on_grid[0].gpu_data,
                        self.fields_on_part[self.fields_on_grid[0]]):
            evt = self.copy.launch_sizes_in_args(p, g, wait_for=wait_evt)
            #evt = self.copy(g, p, wait_for=wait_evt)
            self._init_events[self.fields_on_grid[0]].append(evt)

    def _init_transpose_xy(self, simulation, dtCoeff, split_id, old_dir):
        self._exchange_velocity_buffers(simulation.timeStep * dtCoeff)
        wait_evt = self.fields_on_grid[0].events
        for g, p in zip(self.fields_on_grid[0].gpu_data,
                        self.fields_on_part[self.fields_on_grid[0]]):
            evt = self.transpose_xy(g, p, wait_for=wait_evt)
            self._init_events[self.fields_on_grid[0]].append(evt)

    def _init_transpose_xz(self, simulation, dtCoeff, split_id, old_dir):
        self._exchange_velocity_buffers(simulation.timeStep * dtCoeff)
        wait_evt = self.fields_on_grid[0].events
        for g, p in zip(self.fields_on_grid[0].gpu_data,
                        self.fields_on_part[self.fields_on_grid[0]]):
            evt = self.transpose_xz(g, p, wait_for=wait_evt)
            self._init_events[self.fields_on_grid[0]].append(evt)

    def _compute_advec_comm(self, simulation, dtCoeff, split_id, old_dir):
        dt = simulation.timeStep * dtCoeff
        self._todevice_velocity_buffers()
        wait_evts = self.velocity.events + self._evt_l_v + self._evt_r_v + \
            self._init_events[self.fields_on_grid[0]]
        if self._isMultiScale:
            evt = self.num_advec(
                self.velocity.gpu_data[self.direction],
                self.part_position[0],
                self._cl_v_l_buff,
                self._cl_v_r_buff,
                self.gpu_precision(dt),
                self.gpu_precision(1. / self._v_mesh_size[1]),
                self.gpu_precision(1. / self._v_mesh_size[2]),
                self._cl_mesh_info,
                wait_for=wait_evts)
        else:
            evt = self.num_advec(
                self.velocity.gpu_data[self.direction],
                self.part_position[0],
                self._cl_v_l_buff,
                self._cl_v_r_buff,
                self.gpu_precision(dt),
                self._cl_mesh_info,
                wait_for=wait_evts)
        self._init_events[self.fields_on_grid[0]].append(evt)

    def _num_2k_comm_l(self, wait_list, dt):
        return self.num_remesh_comm_l(
            self.part_position[0],
            self.fields_on_part[self.fields_on_grid[0]][0],
            self._cl_s_l_buff,
            HYSOP_INTEGER(self._s_buff_width_loc_m),
            self._cl_mesh_info,
            wait_for=wait_list)

    def _num_2k_comm_r(self, wait_list, dt):
        return self.num_remesh_comm_r(
            self.part_position[0],
            self.fields_on_part[self.fields_on_grid[0]][0],
            self._cl_s_r_buff,
            HYSOP_INTEGER(self._s_buff_width_loc_p),
            self._cl_mesh_info,
            wait_for=wait_list)

    def _num_2k_comm(self, wait_list, dt):
        return self.num_remesh(
            self.part_position[0],
            self.fields_on_part[self.fields_on_grid[0]][0],
            self.fields_on_grid[0].gpu_data[0],
            self._cl_mesh_info,
            wait_for=wait_list)

    def _num_1k_ms_comm_l(self, wait_list, dt):
        return self.num_advec_and_remesh_comm_l(
            self.velocity.gpu_data[self.direction],
            self._cl_v_l_buff,
            self.fields_on_part[self.fields_on_grid[0]][0],
            self._cl_s_l_buff,
            HYSOP_INTEGER(self._s_buff_width_loc_m),
            self.gpu_precision(dt),
            self.gpu_precision(1. / self._v_mesh_size[1]),
            self.gpu_precision(1. / self._v_mesh_size[2]),
            self._cl_mesh_info,
            wait_for=wait_list)

    def _num_1k_ms_comm_r(self, wait_list, dt):
        return self.num_advec_and_remesh_comm_r(
            self.velocity.gpu_data[self.direction],
            self._cl_v_r_buff,
            self.fields_on_part[self.fields_on_grid[0]][0],
            self._cl_s_r_buff,
            HYSOP_INTEGER(self._s_buff_width_loc_p),
            self.gpu_precision(dt),
            self.gpu_precision(1. / self._v_mesh_size[1]),
            self.gpu_precision(1. / self._v_mesh_size[2]),
            self._cl_mesh_info,
            wait_for=wait_list)

    def _num_1k_ms_comm(self, wait_list, dt):
        return self.num_advec_and_remesh(
            self.velocity.gpu_data[self.direction],
            self._cl_v_l_buff,
            self._cl_v_r_buff,
            self.fields_on_part[self.fields_on_grid[0]][0],
            self.fields_on_grid[0].gpu_data[0],
            self.gpu_precision(dt),
            self.gpu_precision(1. / self._v_mesh_size[1]),
            self.gpu_precision(1. / self._v_mesh_size[2]),
            self._cl_mesh_info,
            wait_for=wait_list)

    def _num_1k_comm_l(self, wait_list, dt):
        return self.num_advec_and_remesh_comm_l(
            self.velocity.gpu_data[self.direction],
            self._cl_v_l_buff,
            self.fields_on_part[self.fields_on_grid[0]][0],
            self._cl_s_l_buff,
            HYSOP_INTEGER(self._s_buff_width_loc_m),
            self.gpu_precision(dt),
            self._cl_mesh_info,
            wait_for=wait_list)

    def _num_1k_comm_r(self, wait_list, dt):
        return self.num_advec_and_remesh_comm_r(
            self.velocity.gpu_data[self.direction],
            self._cl_v_r_buff,
            self.fields_on_part[self.fields_on_grid[0]][0],
            self._cl_s_r_buff,
            HYSOP_INTEGER(self._s_buff_width_loc_p),
            self.gpu_precision(dt),
            self._cl_mesh_info,
            wait_for=wait_list)

    def _num_1k_comm(self, wait_list, dt):
        return self.num_advec_and_remesh(
            self.velocity.gpu_data[self.direction],
            self._cl_v_l_buff,
            self._cl_v_r_buff,
            self.fields_on_part[self.fields_on_grid[0]][0],
            self.fields_on_grid[0].gpu_data[0],
            self.gpu_precision(dt),
            self._cl_mesh_info,
            wait_for=wait_list)

    def _compute_1c_comm(self, simulation, dtCoeff, split_id, old_dir):
        dt = simulation.timeStep * dtCoeff
        if self._is2kernel:
            self._compute_advec_comm(simulation, dtCoeff, split_id, old_dir)
        else:
            self._todevice_velocity_buffers()
        wait_evts = self.velocity.events + \
            self._init_events[self.fields_on_grid[0]] + \
            self.fields_on_grid[0].events
        if not self._is2kernel:
            wait_evts += self._evt_l_v + self._evt_r_v

        # Prepare the MPI receptions
        for b in xrange(self._s_n_blocks_from_l):
            self._l_recv[b] = self._comm.Irecv(
                [self._s_froml_buff_flat[self._s_block_slice_from_l[b]],
                 self._s_elem_block_from_l, HYSOP_MPI_REAL],
                source=self._L_rk, tag=888 + self._L_rk + 19 * b)
        for b in xrange(self._s_n_blocks_from_r):
            self._r_recv[b] = self._comm.Irecv(
                [self._s_fromr_buff_flat[self._s_block_slice_from_r[b]],
                 self._s_elem_block_from_r, HYSOP_MPI_REAL],
                source=self._R_rk, tag=333 + self._R_rk + 17 * b)

        # Fill and get the left buffer
        evt_comm_l = self._num_comm_l(wait_evts, dt)
        s = int(self._s_buff_width_loc_m *
                self.resol_dir[1] * self.resol_dir[2])
        for b in xrange(self._s_n_blocks_to_l):
            self._evt_get_l[b] = cl.enqueue_copy(
                self._queue_comm_m,
                self._s_l_buff, self._cl_s_l_buff,
                host_origin=(b * self._s_block_size_to_l, 0, 0),
                host_pitches=(s * SIZEOF_HYSOP_REAL, 0),
                buffer_origin=(b * self._s_block_size_to_l, 0, 0),
                buffer_pitches=(s * SIZEOF_HYSOP_REAL, 0),
                region=(self._s_block_size_to_l, 1, 1),
                is_blocking=False,
                wait_for=[evt_comm_l])

        # Send the left buffer
        ctime = MPI.Wtime()
        for b in xrange(self._s_n_blocks_to_l):
            self._evt_get_l[b].wait()
            self._l_send[b] = self._comm.Issend(
                [self._s_l_buff[self._s_block_slice_to_l[b]],
                 self._s_elem_block_to_l, HYSOP_MPI_REAL],
                dest=self._L_rk, tag=333 + self._comm_rank + 17 * b)
        ctime_send_l = MPI.Wtime() - ctime

        # Fill and get the right buffer
        evt_comm_r = self._num_comm_r(wait_evts, dt)
        s = int(self._s_buff_width_loc_p *
                self.resol_dir[1] * self.resol_dir[2])
        for b in xrange(self._s_n_blocks_to_r):
            self._evt_get_r[b] = cl.enqueue_copy(
                self._queue_comm_p,
                self._s_r_buff, self._cl_s_r_buff,
                host_origin=(b * self._s_block_size_to_r, 0, 0),
                host_pitches=(s * SIZEOF_HYSOP_REAL, 0),
                buffer_origin=(b * self._s_block_size_to_r, 0, 0),
                buffer_pitches=(s * SIZEOF_HYSOP_REAL, 0),
                region=(self._s_block_size_to_r, 1, 1),
                is_blocking=False,
                wait_for=[evt_comm_r])
        # Send the right buffer
        ctime = MPI.Wtime()
        for b in xrange(self._s_n_blocks_to_r):
            self._evt_get_r[b].wait()
            self._r_send[b] = self._comm.Issend(
                [self._s_r_buff[self._s_block_slice_to_r[b]],
                 self._s_elem_block_to_r, HYSOP_MPI_REAL],
                dest=self._R_rk, tag=888 + self._comm_rank + 19 * b)
        ctime_send_r = MPI.Wtime() - ctime

        # remesh in-domain particles and get left-right layer
        evt = self._num_comm(wait_evts, dt)
        evt_get_locl = cl.enqueue_copy(
            self.cl_env.queue,
            self.fields_on_grid[0].host_data_pinned[0],
            self.fields_on_grid[0].gpu_data[0],
            host_origin=self._origin_locl,
            buffer_origin=self._origin_locl,
            buffer_pitches=self._pitches_dev,
            host_pitches=self._pitches_dev,
            region=self._s_buffer_region_on_l,
            is_blocking=False,
            wait_for=[evt])
        evt_get_locr = cl.enqueue_copy(
            self.cl_env.queue,
            self.fields_on_grid[0].host_data_pinned[0],
            self.fields_on_grid[0].gpu_data[0],
            host_origin=self._origin_locr,
            buffer_origin=self._origin_locr,
            buffer_pitches=self._pitches_dev,
            host_pitches=self._pitches_dev,
            region=self._s_buffer_region_on_r,
            is_blocking=False,
            wait_for=[evt])

        ctime = MPI.Wtime()
        # Wait MPI transfer of data from left, add them to local
        # data and send back to device
        for b in xrange(self._s_n_blocks_to_r):
            self._r_send[b].Wait()
        for b in xrange(self._s_n_blocks_from_l):
            self._l_recv[b].Wait()
        evt_get_locl.wait()
        ctime_wait_l = MPI.Wtime() - ctime

        calctime = MPI.Wtime()
        self._s_locl_buff += self._s_froml_buff
        self.profiler['comm_calc_remesh'] += MPI.Wtime() - calctime
        evt_set_locl = cl.enqueue_copy(
            self.cl_env.queue,
            self.fields_on_grid[0].gpu_data[0],
            self.fields_on_grid[0].host_data_pinned[0],
            host_origin=self._origin_locl,
            buffer_origin=self._origin_locl,
            buffer_pitches=self._pitches_dev,
            host_pitches=self._pitches_dev,
            region=self._s_buffer_region_on_l,
            is_blocking=False)

        # Wait MPI transfer of data from right, add them to local
        # data and send back to device
        ctime = MPI.Wtime()
        for b in xrange(self._s_n_blocks_to_l):
            self._l_send[b].Wait()
        for b in xrange(self._s_n_blocks_from_r):
            self._r_recv[b].Wait()
        evt_get_locr.wait()
        ctime_wait_r = MPI.Wtime() - ctime
        calctime = MPI.Wtime()
        self._s_locr_buff += self._s_fromr_buff
        self.profiler['comm_calc_remesh'] += MPI.Wtime() - calctime
        evt_set_locr = cl.enqueue_copy(
            self.cl_env.queue,
            self.fields_on_grid[0].gpu_data[0],
            self.fields_on_grid[0].host_data_pinned[0],
            host_origin=self._origin_locr,
            buffer_origin=self._origin_locr,
            buffer_pitches=self._pitches_dev,
            host_pitches=self._pitches_dev,
            region=self._s_buffer_region_on_r,
            is_blocking=False)

        if CL_PROFILE:
            evt_set_locl.wait()
            evt_set_locr.wait()

        self.fields_on_grid[0].events.append(evt_set_locr)
        self.fields_on_grid[0].events.append(evt_set_locl)
        self.profiler['comm_cpu_remesh'] += ctime_wait_r + ctime_wait_l + \
            ctime_send_r + ctime_send_l

        if CL_PROFILE:
            rmsh_gpu_time = 0.
            for evt in self._evt_get_l + self._evt_get_r:
                evt.wait()
                rmsh_gpu_time += (evt.profile.end - evt.profile.start) * 1e-9
            self.profiler['comm_gpu_remesh_get'] += rmsh_gpu_time
            rmsh_gpu_time = 0.
            for evt in [evt_get_locr, evt_get_locl]:
                evt.wait()
                rmsh_gpu_time += (evt.profile.end - evt.profile.start) * 1e-9
            self.profiler['comm_gpu_remesh_get_loc'] += rmsh_gpu_time
            rmsh_gpu_time = 0.
            for evt in [evt_set_locl, evt_set_locr]:
                evt.wait()
                rmsh_gpu_time += (evt.profile.end - evt.profile.start) * 1e-9
            self.profiler['comm_gpu_remesh_set_loc'] += rmsh_gpu_time

    @debug
    def finalize(self):
        """
        Cleaning, if required.
        """
        super(MultiGPUParticleAdvection, self).finalize()
        self._s_l_buff.base.release(self.cl_env.queue)
        self._s_r_buff.base.release(self.cl_env.queue)
        self._v_r_buff.base.release(self.cl_env.queue)
        self._v_l_buff.base.release(self.cl_env.queue)

    def get_profiling_info(self):
        super(MultiGPUParticleAdvection, self).get_profiling_info()
        if self._is2kernel:
            for k in (self.num_remesh_comm_l,
                      self.num_remesh_comm_r):
                for p in k.profile:
                    self.profiler += p
        else:
            for k in (self.num_advec_and_remesh_comm_l,
                      self.num_advec_and_remesh_comm_r):
                for p in k.profile:
                    self.profiler += p
