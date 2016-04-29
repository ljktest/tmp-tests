"""
@file gpu/multiresolution_filter.py
Filter values from a fine grid to a coarse grid.
GPU version.
"""
from hysop.constants import debug, np
import hysop.tools.numpywrappers as npw
from hysop.operator.discrete.multiresolution_filter import FilterFineToCoarse
from hysop.gpu.gpu_operator import GPUOperator
from hysop.operator.discrete.discrete import get_extra_args_from_method
from hysop.gpu.gpu_discrete import GPUDiscreteField
from hysop.gpu.gpu_kernel import KernelLauncher
from hysop.methods_keys import Remesh


class GPUFilterFineToCoarse(FilterFineToCoarse, GPUOperator):
    """
    Discretized operator for filtering from fine to coarse grid on GPU.
    """
    @debug
    def __init__(self, field_in, field_out, **kwds):
        super(GPUFilterFineToCoarse, self).__init__(
            field_in, field_out, **kwds)
        assert len(self.field_in) == 1 and len(self.field_out) == 1, \
            "This operator is implemented only for single field"
        self.direction = 0
        self._cl_work_size = 0
        gh_in = self.field_in[0].topology.ghosts()
        resol_in = self._mesh_in.resolution - 2 * gh_in
        resol_out = self._mesh_out.resolution - 2 * self.gh_out
        pts_per_cell = resol_in / resol_out

        GPUOperator.__init__(
            self,
            platform_id=get_extra_args_from_method(self, 'platform_id', None),
            device_id=get_extra_args_from_method(self, 'device_id', None),
            device_type=get_extra_args_from_method(self, 'device_type', None),
            **kwds)

        #GPU allocations
        alloc = not isinstance(self.field_in[0], GPUDiscreteField)
        GPUDiscreteField.fromField(self.cl_env, self.field_in[0],
                                   self.gpu_precision, layout=False)
        if not self.field_in[0].gpu_allocated:
            self.field_in[0].allocate()
        if alloc:
            self.size_global_alloc += self.field_in[0].mem_size

        alloc = not isinstance(self.field_out[0], GPUDiscreteField)
        GPUDiscreteField.fromField(self.cl_env, self.field_out[0],
                                   self.gpu_precision, layout=False)
        if not self.field_out[0].gpu_allocated:
            self.field_out[0].allocate()
        if alloc:
            self.size_global_alloc += self.field_out[0].mem_size

        topo_in = self.field_in[0].topology
        topo_out = self.field_out[0].topology

        self._mesh_size_in = npw.ones(4, dtype=self.gpu_precision)
        self._mesh_size_in[:self.dim] = \
            self._reorderVect(topo_in.mesh.space_step)
        self._mesh_size_out = npw.ones(4, dtype=self.gpu_precision)
        self._mesh_size_out[:self.dim] = \
            self._reorderVect(topo_out.mesh.space_step)
        self._domain_origin = npw.ones(4, dtype=self.gpu_precision)
        self._domain_origin[:self.dim] = self.field_in[0].domain.origin
        shape_in = topo_in.mesh.resolution
        shape_out = topo_out.mesh.resolution
        resol_in = shape_in.copy()
        resol_out = shape_out.copy()
        self.resol_in = npw.dim_ones((self.dim,))
        self.resol_in[:self.dim] = self._reorderVect(shape_in)
        self.resol_out = npw.dim_ones((self.dim,))
        self.resol_out[:self.dim] = self._reorderVect(shape_out)
        self._append_size_constants(resol_in, prefix='NB_IN')
        self._append_size_constants(resol_out, prefix='NB_OUT')
        self._append_size_constants(self.gh_out, prefix='GHOSTS_OUT')
        self._append_size_constants(pts_per_cell, prefix='PTS_PER_CELL')

        # multi-gpu ghosts buffers for communication
        if self._comm_size == 1:
            self._exchange_ghosts = self._exchange_ghosts_local
        else:
            self._exchange_ghosts = self._exchange_ghosts_mpi
            self._gh_from_l = [None] * self.dim
            self._gh_from_r = [None] * self.dim
            self._gh_to_l = [None] * self.dim
            self._gh_to_r = [None] * self.dim
            # self._mpi_to_l = [None] * self.dim
            # self._mpi_to_r = [None] * self.dim
            for d in self._cutdir_list:
                shape = list(self.field_out[0].data[0].shape)
                shape[d] = self.gh_out[d]
                self._gh_from_l[d] = npw.zeros(tuple(shape))
                self._gh_from_r[d] = npw.zeros(tuple(shape))
                self._gh_to_l[d] = npw.zeros(tuple(shape))
                self._gh_to_r[d] = npw.zeros(tuple(shape))

        # # Ghosts temp arrays for the second version of ghosts exchange
        # self.gh_x = npw.zeros((4 * self.gh_out[0], shape_out[1], shape_out[2]))
        # self.gh_y = npw.zeros((shape_out[0], 4 * self.gh_out[1], shape_out[2]))
        # self.gh_z = npw.zeros((shape_out[0], shape_out[1], 4 * self.gh_out[2]))
        # print self.gh_x.shape, self.gh_y.shape, self.gh_z.shape
        # self._pitches_host_x = (int(self.gh_x[:, 0, 0].nbytes),
        #                         int(self.gh_x[:, :, 0].nbytes))
        # self._pitches_host_y = (int(self.gh_y[:, 0, 0].nbytes),
        #                         int(self.gh_y[:, :, 0].nbytes))
        # self._pitches_host_z = (int(self.gh_z[:, 0, 0].nbytes),
        #                         int(self.gh_z[:, :, 0].nbytes))
        # self._pitches_buff = (int(self.field_out[0].data[0][:, 0, 0].nbytes),
        #                       int(self.field_out[0].data[0][:, :, 0].nbytes))

        src, vec, f_space = \
            self._kernel_cfg['fine_to_coarse_filter']
        build_options = self._size_constants
        self._rmsh = self.method[Remesh]()
        gwi, lwi = f_space(self.field_out[0].data[0].shape -
                           2 * topo_out.ghosts(), len(self._rmsh.weights))
        build_options += " -D L_STENCIL=" + str(len(self._rmsh.weights))
        build_options += " -D SHIFT_STENCIL=" + str(self._rmsh.shift)
        build_options += " -D WG=" + str(lwi[0])
        build_options += " -D FORMULA=" + self.method[Remesh].__name__.upper()
        prg = self.cl_env.build_src(src, build_options, vec)
        self.fine_to_coarse = KernelLauncher(
            prg.coarse_to_fine_filter, self.cl_env.queue, gwi, lwi)
        self.initialize = KernelLauncher(
            prg.initialize_output, self.cl_env.queue,
            self.field_out[0].data[0].shape, None)
        self._evts = [None, ] * self.field_in[0].dimension

    def apply(self, simulation=None):
        #evts = []
        self.field_in[0].toHost()
        self.field_in[0].wait()
        for d in xrange(self.field_in[0].nb_components):
            self._evts[d] = []
            self._evts[d].append(
                self.initialize(self.field_out[0].gpu_data[d],
                                wait_for=self.field_out[0].events))
        for iy in xrange(len(self._rmsh.weights)):
            for iz in xrange(len(self._rmsh.weights)):
                for d in xrange(self.field_in[0].nb_components):
                    evt = self.fine_to_coarse(self.field_in[0].gpu_data[d],
                                              self.field_out[0].gpu_data[d],
                                          self.scale_factor,
                                          self._mesh_size_in,
                                          self._mesh_size_out,
                                          self._domain_origin,
                                          np.int32(iy), np.int32(iz),
                                              wait_for=self._evts[d])
                self._evts[d].append(evt)
        # Ghosts values must be exchanged either on process or through mpi
        # communications. Values must be moved to host.
        # We developp 2 versions:
        #  - copy of the entire field data
        #  - rect-copy of only needed data
        # The first one is running much faster than the second because of
        # the use of the mapping of device buffer in host pinned memory.
        # The second version is kept in comments (for sequential case)
        self.field_out[0].toHost()
        self.field_out[0].wait()
        self._exchange_ghosts()
        self.field_out[0].toDevice()

        # # Get ghosts values and in-domain layer
        # # X-direction
        # s_gh = self.gh_out[0]
        # get_gh_xl = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.gh_x, self.field_out[0].gpu_data[0],
        #     host_origin=(0, 0, 0),
        #     buffer_origin=(0, 0, 0),
        #     host_pitches=self._pitches_host_x,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_x[:2 * s_gh, 0, 0].nbytes,
        #             self.gh_x.shape[1],
        #             self.gh_x.shape[2]),
        #     wait_for=evts)
        # get_gh_xr = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.gh_x, self.field_out[0].gpu_data[0],
        #     host_origin=(self.gh_x[:2 * s_gh, 0, 0].nbytes, 0, 0),
        #     buffer_origin=(self.field_out[0].data[0][:, 0, 0].nbytes -
        #                    self.gh_x[:2 * s_gh, 0, 0].nbytes, 0, 0),
        #     host_pitches=self._pitches_host_x,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_x[:2 * s_gh, 0, 0].nbytes,
        #             self.gh_x.shape[1],
        #             self.gh_x.shape[2]),
        #     wait_for=evts)
        # get_gh_xl.wait()
        # get_gh_xr.wait()
        # # Add ghosts contributions in domain layer
        # self.gh_x[2 * s_gh:3 * s_gh, :, :] += \
        #     self.gh_x[0 * s_gh:1 * s_gh, :, :]
        # self.gh_x[1 * s_gh:2 * s_gh, :, :] += \
        #     self.gh_x[3 * s_gh:4 * s_gh, :, :]
        # set_gh_xl = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.field_out[0].gpu_data[0], self.gh_x,
        #     host_origin=(self.gh_x[:1 * s_gh, 0, 0].nbytes, 0, 0),
        #     buffer_origin=(self.gh_x[:1 * s_gh, 0, 0].nbytes, 0, 0),
        #     host_pitches=self._pitches_host_x,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_x[:1 * s_gh, 0, 0].nbytes,
        #             self.gh_x.shape[1],
        #             self.gh_x.shape[2]),
        #     wait_for=evts)
        # set_gh_xr = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.field_out[0].gpu_data[0], self.gh_x,
        #     host_origin=(self.gh_x[:2 * s_gh, 0, 0].nbytes, 0, 0),
        #     buffer_origin=(self.field_out[0].data[0][:, 0, 0].nbytes -
        #                    self.gh_x[:2 * s_gh, 0, 0].nbytes, 0, 0),
        #     host_pitches=self._pitches_host_x,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_x[:1 * s_gh, 0, 0].nbytes,
        #             self.gh_x.shape[1],
        #             self.gh_x.shape[2]),
        #     wait_for=evts)
        # set_gh_xl.wait()
        # set_gh_xr.wait()

        # # Y-direction
        # s_gh = self.gh_out[1]
        # get_gh_yl = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.gh_y, self.field_out[0].gpu_data[0],
        #     host_origin=(0, 0, 0),
        #     buffer_origin=(0, 0, 0),
        #     host_pitches=self._pitches_host_y,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_y[:, 0, 0].nbytes, 2 * s_gh, self.gh_y.shape[2]),
        #     wait_for=evts)
        # get_gh_yr = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.gh_y, self.field_out[0].gpu_data[0],
        #     host_origin=(0, 2 * s_gh, 0),
        #     buffer_origin=(0, self.field_out[0].data[0].shape[1] - 2 * s_gh, 0),
        #     host_pitches=self._pitches_host_y,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_y[:, 0, 0].nbytes, 2 * s_gh, self.gh_y.shape[2]),
        #     wait_for=evts)
        # get_gh_yl.wait()
        # get_gh_yr.wait()
        # # Add ghosts contributions in domain layer
        # self.gh_y[:, 2 * s_gh:3 * s_gh, :] += \
        #     self.gh_y[:, 0 * s_gh:1 * s_gh, :]
        # self.gh_y[:, 1 * s_gh:2 * s_gh, :] += \
        #     self.gh_y[:, 3 * s_gh:4 * s_gh, :]
        # set_gh_yl = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.field_out[0].gpu_data[0], self.gh_y,
        #     host_origin=(0, 1 * s_gh, 0),
        #     buffer_origin=(0, 1 * s_gh, 0),
        #     host_pitches=self._pitches_host_y,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_y[:, 0, 0].nbytes, 1 * s_gh, self.gh_y.shape[2]),
        #     wait_for=evts)
        # set_gh_yr = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.field_out[0].gpu_data[0], self.gh_y,
        #     host_origin=(0, 2 * s_gh, 0),
        #     buffer_origin=(0, self.field_out[0].data[0].shape[1] - 2 * s_gh, 0),
        #     host_pitches=self._pitches_host_y,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_y[:, 0, 0].nbytes, 1 * s_gh, self.gh_y.shape[2]),
        #     wait_for=evts)
        # set_gh_yl.wait()
        # set_gh_yr.wait()

        # # Z-direction
        # s_gh = self.gh_out[2]
        # get_gh_zl = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.gh_z, self.field_out[0].gpu_data[0],
        #     host_origin=(0, 0, 0),
        #     buffer_origin=(0, 0, 0),
        #     host_pitches=self._pitches_host_z,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_z[:, 0, 0].nbytes, self.gh_z.shape[1], 2 * s_gh),
        #     wait_for=evts)
        # get_gh_zr = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.gh_z, self.field_out[0].gpu_data[0],
        #     host_origin=(0, 0, 2 * s_gh),
        #     buffer_origin=(0, 0, self.field_out[0].data[0].shape[2] - 2 * s_gh),
        #     host_pitches=self._pitches_host_z,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_z[:, 0, 0].nbytes, self.gh_z.shape[1], 2 * s_gh),
        #     wait_for=evts)
        # get_gh_zl.wait()
        # get_gh_zr.wait()
        # # Add ghosts contributions in domain layer
        # self.gh_z[:, :, 2 * s_gh:3 * s_gh] += \
        #     self.gh_z[:, :, 0 * s_gh:1 * s_gh]
        # self.gh_z[:, :, 1 * s_gh:2 * s_gh] += \
        #     self.gh_z[:, :, 3 * s_gh:4 * s_gh]
        # set_gh_zl = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.field_out[0].gpu_data[0], self.gh_z,
        #     host_origin=(0, 0, 1 * s_gh),
        #     buffer_origin=(0, 0, 1 * s_gh),
        #     host_pitches=self._pitches_host_z,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_z[:, 0, 0].nbytes, self.gh_z.shape[1], 1 * s_gh),
        #     wait_for=evts)
        # set_gh_zr = cl.enqueue_copy(
        #     self.cl_env.queue,
        #     self.field_out[0].gpu_data[0], self.gh_z,
        #     host_origin=(0, 0, 2 * s_gh),
        #     buffer_origin=(0, 0, self.field_out[0].data[0].shape[2] - 2 * s_gh),
        #     host_pitches=self._pitches_host_z,
        #     buffer_pitches=self._pitches_buff,
        #     region=(self.gh_z[:, 0, 0].nbytes, self.gh_z.shape[1], 1 * s_gh),
        #     wait_for=evts)
        # set_gh_zl.wait()
        # set_gh_zr.wait()

    def get_profiling_info(self):
        for k in (self.fine_to_coarse, self.initialize):
            for p in k.profile:
                self.profiler += p
