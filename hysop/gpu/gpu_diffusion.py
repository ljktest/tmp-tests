"""
@file gpu_diffusion.py

Diffusion on GPU
"""
from hysop.constants import debug, np, S_DIR, HYSOP_MPI_REAL, ORDERMPI, \
    HYSOP_REAL, ORDER
import hysop.tools.numpywrappers as npw
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.operator.discrete.discrete import get_extra_args_from_method
from hysop.gpu import cl
from hysop.gpu.gpu_operator import GPUOperator
from hysop.gpu.gpu_kernel import KernelLauncher
from hysop.gpu.gpu_discrete import GPUDiscreteField
from hysop.tools.profiler import FProfiler
from hysop.mpi.main_var import MPI


class GPUDiffusion(DiscreteOperator, GPUOperator):

    @debug
    def __init__(self, field, viscosity, **kwds):
        super(GPUDiffusion, self).__init__(variables=[field], **kwds)
        ## Discretisation of the solution field
        self.field = self.variables[0]
        ## Viscosity.
        self.viscosity = viscosity
        self.input = [self.field]
        self.output = [self.field]
        self.direction = 0
        self._cl_work_size = 0

        GPUOperator.__init__(
            self,
            platform_id=get_extra_args_from_method(self, 'platform_id', None),
            device_id=get_extra_args_from_method(self, 'device_id', None),
            device_type=get_extra_args_from_method(self, 'device_type', None),
            **kwds)

        ## GPU allocation.
        alloc = not isinstance(self.field, GPUDiscreteField)
        GPUDiscreteField.fromField(self.cl_env, self.field,
                                   self.gpu_precision, simple_layout=False)
        if not self.field.gpu_allocated:
            self.field.allocate()
        if alloc:
            self.size_global_alloc += self.field.mem_size

        self.field_tmp = get_extra_args_from_method(self, 'field_tmp', None),

        topo = self.field.topology
        self._cutdir_list = np.where(topo.cutdir)[0].tolist()
        self._comm = topo.comm
        self._comm_size = self._comm.Get_size()
        self._comm_rank = self._comm.Get_rank()
        if self._comm_size > 1:
            self._to_send = [None] * self.dim
            self._to_recv_buf = [None] * self.dim
            self._to_recv = [None] * self.dim
            self._pitches_host = [None] * self.dim
            self._pitches_buff = [None] * self.dim
            self._region_size = [None] * self.dim
            self._r_orig = [None] * self.dim
            self._br_orig = [None] * self.dim
            self.mpi_type_diff_l = {}
            self.mpi_type_diff_r = {}
            self.profiler += FProfiler('comm_diffusion')
            for d in self._cutdir_list:
                shape = list(self.field.data[0].shape)
                shape_b = list(self.field.data[0].shape)
                start_l = [0, ] * 3
                start_r = [0, ] * 3
                start_r[d] = 1
                shape[d] = 2
                shape_b[d] = 1
                # _to_send[..., 0] contains [..., 0] data
                # _to_send[..., 1] contains [..., Nz-1] data
                # _to_recv[..., 0] contains [..., Nz] data (right ghosts)
                # _to_recv[..., 1] contains [..., -1] data (left ghosts)
                self._to_send[d] = npw.zeros(tuple(shape))
                _to_recv = npw.zeros(tuple(shape))
                self.mpi_type_diff_l[d] = HYSOP_MPI_REAL.Create_subarray(
                    shape, shape_b, start_l, order=ORDERMPI)
                self.mpi_type_diff_l[d].Commit()
                self.mpi_type_diff_r[d] = HYSOP_MPI_REAL.Create_subarray(
                    shape, shape_b, start_r, order=ORDERMPI)
                self.mpi_type_diff_r[d].Commit()
                self._to_recv_buf[d] = self.cl_env.global_allocation(_to_recv)
                self._to_recv[d], evt = cl.enqueue_map_buffer(
                    self.cl_env.queue,
                    self._to_recv_buf[d],
                    offset=0,
                    shape=shape,
                    dtype=HYSOP_REAL,
                    flags=cl.map_flags.READ | cl.map_flags.WRITE,
                    is_blocking=False,
                    order=ORDER)
                evt.wait()
                self._pitches_host[d] = (int(self._to_send[d][:, 0, 0].nbytes),
                                         int(self._to_send[d][:, :, 0].nbytes))
                self._pitches_buff[d] = (int(self.field.data[0][:, 0, 0].nbytes),
                                         int(self.field.data[0][:, :, 0].nbytes))
                cl.enqueue_copy(
                    self.cl_env.queue,
                    self._to_recv_buf[d],
                    self._to_recv[d],
                    buffer_origin=(0, 0, 0),
                    host_origin=(0, 0, 0),
                    region=(self._to_recv[d][0, 0, 0].nbytes, )).wait()
                self._cl_work_size += self._to_recv[d].nbytes

                r_orig = [0, ] * self.dim
                br_orig = [0, ] * self.dim
                r_orig[d] = self.field.data[0].shape[d] - 1
                br_orig[d] = 1
                if d == 0:
                    r_orig[d] *= self._to_send[d][0, 0, 0].nbytes
                    br_orig[d] *= self._to_send[d][0, 0, 0].nbytes
                self._r_orig[d] = tuple(r_orig)
                self._br_orig[d] = tuple(br_orig)
                l_sl = [slice(None), ] * 3
                r_sl = [slice(None), ] * 3
                l_sl[d] = slice(0, 1)
                r_sl[d] = slice(1, 2)
                l_sl = tuple(l_sl)
                r_sl = tuple(r_sl)
                self._region_size[d] = list(self.field.data[0].shape)
                if d == 0:
                    self._region_size[d][0] = self._to_send[d][0, 0, 0].nbytes
                else:
                    self._region_size[d][0] = self._to_send[d][:, 0, 0].nbytes
                    self._region_size[d][d] = 1

            self._compute = self._compute_diffusion_comm
        else:
            self._compute = self._compute_diffusion

        self._mesh_size = npw.ones(4, dtype=self.gpu_precision)
        self._mesh_size[:self.dim] = self._reorderVect(topo.mesh.space_step)
        shape = topo.mesh.resolution
        resol = shape.copy()
        self.resol_dir = npw.dim_ones((self.dim,))
        self.resol_dir[:self.dim] = self._reorderVect(shape)
        self._append_size_constants(resol)

        src, tile_size, nb_part_per_wi, vec, f_space = \
            self._kernel_cfg['diffusion']

        build_options = self._size_constants
        build_options += " -D TILE_SIZE=" + str(tile_size)
        build_options += " -D NB_PART=" + str(nb_part_per_wi)
        build_options += " -D L_WIDTH=" + str(tile_size / nb_part_per_wi)
        for d in xrange(self.dim):
            build_options += " -D CUT_DIR" + S_DIR[d] + "="
            build_options += str(1 if topo.shape[d] > 1 else 0)

        gwi, lwi, blocs_nb = f_space(self.field.data[0].shape,
                                     nb_part_per_wi, tile_size)
        build_options += " -D NB_GROUPS_I={0}".format(blocs_nb[0])
        build_options += " -D NB_GROUPS_II={0}".format(blocs_nb[1])
        prg = self.cl_env.build_src(src, build_options, vec)
        self.num_diffusion = KernelLauncher(
            prg.diffusion, self.cl_env.queue, gwi, lwi)
        self.copy = KernelLauncher(cl.enqueue_copy,
                                   self.cl_env.queue)

    def _compute_diffusion(self, simulation):
        assert self.field_tmp is not None
        wait_evt = self.field.events
        d_evt = self.num_diffusion(
            self.field.gpu_data[0],
            self.field_tmp,
            self.gpu_precision(self.viscosity * simulation.timeStep),
            self._mesh_size,
            wait_for=wait_evt)
        c_evt = self.copy.launch_sizes_in_args(
            self.field.gpu_data[0], self.field_tmp, wait_for=[d_evt])
        #c_evt = cl.enqueue_copy(self.cl_env.queue, self.field.gpu_data[0],
        #                        self.field_tmp, wait_for=[d_evt])
        self.field.events.append(c_evt)

    def set_field_tmp(self, field_tmp):
        self.field_tmp = field_tmp

    def _compute_diffusion_comm(self, simulation):
        assert self.field_tmp is not None
        # Compute OpenCL transfer parameters
        tc = MPI.Wtime()
        topo = self.field.topology
        first_cut_dir = topo.cutdir.tolist().index(True)
        wait_evt = []
        send_l = [None, ] * self.dim
        send_r = [None, ] * self.dim
        recv_l = [None, ] * self.dim
        recv_r = [None, ] * self.dim
        e_l = [None, ] * self.dim
        e_r = [None, ] * self.dim
        for d in self._cutdir_list:
            wait_events = self.field.events
            e_l[d] = cl.enqueue_copy(self.cl_env.queue, self._to_send[d],
                                     self.field.gpu_data[0],
                                     host_origin=(0, 0, 0),
                                     buffer_origin=(0, 0, 0),
                                     host_pitches=self._pitches_host[d],
                                     buffer_pitches=self._pitches_buff[d],
                                     region=tuple(self._region_size[d]),
                                     wait_for=wait_events,
                                     is_blocking=False)
            e_r[d] = cl.enqueue_copy(self.cl_env.queue, self._to_send[d],
                                     self.field.gpu_data[0],
                                     host_origin=self._br_orig[d],
                                     buffer_origin=self._r_orig[d],
                                     host_pitches=self._pitches_host[d],
                                     buffer_pitches=self._pitches_buff[d],
                                     region=tuple(self._region_size[d]),
                                     wait_for=wait_events,
                                     is_blocking=False)

        for d in self._cutdir_list:
            # MPI send
            R_rk = topo.neighbours[1, d - first_cut_dir]
            L_rk = topo.neighbours[0, d - first_cut_dir]
            recv_r[d] = self._comm.Irecv(
                [self._to_recv[d], 1, self.mpi_type_diff_l[d]],
                source=R_rk, tag=123 + R_rk + 19 * d)
            recv_l[d] = self._comm.Irecv(
                [self._to_recv[d], 1, self.mpi_type_diff_r[d]],
                source=L_rk, tag=456 + L_rk + 17 * d)
        for d in self._cutdir_list:
            R_rk = topo.neighbours[1, d - first_cut_dir]
            L_rk = topo.neighbours[0, d - first_cut_dir]
            e_l[d].wait()
            e_r[d].wait()
            send_l[d] = self._comm.Issend(
                [self._to_send[d], 1, self.mpi_type_diff_l[d]],
                dest=L_rk, tag=123 + self._comm_rank + 19 * d)
            send_r[d] = self._comm.Issend(
                [self._to_send[d], 1, self.mpi_type_diff_r[d]],
                dest=R_rk, tag=456 + self._comm_rank + 17 * d)

        for d in self._cutdir_list:
            # _to_recv[..., 0] contains [..., Nz] data (right ghosts)
            # _to_recv[..., 1] contains [..., -1] data (left ghosts)
            send_r[d].Wait()
            send_l[d].Wait()
            recv_r[d].Wait()
            recv_l[d].Wait()
            wait_evt.append(cl.enqueue_copy(self.cl_env.queue,
                                            self._to_recv_buf[d],
                                            self._to_recv[d],
                                            is_blocking=False))
        self.profiler['comm_diffusion'] += MPI.Wtime() - tc

        if len(self._cutdir_list) == 1:
            d_evt = self.num_diffusion(
                self.field.gpu_data[0],
                self._to_recv_buf[self._cutdir_list[0]],
                self.field_tmp,
                self.gpu_precision(self.viscosity * simulation.timeStep),
                self._mesh_size,
                wait_for=wait_evt)
        if len(self._cutdir_list) == 2:
            d_evt = self.num_diffusion(
                self.field.gpu_data[0],
                self._to_recv_buf[self._cutdir_list[0]],
                self._to_recv_buf[self._cutdir_list[1]],
                self.field_tmp,
                self.gpu_precision(self.viscosity * simulation.timeStep),
                self._mesh_size,
                wait_for=wait_evt)
        if len(self._cutdir_list) == 3:
            d_evt = self.num_diffusion(
                self.field.gpu_data[0],
                self._to_recv_buf[self._cutdir_list[0]],
                self._to_recv_buf[self._cutdir_list[1]],
                self._to_recv_buf[self._cutdir_list[2]],
                self.field_tmp,
                self.gpu_precision(self.viscosity * simulation.timeStep),
                self._mesh_size,
                wait_for=wait_evt)
        #c_evt = cl.enqueue_copy(self.cl_env.queue, self.field.gpu_data[0],
        #                        self.field_tmp, wait_for=[d_evt])
        c_evt = self.copy.launch_sizes_in_args(
            self.field.gpu_data[0], self.field_tmp, wait_for=[d_evt])
        self.field.events.append(c_evt)

    def apply(self, simulation):
        self._compute(simulation)

    def get_profiling_info(self):
        for k in [self.num_diffusion, self.copy]:
            if k is not None:
                for p in k.profile:
                    self.profiler += p
