"""
@file multiphase_baroclinic_rhs.py

Multiscale baroclinic term on GPU
"""
from hysop.constants import debug, np, S_DIR, HYSOP_MPI_REAL, ORDERMPI, \
    HYSOP_REAL, ORDER
import hysop.tools.numpywrappers as npw
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.operator.discrete.discrete import get_extra_args_from_method
from hysop.gpu import cl
from hysop.gpu.gpu_operator import GPUOperator
from hysop.gpu.gpu_kernel import KernelListLauncher
from hysop.gpu.gpu_discrete import GPUDiscreteField
from hysop.tools.profiler import FProfiler
from hysop.mpi.main_var import MPI
from hysop.methods_keys import SpaceDiscretisation
from hysop.numerics.finite_differences import FD_C_4


class BaroclinicRHS(DiscreteOperator, GPUOperator):
    """
    This operator computes the right hand side of the baroclinic term:
    \f{eqnarray*}
    \frac{\partial \omega}{\partial t} &=& -\frac{\nabla \rho}{\rho} \times \left(-\frac{\nabla P}{\rho}\right)
    \f}
    This operator works only in a multiscale context and from an input variable containing the
    \f{eqnarray*}
    \left(-\frac{\nabla P}{\rho}\right)
    \f} term. This term needs to be interpolated at fine scale. The implementation
    of such an interpolation is shared memory consuming since each component is to be interpolated.
    We introduce a splitting of the whole term from the different compontents of the pressure gradient term:
    \f{eqnarray*}
    -A\times B =
  \begin{pmatrix}
    0 \\
    -A_Z B_X\\
    A_Y B_X
  \end{pmatrix}+
  \begin{pmatrix}
     A_Z B_Y \\
    0 \\
    -A_X B_Y
  \end{pmatrix}+
  \begin{pmatrix}
    -A_Y B_Z  \\
     A_X B_Z \\
    0
  \end{pmatrix}
    \f}
    Finally, the density field may came from a levelset function.
    By default we assume that the density itself is given in input.
    In some cases, user can give a custom function to compute the
    density from the levelset points by points.
    """
    @debug
    def __init__(self, rhs, rho, gradp, **kwds):
        super(BaroclinicRHS, self).__init__(
            variables=[rhs, rho, gradp], **kwds)
        self.rhs = rhs
        self.rho = rho
        self.gradp = gradp
        self.input = [self.rho, self.gradp]
        self.output = [self.rhs, ]
        self.direction = 0
        self._cl_work_size = 0

        GPUOperator.__init__(
            self,
            platform_id=get_extra_args_from_method(self, 'platform_id', None),
            device_id=get_extra_args_from_method(self, 'device_id', None),
            device_type=get_extra_args_from_method(self, 'device_type', None),
            **kwds)

        # GPU allocation.
        for field in self.variables:
            alloc = not isinstance(field, GPUDiscreteField)
            GPUDiscreteField.fromField(self.cl_env, field,
                                       self.gpu_precision, layout=False)
            if not field.gpu_allocated:
                field.allocate()
            if alloc:
                self.size_global_alloc += field.mem_size

        topo_coarse = self.gradp.topology
        topo_fine = self.rho.topology
        self._cutdir_list = np.where(topo_coarse.cutdir)[0].tolist()
        self._comm = topo_coarse.comm
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
            self.profiler += FProfiler('comm_baroclinic_rhs')
            if 0 in self._cutdir_list:
                raise ValueError("Not yet implemented with comm in X dir")
            for d in self._cutdir_list:
                if self.method[SpaceDiscretisation] == FD_C_4:
                    gh = 2
                else:
                    gh = 1
                shape = list(self.rho.data[0].shape)
                shape_b = list(self.rho.data[0].shape)
                start_l = [0, ] * 3
                start_r = [0, ] * 3
                start_r[d] = gh
                shape[d] = 2 * gh
                shape_b[d] = gh
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
                self._pitches_buff[d] = (int(self.rho.data[0][:, 0, 0].nbytes),
                                         int(self.rho.data[0][:, :, 0].nbytes))
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
                r_orig[d] = self.rho.data[0].shape[d] - gh
                br_orig[d] = gh
                if d == 0:
                    r_orig[d] *= self._to_send[d][0, 0, 0].nbytes
                    br_orig[d] *= self._to_send[d][0, 0, 0].nbytes
                self._r_orig[d] = tuple(r_orig)
                self._br_orig[d] = tuple(br_orig)
                l_sl = [slice(None), ] * 3
                r_sl = [slice(None), ] * 3
                l_sl[d] = slice(0, gh)
                r_sl[d] = slice(gh, 2 * gh)
                l_sl = tuple(l_sl)
                r_sl = tuple(r_sl)
                self._region_size[d] = list(self.rho.data[0].shape)
                if d == 0:
                    self._region_size[d][0] = self._to_send[d][0, 0, 0].nbytes
                else:
                    self._region_size[d][0] = self._to_send[d][:, 0, 0].nbytes
                    self._region_size[d][d] = gh
            self._compute = self._compute_baroclinic_rhs_comm
            if len(self._cutdir_list) == 1:
                self._call_kernel = self._call_kernel_one_ghost
            if len(self._cutdir_list) == 2:
                self._call_kernel = self._call_kernel_two_ghost
        else:
            self._compute = self._compute_baroclinic_rhs

        self._coarse_mesh_size = npw.ones(4, dtype=self.gpu_precision)
        self._fine_mesh_size = npw.ones(4, dtype=self.gpu_precision)
        self._coarse_mesh_size[:self.dim] = \
            self._reorderVect(topo_coarse.mesh.space_step)
        self._fine_mesh_size[:self.dim] = \
            self._reorderVect(topo_fine.mesh.space_step)

        shape_coarse = topo_coarse.mesh.resolution
        resol_coarse = shape_coarse.copy()
        self.resol_coarse = npw.dim_ones((self.dim,))
        self.resol_coarse[:self.dim] = self._reorderVect(shape_coarse)
        self._append_size_constants(resol_coarse, prefix='NB_C')
        compute_coarse = shape_coarse - 2 * topo_coarse.ghosts()
        shape_fine = topo_fine.mesh.resolution
        resol_fine = shape_fine.copy()
        self.resol_fine = npw.dim_ones((self.dim,))
        self.resol_fine[:self.dim] = self._reorderVect(shape_fine)
        self._append_size_constants(resol_fine, prefix='NB_F')
        compute_fine = shape_fine
        pts_per_cell = compute_fine / compute_coarse
        assert np.all(pts_per_cell == pts_per_cell[0]), \
            "Resolutions ratio must be the same in all directions"
        pts_per_cell = pts_per_cell[0]
        if pts_per_cell == 1:
            raise ValueError('Not yet implemented for single scale')

        src, tile_size_c, vec, f_space = \
            self._kernel_cfg['multiphase_baroclinic']
        tile_size_f = tile_size_c * pts_per_cell

        self._append_size_constants(topo_coarse.ghosts(), prefix='GHOSTS_C')
        build_options = self._size_constants
        build_options += " -D C_TILE_SIZE=" + str(tile_size_c)
        build_options += " -D C_TILE_WIDTH=" + str(
            tile_size_c + 2 * topo_coarse.ghosts()[0])
        build_options += " -D C_TILE_HEIGHT=" + str(
            tile_size_c + 2 * topo_coarse.ghosts()[1])
        build_options += " -D F_TILE_SIZE=" + str(tile_size_f)
        build_options += " -D N_PER_CELL=" + str(pts_per_cell)
        for d in xrange(self.dim):
            build_options += " -D CUT_DIR" + S_DIR[d] + "="
            build_options += str(1 if topo_coarse.shape[d] > 1 else 0)
        build_options += " -D FD_ORDER=" + \
            str(self.method[SpaceDiscretisation].__name__)
        build_options += " -D GRADP_COMP=__GRADP_COMPONENT__"
        macros = {'__USER_DENSITY_FUNCTION_FROM_GIVEN_INPUT__':
                  get_extra_args_from_method(self, 'density_func', 'x')}

        gwi, lwi = f_space(compute_coarse, tile_size_c)
        clkernel_baroclinic = [None, ] * self.dim
        for i in xrange(self.dim):
            prg = self.cl_env.build_src(
                src, build_options.replace('__GRADP_COMPONENT__', str(i)),
                vec, macros=macros)
            clkernel_baroclinic[i] = prg.baroclinic_rhs
        self.num_baroclinic = KernelListLauncher(
            clkernel_baroclinic, self.cl_env.queue,
            [gwi, ] * self.dim, [lwi, ] * self.dim)

    def _compute_baroclinic_rhs(self, simulation):
        """Launch kernels without communication"""
        wait_evt = self.rhs.events + self.gradp.events + self.rho.events
        evt_x = self.num_baroclinic(
            0,
            self.rhs.gpu_data[0],
            self.rhs.gpu_data[1],
            self.rhs.gpu_data[2],
            self.rho.gpu_data[0],
            self.gradp.gpu_data[0],
            self._coarse_mesh_size,
            self._fine_mesh_size,
            wait_for=wait_evt)
        evt_y = self.num_baroclinic(
            1,
            self.rhs.gpu_data[0],
            self.rhs.gpu_data[1],
            self.rhs.gpu_data[2],
            self.rho.gpu_data[0],
            self.gradp.gpu_data[1],
            self._coarse_mesh_size,
            self._fine_mesh_size,
            wait_for=wait_evt + [evt_x])
        evt_z = self.num_baroclinic(
            2,
            self.rhs.gpu_data[0],
            self.rhs.gpu_data[1],
            self.rhs.gpu_data[2],
            self.rho.gpu_data[0],
            self.gradp.gpu_data[2],
            self._coarse_mesh_size,
            self._fine_mesh_size,
            wait_for=wait_evt + [evt_y])
        self.rhs.events.append(evt_z)

    def _compute_baroclinic_rhs_comm(self, simulation):
        """Compute operator with communications"""
        tc = MPI.Wtime()
        topo = self.rho.topology
        first_cut_dir = topo.cutdir.tolist().index(True)
        wait_evt = []
        send_l = [None, ] * self.dim
        send_r = [None, ] * self.dim
        recv_l = [None, ] * self.dim
        recv_r = [None, ] * self.dim
        e_l = [None, ] * self.dim
        e_r = [None, ] * self.dim
        for d in self._cutdir_list:
            wait_events = self.rho.events
            e_l[d] = cl.enqueue_copy(self.cl_env.queue, self._to_send[d],
                                     self.rho.gpu_data[0],
                                     host_origin=(0, 0, 0),
                                     buffer_origin=(0, 0, 0),
                                     host_pitches=self._pitches_host[d],
                                     buffer_pitches=self._pitches_buff[d],
                                     region=tuple(self._region_size[d]),
                                     wait_for=wait_events,
                                     is_blocking=False)
            e_r[d] = cl.enqueue_copy(self.cl_env.queue, self._to_send[d],
                                     self.rho.gpu_data[0],
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
        self.profiler['comm_baroclinic_rhs_comm'] += MPI.Wtime() - tc
        self.rhs.events.append(self._call_kernel(wait_evt))

    def _call_kernel_one_ghost(self, wait_evt):
        """Launch kernels with one directions of communication (Z)"""
        evt_x = self.num_baroclinic(
            0,
            self.rhs.gpu_data[0],
            self.rhs.gpu_data[1],
            self.rhs.gpu_data[2],
            self.rho.gpu_data[0],
            self._to_recv_buf[self._cutdir_list[0]],
            self.gradp.gpu_data[0],
            self._coarse_mesh_size,
            self._fine_mesh_size,
            wait_for=wait_evt)
        evt_y = self.num_baroclinic(
            1,
            self.rhs.gpu_data[0],
            self.rhs.gpu_data[1],
            self.rhs.gpu_data[2],
            self.rho.gpu_data[0],
            self._to_recv_buf[self._cutdir_list[0]],
            self.gradp.gpu_data[1],
            self._coarse_mesh_size,
            self._fine_mesh_size,
            wait_for=wait_evt + [evt_x])
        return self.num_baroclinic(
            2,
            self.rhs.gpu_data[0],
            self.rhs.gpu_data[1],
            self.rhs.gpu_data[2],
            self.rho.gpu_data[0],
            self._to_recv_buf[self._cutdir_list[0]],
            self.gradp.gpu_data[2],
            self._coarse_mesh_size,
            self._fine_mesh_size,
            wait_for=wait_evt + [evt_y])

    def _call_kernel_two_ghost(self, wait_evt):
        """Launch kernels with two directions of communication (Y and Z)"""
        evt_x = self.num_baroclinic(
            0,
            self.rhs.gpu_data[0],
            self.rhs.gpu_data[1],
            self.rhs.gpu_data[2],
            self.rho.gpu_data[0],
            self._to_recv_buf[self._cutdir_list[0]],
            self._to_recv_buf[self._cutdir_list[1]],
            self.gradp.gpu_data[0],
            self._coarse_mesh_size,
            self._fine_mesh_size,
            wait_for=wait_evt)
        evt_y = self.num_baroclinic(
            1,
            self.rhs.gpu_data[0],
            self.rhs.gpu_data[1],
            self.rhs.gpu_data[2],
            self.rho.gpu_data[0],
            self._to_recv_buf[self._cutdir_list[0]],
            self._to_recv_buf[self._cutdir_list[1]],
            self.gradp.gpu_data[1],
            self._coarse_mesh_size,
            self._fine_mesh_size,
            wait_for=wait_evt + [evt_x])
        return self.num_baroclinic(
            2,
            self.rhs.gpu_data[0],
            self.rhs.gpu_data[1],
            self.rhs.gpu_data[2],
            self.rho.gpu_data[0],
            self._to_recv_buf[self._cutdir_list[0]],
            self._to_recv_buf[self._cutdir_list[1]],
            self.gradp.gpu_data[2],
            self._coarse_mesh_size,
            self._fine_mesh_size,
            wait_for=wait_evt + [evt_y])


    def apply(self, simulation):
        self._compute(simulation)
