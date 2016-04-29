"""
@file multiresolution_filter.py
Filter values from a fine grid to a coarse grid.
"""
from hysop.constants import debug, np, HYSOP_MPI_REAL
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.tools.profiler import profile
import hysop.tools.numpywrappers as npw
from hysop.methods_keys import Remesh
from hysop.methods import Rmsh_Linear, L2_1
from hysop.numerics.remeshing import RemeshFormula


class FilterFineToCoarse(DiscreteOperator):
    """
    Discretized operator for filtering from fine to coarse grid.
    """
    def __init__(self, field_in, field_out, **kwds):
        self.field_in, self.field_out = field_in, field_out
        super(FilterFineToCoarse, self).__init__(
            variables=self.field_in, **kwds)
        self.input = [self.field_in]
        self.output = [self.field_out]
        self._mesh_in = self.field_in[0].topology.mesh
        self._mesh_out = self.field_out[0].topology.mesh
        self.gh_out = self.field_out[0].topology.ghosts()
        gh_in = self.field_in[0].topology.ghosts()
        if self.method[Remesh] is Rmsh_Linear:
            assert (self.gh_out >= 1).all()
        elif self.method[Remesh] is L2_1:
            assert (self.gh_out >= 2).all()
        else:
            raise ValueError('This scheme (' + self.method[Remesh].__name__ +
                             ') is not yet implemented for filter.')
        resol_in = self._mesh_in.resolution - 2 * gh_in
        resol_out = self._mesh_out.resolution - 2 * self.gh_out
        pts_per_cell = resol_in / resol_out
        assert np.all(pts_per_cell >= 1), \
            "This operator is fine grid to coarse one"
        self.scale_factor = np.prod(self._mesh_in.space_step) / \
            np.prod(self._mesh_out.space_step)

        # multi-gpu ghosts buffers for communication
        self._cutdir_list = np.where(
            self.field_in[0].topology.cutdir)[0].tolist()
        self._comm = self.field_in[0].topology.comm
        self._comm_size = self._comm.Get_size()
        self._comm_rank = self._comm.Get_rank()
        if self._comm_size == 1:
            self._exchange_ghosts = self._exchange_ghosts_local
        else:
            self._exchange_ghosts = self._exchange_ghosts_mpi
            self._gh_from_l = [None] * self._dim
            self._gh_from_r = [None] * self._dim
            self._gh_to_l = [None] * self._dim
            self._gh_to_r = [None] * self._dim
            for d in self._cutdir_list:
                shape = list(self.field_out[0].data[0].shape)
                shape[d] = self.gh_out[d]
                self._gh_from_l[d] = npw.zeros(tuple(shape))
                self._gh_from_r[d] = npw.zeros(tuple(shape))
                self._gh_to_l[d] = npw.zeros(tuple(shape))
                self._gh_to_r[d] = npw.zeros(tuple(shape))

        in_coords = (self._mesh_in.coords - self._mesh_in.origin) / \
            self._mesh_out.space_step
        self.floor_coords = np.array([np.floor(c) for c in in_coords])
        self.dist_coords = in_coords - self.floor_coords
        self._work_weight = np.array(
            [npw.zeros_like(c) for c in self.dist_coords])
        # Slices to serialize concurrent access in coarse grid
        # Several points in fine grid are laying in the same coarse cell
        # The serialization avoid concurrent access.
        self._sl = []
        for ix in xrange(pts_per_cell[0]):
            for iy in xrange(pts_per_cell[1]):
                for iz in xrange(pts_per_cell[2]):
                    self._sl.append((
                        slice(ix + gh_in[0],
                              resol_in[0] + ix + gh_in[0],
                              pts_per_cell[0]),
                        slice(iy + gh_in[1],
                              resol_in[1] + iy + gh_in[1],
                              pts_per_cell[1]),
                        slice(iz + gh_in[2],
                              resol_in[2] + iz + gh_in[2],
                              pts_per_cell[2]),
                    ))
        assert len(self._sl) == np.prod(pts_per_cell)
        # Slice in coarse grid to distribute values
        self._sl_coarse = []
        # Weights associated to offsets in coarse grid
        self._w_coarse = []
        try:
            assert issubclass(self.method[Remesh], RemeshFormula), \
                "This operator works with a RemeshingFormula."
            self._rmsh_method = self.method[Remesh]()
        except KeyError:
            self._rmsh_method = Rmsh_Linear()
        if isinstance(self._rmsh_method, Rmsh_Linear):
            # Linear interpolation
            assert np.all(self.gh_out >= 1), "Output data must have at least" \
                "1 ghosts in all directions (" + str(self.gh_out) + " given)"
        elif isinstance(self._rmsh_method, L2_1):
            # L2_1 interpolation
            assert np.all(self.gh_out >= 2), "Output data must have at least" \
                "2 ghosts in all directions (" + str(self.gh_out) + " given)"
        else:
            raise ValueError("The multiresolution filter is implemented for " +
                             "Linear or L2_1 interpolation (" +
                             str(self.method[Remesh]) + " given).")
        for i_x in xrange(len(self._rmsh_method.weights)):
            for i_y in xrange(len(self._rmsh_method.weights)):
                for i_z in xrange(len(self._rmsh_method.weights)):
                    self._sl_coarse.append((
                        slice(self._mesh_out.iCompute[0].start -
                              self._rmsh_method.shift + i_x,
                              self._mesh_out.iCompute[0].stop -
                              self._rmsh_method.shift + i_x,
                              None),
                        slice(self._mesh_out.iCompute[1].start -
                              self._rmsh_method.shift + i_y,
                              self._mesh_out.iCompute[1].stop -
                              self._rmsh_method.shift + i_y,
                              None),
                        slice(self._mesh_out.iCompute[2].start -
                              self._rmsh_method.shift + i_z,
                              self._mesh_out.iCompute[2].stop -
                              self._rmsh_method.shift + i_z,
                              None)
                    ))
                    self._w_coarse.append(
                        (i_x, i_y, i_z))
        if self._rwork is None:
            self._rwork = npw.zeros(resol_out)

    @debug
    @profile
    def apply(self, simulation=None):
        assert simulation is not None, \
            "Missing simulation value for computation."
        for v_in, v_out in zip(self.field_in, self.field_out):
            for d in xrange(v_in.nb_components):
                for sl_coarse, iw_fun in zip(self._sl_coarse, self._w_coarse):
                    self._work_weight[0] = self._rmsh_method(
                        iw_fun[0], self.dist_coords[0], self._work_weight[0])
                    self._work_weight[1] = self._rmsh_method(
                        iw_fun[1], self.dist_coords[1], self._work_weight[1])
                    self._work_weight[2] = self._rmsh_method(
                        iw_fun[2], self.dist_coords[2], self._work_weight[2])
                    # Loop over fine grid points sharing the same coarse cell
                    for sl in self._sl:
                        self._rwork[...] = v_in[d][sl]
                        # Compute weights
                        self._rwork[...] *= self._work_weight[0][sl[0], :, :]
                        self._rwork[...] *= self._work_weight[1][:, sl[1], :]
                        self._rwork[...] *= self._work_weight[2][:, :, sl[2]]
                        self._rwork[...] *= self.scale_factor
                        # Add contributions in data
                        v_out[d][sl_coarse] += self._rwork[...]
        self._exchange_ghosts()

    def _exchange_ghosts_local_d(self, d):
        """Exchange ghosts values in periodic local array"""
        s_gh = self.gh_out[d]
        sl = [slice(None) for _ in xrange(self._dim)]
        sl_gh = [slice(None) for _ in xrange(self._dim)]
        sl[d] = slice(1 * s_gh, 2 * s_gh)
        sl_gh[d] = slice(-1 * s_gh, None)
        for v_out in self.field_out:
            v_out.data[0][tuple(sl)] += v_out.data[0][tuple(sl_gh)]
        sl[d] = slice(-2 * s_gh, -1 * s_gh)
        sl_gh[d] = slice(0, 1 * s_gh)
        for v_out in self.field_out:
            v_out.data[0][tuple(sl)] += v_out.data[0][tuple(sl_gh)]

    @profile
    def _exchange_ghosts_local(self):
        """Performs ghosts exchange locally in each direction"""
        for d in xrange(self._dim):
            self._exchange_ghosts_local_d(d)

    def _exchange_ghosts_mpi_d(self, d):
        """Exchange ghosts values in parallel"""
        s_gh = self.gh_out[d]
        sl_l = [slice(None) for _ in xrange(self._dim)]
        sl_gh_l = [slice(None) for _ in xrange(self._dim)]
        sl_r = [slice(None) for _ in xrange(self._dim)]
        sl_gh_r = [slice(None) for _ in xrange(self._dim)]
        sl_l[d] = slice(1 * s_gh, 2 * s_gh)
        sl_gh_r[d] = slice(-1 * s_gh, None)
        sl_r[d] = slice(-2 * s_gh, -1 * s_gh)
        sl_gh_l[d] = slice(0, 1 * s_gh)
        for v_out in self.field_out:
            first_cut_dir = v_out.topology.cutdir.tolist().index(True)
            self._gh_to_l[d][...] = v_out.data[0][tuple(sl_gh_l)]
            self._gh_to_r[d][...] = v_out.data[0][tuple(sl_gh_r)]
            r_rk = v_out.topology.neighbours[1, d - first_cut_dir]
            l_rk = v_out.topology.neighbours[0, d - first_cut_dir]
            recv_r = self._comm.Irecv(
                [self._gh_from_r[d], self._gh_from_r[d].size,
                 HYSOP_MPI_REAL],
                source=r_rk, tag=1234 + r_rk + 19 * d)
            recv_l = self._comm.Irecv(
                [self._gh_from_l[d], self._gh_from_l[d].size,
                 HYSOP_MPI_REAL],
                source=l_rk, tag=4321 + l_rk + 17 * d)
            send_l = self._comm.Issend(
                [self._gh_to_l[d], self._gh_to_l[d].size, HYSOP_MPI_REAL],
                dest=l_rk, tag=1234 + self._comm_rank + 19 * d)
            send_r = self._comm.Issend(
                [self._gh_to_r[d], self._gh_to_r[d].size, HYSOP_MPI_REAL],
                dest=r_rk, tag=4321 + self._comm_rank + 17 * d)
            send_r.wait()
            recv_l.wait()
            v_out.data[0][tuple(sl_l)] += self._gh_from_l[d]
            send_l.wait()
            recv_r.wait()
            v_out.data[0][tuple(sl_r)] += self._gh_from_r[d]

    @profile
    def _exchange_ghosts_mpi(self):
        """Performs ghosts exchange either locally or with mpi communications
        in each direction"""
        for d in xrange(self._dim):
            if d in self._cutdir_list:
                self._exchange_ghosts_mpi_d(d)
            else:
                self._exchange_ghosts_local_d(d)
