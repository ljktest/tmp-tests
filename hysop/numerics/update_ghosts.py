"""
@file numerics/update_ghosts.py

Update ghost points for a list of numpy arrays
for a given topology.
"""
from hysop.constants import debug, PERIODIC, HYSOP_MPI_REAL
import hysop.tools.numpywrappers as npw
import numpy as np


class UpdateGhosts(object):
    """
    Ghost points synchronization for a list of numpy arrays
    """

    @debug
    def __init__(self, topo, nbElements):
        """
        Setup for send/recv process of ghosts points for a list
        of numpy arrays, for a given topology.

        @param topology : the topology common to all fields.
        @param nbElements : max number of arrays that will be update
        at each call.
        nbElements and memshape will be used to allocate memory for local
        buffers used for send-recv process.
        """
        ## The mpi topology and mesh distribution
        self.topology = topo
        ## Ghost layer
        self.ghosts = self.topology.mesh.discretization.ghosts
        # Indices of points to be filled from previous neighbour
        # Each component is a slice and corresponds to a direction in
        # the topology grid.
        self._g_fromprevious = []
        # Indices of points to be filled from next neighbour.
        self._g_fromnext = []
        # Indices of points to be sent to next neighbour.
        self._g_tonext = []
        # Indices of points to be sent to previous neighbour.
        self._g_toprevious = []
        # Buffers that contains ghosts points to be sent (list, one per dir)
        self._sendbuffer = []
        # Buffers for reception (list, one per dir)
        self._recvbuffer = []
        # domain dimension
        self._dim = self.topology.domain.dimension

        self._setup_slices()

        ## shape of numpy arrays to be updated.
        self.memshape = tuple(self.topology.mesh.resolution)
        # length of memory required to save one numpy array
        self._memoryblocksize = np.prod(self.memshape)
        ## Number of numpy arrays that will be updated
        self.nbElements = nbElements
        if self.topology.size > 1:  # else no need to set buffers ...
            # Size computation below assumes that what we send in one
            # dir has the same size as what we receive from process in the
            # same dir ...
            exchange_dir = [d for d in xrange(self._dim)
                            if self.topology.cutdir[d]]
            # A temporary array used to calculate slices sizes
            temp = np.zeros(self.memshape, dtype=np.int8)

            for d in exchange_dir:
                buffsize = 0
                buffsize += temp[self._g_tonext[d]].size * self.nbElements
                self._sendbuffer.append(npw.zeros((buffsize)))
                self._recvbuffer.append(npw.zeros((buffsize)))

    def _setup_slices(self):
        """
        Compute slices to send and recieve ghosts values.
        """
        defslice = [slice(None, None, None)] * self._dim
        nogh_slice = [slice(0)] * self._dim
        for d in xrange(self._dim):
            if self.ghosts[d] > 0:
                self._g_fromprevious.append(list(defslice))
                self._g_fromprevious[d][d] = slice(self.ghosts[d])
                self._g_fromnext.append(list(defslice))
                self._g_fromnext[d][d] = slice(-self.ghosts[d], None, None)
                self._g_toprevious.append(list(defslice))
                self._g_toprevious[d][d] = slice(self.ghosts[d],
                                                 2 * self.ghosts[d], None)
                self._g_tonext.append(list(defslice))
                self._g_tonext[d][d] = slice(-2 * self.ghosts[d],
                                             -self.ghosts[d])

                otherDim = [x for x in xrange(self._dim) if x != d]
                for d2 in otherDim:
                    self._g_fromprevious[d][d2] = slice(self.ghosts[d2],
                                                        -self.ghosts[d2])
                    self._g_fromnext[d][d2] = slice(self.ghosts[d2],
                                                    -self.ghosts[d2])
                    self._g_toprevious[d][d2] = slice(self.ghosts[d2],
                                                      -self.ghosts[d2])
                    self._g_tonext[d][d2] = slice(self.ghosts[d2],
                                                  -self.ghosts[d2])
            else:
                self._g_fromprevious.append(list(nogh_slice))
                self._g_fromnext.append(list(nogh_slice))
                self._g_toprevious.append(list(nogh_slice))
                self._g_tonext.append(list(nogh_slice))


    def __call__(self, variables):
        return self.apply(variables)

    def applyBC(self, variables):
        """
        Apply boundary conditions for non-distributed directions (only).
        @param variables : a list of ndarrays
        Note that in distributed directions, BC are automatically set
        during ghosts update (apply).
         """
        assert (self.topology.domain.boundaries == PERIODIC).all(),\
            'Only implemented for periodic boundary conditions.'
        assert isinstance(variables, list)
        dirs = [d for d in xrange(self._dim)
                if self.topology.shape[d] == 1]
        for d in dirs:
            self._applyBC_in_dir(variables, d)

    def _applyBC_in_dir(self, variables, d):
        """Apply periodic boundary condition in direction d."""
        for v in variables:
            assert v.shape == self.memshape
            v[self._g_fromprevious[d]] = v[self._g_tonext[d]]
            v[self._g_fromnext[d]] = v[self._g_toprevious[d]]

    @debug
    def apply(self, variables):
        """
        Compute ghosts values from mpi communications and boundary conditions.
        """
        assert isinstance(variables, list)
        exchange_dir = []
        if self.topology.size > 1:
            exchange_dir = [d for d in xrange(self._dim)
                            if self.topology.cutdir[d]]
        i = 0
        for d in exchange_dir:
            self._apply_in_dir(variables, d, i)
            # update index in neighbours list
            i += 1
            # End of loop through send/recv directions.
        # Apply boundary conditions for non-distributed directions
        self.applyBC(variables)

    def _apply_in_dir(self, variables, d, i):
        """Communicate ghosts values in direction d for neighbour
        in direction i of the topology"""
        comm = self.topology.comm
        rank = self.topology.rank
        neighbours = self.topology.neighbours
        # 1 - Fill in buffers
        # Loop through all variables that are distributed
        pos = 0
        nextpos = 0
        for v in variables:
            assert v.shape == self.memshape
            nextpos += v[self._g_tonext[d]].size
            self._sendbuffer[i][pos:nextpos] = v[self._g_tonext[d]].flat
            pos = nextpos

        # 2 - Send to next receive from previous
        dest_rk = neighbours[1, i]
        from_rk = neighbours[0, i]
        comm.Sendrecv([self._sendbuffer[i], HYSOP_MPI_REAL],
                      dest=dest_rk, sendtag=rank,
                      recvbuf=self._recvbuffer[i],
                      source=from_rk, recvtag=from_rk)

        # 3 - Print recvbuffer back to variables and update sendbuffer
        # for next send
        pos = 0
        nextpos = 0
        for v in variables:
            nextpos += v[self._g_fromprevious[d]].size
            v[self._g_fromprevious[d]].flat = \
                self._recvbuffer[i][pos:nextpos]
            self._sendbuffer[i][pos:nextpos] = \
                v[self._g_toprevious[d]].flat
            pos = nextpos

        # 4 -Send to previous and receive from next
        dest_rk = neighbours[0, i]
        from_rk = neighbours[1, i]
        comm.Sendrecv([self._sendbuffer[i], HYSOP_MPI_REAL],
                      dest=dest_rk, sendtag=rank,
                      recvbuf=self._recvbuffer[i],
                      source=from_rk, recvtag=from_rk)
        # 5 - Print recvbuffer back to variables.
        pos = 0
        nextpos = 0
        for v in variables:
            nextpos += v[self._g_fromprevious[d]].size
            v[self._g_fromnext[d]].flat = \
                self._recvbuffer[i][pos:nextpos]
            pos = nextpos


class UpdateGhostsFull(UpdateGhosts):
    """
    Ghost points synchronization for a list of numpy arrays
    """

    @debug
    def __init__(self, topo, nbElements):
        """
        Setup for send/recv process of ghosts points for a list
        of numpy arrays, for a given topology.

        @param topology : the topology common to all fields.
        @param nbElements : max number of arrays that will be update
        at each call.
        nbElements and memshape will be used to allocate memory for local
        buffers used for send-recv process.
        This version differs from UpdateGhosts by computing also ghosts values
        in edges and corners of the domain. The directions are computed in reversed order.
        """
        super(UpdateGhostsFull, self).__init__(topo, nbElements)

    def _setup_slices(self):
        """
        Computes slices to send and recieve ghosts values.
        It assumes that directions are computed from an xrange() loop so that
        ghosts in previous directions are completed.
        """
        defslice = [slice(None, None, None)] * self._dim
        nogh_slice = [slice(0)] * self._dim
        for d in xrange(self._dim):
            if self.ghosts[d] > 0:
                self._g_fromprevious.append(list(defslice))
                self._g_fromprevious[d][d] = slice(self.ghosts[d])
                self._g_fromnext.append(list(defslice))
                self._g_fromnext[d][d] = slice(-self.ghosts[d], None, None)
                self._g_toprevious.append(list(defslice))
                self._g_toprevious[d][d] = slice(self.ghosts[d],
                                                 2 * self.ghosts[d], None)
                self._g_tonext.append(list(defslice))
                self._g_tonext[d][d] = slice(-2 * self.ghosts[d],
                                             -self.ghosts[d])

                ## Slices for other directions corresponding to directions not
                ## yet exchanged : x > d. For directions x < d, the slices is a
                ## full slice that includes ghosts. This assumes that directions
                ## x < d have already been computed (by communications or local
                ## exchanges)
                #                if d == 0:
                otherDim = [x for x in xrange(self._dim) if x > d]
                for d2 in otherDim:
                    self._g_fromprevious[d][d2] = slice(self.ghosts[d2],
                                                        -self.ghosts[d2])
                    self._g_fromnext[d][d2] = slice(self.ghosts[d2],
                                                    -self.ghosts[d2])
                    self._g_toprevious[d][d2] = slice(self.ghosts[d2],
                                                      -self.ghosts[d2])
                    self._g_tonext[d][d2] = slice(self.ghosts[d2],
                                                  -self.ghosts[d2])
            else:
                self._g_fromprevious.append(list(nogh_slice))
                self._g_fromnext.append(list(nogh_slice))
                self._g_toprevious.append(list(nogh_slice))
                self._g_tonext.append(list(nogh_slice))

    @debug
    def apply(self, variables):
        """Apply either mpi communications
        or local boundary conditions to fill ghosts.
        Loop over directions and switch among local BC or mpi comm.
        """
        assert isinstance(variables, list)
        exchange_dir = []
        if self.topology.size > 1:
            exchange_dir = [d for d in xrange(self._dim)
                            if self.topology.cutdir[d]]
        local_bc_dir = [d for d in xrange(self._dim)
                        if self.topology.shape[d] == 1]
        assert len(exchange_dir) + len(local_bc_dir) == self._dim

        i = 0
        for d in xrange(self._dim):
            if d in local_bc_dir:
                self._applyBC_in_dir(variables, d)
            elif d in exchange_dir:
                self._apply_in_dir(variables, d, i)
                # update index in neighbours list
                i += 1
                # End of loop through send/recv directions.
