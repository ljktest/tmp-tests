"""To define a restriction of a Mesh
(like a subset for a domain)

"""
import hysop.tools.numpywrappers as npw
from hysop.tools.parameters import Discretization
import numpy as np
from hysop.tools.misc import utils


class SubMesh(object):
    """
    A subset of a predefined (distributed) mesh.
    """

    def __init__(self, mesh, substart, subend):
        """
        Parameters
        ----------
        mesh : :class:`~hysop.domain.mesh.Mesh`
            the parent mesh
        substart : list or array of int
            indices in the global grid of the lowest point of this submesh
        subend : list or array of int
            indices of the 'highest' point of this submesh, in the global grid.

        Warning : subend/substart are global values, that do not depend on the
        mpi distribution of data.

        todo : a proper scheme to clarify all the notations for meshes
        (global/local start end and so on).
        """
        # Note : all variables with 'global' prefix are related
        # to the global grid, that is the mesh defined on the whole
        # domain. These variables do not depend on the mpi distribution.

        # ---
        # Attributes relative to the global mesh

        # parent mesh
        self.mesh = mesh
        # dimension of the submesh
        self._dim = self.mesh.domain.dimension

        # Index of the lowest point of the global submesh in the global grid
        # of its parent
        self.substart = npw.asdimarray(substart)
        # Index of the 'highest' point of the global submesh
        # in the global grid of its parent
        self.subend = npw.asdimarray(subend)

        # position of the submesh in the global grid of its parent mesh.
        global_position_in_parent = [slice(substart[d], self.subend[d] + 1)
                                     for d in xrange(self._dim)]
        hh = self.mesh.space_step
        # Coordinates of the lowest point of this submesh
        self.global_origin = self.substart * hh + self.mesh.domain.origin
        # Length of this submesh
        self.global_length = (self.subend - self.substart) * hh

        # Warning : we must not overpass the parent global discretization.
        gres = self.subend - self.substart + 1
        # directions where length is 0, i.e. directions 'normal' to
        # the submesh.
        self._n_dir = np.where(gres == 1)[0]
        # discretization of the subset
        # Warning : at the time, no ghosts on the submesh!
        self.discretization = Discretization(gres)
        # Find which part of submesh is on the current process and
        # find its computational points. Warning:
        # the indices of computational points must be
        # relative to the parent mesh local grid!
        sl = utils.intersl(global_position_in_parent, self.mesh.position)
        # Bool to check if this process holds the end (in any direction)
        # of the domain. Useful for proper integration on this subset.
        is_last = [False, ] * self._dim

        # Check if a part of the submesh is present on the current proc.
        self.on_proc = sl is not None

        if self.on_proc:
            # Is this mesh on the last process in some direction in the
            # mpi grid of process?
            is_last = np.asarray([self.subend[d] < sl[d].stop
                                  for d in xrange(self._dim)])
            # position of the LOCAL submesh in the global grid
            # of the parent mesh
            self.position_in_parent = [s for s in sl]
            # Indices of the points of the submesh, relative to
            # the LOCAL array
            self.iCompute = self.mesh.convert2local(self.position_in_parent)

            # Resolution of the local submesh
            self.resolution = [self.iCompute[d].stop - self.iCompute[d].start
                               for d in xrange(self._dim)]

            # Same as self.iCompute but recomputed to be used
            # for integration on the submesh
            self.ind4integ = self.mesh.compute_integ_point(is_last,
                                                           self.iCompute,
                                                           self._n_dir)
            start = [self.position_in_parent[d].start - self.substart[d]
                     for d in xrange(self._dim)]
            # position of the LOCAL submesh in the global grid
            # of the submesh (not the grid of the parent mesh!)
            self.position = [slice(start[d], start[d] + self.resolution[d])
                             for d in xrange(self._dim)]
        else:
            self.position_in_parent = None
            self.position = None
            self.iCompute = [slice(0, 0), ] * self._dim
            self.ind4integ = self.iCompute
            self.resolution = [0, ] * self._dim

        # Shift between local submesh and local parent mesh.
        self.local_start = [self.iCompute[d].start for d in xrange(self._dim)]

        # Coordinates of the points of the local submesh
        self.coords = self.mesh.reduce_coords(self.mesh.coords,
                                              self.iCompute)
        self.coords4int = self.mesh.reduce_coords(self.mesh.coords,
                                                  self.ind4integ)

    def check_boundaries(self):
        """
        Special care when some boundaries of the submesh are on the
        upper boundaries of the parent mesh.
        Remind that for periodic bc, such a point does not really
        exists in the parent mesh.
        """
        # List of directions which are periodic
        periodic_dir = self.mesh.domain.i_periodic_boundaries()
        if len(periodic_dir) > 0:
            ll = np.where(
                self.subend ==
                self.mesh.global_indices(self.mesh.domain.end))[0]
            return (ll != self._n_dir).all()

        return True

    def global_resolution(self):
        """return the resolution of the global grid (on the whole
        domain, whatever the mpi distribution is).
        """
        return self.discretization.resolution

    def cx(self):
        return self.coords[0][:, ...]

    def cy(self):
        assert self._dim > 1
        return self.coords[1][0, :, ...]

    def cz(self):
        assert self._dim > 2
        return self.coords[2][0, 0, :]

    def chi(self, *args):
        """
        indicator function for points inside this submesh.
        This is only useful when one require the computation
        of the intersection of a regular subset and a sphere-like
        subset.
        See intersection and subtract methods in Subset class.

        param : tuple of coordinates (like topo.mesh.coords)

        returns : an array of boolean (True if inside)
        """
        assert len(args) == self._dim
        if not self.on_proc:
            return False
        origin = [self.coords[d].flat[0] for d in xrange(self._dim)]
        end = [self.coords[d].flat[-1] for d in xrange(self._dim)]
        c1 = [np.logical_and(args[d] >= origin[d],
                             args[d] <= end[d]) for d in xrange(self._dim)]
        for i in xrange(self._dim - 1):
            c1[i + 1] = np.logical_and(c1[i], c1[i + 1])
        return c1[-1]
