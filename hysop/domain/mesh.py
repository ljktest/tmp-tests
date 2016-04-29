"""local (mpi process) cartesian grid.

See also
--------

* :class:`~hysop.mpi.topology.Cartesian`
* :ref:`topologies` in HySoP user guide.

"""
from hysop.constants import debug
import hysop.tools.numpywrappers as npw
from hysop.tools.parameters import Discretization
import numpy as np
from hysop.tools.misc import utils


class Mesh(object):
    """A local cartesian grid, defined on each mpi process.

    For example, consider a 1D domain and a global resolution with N+1 points
    and a ghost layer with 2 points::

        N = 9
        dom = Box(dimension=1)
        discr = Discretization([N + 1], [2])
        m = Mesh(dom, discr, resol, start)

    will describe a grid on the current process, starting at point
    of index start in the global mesh,
    with a local resolution equal to resol.

    Usually, Mesh creation is an internal process
    leaded by the domain and its topologies.
    Just call something like::

        dom = Box(dimension=1)
        dom.create_topology(...)

    and you'll get all the local meshes.

    For example, with 2 procs:

    global grid (node number):        0 1 2 3 4 5 6 7 8

    proc 0 (global indices):      X X 0 1 2 3 X X
           (local indices) :      0 1 2 3 4 5 6 7
    proc 1 :                                  X X 4 5 6 7 X X
                                              0 1 2 3 4 5 6 7
    with 'X' for ghost points.

    - Node '8' of the global grid is not represented on local mesh,
    because of periodicity. N8 == N0

    - on proc 1, we have:
       - local resolution = 8
       - global_start = 4
       - 'computation nodes' = 2, 3, 4, 5
       - 'ghost nodes' = 0, 1, 6, 7

    Remarks:
        - all 'global' values refer to the discretization parameter.
        For example 'global start' = 2 means a point of index 2 in the
        global resolution.
        - only periodic grid are considered
    """

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @debug
    def __init__(self, parent, discretization, resolution, global_start):
        """
        Parameters
        -----------
        parent : :class:`hysop.domain.domain.Domain`
            the geometry on which this mesh is defined
            (it must be a rectangle or a rectangular cuboid)
        discretization : :class:`hysop.tools.parameters.Discretization`
            describes the global discretization of the domain
            (resolution and ghost layer)
        resolution : list or array of int
            local resolution of this mesh, INCLUDING ghost points
        global_start : list or array of int
            indices in the global mesh of the lowest point of the local mesh.

        Attributes
        ----------

        Notes
        -----
        This is a class mainly for internal use not supposed to be called
        directly by user but from domain during topology creation.
        """

        # Note : all variables with 'global' prefix are related
        # to the global grid, that is the mesh defined on the whole
        # domain. These variables do not depend on the mpi distribution.

        # ---
        # Attributes relative to the global mesh

        # geometry of reference for the mesh
        self.domain = parent
        # dimension of the mesh
        self._dim = self.domain.dimension

        assert isinstance(discretization, Discretization)
        # The global discretization of reference for this mesh.
        # It defines the resolution of the grid on the whole domain and
        # the number of ghost points on each subdomain.
        self.discretization = discretization
        # Mesh step size
        self.space_step = self.domain.length / (self.discretization.resolution
                                                - 1)
        # Coordinates of the origin of the mesh
        self.global_origin = self.domain.origin
        # Length of this mesh
        self.global_length = self.domain.length

        # ---
        # Attributes relative to the distributed mesh
        # i.e. the part of the mesh on the current mpi process
        # ---

        # Local resolution of this mesh, INCLUDING ghost points
        self.resolution = npw.const_dimarray(resolution)

        # Position of the local mesh in the global grid
        self.position = []
        # position (index) of the first computational (i.e. no ghosts)
        # point of this mesh in the global grid.
        start = npw.asdimarray(global_start)
        # last computational point, global index
        ghosts = self.discretization.ghosts
        # shift between local and global index,
        # such that: iglobal + shift = ilocal.
        # Usefull for start() and end() methods.
        self._shift = ghosts - start
        stop = start + self.resolution - 2 * ghosts
        self.position = [slice(start[d], stop[d]) for d in xrange(self._dim)]
        # Coordinates of the "lowest" point
        # of the local mesh (including ghosts)
        self.origin = self.domain.origin.copy()
        self.origin += self.space_step * (start - ghosts)
        # Coordinates of the last point of the mesh (including ghosts)
        self.end = self.origin + self.space_step * (self.resolution - 1)
        # position (index) of the first computational (i.e. no ghosts), local.
        start = self.discretization.ghosts
        # last computational point, local index
        stop = self.resolution - start
        # Local position of the computational points
        self.iCompute = [slice(start[d], stop[d])
                         for d in xrange(self._dim)]

        # coordinates of the points of the grid
        self.coords = np.ix_(*(np.linspace(self.origin[d],
                                           self.end[d], self.resolution[d])
                               for d in xrange(self._dim)))
        # coordinates of the 'computational' points, i.e. excluding ghosts.
        self.compute_coords = self.reduce_coords(self.coords, self.iCompute)

        gend = self.discretization.resolution - 1
        # Is this mesh on the last process in some direction in the
        # mpi grid of process?
        is_last = np.asarray([gend[d] < self.position[d].stop
                              for d in xrange(self._dim)])
        # indices of points that must be used to compute integrals on this mesh
        self.ind4integ = self.compute_integ_point(is_last, self.iCompute)
        self.coords4int = self.reduce_coords(self.coords, self.ind4integ)
        # True if the current mesh is locally defined on the current mpi proc.
        self.on_proc = True

    @staticmethod
    def compute_integ_point(is_last, ic, n_dir=None):
        """Compute indices corresponding to integration points
        Parameters
        ----------
        is_last : numpy array of bool
            is_last[d] = True means the process is on the top
            boundary of the mesh, in direction d.
        ic : tuple of indices
            indices of the mesh
        n_dir : array of int
            direction where lengthes of the mesh are null.
        """
        dim = len(ic)
        # We must find which points must be used
        # when we integrate on this submesh
        stops = npw.asdimarray([ic[d].stop for d in xrange(dim)])
        # when 'is_last', the last point must be removed for integration
        stops[is_last] -= 1
        # and finally, for direction where subset length is zero,
        # we must increase stop, else integral will always be zero!
        if n_dir is not None:
            stops[n_dir] = npw.asdimarray([ic[d].start + 1 for d in n_dir])

        return [slice(ic[d].start,
                stops[d]) for d in xrange(dim)]

    @staticmethod
    def reduce_coords(coords, reduced_index):
        """Compute a reduced set of coordinates

        Parameters
        ----------
        coords : tuple of arrays
            the original coordinates
        reduce : list of slices
            indices of points for which reduced coordinates
            are required.

        Returns a tuple-like set of coordinates.

        """
        assert isinstance(coords, tuple)
        assert isinstance(reduced_index, list)
        dim = len(coords)
        shapes = [list(coords[i].shape) for i in xrange(dim)]
        res = [reduced_index[i].stop - reduced_index[i].start
               for i in xrange(dim)]
        for i in xrange(dim):
            shapes[i][i] = res[i]
        shapes = tuple(shapes)
        return [coords[i].flat[reduced_index[i]].reshape(shapes[i])
                for i in xrange(dim)]

    def local_indices(self, point):
        """
        returns indices of the point of coordinates (close to) tab = x, y, z
        If (x, y, z) is not a grid point, it returns the closest grid point.
        """
        point = npw.asrealarray(point)
        if self.is_inside(point):
            return npw.asdimarray(np.rint((point - self.origin)
                                          / self.space_step))
        else:
            return False

    def global_indices(self, point):
        point = npw.asrealarray(point)
        return npw.asdimarray(np.rint((point - self.domain.origin)
                                      / self.space_step))

    def is_inside(self, point):
        """True if the point belongs to volume or surface
        described by the mesh

        :param point: list or numpy array,
           coordinates of a point
        """
        point = npw.asrealarray(point)
        return ((point - self.origin) >= 0.).all() and ((self.end
                                                         - point) >= 0.).all()

    def __str__(self):
        """
        mesh display
        """
        s = 'Local mesh: \n'
        s += ' - resolution : ' + str(self.resolution) + '\n'
        s += ' - position in global mesh: ' + str(self.position) + '\n'
        s += ' - global discretization : ' + str(self.discretization) + '\n'
        s += ' - computation points :' + str(self.iCompute) + '\n'
        return s

    def convert2local(self, sl):
        """convert indices from global mesh to local one

        :param sl: list of slices (one slice per dir) of global indices
        something like [(start_dir1, end_dir1), ...]

        Returns the same kind of list but in local coordinates
        """
        tmp = utils.intersl(sl, self.position)
        if tmp is not None:
            return [slice(tmp[i].start + self._shift[i],
                          tmp[i].stop + self._shift[i])
                    for i in xrange(self._dim)]
        else:
            return None

    def convert2global(self, sl):
        """convert indices from local mesh to global one

        :param sl: list of slices (one slice per dir) of local indices
        something like [(start_dir1, end_dir1), ...]

        Returns the same kind of list but in global coordinates
        """
        return [slice(sl[i].start - self._shift[i],
                      sl[i].stop - self._shift[i])
                for i in xrange(self._dim)]

    def __eq__(self, other):
        """True if meshes are equal.
        Two meshes are equal if they have the same
        global resolution, the same ghost layer and the same
        local resolution is the same
        """
        if self.__class__ != other.__class__:
            return False
        return self.discretization == other.discretization and\
            npw.equal(self.resolution, other.resolution).all()

    def start(self):
        """Returns indices of the lowest point of the local mesh
        in the global grid
        """
        return npw.asdimarray([p for p in
                               (self.position[d].start
                                for d in xrange(self._dim))])

    def stop(self):
        """Returns indices + 1 of the 'highest' point of the local mesh
        in the global grid.

        Warning: made to be used in slices :
        slice(mesh.start()[d], mesh.stop()[d]) represents the
        points of the current process mesh, indices given
        in the global grid.
        """
        return npw.asdimarray([p for p in
                               (self.position[d].stop
                                for d in xrange(self._dim))])

    def global_resolution(self):
        """Returns the resolution of the global grid (on the whole
        domain, whatever the mpi distribution is).
        Warning : this depends on the type of boundary conditions.
        For periodic bc, this corresponds to the user-defined discretization
        minus 1.
        """
        return self.discretization.resolution - 1

    def local_shift(self, indices):
        shift = [self.iCompute[d].start for d in xrange(self._dim)]
        return tuple([indices[d] + shift[d] for d in xrange(self._dim)])
