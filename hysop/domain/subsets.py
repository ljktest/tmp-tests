"""Subsets of a given domain:

* :class:`~hysop.domain.subsets.Sphere`,
* :class:`~hysop.domain.subsets.HemiSphere`,
* :class:`~hysop.domain.subsets.Cylinder`,
* :class:`~hysop.domain.subsets.HemiCylinder`,
* :class:`~hysop.domain.subsets.SubBox`,
* :class:`~hysop.domain.subsets.Subset` (abstract base class).

See also
--------

* :class:`~hysop.domain.porous.Porous` for porous (multi-layers) subsets.
* :ref:`subsets` in HySoP user guide.

"""
from hysop.domain.domain import Domain
import numpy as np
from hysop.mpi.topology import Cartesian
from hysop.fields.discrete import DiscreteField
import hysop.tools.numpywrappers as npw
from hysop.fields.continuous import Field
from hysop.domain.submesh import SubMesh
import numpy.linalg as la
from hysop.mpi import MPI


class Subset(object):
    """
    A subset is a geometry defined inside a domain.
    Given a topology on the parent domain, the subset must
    provide some lists of indices to allow the computation of a discrete
    field on the subset, with something like

    data[subset.ind[topo] = ...
    or
    data[subset.tab] = ...

    There are two types of subsets:
    - 'regular' ones, those on which a regular mesh can be defined
    - others, like spheres, cylinders ...

    Subsets have an attribute 'ind' which is a dictionnary of tuples
    representing the points inside the subset, (keys = topology) such that:
    ind[topo] = (i_x, i_y, i_z),
    with i_x[i], i_y[i], i_z[i] for each index i being the indices in the
    local grid of a point inside the subset.

    It means that for any discrete field df,
    df[ind[topo]] represents the grid values of dd inside the subset.

    """

    _TOLCOEF_ = 0.

    def __init__(self, parent, chi=None, group=None):
        """
        Parameters
        ----------

        parent : :class:`~hysop.domain.box.Box`
            the domain in which the subset is defined
        func : a python function
            indicator function of the domain.
        group : a list of Subsets, optional
            useful to build a new subset from the union or
            intersection of subsets.

        Attributes
        ----------
        ind : dictionnary
            indices of points inside the subset, for a given topology.
            Keys = topology, values = indices, as tuple or arrays.

        Notes
        -----

        func argument list must start with space coordinates,
        e.g. for a 3D domain something like::

            def myfunc(x, y, z, ...):
                ...

        """
        assert isinstance(parent, Domain)
        self._parent = parent
        # dictionnary of indices of points inside the subsets.
        # Keys = topology, Values = tuple of arrays.
        self.ind = {}
        # Dictionnary (key = topo), on_proc[topo] = True
        # if the subset has points on the current mpi process.
        self.on_proc = {}
        # indicator function of the subset
        self._is_inside = chi
        self.is_porous = False
        # list of space direction, used for integration.
        self.t_dir = [d for d in xrange(parent.dimension)]
        # dict of mpi communicators, keys=topo
        # such that self.subcomm[topo] represents the mpi processes
        # where on_proc[topo] is true. Useful to reduce collective comm
        # on the subset
        self.subcomm = {}
        self.max_subcoords = {}
        # a list of subsets used to build this subset, from union
        # or intersection
        self._group = group

    def chi(self, topo, *args):
        """Indicator function of the subset

        Returns
        -------
        array of bool

        """
        tol = la.norm(topo.mesh.space_step) * self._TOLCOEF_
        return self._is_inside(*args) <= tol

    def _set_chi(self, chi):
        """Reset indicator function

        Mostly for internal setup (during porous obst. init), so do not
        use it or use it with care ...
        """
        self._is_inside = chi

    def get_chi(self):
        """Get indicator function

        Mostly for internal setup (during porous obst. init), so do not
        use it or use it with care ...
        """
        return self._is_inside

    def discretize(self, topo):
        """
        Create the list of indices for points inside the subset
        for a given topo.

        :param topo: :class:`hysop.mpi.topology.Cartesian`

        Returns
        -------
        tuple of indices
           indices of points inside the domain, np.where result.

        """
        assert isinstance(topo, Cartesian)
        if topo not in self.ind:
            self.ind[topo] = \
                [np.where(self.chi(topo, *topo.mesh.compute_coords))]
            self.ind[topo][0] = topo.mesh.local_shift(self.ind[topo][0])
            self._reduce_topology(topo)

        return self.ind[topo]

    def discretize_group(self, topo, union=True):
        """Build a subset from a group of subsets
        Parameters
        ----------
        topo : :class:`hysop.mpi.topology.Cartesian`
        union : bool, optional
            compute union of subsets if True, else intersection
        """
        if union:
            self.ind[topo] = [self.union(self._group, topo)]
        else:
            self.ind[topo] = [self.intersection(self._group, topo)]
        self._reduce_topology(topo)
        return self.ind[topo]

    def _reduce_topology(self, topo):
        """Find the reduced mpi communicator that handles
        all points of this subset.
        """
        dim = self._parent.dimension
        self.on_proc[topo] = (np.asarray([len(self.ind[topo][0][i])
                                          for i in xrange(dim)])
                              != 0).all()

        plist = np.asarray(topo.comm.allgather(self.on_proc[topo]),
                           dtype=np.bool)
        gtopo = topo.comm.Get_group()
        rks = np.where(plist)[0]
        subgroup = gtopo.Incl(rks)
        self.subcomm[topo] = topo.comm.Create(subgroup)
        self.max_subcoords[topo] = None
        if self.on_proc[topo]:
            self.max_subcoords[topo] = topo.proc_coords.copy()
            self.subcomm[topo].Allreduce(topo.proc_coords,
                                         self.max_subcoords[topo],
                                         op=MPI.MAX)

    @staticmethod
    def intersection(slist, topo):
        """Compute the intersection of subsets

        Parameters
        ----------
        slist : a list of :class:`~hysop.domain.subsets.Subset`
        topo : :class:`~hysop.mpi.topology.Cartesian`

        Returns
        -------
        list of tuples
            the intersection of the subsets in slist

        """
        c0 = Subset.intersection_as_bool(slist, topo)
        return topo.mesh.local_shift(np.where(c0))

    @staticmethod
    def intersection_as_bool(slist, topo):
        """Compute the intersection of subsets

        Parameters
        ----------
        slist : a list of :class:`~hysop.domain.subsets.Subset`
        topo : :class:`~hysop.mpi.topology.Cartesian`

        Returns
        -------
        numpy array of boolean
            the intersection of the subsets in slist

        """
        sref = slist[0]
        ## if len(slist) == 1:
        ##     return sref.ind[topo]
        coords = topo.mesh.compute_coords
        c0 = sref.chi(topo, *coords)
        for s in slist[1:]:
            c1 = s.chi(topo, *coords)
            c0 = np.logical_and(c0, c1)
        return c0

    @staticmethod
    def intersection_of_list_of_sets(s1, s2, topo):
        """Compute the intersection of two lists, each list
        being the union of a set of subsets.

        Parameters
        ----------
        s1, s2 : lists of :class:`~hysop.domain.subsets.Subset`
        topo : :class:`~hysop.mpi.topology.Cartesian`

        Returns
        -------
        numpy array of boolean
            the intersection of the subsets in slist
        """
        c1 = Subset.union_as_bool(s1, topo)
        c2 = Subset.union_as_bool(s2, topo)
        return topo.mesh.local_shift(np.where(np.logical_and(c1, c2)))

    @staticmethod
    def union(slist, topo):
        """Union of subsets

        Parameters
        ----------
        slist : list of :class:`~hysop.domain.subsets.Subset`
        topo : :class:`~hysop.mpi.topology.Cartesian`

        Returns
        --------
        the union of a set of subsets for a
        given topo as a tuple of arrays which gives
        the indexes of points inside the union.
        """
        return topo.mesh.local_shift(
            np.where(Subset.union_as_bool(slist, topo)))

    @staticmethod
    def union_as_bool(slist, topo):
        """Union of subsets

        Parameters
        ----------
        slist : list of :class:`~hysop.domain.subsets.Subset`
        topo : :class:`~hysop.mpi.topology.Cartesian`

        Returns
        --------
        the union of a set of subsets for a
        given topo as an array of bool, True
        for points inside the union.
        """
        sref = slist[0]
        coords = topo.mesh.compute_coords
        c0 = sref.chi(topo, *coords)
        if len(slist) == 1:
            return c0

        for s in slist[1:]:
            c1 = s.chi(topo, *coords)
            c0 = np.logical_or(c0, c1)
        return c0

    @staticmethod
    def subtract(s1, s2, topo):
        """Difference of subsets

        Parameters
        ----------
        s1, s2 : :class:`~hysop.domain.subsets.Subset`
        topo : :class:`~hysop.mpi.topology.Cartesian`

        Returns
        --------
        points in s1 - s2 as a tuple of arrays of indices.
        """
        return topo.mesh.local_shift(
            np.where(Subset.subtract_as_bool(s1, s2, topo)))

    @staticmethod
    def subtract_as_bool(s1, s2, topo):
        """Difference of subsets

        Parameters
        ----------
        s1, s2 : :class:`~hysop.domain.subsets.Subset`
        topo : :class:`~hysop.mpi.topology.Cartesian`

        Returns
        --------
        points in s1 - s2 as an array of boolean
        """
        coords = topo.mesh.compute_coords
        c1 = s1.chi(topo, *coords)
        c2 = np.logical_not(s2.chi(topo, *coords))
        return np.logical_and(c1, c2)

    @staticmethod
    def subtract_list_of_sets(s1, s2, topo):
        """Difference of subsets

        Parameters
        ----------
        s1, s2 : list of :class:`~hysop.domain.subsets.Subset`
        topo : :class:`~hysop.mpi.topology.Cartesian`

        Returns
        --------
        points in s1 - s2 as a tuple of arrays of indices.
        """
        c1 = Subset.union_as_bool(s1, topo)
        c2 = np.logical_not(Subset.union_as_bool(s2, topo))
        return topo.mesh.local_shift(np.where(np.logical_and(c1, c2)))

    def integrate_field_allc(self, field, topo, root=None):
        """Field integration

        Parameters
        ----------
        field : :class:`~hysop.fields.continuous.Field`
            a field to be integrated on the box
        topo : :class:`~hysop.mpi.topology.Cartesian`
            set mesh/topology used for integration
        root : int, optional
            rank of the leading mpi process (to collect data)
            If None reduction is done on all processes from topo.

        Returns
        --------
        a numpy array, with res[i] = integral of component i
        of the input field over the current subset.
        """
        res = npw.zeros(field.nb_components)
        gres = npw.zeros(field.nb_components)
        for i in xrange(res.size):
            res[i] = self.integrate_field_on_proc(field, topo, component=i)
        if root is None:
            topo.comm.Allreduce(res, gres)
        else:
            topo.comm.Reduce(res, gres, root=root)

        return gres

    def integrate_field(self, field, topo, component=0, root=None):
        """Field integration

        Parameters
        ----------
        field : :class:`~hysop.fields.continuous.Field`
            a field to be integrated on the box
        topo : :class:`~hysop.mpi.topology.Cartesian`
            set mesh/topology used for integration
        component : int, optional
            number of the field component to be integrated
        root : int, optional
            rank of the leading mpi process (to collect data)
            If None reduction is done on all processes from topo.

        Returns
        --------
        double, integral of a component
        of the input field over the current subset.
        """
        res = self.integrate_field_on_proc(field, topo, component)
        if root is None:
            return topo.comm.allreduce(res)
        else:
            return topo.comm.reduce(res, root=root)

    def integrate_field_on_proc(self, field, topo, component=0):
        """Field integration

        Parameters
        ----------
        field : :class:`~hysop.fields.continuous.Field`
            a field to be integrated on the box
        topo : :class:`~hysop.mpi.topology.Cartesian`
            set mesh/topology used for integration
        component : int, optional
            number of the field component to be integrated

        Returns
        --------
        double, integral of a component
        of the input field over the current subset, on the current process
        (i.e. no mpi reduce over all processes)
        """
        assert isinstance(field, Field)
        assert isinstance(topo, Cartesian)
        discr_f = field.discretize(topo)
        dvol = npw.prod(topo.mesh.space_step[self.t_dir])
        result = npw.real_sum(discr_f[component][self.ind[topo][0]])
        result *= dvol
        return result

    def integrate_dfield_allc(self, field, root=None):
        """Field integration

        Parameters
        ----------
        field : :class:`~hysop.fields.discrete.DiscreteField`
            a field to be integrated on the box
        root : int, optional
            rank of the leading mpi process (to collect data)
            If None reduction is done on all processes from topo.

        Returns
        --------
        a numpy array, with res[i] = integral of component i
        of the input field over the current subset.
        """
        res = npw.zeros(field.nb_components)
        gres = npw.zeros(field.nb_components)
        for i in xrange(res.size):
            res[i] = self.integrate_dfield_on_proc(field, component=i)
        if root is None:
            field.topology.comm.Allreduce(res, gres)
        else:
            field.topology.comm.Reduce(res, gres, root=root)

        return gres

    def integrate_dfield(self, field, component=0, root=None):
        """Field integration

        Parameters
        ----------
        field : :class:`~hysop.fields.discrete.DiscreteField`
            a field to be integrated on the box
        component : int, optional
            number of the field component to be integrated
        root : int, optional
            rank of the leading mpi process (to collect data)
            If None reduction is done on all processes from topo.

        Returns
        --------
        double, integral of a component
        of the input field over the current subset.
        """
        res = self.integrate_dfield_on_proc(field, component)
        if root is None:
            return field.topology.comm.allreduce(res)
        else:
            return field.topology.comm.reduce(res, root=root)

    def integrate_dfield_on_proc(self, field, component=0):
        """Field integration

        Parameters
        ----------
        field : :class:`~hysop.fields.discrete.DiscreteField`
            a field to be integrated on the box
        component : int, optional
            number of the field component to be integrated

        Returns
        --------
        double, integral of a component
        of the input field over the current subset, on the current process
        (i.e. no mpi reduce over all processes)
        """
        assert isinstance(field, DiscreteField)
        topo = field.topology
        dvol = npw.prod(topo.mesh.space_step[self.t_dir])
        result = npw.real_sum(field[component][self.ind[topo][0]])
        result *= dvol
        return result


class Sphere(Subset):
    """Spherical domain.
    """

    def __init__(self, origin, radius=1.0, **kwds):
        """
        Parameters
        ----------
        origin : :class:`~hysop.domain.subsets.Subset`

        origin : list or array
            position of the center
        radius : double, optional
        kwds : base class parameters

        """
        def dist(*args):
            size = len(args)
            return npw.asarray(np.sqrt(sum([(args[d] - self.origin[d]) ** 2
                                            for d in xrange(size)]))
                               - self.radius)

        super(Sphere, self).__init__(chi=dist, **kwds)
        # Radius of the sphere
        self.radius = radius
        # Center position
        self.origin = npw.asrealarray(origin).copy()

    def __str__(self):
        s = self.__class__.__name__ + ' of radius ' + str(self.radius)
        s += ' and center position ' + str(self.origin)
        return s


class HemiSphere(Sphere):
    """HemiSpherical domain.
    Area defined by the intersection of a sphere and a box.
    The box is defined with :
    - cutdir, normal direction to a plan
    - cutpos, position of this plan along the 'cutdir" axis
    - all points of the domain where x < xs.
    """
    def __init__(self, cutpos=None, cutdir=0, **kwds):
        """
        Parameters
        ----------
        cutpos : list or array of coordinates
            position of the cutting plane
        cutdir : real, optional
            direction of the normal to the cutting plane.
            Default = x-direction (0).
        """
        super(HemiSphere, self).__init__(**kwds)
        # direction of normal to the cutting plane
        self.cutdir = cutdir
        if cutpos is None:
            cutpos = self.origin[self.cutdir]

        def left_box(x):
            return x - cutpos
        self.LeftBox = left_box

    def chi(self, topo, *args):
        """Indicator function of the subset

        Returns
        -------
        array of bool

        """
        tol = la.norm(topo.mesh.space_step) * self._TOLCOEF_
        return np.logical_and(
            self._is_inside(*args) <= tol,
            self.LeftBox(args[self.cutdir]) <=
            (topo.mesh.space_step[self.cutdir] * 0.5))


class Cylinder(Subset):
    """Cylinder-like domain.
    """

    def __init__(self, origin, radius=1.0, axis=1, **kwds):
        """
        Parameters
        ----------
        origin : list or array
            coordinates of the center
        radius : double, optional
             default = 1.
        axis : int, optional
           direction of the main axis of the cylinder, default=1 (y)

        """

        def dist(*args):
            size = len(self._dirs)
            return npw.asarray(np.sqrt(sum([(args[self._dirs[d]] -
                                             self.origin[self._dirs[d]]) ** 2
                                            for d in xrange(size)]))
                               - self.radius)

        super(Cylinder, self).__init__(chi=dist, **kwds)
        # Radius of the cylinder
        self.radius = radius
        # Main axis position
        self.origin = npw.asrealarray(origin).copy()
        # direction of the main axis of the cylinder
        self.axis = axis
        dim = self._parent.dimension
        dirs = np.arange(dim)
        self._dirs = np.where(dirs != self.axis)[0]

    def chi(self, topo, *args):
        """Indicator function of the subset

        Returns
        -------
        array of bool

        """
        tol = la.norm(topo.mesh.space_step) * self._TOLCOEF_
        return np.logical_and(self._is_inside(*args) <= tol,
                              args[self.axis] ==
                              args[self.axis])

    def __str__(self):
        s = self.__class__.__name__ + ' of radius ' + str(self.radius)
        s += ' and center position ' + str(self.origin)
        return s


class HemiCylinder(Cylinder):
    """Half cylinder domain.
    """
    def __init__(self, cutpos=None, cutdir=0, **kwds):
        """A cylinder cut by a plane (normal to one axis dir).

        Parameters
        ----------
        cutpos : list or array of coordinates
            position of the cutting plane
        cutdir : real, optional
            direction of the normal to the cutting plane.
            Default = x-direction (0).

        """
        super(HemiCylinder, self).__init__(**kwds)
        # direction of normal to the cutting plane
        self.cutdir = cutdir
        if cutpos is None:
            cutpos = self.origin[self.cutdir]

        def left_box(x):
            return x - cutpos
        self.LeftBox = left_box

    def chi(self, topo, *args):
        """Indicator function of the subset

        Returns
        -------
        array of bool

        """
        tol = la.norm(topo.mesh.space_step) * self._TOLCOEF_
        return (np.logical_and(
            np.logical_and(self._is_inside(*args) <= tol,
                           self.LeftBox(args[self.cutdir])
                           <= topo.mesh.space_step[self.cutdir]),
            args[self.axis] == args[self.axis]))


class SubBox(Subset):
    """
    A rectangle (in 2 or 3D space), defined by the coordinates of
    its lowest point, its lenghts and its normal.
    """
    def __init__(self, origin, length, normal=1, **kwds):
        """
        Parameters
        ----------
        origin : list or array of double
            position of the lowest point of the box
        length : list or array of double
            lengthes of the sides of the box
        normal : int = 1 or -1, optional
            direction of the outward normal. Only makes
            sense when the 'box' is a plane.
        **kwds : extra args for parent class

        """
        super(SubBox, self).__init__(**kwds)
        # Dictionnary of hysop.domain.mesh.Submesh, keys=topo, values=mesh
        self.mesh = {}
        # coordinates of the lowest point of this subset
        self.origin = npw.asrealarray(origin).copy()
        # length of this subset
        self.length = npw.asrealarray(length).copy()
        # coordinates of the 'highest' point of this subset
        self.end = self.origin + self.length
        # coordinate axes belonging to the subset
        self.t_dir = np.where(self.length != 0)[0]
        # coordinate axe normal to the subset
        self.n_dir = np.where(self.length == 0)[0]
        # direction of the outward unit normal (+ or - 1)
        # Useful when the surface belong to a control box.
        self.normal = normal
        msg = 'Subset error, the origin is outside of the domain.'
        assert ((self.origin - self._parent.origin) >= 0).all(), msg
        msg = 'Subset error, the subset is too large for the domain.'
        assert ((self._parent.end - self.end) >= 0).all(), msg
        # dict of coordinates of the lengthes of the subset (values)
        # for a given topo (keys). If the origin/length does not
        # fit exactly with the mesh, there may be a small shift of
        # these values.
        self.real_length = {}
        # dict of coordinates of the origin of the subset (values)
        # for a given topo (keys). If the origin/length does not
        # fit exactly with the mesh, there may be a small shift of
        # these values.
        self.real_orig = {}

    def discretize(self, topo):
        """
        Compute a local submesh on the subset, for a given topology

        :param topo: :class:`~hysop.mpi.topology.Cartesian`
        """
        assert isinstance(topo, Cartesian)
        if topo in self.mesh:
            return self.ind[topo]

        # Find the indices of starting/ending points in the global_end
        # grid of the mesh of topo.
        gstart = topo.mesh.global_indices(self.origin)
        gend = topo.mesh.global_indices(self.origin + self.length)
        # create a submesh
        self.mesh[topo] = SubMesh(topo.mesh, gstart, gend)
        self.real_length[topo] = self.mesh[topo].global_length
        self.real_orig[topo] = self.mesh[topo].global_origin
        self.ind[topo] = [self.mesh[topo].iCompute]
        self._reduce_topology(topo)
        return self.ind[topo]

    def chi(self, topo, *args):
        """
        indicator function for points inside this submesh.
        This is only useful when one require the computation
        of the intersection of a regular subset and a sphere-like
        subset.
        See intersection and subtract methods in Subset class.

        param : tuple of coordinates (like topo.mesh.coords)

        returns : an array of boolean (True if inside)
        """
        msg = 'You must discretize the SubBox before any call to chi function.'
        assert topo in self.mesh, msg
        return self.mesh[topo].chi(*args)

    def _reduce_topology(self, topo):
        """Find the reduced mpi communicator that handles
        all points of this subset.
        """
        self.on_proc[topo] = self.mesh[topo].on_proc
        plist = np.asarray(topo.comm.allgather(self.on_proc[topo]),
                           dtype=np.bool)
        gtopo = topo.comm.Get_group()
        rks = np.where(plist)[0]
        subgroup = gtopo.Incl(rks)
        self.subcomm[topo] = topo.comm.Create(subgroup)
        self.max_subcoords[topo] = None
        if self.on_proc[topo]:
            self.max_subcoords[topo] = topo.proc_coords.copy()
            self.subcomm[topo].Allreduce(topo.proc_coords,
                                         self.max_subcoords[topo],
                                         op=MPI.MAX)

    def integrate_field_on_proc(self, field, topo, component=0):
        """Field integration

        Parameters
        ----------
        field : :class:`~hysop.field.continuous.Field`
            a field to be integrated on the box
        topo : :class:`~hysop.mpi.topology.Cartesian`
            set mesh/topology used for integration
        component : int, optional
            number of the field component to be integrated

        Returns
        --------
        integral of a component of the input field over the current subset,
        on the current process
        (i.e. no mpi reduce over all processes) for the discretization
        given by topo..
        """
        assert isinstance(field, Field)
        assert isinstance(topo, Cartesian)
        df = field.discretize(topo)
        dvol = npw.prod(topo.mesh.space_step[self.t_dir])
        result = npw.real_sum(df[component][self.mesh[topo].ind4integ])
        result *= dvol
        return result

    def integrate_func(self, func, topo, nbc, root=None):
        """Function integration

        Parameters
        ----------
        func: python function of space coordinates
        topo: :class:`~hysop.mpi.topology.Cartesian`
            set mesh/topology used for integration
        nbc : int
            number of components of the return value from func

        Returns
        -------
        integral of a component of the input field
        over the current subset, for the discretization given by
        topo
        """
        res = npw.zeros(nbc)
        gres = npw.zeros(nbc)
        for i in xrange(res.size):
            res[i] = self.integrate_func_on_proc(func, topo)
        if root is None:
            topo.comm.Allreduce(res, gres)
        else:
            topo.comm.Reduce(res, gres, root=root)
        return gres

    def integrate_func_on_proc(self, func, topo):
        """Function local integration

        Parameters
        ----------
        func: python function of space coordinates
        topo: :class:`~hysop.mpi.topology.Cartesian`
            set mesh/topology used for integration

        Returns
        --------
        integral of the function on the subset on the current process
        (i.e. no mpi reduce over all processes)
        """
        assert hasattr(func, '__call__')
        assert isinstance(topo, Cartesian)
        vfunc = np.vectorize(func)
        if self.mesh[topo].on_proc:
            result = npw.real_sum(vfunc(*self.mesh[topo].coords4int))
        else:
            result = 0.
        dvol = npw.prod(topo.mesh.space_step)
        result *= dvol
        return result

    def integrate_dfield_on_proc(self, field, component=0):
        """Discrete field local integration

        Parameters
        ----------
        field : :class:`~hysop.field.discrete.DiscreteField`
            a field to be integrated on the box
        component : int, optional
            number of the field component to be integrated
        integrate the field on the subset on the current process
        (i.e. no mpi reduce over all processes) for the discretization
        given by topo.

        Returns
        --------
        integral of the field on the subset on the current process
        (i.e. no mpi reduce over all processes)
        """
        assert isinstance(field, DiscreteField)
        topo = field.topology
        dvol = npw.prod(topo.mesh.space_step[self.t_dir])
        result = npw.real_sum(field[component][self.mesh[topo].ind4integ])
        result *= dvol
        return result
