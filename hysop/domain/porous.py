"""Porous (multi-layers) subsets:

.. currentmodule hysop.domain.porous
* :class:`~Porous`
* :class:`~BiPole`
* :class:`~QuadriPole`
* :class:`~RingPole`
* :class:`~Ring`

See also
--------

* :class:`~hysop.domain.subsets.Subset` for standard subsets.
* :ref:`subsets` in HySoP user guide.

"""
from hysop.domain.subsets import Subset, Sphere, HemiSphere
from hysop.domain.subsets import Cylinder, HemiCylinder, SubBox
import hysop.tools.numpywrappers as npw
from hysop.mpi.topology import Cartesian
import numpy as np


class Porous(Subset):
    """Porous subset, with successive layers (onion-like indeed ...)
    """

    # List of authorized type for the basic geometry of the subset
    _authorized_sources = [Cylinder, Sphere, HemiSphere, HemiCylinder]

    def __init__(self, origin, source, layers, **kwds):
        """
        Parameters
        -----------
        source : :class:`~hysop.domain.subsets.Subset`
            geometry of the object. The only authorized sources are :
            Cylinder, Sphere, HemiSphere, HemiCylinder
        origin : list or array of coordinates
            position of the center of the source
        layers : list of real
            width of each layer (starting from outside layer)
        """
        # the external layer of the porous subset.
        # Note FP : we need this envelop to set a proper
        # indicator function for the porous subset, since this
        # function is used in all the union, intersection ...
        # methods.
        # is_inside will return true for any point inside this
        # envelop.
        envelop = source(parent=kwds['parent'], origin=origin,
                         radius=sum(layers))
        assert 'chi' not in kwds
        super(Porous, self).__init__(chi=envelop.get_chi(), **kwds)
        # Center position
        self.origin = npw.asrealarray(origin).copy()
        self.is_porous = True
        self._source = source
        assert self._source.__mro__[0] in self._authorized_sources
        assert isinstance(layers, list)
        # width of porous layers on the source
        self.layers = layers

    def discretize(self, topo):
        assert isinstance(topo, Cartesian)
        if topo in self.ind:
            return self.ind[topo]

        poles = self._extrude(topo)
        # a subbox used to reduced the number size of coords
        # during dist computation.
        max_radius = sum(self.layers)
        nb_layers = len(self.layers)
        if len(poles) > 0:
            self.ind[topo] = [None] * (nb_layers + 1)
        else:
            self.ind[topo] = [None] * nb_layers

        if nb_layers == 1:
            # pathologic case where the subset has only one layer
            out_set = self._source(parent=self._parent, origin=self.origin,
                                   radius=max_radius)
            if len(poles) == 0:
                self.ind[topo] = out_set.discretize(topo)
            else:
                # First ind list is the whole source minus the 'poles'
                self.ind[topo][0] = Subset.subtract_list_of_sets([out_set],
                                                                 poles,
                                                                 topo)
                # Second ind list is the intersection of the source and of the
                # union of the poles
                self.ind[topo][1] = Subset.intersection_of_list_of_sets(
                    [out_set], poles, topo)
        else:
            out_radius = max_radius
            for i in xrange(nb_layers - 1):
                in_radius = out_radius - self.layers[i]
                in_set = self._source(parent=self._parent, origin=self.origin,
                                      radius=in_radius)
                out_set = self._source(parent=self._parent, origin=self.origin,
                                       radius=out_radius)
                setlist = [in_set] + poles
                # subtract setlist from out_set i.e. compute the
                # union of the two boxes (top/down) and of
                # the internal set and subtract this union from
                # the outer set.
                self.ind[topo][i] = Subset.subtract_list_of_sets([out_set],
                                                                 setlist, topo)
                out_radius = in_radius
            if len(poles) == 0:
                self.ind[topo][-1] = in_set.discretize(topo)[0]
            else:
                self.ind[topo][-2] = in_set.discretize(topo)[0]
                # The last ind. list is the intersection of out_set and
                # of the union of the two boxes
                out_set = self._source(parent=self._parent, origin=self.origin,
                                       radius=max_radius)
                self.ind[topo][-1] = self.intersection_of_list_of_sets(
                    [out_set], poles, topo)

        self._reduce_topology(topo)
        return self.ind[topo]

    def _extrude(self, topo):
        """
        Compute the list of subsets that must be subtracted from
        the source to compute the final object
        """
        assert isinstance(topo, Cartesian)
        return []


class BiPole(Porous):
    """Intersection of a subset with two boxes at the poles
    """
    def __init__(self, poles_thickness, poles_dir=None, **kwds):
        """

        Parameters
        ----------
        poles_thickness : real or list or array of real numbers
            thickness of layer(s) in direction given by poles_dir.
            Use poles_thickness = value or [value]
            if all poles have the same width.
            Or, to set bottom pole to v1 and top pole to v2
            poles_thickness = [v1, v2].
        poles_dir : int
            direction for which poles are 'computed'. Default = last dir.
        kwds : args for base class

        Notes
        -----

        An example to create poles in direction 2, i.e. z,
        of thickness 0.2 on top and 0.1 at the bottom of an HemiSphere::

            obst = BiPole(..., source=HemiSphere, poles_thickness=[0.1, 0.2],
                          poles_dir=[2])

        """
        super(BiPole, self).__init__(**kwds)
        if not isinstance(poles_thickness, list):
            self._thicknesses = [poles_thickness] * 2
        elif len(poles_thickness) == 1:
            self._thicknesses = [poles_thickness[0]] * 2
        else:
            assert len(poles_thickness) == 2
            self._thicknesses = list(poles_thickness)
        if poles_dir is None:
            poles_dir = self._parent.dimension - 1
        self._ndir = poles_dir

    def _extrude(self, topo):
        dim = self._parent.dimension
        # a subbox used to reduced the number size of coords
        # during dist computation.
        max_radius = sum(self.layers)
        # create top/bottom boxes to compute
        # intersection with self._source
        # dimension of top/bottom boxes
        downpos = self.origin - max_radius
        lbox = [2 * max_radius, ] * dim
        lbox[self._ndir] = self._thicknesses[0]
        if self._source is Cylinder or self._source is HemiCylinder:
            downpos[1] = self._parent.origin[1]
            lbox[1] = self._parent.length[1]
        downbox = SubBox(parent=self._parent, length=lbox, origin=downpos)
        downbox.discretize(topo)
        toppos = downpos.copy()
        toppos[self._ndir] += 2 * max_radius - self._thicknesses[-1]
        lbox = [2 * max_radius, ] * dim
        lbox[self._ndir] = self._thicknesses[-1]
        if self._source is Cylinder or self._source is HemiCylinder:
            lbox[1] = self._parent.length[1]
        topbox = SubBox(parent=self._parent, length=lbox, origin=toppos)
        topbox.discretize(topo)
        return [downbox, topbox]


class QuadriPole(Porous):
    """Intersection of a subset with boxes at the poles on y and z axis
    """

    _authorized_sources = [Sphere, HemiSphere]

    def __init__(self, poles_thickness, **kwds):
        """

        Parameters
        ----------
        poles_thickness : real or list or array of real numbers
            thickness of layer(s) in direction given by poles_dir.
            Use poles_thickness = value or [v1, v2]
            if all poles in one direction have the same width.
            Or, to set bottom/top poles to v1/v2 in y dir and
            bottom/top poles to v3/v4 in z dir,
            poles_thickness = [v1, v2, v3, v4].

        kwds : args for base class

        Notes
        -----

        An example to create poles in direction 2, i.e. z,
        of thickness 0.2 on top and 0.1 at the bottom, on an HemiSphere::

            obst = QuadriPole(..., source=HemiSphere,
                              poles_thickness=[0.1, 0.2])

        """
        super(QuadriPole, self).__init__(**kwds)
        msg = 'This class can not be used in 1 or 2D.'
        assert self._parent.dimension == 3, msg
        assert isinstance(poles_thickness, list)
        self._thicknesses = [0.] * 4
        if len(poles_thickness) == 2:
            self._thicknesses[:2] = [poles_thickness[0]] * 2
            self._thicknesses[2:] = [poles_thickness[1]] * 2
        else:
            assert len(poles_thickness) == 4
            self._thicknesses = list(poles_thickness)

    def _extrude(self, topo):
        # a subbox used to reduced the number size of coords
        # during dist computation.
        max_radius = sum(self.layers)
        dim = self._parent.dimension
        # create top/bottom boxes to compute
        # intersection with self._source
        # dimension of top/bottom boxes
        poles = [None] * 4
        p = 0
        for ndir in xrange(1, 3):
            current = 2 * p
            downpos = self.origin - max_radius
            lbox = [2 * max_radius, ] * dim
            lbox[ndir] = self._thicknesses[current]
            if self._source is Cylinder or self._source is HemiCylinder:
                downpos[1] = self._parent.origin[1]
                lbox[1] = self._parent.length[1]
            poles[current] = SubBox(parent=self._parent,
                                    length=lbox, origin=downpos)
            poles[current].discretize(topo)
            current = 2 * p + 1
            toppos = downpos.copy()
            toppos[ndir] += 2 * max_radius - self._thicknesses[current]
            lbox = [2 * max_radius, ] * dim
            lbox[ndir] = self._thicknesses[current]
            if self._source is Cylinder or self._source is HemiCylinder:
                lbox[1] = self._parent.length[1]
            poles[current] = SubBox(parent=self._parent,
                                    length=lbox, origin=toppos)
            poles[current].discretize(topo)
            p += 1
        return poles


class RingPole(Porous):
    """A sphere or hemisphere with a ring-shape subset
    """
    _authorized_sources = [Sphere, HemiSphere]

    def __init__(self, ring_width, **kwds):
        """

        Parameters
        ----------
        ring_width : real
            width of the ring
        kwds : args for base class
        """
        super(RingPole, self).__init__(**kwds)
        self._thicknesses = ring_width

    def _extrude(self, topo):
        # a subbox used to reduced the number size of coords
        # during dist computation.
        max_radius = sum(self.layers)

        # create a cylinder to "extrude" the central part of the ring
        return [Cylinder(parent=self._parent, origin=self.origin,
                         radius=max_radius - self._thicknesses,
                         axis=0)]

    def discretize(self, topo):
        assert isinstance(topo, Cartesian)
        if topo in self.ind:
            return self.ind[topo]
        poles = self._extrude(topo)
        max_radius = sum(self.layers)

        nb_layers = len(self.layers)
        if len(poles) > 0:
            self.ind[topo] = [None] * (nb_layers + 1)
        else:
            self.ind[topo] = [None] * nb_layers

        if nb_layers == 1:
            # pathologic case where the subset has only one layer
            out_set = self._source(parent=self._parent, origin=self.origin,
                                   radius=max_radius)
            # First ind list is the intersection of the source and the cylinder
            self.ind[topo][0] = Subset.intersection_of_list_of_sets([out_set],
                                                                    poles,
                                                                    topo)
            # Second ind list is the source minus the cylinder
            self.ind[topo][1] = Subset.subtract_list_of_sets([out_set], poles,
                                                             topo)
        else:
            out_radius = max_radius
            for i in xrange(nb_layers - 1):
                in_radius = out_radius - self.layers[i]
                in_set = self._source(parent=self._parent, origin=self.origin,
                                      radius=in_radius)
                out_set = self._source(parent=self._parent, origin=self.origin,
                                       radius=out_radius)
                setlist = [in_set] + poles
                # subtract setlist from out_set i.e. compute the
                # union of the two boxes (top/down) and of
                # the internal set and subtract this union from
                # the outer set.
                self.ind[topo][i] = Subset.intersection_of_list_of_sets(
                    [out_set], setlist, topo)
                out_radius = in_radius
            if len(poles) == 0:
                self.ind[topo][-1] = in_set.discretize(topo)[0]
            else:
                self.ind[topo][-2] = in_set.discretize(topo)[0]
                # The last ind. list is the intersection of out_set and
                # of the union of the two boxes
                out_set = self._source(parent=self._parent, origin=self.origin,
                                       radius=max_radius)
                self.ind[topo][-1] = self.subtract_list_of_sets(
                    [out_set], poles, topo)

        self._reduce_topology(topo)
        return self.ind[topo]


class Ring(Porous):
    """A sphere or hemisphere with a ring-shape subset
    """

    def __init__(self, ring_width, **kwds):
        """

        Parameters
        ----------
        ring_width : real or list or array of real numbers
            lengthes of the ring (in x and y dir).
        kwds : args for base class
        """
        super(Ring, self).__init__(**kwds)
        assert isinstance(ring_width, list)
        self._thicknesses = ring_width

    def discretize(self, topo):
        assert isinstance(topo, Cartesian)
        if topo in self.ind:
            return self.ind[topo]
        dim = self._parent.dimension
        max_radius = sum(self.layers)
        # create a box that intersect with the sphere to create the ring.
        posbox = self.origin - max_radius
        posbox[0] = self.origin[0] - self._thicknesses[0]
        lbox = [2 * max_radius, ] * dim
        lbox[0] = self._thicknesses[0]
        box = SubBox(parent=self._parent, length=lbox, origin=posbox)
        box.discretize(topo)
        # create a cylinder to "extrude" the central part of the ring
        cyl = Cylinder(parent=self._parent, origin=self.origin,
                       radius=max_radius - self._thicknesses[1],
                       axis=0)
        cyl.discretize(topo)
        max_radius = sum(self.layers)

        iring = Subset.subtract_as_bool(box, cyl, topo)
        dim = self._parent.dimension

        nb_layers = len(self.layers)
        self.ind[topo] = [None] * (nb_layers + 1)
        if nb_layers == 1:
            out_set = self._source(parent=self._parent, origin=self.origin,
                                   radius=max_radius)
            iout = Subset.union_as_bool([out_set], topo)
            # pathologic case where the subset has only one layer
            self.ind[topo][0] = topo.mesh.local_shift(
                np.where(np.logical_and(iout, np.logical_not(iring))))
            self.ind[topo][1] = topo.mesh.local_shift(
                np.where(np.logical_and(iring, iout)))
        else:
            out_radius = max_radius
            for i in xrange(nb_layers - 1):
                in_radius = out_radius - self.layers[i]
                in_set = self._source(parent=self._parent, origin=self.origin,
                                      radius=in_radius)
                out_set = self._source(parent=self._parent, origin=self.origin,
                                       radius=out_radius)
                self.ind[topo][i] = Subset.subtract_as_bool(
                    out_set, in_set, topo)
                self.ind[topo][i] = topo.mesh.local_shift(np.where(
                    np.logical_and(self.ind[topo][i], np.logical_not(iring))))
                out_radius = in_radius
            ind = Subset.union_as_bool([in_set], topo)
            self.ind[topo][-2] = topo.mesh.local_shift(np.where(
                np.logical_and(ind, np.logical_not(iring))))
            out_set = self._source(parent=self._parent, origin=self.origin,
                                   radius=max_radius)
            ind = Subset.union_as_bool([out_set], topo)
            self.ind[topo][-1] = topo.mesh.local_shift(
                np.where(np.logical_and(iring, ind)))

        self._reduce_topology(topo)
        return self.ind[topo]
