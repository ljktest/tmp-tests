"""Define a volume of control inside a domain (volume + all faces)"""
from hysop.domain.subsets import SubBox
import numpy as np
import hysop.tools.numpywrappers as npw


class ControlBox(SubBox):
    """
    Build a sub-domain, box-shaped
    ==> define set of indices inside this domain (ind member)
    and set of indices belonging to surfaces of this domain (slices members).
    Useful to define control volume to perform integration.
    """
    def __init__(self, **kwds):
        """
        Same parameters as for :class:`hysop.domain.subsets.SubBox`.
        """
        super(ControlBox, self).__init__(**kwds)

        # We must have a real box, not a plane ...
        assert len(self.t_dir) == len(self.length)

        self.surf = [None] * len(self.length) * 2

    def discretize(self, topo):
        """Create a sub meshes for topo inside the control box and
        on its faces.
        """
        # Create the mesh for the whole box
        super(ControlBox, self).discretize(topo)
        # Create a mesh for each side
        dim = topo.domain.dimension
        ilist = np.arange(dim)
        for direction in xrange(dim):
            ndir = np.where(ilist == direction)[0]
            length = self.real_length[topo].copy()
            length[ndir] = 0.0
            orig = self.real_orig[topo].copy()
            self.surf[2 * direction] = \
                SubBox(origin=orig, length=length, parent=self._parent,
                       normal=-1)
            orig = self.real_orig[topo].copy()
            orig[ndir] += self.real_length[topo][ndir]
            self.surf[2 * direction + 1] = \
                SubBox(origin=orig, length=length, parent=self._parent,
                       normal=1)

        for s in self.surf:
            s.discretize(topo)
        return self.ind[topo]

    def _check_boundaries(self, surf, topo):
        """
        Special care when some boundaries of the control box are on the
        upper boundaries of the domain.
        Remind that for periodic bc, such surfaces does not really
        exists in the parent mesh.
        """
        return surf.mesh[topo].check_boundaries()

    def integrate_on_faces(self, field, topo, list_dir,
                           component=0, root=None):
        """Integration of a field on one or more faces of the box

        Parameters
        ----------
        field : :class:`~hysop.fields.continuous.Field`
            a field to be integrated on the box
        topo : :class:`~hysop.mpi.topology.Cartesian`
            set mesh/topology used for integration
        list_dir : list of int
            indices of faces on which integration is required
            0 : normal to xdir, lower surf,
            1 : normal to xdir, upper surf (in x dir)
            2 : normal to ydir, lower surf, and so on.
        component : int, optional
            number of the field component to be integrated
        root : int, optional
            rank of the leading mpi process (to collect data)
            If None reduction is done on all processes from topo.

        Returns a numpy array, with res = sum
        of the integrals of a component of field on all surf in list_dir
        """
        res = 0.
        for ndir in list_dir:
            surf = self.surf[ndir]
            msg = 'Control Box error : surface out of bounds.'
            assert self._check_boundaries(surf, topo), msg
            res += surf.integrate_field_on_proc(field, topo, component)
        if root is None:
            return topo.comm.allreduce(res)
        else:
            return topo.comm.reduce(res, root=root)

    def integrate_on_faces_allc(self, field, topo, list_dir, root=None):
        """Integration of a field on one or more faces of the box

        Parameters
        ----------
        field : :class:`~hysop.fields.continuous.Field`
            a field to be integrated on the box
        topo : :class:`~hysop.mpi.topology.Cartesian`
            set mesh/topology used for integration
        list_dir : list of int
            indices of faces on which integration is required
            0 : normal to xdir, lower surf,
            1 : normal to xdir, upper surf (in x dir)
            2 : normal to ydir, lower surf, and so on.
        root : int, optional
            rank of the leading mpi process (to collect data)
            If None reduction is done on all processes from topo.

        Returns a numpy array, with res[i] = sum
        of the integrals of component i of field on all surf in list_dir
        """
        nbc = field.nb_components
        res = npw.zeros(nbc)
        gres = npw.zeros(nbc)
        for ndir in list_dir:
            surf = self.surf[ndir]
            assert self._check_boundaries(surf, topo)
            for i in xrange(nbc):
                res[i] += surf.integrate_field_on_proc(field, topo, i)
        if root is None:
            topo.comm.Allreduce(res, gres)
        else:
            topo.comm.Reduce(res, gres, root=root)
        return gres
