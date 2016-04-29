# -*- coding: utf-8 -*-
"""Methods to compute drag and lift for a flow around a body.

.. currentmodule:: hysop.operator

* :class:`~drag_and_lift.MomentumForces` : Momentum Formula
* :class:`~drag_and_lift.NocaForces` : Noca formulation
 (formulation = 1, 2 or 3)

See :ref:`forces`.


"""
from hysop.operator.computational import Computational
from hysop.operator.continuous import opsetup
from abc import ABCMeta, abstractmethod
from hysop.domain.control_box import ControlBox
import numpy as np


class Forces(Computational):
    """Abstract interface to classes dedicated to drag/lift computation
    for a flow around a predefined obstacle.


    """

    __metaclass__ = ABCMeta

    def __init__(self, obstacles, normalization=1., **kwds):
        """

        Parameters
        ----------
        obstacles : list of :class:`hysop.domain.obstacles`
            list of bodies inside the flow
        normalization : double, optional
            a normalization coefficient applied to the force, default = 1.
        kwds : arguments passed to base class.


        """
        super(Forces, self).__init__(**kwds)
        self.input = self.variables
        # List of hysop.domain.subsets, obstacles to the flow
        self.obstacles = obstacles
        # Normalizing coefficient for forces
        # (based on the physics of the flow)
        self.normalization = normalization
        # Minimal length of ghost layer.
        # This obviously depends on the formulation used for the force.
        self._min_ghosts = 0

    def discretize(self):
        super(Forces, self)._standard_discretize(self._min_ghosts)
        # all variables must have the same resolution
        assert self._single_topo, 'multi-resolution case not allowed.'

    @abstractmethod
    @opsetup
    def setup(self, rwork=None, iwork=None):
        pass

    def drag(self):
        """

        Returns
        -------
        HYSOP_REAL
            the last computed value for drag force
        """
        return self.discrete_op.force[0]

    def lift(self):
        """
        Returns
        -------

        np.array
            the last computed values for lift forces
        """
        return self.discrete_op.force[1:]

    def forces(self):
        """
        Returns
        -------

        np.array
            the last computed values for forces (drag, lifts)
        """
        return self.discrete_op.force


class MomentumForces(Forces):
    """
    Computation of forces (drag and lift) around an obstacle using
    Momentum (Heloise) formula

    """

    def __init__(self, velocity, penalisation_coeff, **kwds):
        """
        Parameters
        -----------
        velocity : :class:`hysop.field.continuous.Field`
            the velocity field
        penalisation_coeff : double
            coeff used to penalise velocity before force computation
        kwds : arguments passed to drag_and_lift.Forces base class.


        See :ref:`forces`.

        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        self.velocity = velocity
        super(MomentumForces, self).__init__(variables=[velocity], **kwds)
        # Penalisation coefficient value
        msg = 'penalisation coeff must be a list of values.'
        assert isinstance(penalisation_coeff, list), msg
        self._coeff = penalisation_coeff
        self._min_ghosts = 1

    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:
            from hysop.operator.discrete.drag_and_lift \
                import MomentumForces as DiscreteForce
            # build the discrete operator
            self.discrete_op = DiscreteForce(
                velocity=self.discreteFields[self.velocity],
                penalisation_coeff=self._coeff,
                obstacles=self.obstacles,
                normalization=self.normalization)

            # output setup
            self._set_io('drag_and_lift', (1, 4))
            self.discrete_op.setWriter(self._writer)
            self._is_uptodate = True


class NocaForces(Forces):
    """
    Computation of forces (drag and lift) around an obstacle using
    Noca's formula
    (See Noca99 or Plouhmans, 2002, Journal of Computational Physics)
    """

    def __init__(self, velocity, vorticity, nu, formulation=1,
                 volume_of_control=None, surfdir=None, **kwds):
        """
        Parameters
        -----------
        velocity : :class:`hysop.field.continuous.Field`
            the velocity field
        vorticity : :class:`hysop.field.continuous.Field`
            vorticity field inside the domain
        nu : double
            viscosity
        formulation : int, optional
            Noca formulation (1, 2 or 3, corresponds to equations
            I, II and III in Noca's paper)
        volume_of_control: :class:`~hysop.domain.subset.controlBox.ControlBox`,
        optional
            an optional subset of the domain, on which forces will be computed,
            useful to reduce computational cost
        surfdir : python list, optional
            indices of the surfaces on which forces are computed,
            see example below.
            Default = only surfaces normal to x direction.
        kwds : arguments passed to drag_and_lift.Forces base class.

        Attributes
        ----------
        nu : double
            viscosity


        Examples
        --------

        >> op = NocaForces(velocity=v, vorticity=w, nu=0.3, surfdir=[0, 1],
                           obstacles=[])

        Compute the integral of surface in Noca's formula only for
        surfaces normal to x and y axis.

        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(NocaForces, self).__init__(variables=[velocity, vorticity],
                                         **kwds)

        self.velocity = velocity
        self.vorticity = vorticity
        # viscosity
        self.nu = nu

        # setup for finite differences
        if self.method is None:
            import hysop.default_methods as default
            self.method = default.FORCES
        from hysop.methods_keys import SpaceDiscretisation
        from hysop.numerics.finite_differences import FD_C_4, FD_C_2
        assert SpaceDiscretisation in self.method.keys()
        if SpaceDiscretisation is FD_C_2:
            self._min_ghosts = 1
        elif SpaceDiscretisation is FD_C_4:
            self._min_ghosts = 2

        if surfdir is None:
            surfdir = [0, 1]
        # Directions where integrals on surfaces are computed
        self._surfdir = surfdir

        # The volume of control, in which forces are computed
        if volume_of_control is None:
            lr = self.domain.length * 0.9
            xr = self.domain.origin + 0.04 * self.domain.length
            volume_of_control = ControlBox(parent=self.domain,
                                           origin=xr, length=lr)
        self.voc = volume_of_control

        if formulation == 1:
            from hysop.operator.discrete.drag_and_lift import NocaI
            self.formulation = NocaI
        elif formulation == 2:
            from hysop.operator.discrete.drag_and_lift import NocaII
            self.formulation = NocaII
        elif formulation == 3:
            from hysop.operator.discrete.drag_and_lift import NocaIII
            self.formulation = NocaIII
        else:
            raise ValueError("Unknown formulation for Noca formula")

    def get_work_properties(self):
        if not self._is_discretized:
            msg = 'The operator must be discretized '
            msg += 'before any call to this function.'
            raise RuntimeError(msg)

        shape_v = [None, ] * (self.domain.dimension + 1)
        slist = self.voc.surf
        toporef = self.discreteFields[self.velocity].topology
        for i in xrange(self.domain.dimension):
            v_ind = slist[2 * i].mesh[toporef].ind4integ
            shape_v[i] = self.velocity.data[i][v_ind].shape
        v_ind = self.voc.mesh[toporef].ind4integ
        shape_v[-1] = self.velocity.data[0][v_ind].shape
        # setup for rwork, iwork is useless.
        return {'rwork': shape_v, 'iwork': None}

    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:
            topo = self.discreteFields[self.velocity].topology
            self.voc.discretize(topo)
            self.discrete_op = self.formulation(
                velocity=self.discreteFields[self.velocity],
                vorticity=self.discreteFields[self.vorticity],
                nu=self.nu,
                volume_of_control=self.voc,
                surfdir=self._surfdir,
                obstacles=self.obstacles,
                normalization=self.normalization)

            # output setup
            self._set_io('drag_and_lift', (1, 4))
            self.discrete_op.setWriter(self._writer)
            self._is_uptodate = True
