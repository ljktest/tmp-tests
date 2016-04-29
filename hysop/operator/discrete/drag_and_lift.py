# -*- coding: utf-8 -*-
"""Discrete operators to compute drag and lift forces
"""
from hysop.numerics.update_ghosts import UpdateGhosts
from hysop.operator.discrete.discrete import DiscreteOperator
import hysop.tools.numpywrappers as npw
from abc import ABCMeta, abstractmethod
from hysop.numerics.utils import Utils
from hysop.constants import XDIR, YDIR, ZDIR
from hysop.domain.control_box import ControlBox
from hysop.domain.subsets import Subset
from hysop.numerics.differential_operations import Laplacian
from hysop.numerics.finite_differences import FD_C_2
import numpy as np


class Forces(DiscreteOperator):
    """
    Compute drag and lift using Noca's formula.
    See Noca99 or Plouhmans, 2002, Journal of Computational Physics
    The present class implements formula (52) of Plouhmans2002.
    Integral inside the obstacle is not taken into account.
    """
    __metaclass__ = ABCMeta

    def __init__(self, obstacles=None, normalization=1., **kwds):
        """
        Parameters
        ----------
        obstacles : list of :class:`~hysop.domain.subsets.Subset`
            list of bodies inside the flow
        normalization : double, optional
            a normalization coefficient applied to the force, default = 1.
        kwds : arguments passed to base class.

        Attributes
        ----------
        force : numpy array
            drag and lift forces

        """
        # true if the operator needs to work on the current process.
        # Updated in derived class.
        self._on_proc = True
        # deal with obstacles, volume of control ...

        self._indices = self._init_indices(obstacles)

        super(Forces, self).__init__(**kwds)
        # topology is common to all variables
        self.input = self.variables
        self._topology = self.input[0].topology
        # elem. vol
        self._dvol = npw.prod(self._topology.mesh.space_step)

        msg = 'Force computation undefined for domain of dimension 1.'
        assert self._dim > 1, msg

        # Local buffers, used for time-derivative computation
        self._previous = npw.zeros(self._dim)
        self._buffer = npw.zeros(self._dim)
        # The force we want to compute (lift(s) and drag)
        self.force = npw.zeros(self._dim)

        # list of np arrays to be synchronized
        self._datalist = []
        for v in self.input:
            self._datalist += v.data
        nbc = len(self._datalist)
        # Ghost points synchronizer
        self._synchronize = UpdateGhosts(self._topology, nbc)

        # Normalizing coefficient for forces
        # (based on the physics of the flow)
        self._normalization = normalization

        # Which formula must be used to compute the forces.
        # Must be set in derived class.
        self._formula = lambda dt: 0

        # Set how reduction will be performed
        # Default = reduction over all process.
        # \todo : add param to choose this option
        self.mpi_sum = self._mpi_allsum
        # A 'reduced' communicator used to mpi-reduce the force.
        # Set in derived class
        self._subcomm = None

    @abstractmethod
    def _init_indices(self, obstacles):
        """
        Parameters
        -----------
        obstacles : a list of :class:`hysop.domain.subsets.Subset`

        Returns
        -------
        a list of np arrays
            points indices (like result from np.where)

        Discretize obstacles, volume of control ... and
        compute a list of points representing these sets.
        What is inside indices depends on the chosen method.
        See derived class 'init_indices' function for details.
        """

    def _mpi_allsum(self):
        """
        Performs MPI reduction (sum result value over all process)
        All process get the result of the sum.
        """
        self.force = self._topology.comm.allreduce(self.force)

    def _mpi_sum(self, root=0):
        """
        Performs MPI reduction (sum result value over all process)
        Result send only to 'root' process.

        : param root : int
            number of the process which collect the result.

        """
        self.force = self._topology.comm.reduce(self.force, root=root)

    def apply(self, simulation=None):
        """Compute forces

        :param simulation: :class:`~hysop.problem.simulation.Simulation`

        """
        assert simulation is not None,\
            "Simulation parameter is required for Forces apply."
        # Synchro of ghost points is required for fd schemes
        self._synchronize(self._datalist)
        # Compute forces locally
        dt = simulation.timeStep
        if not self._on_proc:
            self._buffer[...] = 0.0
            self.force[...] = 0.0
            self._previous[...] = 0.0
        else:
            self._formula(dt)
        # Reduce results over MPI processes
        self.mpi_sum()
        # normalization of the forces --> cD, cL, cZ
        self.force *= self._normalization
        # Print results, if required
        ite = simulation.currentIteration
        if self._writer is not None and self._writer.do_write(ite):
            self._writer.buffer[0, 0] = simulation.time
            self._writer.buffer[0, 1:] = self.force
            self._writer.write()


class MomentumForces(Forces):
    """
    Compute drag and lift using Noca's formula.
    See Noca99 or Plouhmans, 2002, Journal of Computational Physics
    The present class implements formula (52) of Plouhmans2002.
    Integral inside the obstacle is not taken into account.
    """
    def __init__(self, velocity, penalisation_coeff, **kwds):
        """
        Parameters
        -----------
        velocity : :class:`hysop.field.discrete.DiscreteField`
            the velocity field
        penalisation_coeff : list of double
            coeff used to penalise velocity before force computation
        kwds : arguments passed to drag_and_lift.Forces base class.

        See :ref:`forces`.

        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        # discrete velocity field
        self.velocity = velocity
        msg = 'penalisation coeff must be a list of values.'
        assert isinstance(penalisation_coeff, list), msg
        # penalisation coefficient(s)
        self._coeff = penalisation_coeff

        super(MomentumForces, self).__init__(variables=[velocity], **kwds)

        # formula used to compute drag and lift
        self._formula = self._momentum

        # Check ghost points
        assert (self._topology.ghosts() >= 1).all()

    def _set_work_arrays(self, rwork=None, iwork=None):
        """Set or check rwork.

        rwork will be required only in formulations where
        an integral over the volume of the control box is computed.
        """
        # !!! Velocity must be set before a call to this function
        # and so before base class initialization !!!

        # The only required buffer if for integral on the volume
        # of control.
        if not self._on_proc:
            self._rwork = [None]
            return

        size = 0
        for ind in self._indices:
            size = np.maximum(self.velocity.data[0][ind].size, size)
        shape = (size,)
        if rwork is None:
            self._rwork = [npw.zeros(shape)]
        else:
            assert isinstance(rwork, list), 'rwork must be a list.'
            assert len(self._rwork) == 1
            assert self._rwork[0].shape == shape

    def _init_indices(self, obstacles):
        msg = 'obstacles arg must be a list.'
        assert isinstance(obstacles, list), msg
        # only one obstacle allowed for the moment
        assert len(obstacles) == 1
        obst = obstacles[0]
        toporef = self.velocity.topology
        obst.discretize(toporef)
        self._on_proc = obst.on_proc[toporef]
        # mpi communicator
        #self._subcomm = obstacles[0].subcomm[self._topology]
        # Return the list of indices of points inside the obstacle
        return obst.ind[toporef]

    def _momentum(self, dt):
        # -- Integration over the obstacle --
        # the force has to be set to 0 before computation
        self.force[...] = 0.0
        for d in xrange(self._dim):
            # buff is initialized to component d of
            # the velocity, inside the obstacle and to
            # zero elsewhere.
            # For each area of the considered obstacle:
            for i in xrange(len(self._indices)):
                ind = self._indices[i]
                subshape = self.velocity.data[d][ind].shape
                lbuff = np.prod(subshape)
                buff = self._rwork[0][:lbuff].reshape(subshape)
                coeff = self._coeff[i] / (1. + self._coeff[i] * dt)
                buff[...] = coeff * self.velocity.data[d][ind]
                self.force[d] += npw.real_sum(buff)

        self.force *= self._dvol


class NocaForces(Forces):
    """
    Compute drag and lift using Noca's formula.
    See Noca99 or Plouhmans, 2002, Journal of Computational Physics
    The present class implements formula (52) of Plouhmans2002.
    Integral inside the obstacle is not taken into account.
    """

    __metaclass__ = ABCMeta

    def __init__(self, velocity, vorticity, nu, volume_of_control,
                 surfdir=None, **kwds):
        """
        Parameters
        -----------
        velocity : :class:`hysop.field.discrete.DiscreteField`
            the velocity field
        vorticity : :class:`hysop.field.discrete.DiscreteField`
            vorticity field inside the domain
        nu : double
            viscosity
        volume_of_control : :class:`~hysop.domain.subset.controlBox.ControlBox`
            a subset of the domain, on which forces will be computed,
            useful to reduce computational cost
        surfdir : python list, optional
            indices of the surfaces on which forces are computed,
            0, 1 = bottom/top faces in xdir, 2,3 in ydir ...
            Default = only surfaces normal to x direction.
        kwds : arguments passed to drag_and_lift.Forces base class.

        """
        # A volume of control, in which forces are computed
        self._voc = volume_of_control
        # discrete velocity field
        self.velocity = velocity
        # discrete vorticity field
        self.vorticity = vorticity

        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(NocaForces, self).__init__(variables=[velocity, vorticity],
                                         **kwds)
        # Coef in the Noca formula
        self._coeff = 1. / (self._dim - 1)
        # viscosity
        self.nu = nu

        # connect to noca's formula
        self._formula = self._noca
        # Set mpi comm
        self._subcomm = self._voc.subcomm[self._topology]

        if surfdir is None:
            surfdir = [0, 1]
        # Faces where integrals on surfaces are computed
        # order : [xmin, xmax, ymin, ymax, ...]
        # So if you need computation in x and z directions,
        # surfdir = [0, 1, 4, 5]
        self._surfdir = surfdir
        # buffers used to compute d/dt velocity
        self._previous_velo = [None] * self._dim * 2 * self._dim
        self._surf_buffer = None
        self._init_surf_buffers()

    def _init_indices(self, obstacles):
        """
        Compute a list of indices corresponding to points inside
        the volume of control minus those inside the obstacles

        Parameters
        ----------
        obstacles: list of :class:`~hysop.domain.subsets.Subset`
            obstacles in the flow

        Returns
        -------
        list of indices of points inside those obstacles

        """
        assert isinstance(self._voc, ControlBox)
        toporef = self.velocity.topology
        self._on_proc = self._voc.on_proc[toporef]
        # no obstacle in the box, just for test purpose.
        if obstacles is None or len(obstacles) == 0:
            return self._voc.ind[toporef]
        else:
            msg = 'obstacles arg must be a list.'
            assert isinstance(obstacles, list), msg
            for obs in obstacles:
                obs.discretize(toporef)
            # return indices of points inside the box, excluding points
            # inside the obstacles.
            return Subset.subtract_list_of_sets([self._voc], obstacles,
                                                toporef)

    def _update_surf_buffers(self):
        """
        Set local buffers values (self._previous_velo)
        previous_velo is used to compute dv/dt on the surfaces, so we need
        to save velocity at the current time step for computation at the next
        time step.
        """
        pass

    def _init_surf_buffers(self):
        """Allocate memory for local buffers (used to compute
        time derivative of the velocity on the surfaces of
        the volume of control)
        """
        # This Noca's formulation uses only 'gamma_common'
        # which require a buffer/direction of integration
        # of size shape(v) on the surface and unlocked.
        # Surface in the same direction can use the same buffer.
        toporef = self.velocity.topology
        subsize = 0
        for s_id in self._surfdir:
            surf = self._voc.surf[s_id]
            ind = surf.mesh[toporef].ind4integ
            subsize = np.maximum(self.velocity.data[0][ind].size, subsize)
        self._surf_buffer = npw.zeros(2 * subsize)

    def _compute_gamma_common(self, surf, res):
        """
        Computation of the part common to the 3 Noca's
        formulations for the local integral
        on a surface of the control box.

        Parameters
        ----------
        surf : :class:`~hysop.domain.subset.boxes.SubBox`
            The surface on which the integral is computed
        s_id : int
            index of the surface in buffer list
        res : np array
            in, out parameter

        Returns
        -------
        res
            value of the integral on surf

        Notes
        -----
        * res input will be completely erased and recomputed
        * this function uses self._previous_velo[s_id * self._dim] buffer
        which must be of shape equal to the resolution of the input surface.
        * finite differences are used to compute Laplacian and other
          derivatives.

        """
        res[...] = 0.0
        if not surf.on_proc[self._topology]:
            return res
        # Get indices for integration on the surface
        ind = surf.mesh[self._topology].ind4integ
        # i_n : normal dir
        # i_t : other dirs
        i_n = surf.n_dir
        i_t = surf.t_dir
        # coordinates of points in the surface (for integration)
        coords = surf.mesh[self._topology].coords4int
        # list of array for discrete velocity field
        vd = self.velocity.data
        normal = surf.normal

        # --- First part of the integral ---
        #  int(0.5 (uu)n - (nu)u)
        # i_n component
        subshape = vd[i_n][ind].shape
        subsize = vd[i_n][ind].size
        buff = self._surf_buffer[:subsize].reshape(subshape)
        for j in i_t:
            np.multiply(vd[j][ind], vd[j][ind], buff)
            res[i_n] += npw.real_sum(buff)
        np.multiply(vd[i_n][ind], vd[i_n][ind], buff)
        res[i_n] -= npw.real_sum(buff)
        res[i_n] *= 0.5 * normal

        # other components
        for j in i_t:
            np.multiply(vd[i_n][ind], vd[j][ind], buff)
            res[j] = - normal * npw.real_sum(buff)

        # --- Second part of integral on surface ---
        #  1/(dim - 1) * int( (nw)(x X u) - (nu)(x X w))
        x0 = coords[i_n].flat[0]
        buff2 = self._surf_buffer[subsize:2 * subsize].reshape(subshape)
        # Indices used for cross-product
        j1 = [YDIR, ZDIR, XDIR]
        j2 = [ZDIR, XDIR, YDIR]
        wd = self.vorticity.data
        for j in i_t:
            np.multiply(vd[j2[j]][ind], wd[j1[j]][ind], buff)
            np.multiply(vd[j1[j]][ind], wd[j2[j]][ind], buff2)
            np.subtract(buff, buff2, buff)
            res[j] += x0 * normal * self._coeff * npw.real_sum(buff)
            np.multiply(coords[j], buff, buff)
            res[i_n] -= self._coeff * normal * npw.real_sum(buff)

        # Last part
        # Update fd schemes in order to compute laplacian and other derivatives
        # only on the surface (i.e. for list of indices in sl)

        # function to compute the laplacian
        # of a scalar field. Default fd scheme.
        laplacian = Laplacian(topo=self._topology, indices=ind,
                              reduce_output_shape=True)
        for j in i_t:
            [buff] = laplacian(vd[j:j + 1], [buff])
            res[j] -= self._coeff * self.nu * normal * x0 * npw.real_sum(buff)
            np.multiply(coords[j], buff, buff)
            res[i_n] += self._coeff * self.nu * normal * npw.real_sum(buff)
        # function used to compute first derivative of
        # a scalar field in a given direction.
        # Default = FD_C_2. Todo : set this as an input method value.
        fd_scheme = FD_C_2(self._topology.mesh.space_step, ind,
                           reduce_output_shape=True)
        fd_scheme.compute(vd[i_n], i_n, buff)
        res[i_n] += 2.0 * normal * self.nu * npw.real_sum(buff)
        for j in i_t:
            fd_scheme.compute(vd[i_n], j, buff)
            res[j] += normal * self.nu * npw.real_sum(buff)
            fd_scheme.compute(vd[j], i_n, buff)
            res[j] += normal * self.nu * npw.real_sum(buff)

        return res

    @abstractmethod
    def _noca(self, dt):
        """Computes local values of the forces

        Parameters
        ----------
        dt : double
            current time step

        Returns
        -------
        array of double
            the local (i.e. current mpi process) forces
        """


class NocaI(NocaForces):
    """Noca, "Impulse Equation" from Noca99
    """
    def _set_work_arrays(self, rwork=None, iwork=None):
        """Set or check rwork.

        rwork will be required only in formulations where
        an integral over the volume of the control box is computed,
        Noca I and Noca II.
        """
        # !!! Velocity must be set before a call to this function
        # and so before base class initialization !!!

        # The only required buffer if for integral on the volume
        # of control.
        toporef = self.velocity.topology
        v_ind = self._voc.mesh[toporef].ind4integ
        shape_v = self.velocity.data[0][v_ind].shape
        # setup for rwork, iwork is useless.
        if rwork is None:
            # ---  Local allocation ---
            self._rwork = [npw.zeros(shape_v)]
        else:
            assert isinstance(rwork, list), 'rwork must be a list.'
            # --- External rwork ---
            self._rwork = rwork
            assert len(self._rwork) == 1
            assert self._rwork[0].shape == shape_v

    def _noca(self, dt):
        """
        Computes local values of the forces using formula 2.1
        ("Impulse Equation") from :cite:`Noca-1999`
        :parameter dt: double
            current time step

        Returns
        -------
        np array
            local (i.e. current mpi process) forces
        """
        # -- Integration over the volume of control --
        # -1/(N-1) . d/dt int(x ^ w)
        mesh = self._voc.mesh[self._topology]
        coords = mesh.coords4int
        ind = mesh.ind4integ
        if self._dim == 2:
            wz = self.vorticity.data[0]
            np.multiply(coords[YDIR], wz[ind], self._rwork[-1])
            self._buffer[0] = npw.real_sum(self._rwork[-1])
            np.multiply(coords[XDIR], wz[ind], self._rwork[-1])
            self._buffer[1] = -npw.real_sum(self._rwork[-1])
        elif self._dim == 3:
            self._rwork[-1][...] = 0.
            self._buffer[...] = Utils.sum_cross_product(coords,
                                                        self.vorticity.data,
                                                        ind, self._rwork[-1])
        self._buffer[...] *= self._dvol
        self.force[...] = -1. / dt * self._coeff * (self._buffer
                                                    - self._previous)
        # Update previous for next time step ...
        self._previous[...] = self._buffer[...]

        # -- Integrals on surfaces --
        # Only on surf. normal to dir in self._surfdir.
        for s_id in self._surfdir:
            s_x = self._voc.surf[s_id]
            i_t = s_x.t_dir
            # cell surface
            dsurf = npw.prod(self._topology.mesh.space_step[i_t])
            # The 'common' part (same in all Noca's formula)
            self._buffer = self._compute_gamma_common(s_x, self._buffer)
            self.force += self._buffer * dsurf


class NocaII(NocaForces):

    def _init_surf_buffers(self):
        """Allocate memory for local buffers (used to compute
        time derivative of the velocity on the surfaces of
        the volume of control)
        """
        # Buffers are :
        # - used in gamma momentum
        # - used in gamma common
        # - update with local velocity and locked till next gamma_momentum.
        # For each surface, a buffer is required for each velocity component
        # in tangential directions to the surface.
        # For example, in 3D, with integration only in xdir, 2 * 2 buffers are
        # required.

        toporef = self.velocity.topology
        subsize = 0
        for s_id in self._surfdir:
            surf = self._voc.surf[s_id]
            ind = surf.mesh[toporef].ind4integ
            subsize = np.maximum(self.velocity.data[0][ind].size, subsize)
        self._surf_buffer = npw.zeros(2 * subsize)
        # For each surface ...
        for s_id in self._surfdir:
            ind = self._voc.surf[s_id].mesh[self._topology].ind4integ
            shape = self.velocity.data[0][ind].shape
            i_t = self._voc.surf[s_id].t_dir
            # for each tangential direction ...
            for d in i_t:
                pos = s_id * self._dim + d
                self._previous_velo[pos] = npw.zeros(shape)

    def _update_surf_buffers(self):
        """Set local buffers values (self._previous_velo)
        previous_velo is used to compute dv/dt on the surfaces, so we need
        to save velocity at the current time step for computation at the next
        time step.
        After a call to this function, the buffer is "locked" until next call
        to _noca.
        """
        # Done only for surfaces on which integration is performed
        for s_id in self._surfdir:
            surf = self._voc.surf[s_id]
            if surf.on_proc[self._topology]:
                i_t = self._voc.surf[s_id].t_dir
                ind = self._voc.surf[s_id].mesh[self._topology].ind4integ
                for d in i_t:
                    pos = s_id * self._dim + d
                    self._previous_velo[pos][...] = self.velocity.data[d][ind]
                    npw.lock(self._previous_velo[pos])

    def _noca(self, dt):
        """
        Computes local values of the forces using formula 2.5
        ("Momentum Equation") from :cite:`Noca-1999`

        :parameter dt: double
            current time step

        Returns
        -------
        np array
            local (i.e. current mpi process) forces
        """
        # -- Integration over the volume of control --
        # -d/dt int(v)
        nbc = self.velocity.nb_components
        self._buffer[...] = \
            [self._voc.integrate_dfield_on_proc(self.velocity,
                                                component=d)
             for d in xrange(nbc)]
        self.force[...] = -1. / dt * (self._buffer - self._previous)

        # Update previous for next time step ...
        self._previous[...] = self._buffer[...]

        # -- Integrals on surfaces --
        # Only on surf. normal to dir in self._surfdir.
        for s_id in self._surfdir:
            s_x = self._voc.surf[s_id]
            if s_x.on_proc[self._topology]:
                i_t = s_x.t_dir
                # cell surface
                dsurf = npw.prod(self._topology.mesh.space_step[i_t])
                # First, part relative to "du/dt"
                self._buffer = self._compute_gamma_momentum(dt, s_x, s_id,
                                                            self._buffer)
                self.force += self._buffer * dsurf
                # Then the 'common' part (same in all Noca's formula)
                self._buffer = self._compute_gamma_common(s_x, self._buffer)
                self.force += self._buffer * dsurf
        # Prepare next step
        self._update_surf_buffers()

    def _compute_gamma_momentum(self, dt, surf, s_id, res):
        """
        Partial computation of gamma in Noca's "momentum formulation",
        on a surface of the control box.

        It corresponds to the terms in the second line of gamma_mom
        in formula 2.5 of :cite:`Noca-1999`.

        Parameters
        ----------
        dt: double
            time step
        surf : :class:`~hysop.domain.subset.boxes.SubBox`
            The surface on which the integral is computed
        s_id : int
            index of the surface in buffer list
        res : np array
            in, out parameter.


        Returns
        -------
        res
            value of the integral on surf

        Notes
        -----
        * res input will be completely erased and recomputed
        * this function uses self._previous_velo[s_id * self._dim + j] buffers,
        j = tangential dirs to the surface.
        which must be of shape equal to the resolution of the input surface.
        """
        res[...] = 0.

        # i_n : normal dir
        # i_t : other dirs
        i_n = surf.n_dir
        i_t = surf.t_dir
        coords = surf.mesh[self._topology].coords4int
        ind = surf.mesh[self._topology].ind4integ
        x0 = coords[i_n].flatten()
        # We want to compute:
        # res = -1/(d - 1) * integrate_on_surf(
        #         (x.du/dt)n - (x.n) du/dt)
        # n : normal, d : dimension of the domain, u : velocity

        # (x.du/dt)n - (x.n) du/dt
        coeff = self._coeff * surf.normal * 1. / dt
        for j in i_t:
            # compute d(velocity_it)/dt on surf in buff
            buff = self._previous_velo[s_id * self._dim + j]
            npw.unlock(buff)
            np.subtract(self.velocity.data[j][ind], buff, buff)
            res[j] = coeff * x0 * npw.real_sum(buff)
            np.multiply(coords[j], buff, buff)
            res[i_n] -= coeff * npw.real_sum(buff)
        return res


class NocaIII(NocaForces):

    def _init_surf_buffers(self):
        """Allocate memory for local buffers (used to compute
        time derivative of the velocity on the surfaces of
        the volume of control)
        """
        # Buffers are :
        # - used in gamma momentum and unlocked
        # - used in gamma flux
        # - update with local velocity and locked till next gamma_momentum.
        # For each surface, a buffer is required for each velocity component
        # For example, in 3D, with integration only in xdir, 2 * 3 buffers are
        # required.
        toporef = self.velocity.topology
        subsize = 0
        for s_id in self._surfdir:
            surf = self._voc.surf[s_id]
            ind = surf.mesh[toporef].ind4integ
            subsize = np.maximum(self.velocity.data[0][ind].size, subsize)
        self._surf_buffer = npw.zeros(2 * subsize)
        for s_id in self._surfdir:
            ind = self._voc.surf[s_id].mesh[self._topology].ind4integ
            shape = self.velocity.data[0][ind].shape
            i_n = self._voc.surf[s_id].n_dir
            pos = s_id * self._dim + i_n
            self._previous_velo[pos] = npw.zeros(shape)
            i_t = self._voc.surf[s_id].t_dir
            # for each tangential direction ...
            for d in i_t:
                pos = s_id * self._dim + d
                self._previous_velo[pos] = npw.zeros(shape)

    def _update_surf_buffers(self):
        """Set local buffers values (self._previous_velo).
        previous_velo is used to compute dv/dt on the surfaces, so we need
        to save velocity at the current time step for computation at the next
        time step.
        After a call to this function, the buffer is "locked" until next call
        to _noca.
        """
        for s_id in self._surfdir:
            surf = self._voc.surf[s_id]
            if surf.on_proc[self._topology]:
                ind = self._voc.surf[s_id].mesh[self._topology].ind4integ
                i_n = self._voc.surf[s_id].n_dir
                pos = s_id * self._dim + i_n
                # update v component normal to surf
                self._previous_velo[pos][...] = self.velocity.data[i_n][ind]
                # lock
                npw.lock(self._previous_velo[pos])
                i_t = self._voc.surf[s_id].t_dir
                # for each tangential direction ...
                for d in i_t:
                    pos = s_id * self._dim + d
                    # update v components tangent to the surface
                    self._previous_velo[pos][...] = self.velocity.data[d][ind]
                    # lock
                    npw.lock(self._previous_velo[pos])

    def _noca(self, dt):
        """
        Computes local values of the forces using formula 2.10
        ("Flux Equation") from :cite:`Noca-1999`

        :parameter dt: double
            current time step

        Returns
        -------
        np array
            local (i.e. current mpi process) forces
        """
        self.force[...] = 0.
        # -- Integrals on surfaces --
        # Only on surf. normal to dir in self._surfdir.
        for s_id in self._surfdir:
            s_x = self._voc.surf[s_id]
            if s_x.on_proc[self._topology]:
                i_t = s_x.t_dir
                # cell surface
                dsurf = npw.prod(self._topology.mesh.space_step[i_t])
                # First, part relative to "du/dt"
                self._buffer = self._compute_gamma_flux(dt, s_x, s_id,
                                                    self._buffer)
                self.force += self._buffer * dsurf
                self._buffer = self._compute_gamma_common(s_x, self._buffer)
                self.force += self._buffer * dsurf
        self._update_surf_buffers()

    def _compute_gamma_flux(self, dt, surf, s_id, res):
        """
        Partial computation of gamma in Noca's "flux formulation",
        on a surface of the control box.

        It corresponds to the terms in the second line of gamma_flux
        in formula 2.10 of :cite:`Noca-1999`.

        Parameters
        ----------
        dt: double
            time step
        surf : :class:`~hysop.domain.subset.boxes.SubBox`
            The surface on which the integral is computed
        s_id : int
            index of the surface in buffer list
        res : np array
            in, out parameter.

        Returns
        -------
        res
            value of the integral on surf

        Notes
        -----
        * this function uses self._previous_velo[s_id * self._dim + j] buffers,
        j = xrange(domain.dimension)
        which must be of shape equal to the resolution of the input surface.
        """
        res[...] = 0.
        # i_n : normal dir
        # i_t : other dirs
        i_n = surf.n_dir
        i_t = surf.t_dir
        coords = surf.mesh[self._topology].coords4int
        ind = surf.mesh[self._topology].ind4integ
        x0 = coords[i_n].flat[0]
        # We want to compute:
        # res = -1/(d - 1) * integrate_on_surf(
        #         (x.du/dt)n - (x.n) du/dt
        #         + (d-1) * (du/dt.n).x)
        # n : normal, d : dimension of the domain, u : velocity

        # buff = d(velocity_in) /dt on surf
        buff = self._previous_velo[s_id * self._dim + i_n]
        npw.unlock(buff)
        np.subtract(self.velocity.data[i_n][ind], buff, buff)
        coeff = surf.normal * 1. / dt
        # -(n.du/dt).x
        res[i_n] = - coeff * x0 * npw.real_sum(buff)
        for j in i_t:
            res[j] = - coeff * npw.real_sum(coords[j] * buff)

        # (x.du/dt)n - (x.n) du/dt
        coeff = self._coeff * surf.normal * 1. / dt
        for j in i_t:
            # compute d(velocity_it)/dt on surf in buff
            buff = self._previous_velo[s_id * self._dim + j]
            npw.unlock(buff)
            np.subtract(self.velocity.data[j][ind], buff, buff)
            res[j] += coeff * x0 * npw.real_sum(buff)
            np.multiply(coords[j], buff, buff)
            res[i_n] -= coeff * npw.real_sum(buff)
        return res
