# -*- coding: utf-8 -*-
"""
@file operator/discrete/velocity_correction.py

Correction of the velocity field.
"""

from hysop.constants import debug
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.fields.variable_parameter import VariableParameter
from hysop.tools.profiler import profile
import hysop.tools.numpywrappers as npw
from hysop.constants import XDIR, YDIR, ZDIR


class VelocityCorrection_D(DiscreteOperator):
    """
    The velocity field is corrected after solving the
    Poisson equation. For more details about calculations,
    see the "velocity_correction.pdf" explanations document
    in Docs.
    """

    @debug
    def __init__(self, velocity, vorticity, req_flowrate, cb, **kwds):
        """
        @param[in, out] velocity field to be corrected
        @param[in] vorticity field used to compute correction
        @param[in] req_flowrate : required value for the
        flowrate (VariableParameter object)
        @param[in] surf : surface (hysop.domain.obstacle.planes.SubPlane)
        used to compute reference flow rates. Default = surface at x_origin,
        normal to x-dir.
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(VelocityCorrection_D, self).__init__(
            variables=[velocity, vorticity], **kwds)
        ## velocity discrete field
        self.velocity = velocity
        ## vorticity discrete field
        self.vorticity = vorticity
        ## domain dimension
        self.dim = self.velocity.domain.dimension
        # If 2D problem, vorticity must be a scalar
        if self.dim == 2:
            assert self.vorticity.nb_components == 1
        assert (self.dim >= 2),\
            "Wrong problem dimension: only 2D and 3D cases are implemented."

        self.input = self.variables
        self.output = [self.velocity]
        ## A reference topology
        self.topo = self.velocity.topology
        ## Volume of control
        self.cb = cb
        self.cb.discretize(self.topo)
        # A reference surface, i.e. input surface for flow in x direction
        self._in_surf = cb.surf[XDIR]

        sdirs = self._in_surf.t_dir
        # Compute 1./ds and 1./dv ...
        cb_length = self.cb.real_length[self.topo]
        self._inv_ds = 1. / npw.prod(cb_length[sdirs])
        self._inv_dvol = 1. / npw.prod(cb_length)
        ## Expected value for the flow rate through self.surfRef
        self.req_flowrate = req_flowrate
        assert isinstance(self.req_flowrate, VariableParameter),\
            "the required flowrate must be a VariableParameter object."
        ## The correction that must be applied on each
        ## component of the velocity.
        self.velocity_shift = npw.zeros(self.dim)
        nbf = self.velocity.nb_components + self.vorticity.nb_components
        # temp buffer, used to save flow rates and mean
        # values of vorticity
        self.rates = npw.zeros(nbf)
        self.req_flowrate_val = None

        spaceStep = self.topo.mesh.space_step
        lengths = self.topo.domain.length
        self.coeff_mean = npw.prod(spaceStep) / npw.prod(lengths)
        x0 = self._in_surf.real_orig[self.topo][XDIR]
        # Compute X - X0, x0 being the coordinate of the 'entry'
        # surface for the flow.
        self.x_coord = self.topo.mesh.coords[XDIR] - x0

    def computeCorrection(self):
        """
        Compute the required correction for the current state
        but do not apply it onto velocity.
        """
        ## Computation of the flowrates evaluated from
        ## current (ie non corrected) velocity
        nbf = self.velocity.nb_components + self.vorticity.nb_components
        localrates = npw.zeros((nbf))
        for i in xrange(self.velocity.nb_components):
            localrates[i] = self._in_surf.integrate_dfield_on_proc(
                self.velocity, component=i)
        start = self.velocity.nb_components
        ## Integrate vorticity over the whole domain
        for i in xrange(self.vorticity.nb_components):
            localrates[start + i] = self.cb.integrate_dfield_on_proc(
                self.vorticity, component=i)

        # MPI reduction for rates
        # rates = [flowrate[X], flowrate[Y], flowrate[Z],
        #          vort_mean[X], ..., vort_mean[Z]]
        # or (in 2D) = [flowrate[X], flowrate[Y], vort_mean]
        self.rates[...] = 0.0
        self.velocity.topology.comm.Allreduce(localrates, self.rates)

        self.rates[:start] *= self._inv_ds
        self.rates[start:] *= self._inv_dvol
 
        # Set velocity_shift == [Vx_shift, vort_mean[Y], vort_mean[Z]]
        # or (in 2D) velocity_shift == [Vx_shift, vort_mean]
        # Velocity shift for main dir component
        self.velocity_shift[XDIR] = self.req_flowrate_val[XDIR]\
            - self.rates[XDIR]
        # Shifts in other directions depend on x coord
        # and will be computed during apply.

    @debug
    @profile
    def apply(self, simulation=None):
        # the required flowrate value is updated (depending on time)
        self.req_flowrate.update(simulation)

        # warning : the flow rate value is divided by surf.
        self.req_flowrate_val = self.req_flowrate[self.req_flowrate.name] \
            * self._inv_ds
        # Computation of the required velocity shift
        # for the current state
        self.computeCorrection()
        iCompute = self.topo.mesh.iCompute

        # Apply shift to velocity
        self.velocity[XDIR][iCompute] += self.velocity_shift[XDIR]
        start = self.velocity.nb_components
        # reminder : self.rates =[vx_shift, flowrates[Y], flowrate[Z],
        #                         vort_mean[X], vort_mean[Y], vort_mean[Z]]
        # or (in 2D) [vx_shift, flowrates[Y], vort_mean]
        vort_mean = self.rates[start:]
        ite = simulation.currentIteration
        if self._writer is not None and self._writer.do_write(ite):
            self._writer.buffer[0, 0] = simulation.time
            self._writer.buffer[0, 1] = ite
            self._writer.buffer[0, 2:] = vort_mean[...]
            self._writer.write()

        if self.dim == 2:
            # Correction of the Y-velocity component
            self.velocity[YDIR][...] += self.req_flowrate_val[YDIR] + \
                vort_mean[XDIR] * self.x_coord - self.rates[YDIR]

        elif self.dim == 3:
            # Correction of the Y and Z-velocity components
            self.velocity[YDIR][...] += self.req_flowrate_val[YDIR] + \
                vort_mean[ZDIR] * self.x_coord - self.rates[YDIR]
            self.velocity[ZDIR][...] += self.req_flowrate_val[ZDIR] - \
                vort_mean[YDIR] * self.x_coord - self.rates[ZDIR]
