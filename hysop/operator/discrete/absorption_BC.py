# -*- coding: utf-8 -*-
"""
@file operator/discrete/absorption_BC.py

Operator to kill the vorticity at the outlet boundary 
(i.e. removal of the periodic BC in the flow direction 
by vorticity absorption in order to set the far field 
velocity to u_inf at the inlet)
"""

from hysop.constants import debug, np
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.fields.variable_parameter import VariableParameter
from hysop.tools.profiler import profile
import hysop.tools.numpywrappers as npw
from hysop.constants import XDIR, YDIR, ZDIR


class AbsorptionBC_D(DiscreteOperator):
    """
    The periodic boundary condition is modified at the outlet
    in the flow direction in order to discard 
    in the downstream region the eddies coming 
    periodically from the outlet. 
    The vorticity absorption conserves div(omega)=0.
    The far field velocity is set to u_inf at the inlet.
    """

    @debug
    def __init__(self, velocity, vorticity, req_flowrate, 
                 x_coords_absorp, cb, **kwds):
        """
        @param[in] velocity field
        @param[in, out] vorticity field to absorbe
        @param[in] req_flowrate : required value for the flowrate 
        (used to set the u_inf velocity value at the inlet)
        @param x_coords_absorp : array containing the x-coordinates delimitating 
        the absorption domain ([x_beginning, x_end])
        @param cb : control box for inlet surface computation
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(AbsorptionBC_D, self).__init__(
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
        self.output = [self.vorticity]
        ## A reference topology
        self.topo = self.vorticity.topology
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
        self.req_flowrate_val = None
        ## x-coordinates delimitating the absorption band at the outlet
        self.x_coords_absorp = x_coords_absorp

        ## setup for the absorption filter definition
        self.xb = self.x_coords_absorp[0]
        self.xe = self.x_coords_absorp[1]
        self.xc = self.xb + (self.xe - self.xb) / 2.0
        self.eps = 10.0
        self.coeff = 1.0 / (np.tanh(self.eps * (self.xb - self.xc)) - 
                            np.tanh(self.eps * (self.xe - self.xc)))
        self.coords = self.topo.mesh.coords

    @debug
    @profile
    def apply(self, simulation=None):
        # the required flowrate value is updated (depending on time)
        self.req_flowrate.update(simulation)
        # \warning : the flow rate value is divided by area of input surf.
        self.req_flowrate_val = self.req_flowrate[self.req_flowrate.name] \
            * self._inv_ds

        # Definition of the filter function (for smooth vorticity absorption)
        self._filter = npw.ones_like(self.vorticity.data[0])
        indFilter = np.where(np.logical_and(self.coords[0][:,0,0] >= self.xb, 
                                            self.coords[0][:,0,0] <= self.xe))
#        indFilterZero = np.where(self.coords[0][:,0,0] > self.xe)
        FiltFormula = np.tanh(self.eps * (self.coords[0][indFilter] - 
                                          self.xc))
        FiltFormula -= np.tanh(self.eps * (self.xe - self.xc))
        FiltFormula *= self.coeff
        self._filter[indFilter,:,:] = FiltFormula
#        self._filter[indFilterZero] = 0.0

        # Beginning of divergence free vorticity absorption 
        # for non-periodic inlet BC
        for d in xrange(self.vorticity.nb_components):
            self.vorticity[d][...] *= self._filter[...]

        # Definition of the X-derivative of the filter function
        self._filter = npw.zeros_like(self.vorticity.data[0])
        indFilter = np.where(np.logical_and(self.coords[0][:,0,0] >= self.xb, 
                                            self.coords[0][:,0,0] <= self.xe))
        FiltFormula = self.eps * (1.0 - np.tanh(self.eps * 
                                               (self.coords[0][indFilter] - 
                                                self.xc)) ** 2)
        FiltFormula *= self.coeff
        self._filter[indFilter] = FiltFormula

        # End of divergence free vorticity absorption for non-periodic inlet BC
        self.vorticity.data[YDIR][...] += (- self._filter[...] *
                                           self.velocity[ZDIR][...]) + \
                                          (self._filter[...] *
                                           self.req_flowrate_val[ZDIR])
        self.vorticity.data[ZDIR][...] += (self._filter[...] *
                                           self.velocity[YDIR][...]) - \
                                          (self._filter[...] *
                                           self.req_flowrate_val[YDIR])
