# -*- coding: utf-8 -*-
"""Operator to kill the vorticity at the outlet boundary
(i.e. removal of the periodic BC in the flow direction
by vorticity absorption in order to set the far field
velocity to u_inf at the inlet)

"""
from hysop.constants import debug
from hysop.operator.discrete.absorption_BC import AbsorptionBC_D
from hysop.operator.computational import Computational
from hysop.domain.control_box import ControlBox
from hysop.operator.continuous import opsetup


class AbsorptionBC(Computational):
    """
    The periodic boundary condition is modified at the outlet
    in the flow direction in order to discard 
    in the dowstream region the eddies coming 
    periodically from the oulet. 
    The far field velocity is set to u_inf at the inlet.
    """

    @debug
    def __init__(self, velocity, vorticity, req_flowrate,
                 x_coords_absorp, **kwds):
        """
        @param[in] velocity field
        @param[in, out] vorticity field to absorbe
        @param[in] req_flowrate : required value for the flowrate 
        (used to set the u_inf velocity value at the inlet)
        @param x_coords_absorp : array containing the x-coordinates delimitating 
        the absorption domain ([x_beginning, x_end])
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(AbsorptionBC, self).__init__(variables=[velocity,
                                                        vorticity], **kwds)
        ## velocity variable
        self.velocity = velocity
        ## vorticity variable
        self.vorticity = vorticity

        self.input = [self.velocity, self.vorticity]
        self.output = [self.vorticity]
        ## Expected value for the flow rate through input surface
        self.req_flowrate = req_flowrate
        ## x-coordinates delimitating the absorption band at the outlet
        self.x_coords_absorp = x_coords_absorp
        dom = self.velocity.domain
        self.cb = ControlBox(origin=dom.origin, length=dom.length,
                             parent=dom)
        ## Extra parameters that may be required for discrete operator
        ## (at the time, only io_params)
        self.config = kwds

    def discretize(self):
        super(AbsorptionBC, self)._standard_discretize()
        assert self._single_topo, 'Multi-resolution case is not allowed.'

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:
            self.discrete_op =\
                AbsorptionBC_D(self.discreteFields[self.velocity],
                               self.discreteFields[self.vorticity],
                               req_flowrate=self.req_flowrate, 
                               x_coords_absorp=self.x_coords_absorp, 
                               cb=self.cb, rwork=rwork, iwork=iwork)
            # Output setup
            self._set_io('absorption_BC', (1, 2 + self.domain.dimension))
            self.discrete_op.setWriter(self._writer)
            self._is_uptodate = True
