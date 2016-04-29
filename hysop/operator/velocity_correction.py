# -*- coding: utf-8 -*-
"""
@file operator/velocity_correction.py

Operator to shift velocity to fit with a required input flowrate.

"""
from hysop.constants import debug
from hysop.operator.discrete.velocity_correction import VelocityCorrection_D
from hysop.operator.computational import Computational
from hysop.domain.control_box import ControlBox
from hysop.operator.continuous import opsetup


class VelocityCorrection(Computational):
    """
    The velocity field is corrected after solving the
    Poisson equation. For more details about calculations,
    see the "velocity_correction.pdf" explanations document
    in Docs.
    """

    @debug
    def __init__(self, velocity, vorticity, req_flowrate, **kwds):
        """
        Corrects the values of the velocity field after
        solving Poisson equation in order to prescribe proper
        mean flow and ensure the desired inlet flowrate.

        @param[in, out] velocity field to be corrected
        @param[in] vorticity field used to compute correction
        @param resolutions : grid resolutions of velocity and vorticity
        @param[in] req_flowrate : required value for the flowrate
        @param topo : a predefined topology to discretize velocity/vorticity
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(VelocityCorrection, self).__init__(variables=[velocity,
                                                            vorticity], **kwds)
        ## velocity variable (vector)
        self.velocity = velocity
        ## velocity variable
        self.vorticity = vorticity

        self.input = [self.velocity, self.vorticity]
        self.output = [self.velocity]
        ## Expected value for the flow rate through input surface
        self.req_flowrate = req_flowrate
        dom = self.velocity.domain
        self.cb = ControlBox(origin=dom.origin, length=dom.length,
                             parent=dom)
        ## Extra parameters that may be required for discrete operator
        ## (at the time, only io_params)
        self.config = kwds

    def discretize(self):
        super(VelocityCorrection, self)._standard_discretize()
        assert self._single_topo, 'Multi-resolution case is not allowed.'

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:
            self.discrete_op =\
                VelocityCorrection_D(self.discreteFields[self.velocity],
                                     self.discreteFields[self.vorticity],
                                     self.req_flowrate, self.cb, rwork=rwork,
                                     iwork=iwork)
            # Output setup
            self._set_io('velocity_correction', (1, 2 + self.domain.dimension))
            self.discrete_op.setWriter(self._writer)
            self._is_uptodate = True

    def computeCorrection(self):
        """
        Compute the required correction for the current state
        but do not apply it onto velocity.
        """
        self.discrete_op.computeCorrection()
