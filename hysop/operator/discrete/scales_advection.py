# -*- coding: utf-8 -*-
"""
@file scales_advection.py
Discrete Advection operator based on scales library (Jean-Baptiste)

"""
try:
    from hysop.f2hysop import scales2py
except ImportError:
    from hysop.fakef2py import scales2py
from hysop.operator.discrete.discrete import DiscreteOperator
from hysop.methods_keys import MultiScale
from hysop.constants import debug
import math
ceil = math.ceil


class ScalesAdvection(DiscreteOperator):
    """
    Advection process, based on scales library.
    """

    @debug
    def __init__(self, velocity, advected_fields, **kwds):
        """
        Constructor.
        @param velocity discrete field
        @param advected_fields : list of discrete fields to be advected
        """
        ## Advection velocity
        self.velocity = velocity
        if 'variables' in kwds:
            assert advected_fields is None, 'too many input arguments.'
            self.advected_fields = kwds['variables'].keys()
            kwds['variables'][self.velocity] = kwds['discretization']
            kwds.pop('discretization')
            super(ScalesAdvection, self).__init__(**kwds)
        else:
            v = [self.velocity]
            if isinstance(advected_fields, list):
                self.advected_fields = advected_fields
            else:
                self.advected_fields = [advected_fields]
            v += self.advected_fields
            super(ScalesAdvection, self).__init__(variables=v, **kwds)

        self.input = [self.velocity]
        self.output = self.advected_fields

        # Scales functions for each field (depending if vector)
        self._scales_func = []
        isMultiscale = self.method[MultiScale] is not None
        for adF in self.advected_fields:
            if adF.nb_components == 3:
                if isMultiscale:
                    # 3D interpolation of the velocity before advection
                    self._scales_func.append(
                        scales2py.solve_advection_inter_basic_vect)
                    # Other interpolation only 2D interpolation first and
                    # 1D interpolations before advections in each direction
                    # (slower than basic): solve_advection_inter
                else:
                    self._scales_func.append(scales2py.solve_advection_vect)
            else:
                if isMultiscale:
                    self._scales_func.append(
                        scales2py.solve_advection_inter_basic)
                else:
                    self._scales_func.append(scales2py.solve_advection)

    @debug
    def apply(self, simulation=None):
        assert simulation is not None, \
            "Missing simulation value for computation."

        dt = simulation.timeStep
        # Call scales advection
        for adF, fun in zip(self.advected_fields, self._scales_func):
            adF = fun(dt, self.velocity.data[0],
                      self.velocity.data[1],
                      self.velocity.data[2],
                      *adF)

    def finalize(self):
        """
        \todo check memory deallocation in scales???
        """
        DiscreteOperator.finalize(self)
