# -*- coding: utf-8 -*-
"""
@file energy_enstrophy.py
Compute Energy and Enstrophy
"""
from hysop.operator.discrete.energy_enstrophy import EnergyEnstrophy as DEE
from hysop.operator.computational import Computational
from hysop.operator.continuous import opsetup


class EnergyEnstrophy(Computational):
    """
    Computes enstrophy and the kinetic energy
    \f{eqnarray*}
    enstrophy = \frac{1}{\Omega}\int_\Omega \omega^2 d\Omega
    \f} with \f$\Omega\f$ the volume or surface of the physical domain
    \f$\omega\f$ the vorticity and
    \f{eqnarray*}
    energy = \frac{1}{2\Omega}\int_\Omega v^2 d\Omega
    \f}
    """

    def __init__(self, velocity, vorticity, is_normalized=True, **kwds):
        """
        Constructor.
        @param velocity field
        @param vorticity field
        @param isNormalized : boolean indicating whether the enstrophy
        and energy values have to be normalized by the domain lengths.

        Default file name = 'energy_enstrophy.dat'
        See hysop.tools.io_utils.Writer for details
        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(EnergyEnstrophy, self).__init__(variables=[velocity, vorticity],
                                              **kwds)
        ## velocity field
        self.velocity = velocity
        ## vorticity field
        self.vorticity = vorticity
        ## are the energy end enstrophy values normalized by domain lengths ?
        self.is_normalized = is_normalized
        ## self._buffer_1 = 0.
        ## self._buffer_2 = 0.
        self.input = [velocity, vorticity]
        self.output = []

    def get_work_properties(self):
        if not self._is_discretized:
            msg = 'The operator must be discretized '
            msg += 'before any call to this function.'
            raise RuntimeError(msg)
        vd = self.discreteFields[self.velocity]
        wd = self.discreteFields[self.vorticity]
        v_ind = vd.topology.mesh.iCompute
        w_ind = wd.topology.mesh.iCompute
        shape_v = vd[0][v_ind].shape
        shape_w = wd[0][w_ind].shape
        if shape_v == shape_w:
            return {'rwork': [shape_v], 'iwork': None}
        else:
            return {'rwork': [shape_v, shape_w], 'iwork': None}

    @opsetup
    def setup(self, rwork=None, iwork=None):
        if not self._is_uptodate:

            self.discrete_op = DEE(self.discreteFields[self.velocity],
                                        self.discreteFields[self.vorticity],
                                        self.is_normalized,
                                        rwork=rwork)
            # Output setup
            self._set_io('energy_enstrophy', (1, 3))
            self.discrete_op.setWriter(self._writer)
            self._is_uptodate = True

    def energy(self):
        """
        Return last computed value of the energy
        """
        return self.discrete_op.energy

    def enstrophy(self):
        """
        Return last computed value of the enstrophy
        """
        return self.discrete_op.enstrophy
