"""
@file spectrum.py
"""
from hysop.operator.computational import Computational
from hysop.operator.discrete.spectrum import FFTSpectrum
from hysop.constants import debug
from hysop.operator.continuous import opsetup


class Spectrum(Computational):
    """
    Fourier spectrum computation of a scalar field.
    """

    def __init__(self, field, prefix=None, **kwds):
        """
        Constructor for Spectrum operator
        @param[in] field : field to compute
        """
        super(Spectrum, self).__init__(variables=[field], **kwds)
        self.field = field
        self.input = [field]
        self._prefix = prefix

    def discretize(self):
        super(Spectrum, self)._fftw_discretize()

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        self.discrete_op = FFTSpectrum(self.discreteFields[self.field],
                                       method=self.method,
                                       prefix=self._prefix)
        self._is_uptodate = True
