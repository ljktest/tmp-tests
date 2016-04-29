from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.spectrum import Spectrum
from hysop.tools.parameters import Discretization
from hysop.problem.simulation import Simulation
from hysop.operator.hdf_io import HDF_Writer
import numpy as np
from hysop.tools.io_utils import IO
pi = np.pi
sin = np.sin
cos = np.cos


def computeScal(res, x, y, z, t):
    res[0][...] = z * sin((3*pi*x)*(2*pi*y))
    return res


def test_spectrum():
    dom = Box(origin=[0., 0., 0.], length=[1., 1., 1.])
    field = Field(domain=dom, name='Field',
                  is_vector=False, formula=computeScal)
    d3D = Discretization([257, 257, 257])

    op = Spectrum(field, discretization=d3D)
    op.discretize()
    op.setup()
    topo = op.discreteFields[field].topology
    field.initialize(topo=topo)
    simu = Simulation(nbIter=1)
    simu.initialize()
    op.apply(simu)


