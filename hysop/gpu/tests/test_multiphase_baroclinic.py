
"""Testing baroclinic right hand side vector computing"""
from hysop.problem.simulation import Simulation
from hysop.tools.parameters import Discretization, MPIParams
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.operator.multiphase_baroclinic_rhs import MultiphaseBaroclinicRHS
from hysop.numerics.finite_differences import FD_C_4, FD_C_2
import hysop.tools.numpywrappers as npw
import numpy as np
from hysop.methods_keys import Support, SpaceDiscretisation, ExtraArgs
from hysop.constants import HYSOP_REAL

def test_baroclinic_rhs():
    def func(res, x, y, z, t=0):
        res[0][...] = np.cos(2. * np.pi * x) * \
                      np.sin(2. * np.pi * y) * \
                      np.cos(2. * np.pi * z)
        return res

    def grad_func(res, x, y, z, t=0):
        res[0][...] = -2 * np.pi * np.sin(2. * np.pi * x) * \
                      np.sin(2. * np.pi * y) * \
                      np.cos(2. * np.pi * z)
        res[1][...] = np.cos(2. * np.pi * x) * \
                      2 * np.pi * np.cos(2. * np.pi * y)* \
                      np.cos(2. * np.pi * z)
        res[2][...] = np.cos(2. * np.pi * x) * \
                      np.sin(2. * np.pi * y) * \
                      -2 * np.pi * np.sin(2. * np.pi * z)
        return res

    def vfunc(res, x, y, z, t=0):
        res[0][...] = np.sin(2. * np.pi * x)
        res[1][...] = 2.5 * np.cos(2. * np.pi * y)
        res[2][...] = -2. * np.sin(2. * np.pi * z)
        return res

    call_operator(func, grad_func, vfunc)


def test_baroclinic_rhs_nonperiodic():
    def func(res, x, y, z, t=0):
        res[0][...] = x * y * z
        return res

    def grad_func(res, x, y, z, t=0):
        """Values obtained for non periodic function with periodic FD scheme"""
        res[2][...] = x * y
        res[2][:,:,np.bitwise_or(z<2./256., z>253./256.)[0,0,:]] = x * y
        res[2][:,:,np.bitwise_or(z<1./256., z>254./256.)[0,0,:]] = -127. * x * y
        res[2][:,:,np.bitwise_or(z<2./256., z>253./256.)[0,0,:]] = 22.333333333333332 * x * y
        res[2][:,:,np.bitwise_or(z<1./256., z>254./256.)[0,0,:]] = -148.33333333333334 * x * y

        res[1][...] = x * z
        res[1][:,np.bitwise_or(y<2./256., y>253./256.)[0,:,0],:] = x * z
        res[1][:,np.bitwise_or(y<1./256., y>254./256.)[0,:,0],:] = -127. * x * z
        res[1][:,np.bitwise_or(y<2./256., y>253./256.)[0,:,0],:] = 22.333333333333332 * x * z
        res[1][:,np.bitwise_or(y<1./256., y>254./256.)[0,:,0],:] = -148.33333333333334 * x * z

        res[0][...] = y * z
        res[0][np.bitwise_or(x<2./256., x>253./256.)[:,0,0],:,:] = y * z
        res[0][np.bitwise_or(x<1./256., x>254./256.)[:,0,0],:,:] = -127. * y * z
        res[0][np.bitwise_or(x<2./256., x>253./256.)[:,0,0],:,:] = 22.333333333333332 * y * z
        res[0][np.bitwise_or(x<1./256., x>254./256.)[:,0,0],:,:] = -148.33333333333334 * y * z
        return res

    def vfunc(res, x, y, z, t=0):
        res[0][...] = np.sin(2. * np.pi * x)
        res[1][...] = 2.5 * np.cos(2. * np.pi * y)
        res[2][...] = -2. * np.sin(2. * np.pi * z)
        return res

    call_operator(func, grad_func, vfunc)


def call_operator(func, grad_func, vfunc):
    """Call the baroclinic rhs operator from given initialization functions"""
    simu = Simulation(tinit=0.0, tend=1.0, timeStep=0.1, iterMax=1)
    box = Box()
    rhs = Field(box, is_vector=True, name='rhs')
    gradp = Field(box, is_vector=True, formula=vfunc, name='gradp')
    rho = Field(box, is_vector=False, formula=func, name='rho')
    gradrho = Field(box, is_vector=True, formula=grad_func, name='gradrho')
    gradp_fine = Field(box, is_vector=True, formula=vfunc, name='fine')
    true_rhs = Field(box, is_vector=True, name='ref')
    d_fine = Discretization([257, 257, 257])
    d_coarse = Discretization([129, 129, 129], ghosts=[2, 2, 2])
    op = MultiphaseBaroclinicRHS(rhs, rho, gradp,
                                 variables={rhs: d_fine,
                                            gradp: d_coarse,
                                            rho: d_fine},
                                 method={Support: 'gpu',
                                         SpaceDiscretisation: FD_C_4,
                                         ExtraArgs: {'density_func': 'x', }})
    op.discretize()
    op.setup()
    topo_coarse = op.discreteFields[gradp].topology
    topo_fine = op.discreteFields[rho].topology
    d_rhs = rhs.discreteFields[topo_fine]
    d_gradp = gradp.discreteFields[topo_coarse]
    d_rho = rho.discreteFields[topo_fine]
    rhs.initialize(topo=topo_fine)
    gradp.initialize(topo=topo_coarse)
    rho.initialize(topo=topo_fine)
    op.apply(simu)
    d_rhs.toHost()
    d_rhs.wait()

    gradrho.initialize(topo=topo_fine)
    d_gradrho = gradrho.discreteFields[topo_fine]
    gradp_fine.initialize(topo=topo_fine)
    d_gradp_fine = gradp_fine.discreteFields[topo_fine]
    true_rhs.initialize(topo=topo_fine)
    d_true_rhs = true_rhs.discreteFields[topo_fine]
    d_true_rhs[0] = d_gradrho[2] * d_gradp_fine[1] - \
        d_gradrho[1] * d_gradp_fine[2]
    d_true_rhs[1] = d_gradrho[0] * d_gradp_fine[2] - \
        d_gradrho[2] * d_gradp_fine[0]
    d_true_rhs[2] = d_gradrho[1] * d_gradp_fine[0] - \
        d_gradrho[0] * d_gradp_fine[1]

    max_val = [np.max(np.abs(r)) for r in d_true_rhs]

    print np.max(np.abs(d_true_rhs[0]-d_rhs[0]) / max_val[0])
    print np.where((np.abs(d_true_rhs[0]-d_rhs[0]) / max_val[0]) > 0.4)
    assert np.allclose(d_rhs[0] / max_val[0], d_true_rhs[0] / max_val[0],
                       atol=1e-8 if HYSOP_REAL != np.float32 else 5e-4)
    assert np.allclose(d_rhs[1] / max_val[1], d_true_rhs[1] / max_val[1],
                       atol=1e-8 if HYSOP_REAL != np.float32 else 5e-4)
    assert np.allclose(d_rhs[2] / max_val[2], d_true_rhs[2] / max_val[2],
                       atol=1e-8 if HYSOP_REAL != np.float32 else 5e-4)

