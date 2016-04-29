"""
@file remeshing.py
"""
from hysop.constants import np, HYSOP_INDEX
from hysop.numerics.method import NumMethod
import hysop.tools.numpywrappers as npw


class Remeshing(NumMethod):
    """Remshing"""

    def __init__(self, kernel, dim, topo, d, work, iwork):
        """
        Create a remeshing numeric method based on given formula.
        @param dim : problem dimension
        @param dx : mesh space step
        @param origin : mesh lower point
        @param formula : Remeshing formula to use.
        @param work : Work arrays (floats)
        @param iwork : Work arrays (integers)
        work must be a list containing (2 elements):
          - numpy float array like y[0] (used to store particles contributions)
          - numpy float array like y[0] (used to store distance to grid points)
        iwork must be a list containing (dim elements( in the order):):
          - numpy integer array of shape like y[0]
          - numpy integer array of shape like y[0] (if in 2D and 3D)
          - numpy integer array of shape like y[0] (if in 3D)
        Availables formulas :
          - 'l2_1' : Labmda2,1 : (=M'4) 4 point formula, C1 regularity
          - 'l2_2' : Labmda2,2 : 4 point formula, C2 regularity
          - 'l4_2' : Labmda4,2 : (=M'6) 6 point formula, C2 regularity
          - 'l4_3' : Labmda4,3 : 6 point formula, C3 regularity
          - 'l4_4' : Labmda4,4 : 6 point formula, C4 regularity
          - 'l6_3' : Labmda6,3 : 8 point formula, C3 regularity
          - 'l6_4' : Labmda6,4 : 8 point formula, C4 regularity
          - 'l6_5' : Labmda6,5 : 8 point formula, C5 regularity
          - 'l6_6' : Labmda6,6 : 8 point formula, C6 regularity
          - 'l8_4' : Labmda8,4 : 10 point formula, C4 regularity
          - 'm8prime' : M8prime formula
        """
        NumMethod.__init__(self)
        self._kernel = kernel()
        self.dir = d
        self._dim = dim
        self.work = work
        self.iwork = iwork
        assert len(self.work) == 2
        assert len(self.iwork) == self._dim
        self.shift = self._kernel.shift
        self.weights = self._kernel.weights
        self.topo = topo
        self._slice_all = [slice(None, None, None)
                           for dd in xrange(dim)]

        if self._dim == 3:
            if self.dir == 0:
                self._affect_working_arrays = self._affect_work_3D_X
            if self.dir == 1:
                self._affect_working_arrays = self._affect_work_3D_Y
            if self.dir == 2:
                self._affect_working_arrays = self._affect_work_3D_Z
        if self._dim == 2:
            if self.dir == 0:
                self._affect_working_arrays = self._affect_work_2D_X
            if self.dir == 1:
                self._affect_working_arrays = self._affect_work_2D_Y
        if self._dim == 1:
            if self.dir == 0:
                self._affect_working_arrays = self._affect_work_1D

    def slice_i_along_d(self, i, d):
        l = list(self._slice_all)
        l[d] = i
        return tuple(l)

    @staticmethod
    def getWorkLengths(nb_components=None, domain_dim=1):
        return 2, domain_dim

    def _affect_work_1D(self, resol):
        return (self.work[0], self.work[1], tuple(self.iwork))

    def _affect_work_2D_X(self, resol):
        self.iwork[1][...] = np.indices((resol[1],))[0].astype(
            HYSOP_INDEX)[np.newaxis, :]
        return (self.work[0], self.work[1], tuple(self.iwork))

    def _affect_work_2D_Y(self, resol):
        self.iwork[0][...] = np.indices((resol[0],))[0].astype(
            HYSOP_INDEX)[:, np.newaxis]
        return (self.work[0], self.work[1], tuple(self.iwork))

    def _affect_work_3D_X(self, resol):
        self.iwork[1][...] = np.indices((resol[1],))[0].astype(
            HYSOP_INDEX)[np.newaxis, :, np.newaxis]
        self.iwork[2][...] = np.indices((resol[2],))[0].astype(
            HYSOP_INDEX)[np.newaxis, np.newaxis, :]
        return (self.work[0], self.work[1], tuple(self.iwork))

    def _affect_work_3D_Y(self, resol):
        self.iwork[0][...] = np.indices((resol[0],))[0].astype(
            HYSOP_INDEX)[:, np.newaxis, np.newaxis]
        self.iwork[2][...] = np.indices((resol[2],))[0].astype(
            HYSOP_INDEX)[np.newaxis, np.newaxis, :]
        return (self.work[0], self.work[1], tuple(self.iwork))

    def _affect_work_3D_Z(self, resol):
        self.iwork[0][...] = np.indices((resol[0],))[0].astype(
            HYSOP_INDEX)[:, np.newaxis, np.newaxis]
        self.iwork[1][...] = np.indices((resol[1],))[0].astype(
            HYSOP_INDEX)[np.newaxis, :, np.newaxis]
        return (self.work[0], self.work[1], tuple(self.iwork))

    def __call__(self, ppos, pscal, result):
        """
        Remesh particles at position p_pos with scalar p_scal along
        direction d.

        @param p_pos : particle position
        @param p_scal : particle scalar
        @param result : a predefined list of numpy arrays to solve the result
        @return remeshed scalar on grid
        """
        d = self.dir
        mesh = self.topo.mesh
        resolution = self.topo.mesh.discretization.resolution
        origin = self.topo.domain.origin
        dx = mesh.space_step
        tmp, i_y, index = self._affect_working_arrays(mesh.resolution)

        floor = result  # use res array as working array
        floor[...] = (ppos - origin[d]) / dx[d]
        i_y[...] = floor
        floor[...] = np.floor(floor)
        i_y[...] -= floor

        # Gobal indices
        index[d][...] = (floor.astype(HYSOP_INDEX) - self.shift) \
            % (resolution[d] - 1)
        result[...] = 0.  # reset res array (no more uses to floor variable)
        for w_id, w in enumerate(self.weights):
            if w_id > 0:
                index[d][...] = (index[d] + 1) % (resolution[d] - 1)
            tmp[...] = self._kernel(w_id, i_y, tmp)
            tmp *= pscal
            for i in xrange(mesh.resolution[d]):
                sl = self.slice_i_along_d(i, d)
                index_sl = tuple([ind[sl] for ind in index])
                result[index_sl] = result[index_sl] + tmp[sl]

        return result


class RemeshFormula(object):
    """Abstract class for remeshing formulas"""
    def __init__(self):
        self.shift = 0
        self.weights = None

    def __call__(self, w, x, res):
        """Compute remeshing weights."""
        res[...] = self.weights[w][0]
        for c in self.weights[w][1:]:
            res[...] *= x
            res[...] += c
        return res


class Linear(RemeshFormula):
    """Linear kernel."""
    def __init__(self):
        super(Linear, self).__init__()
        self.shift = 0
        self.weights = [
            npw.asrealarray([-1, 1]),
            npw.asrealarray([1, 0]),
            ]


class L2_1(RemeshFormula):
    """L2_1 kernel."""
    def __init__(self):
        super(L2_1, self).__init__()
        self.shift = 1
        self.weights = [
            npw.asrealarray([-1, 2, -1, 0]) / 2.,
            npw.asrealarray([3, -5, 0, 2]) / 2.,
            npw.asrealarray([-3, 4, 1, 0]) / 2.,
            npw.asrealarray([1, -1, 0, 0]) / 2.,
            ]


class L2_2(RemeshFormula):
    """L2_2 kernel."""
    def __init__(self):
        super(L2_2, self).__init__()
        self.shift = 1
        self.weights = [
            npw.asrealarray([2, -5, 3, 1, -1, 0]) / 2.,
            npw.asrealarray([-6, 15, -9, -2, 0, 2]) / 2.,
            npw.asrealarray([6, -15, 9, 1, 1, 0]) / 2.,
            npw.asrealarray([-2, 5, -3, 0, 0, 0]) / 2.,
            ]


class L2_3(RemeshFormula):
    """L2_3 kernel."""
    def __init__(self):
        super(L2_3, self).__init__()
        self.shift = 1
        self.weights = [
            npw.asrealarray([-6, 21, -25, 10, 0, 1, -1, 0]) / 2.,
            npw.asrealarray([18, -63, 75, -30, 0, -2, 0, 2]) / 2.,
            npw.asrealarray([-18, 63, -75, 30, 0, 1, 1, 0]) / 2.,
            npw.asrealarray([6, -21, 25, -10, 0, 0, 0, 0]) / 2.,
            ]


class L2_4(RemeshFormula):
    """L2_4 kernel."""
    def __init__(self):
        super(L2_4, self).__init__()
        self.shift = 1
        self.weights = [
            npw.asrealarray([20, -90, 154, -119, 35, 0, 0, 1, -1, 0]) / 2.,
            npw.asrealarray([-60, 270, -462, 357, -105, 0, 0, -2, 0, 2]) / 2.,
            npw.asrealarray([60, -270, 462, -357, 105, 0, 0, 1, 1, 0]) / 2.,
            npw.asrealarray([-20, 90, -154, 119, -35, 0, 0, 0, 0, 0]) / 2.,
            ]


class L4_2(RemeshFormula):
    """L4_2 kernel."""
    def __init__(self):
        super(L4_2, self).__init__()
        self.shift = 2
        self.weights = [
            npw.asrealarray([-5, 13, -9, -1, 2, 0]) / 24.,
            npw.asrealarray([25, -64, 39, 16, -16, 0]) / 24.,
            npw.asrealarray([-50, 126, -70, -30, 0, 24]) / 24.,
            npw.asrealarray([50, -124, 66, 16, 16, 0]) / 24.,
            npw.asrealarray([-25, 61, -33, -1, -2, 0]) / 24.,
            npw.asrealarray([5, -12, 7, 0, 0, 0]) / 24.,
            ]


class L4_3(RemeshFormula):
    """L4_3 kernel."""
    def __init__(self):
        super(L4_3, self).__init__()
        self.shift = 2
        self.weights = [
            npw.asrealarray([14, -49, 58, -22, -2, -1, 2, 0]) / 24.,
            npw.asrealarray([-70, 245, -290, 111, 4, 16, -16, 0]) / 24.,
            npw.asrealarray([140, -490, 580, -224, 0, -30, 0, 24]) / 24.,
            npw.asrealarray([-140, 490, -580, 226, -4, 16, 16, 0]) / 24.,
            npw.asrealarray([70, -245, 290, -114, 2, -1, -2, 0]) / 24.,
            npw.asrealarray([-14, 49, -58, 23, 0, 0, 0, 0]) / 24.,
            ]


class L4_4(RemeshFormula):
    """L4_4 kernel."""
    def __init__(self):
        super(L4_4, self).__init__()
        self.shift = 2
        self.weights = [
            npw.asrealarray([-46, 207, -354, 273, -80, 1, -2, -1, 2, 0]) / 24.,
            npw.asrealarray([230, -1035, 1770, -1365, 400, -4, 4, 16, -16, 0]) / 24.,
            npw.asrealarray([-460, 2070, -3540, 2730, -800, 6, 0, -30, 0, 24]) / 24.,
            npw.asrealarray([460, -2070, 3540, -2730, 800, -4, -4, 16, 16, 0]) / 24.,
            npw.asrealarray([-230, 1035, -1770, 1365, -400, 1, 2, -1, -2, 0]) / 24.,
            npw.asrealarray([46, -207, 354, -273, 80, 0, 0, 0, 0, 0]) / 24.,
            ]


class M8Prime(RemeshFormula):
    """M8Prime kernel."""
    def __init__(self):
        super(M8Prime, self).__init__()
        self.shift = 3
        self.weights = [
            npw.asrealarray([-10, 21, 28, -105, 70, 35, -56, 17]) / 3360.,
            npw.asrealarray([70, -175, -140, 770, -560, -350, 504, -102]) / 3360.,
            npw.asrealarray([-210, 609, 224, -2135, 910, 2765, -2520, 255]) / 3360.,
            npw.asrealarray([350, -1155, 0, 2940, 0, -4900, 0, 3020]) / 3360.,
            npw.asrealarray([-350, 1295, -420, -2135, -910, 2765, 2520, 255]) / 3360.,
            npw.asrealarray([210, -861, 532, 770, 560, -350, -504, -102]) / 3360.,
            npw.asrealarray([-70, 315, -280, -105, -70, 35, 56, 17]) / 3360.,
            npw.asrealarray([10, -49, 56, 0, 0, 0, 0, 0]) / 3360.,
            ]


class L6_3(RemeshFormula):
    """L6_3 kernel."""
    def __init__(self):
        super(L6_3, self).__init__()
        self.shift = 3
        self.weights = [
            npw.asrealarray([-89, 312, -370, 140, 15, 4, -12, 0]) / 720.,
            npw.asrealarray([623, -2183, 2581, -955, -120, -54, 108, 0]) / 720.,
            npw.asrealarray([-1869, 6546, -7722, 2850, 195, 540, -540, 0]) / 720.,
            npw.asrealarray([3115, -10905, 12845, -4795, 0, -980, 0, 720]) / 720.,
            npw.asrealarray([-3115, 10900, -12830, 4880, -195, 540, 540, 0]) / 720.,
            npw.asrealarray([1869, -6537, 7695, -2985, 120, -54, -108, 0]) / 720.,
            npw.asrealarray([-623, 2178, -2566, 1010, -15, 4, 12, 0]) / 720.,
            npw.asrealarray([89, -311, 367, -145, 0, 0, 0, 0]) / 720.,
            ]


class L6_4(RemeshFormula):
    """L6_4 kernel."""
    def __init__(self):
        super(L6_4, self).__init__()
        self.shift = 3
        self.weights = [
            npw.asrealarray([290, -1305, 2231, -1718, 500, -5, 15, 4, -12, 0]) / 720.,
            npw.asrealarray([-2030, 9135, -15617, 12027, -3509, 60, -120, -54, 108, 0]) / 720.,
            npw.asrealarray([6090, -27405, 46851, -36084, 10548, -195, 195, 540, -540, 0]) / 720.,
            npw.asrealarray([-10150, 45675, -78085, 60145, -17605, 280, 0, -980, 0, 720]) / 720.,
            npw.asrealarray([10150, -45675, 78085, -60150, 17620, -195, -195, 540, 540, 0]) / 720.,
            npw.asrealarray([-6090, 27405, -46851, 36093, -10575, 60, 120, -54, -108, 0]) / 720.,
            npw.asrealarray([2030, -9135, 15617, -12032, 3524, -5, -15, 4, 12, 0]) / 720.,
            npw.asrealarray([-290, 1305, -2231, 1719, -503, 0, 0, 0, 0, 0]) / 720.,
            ]


class L6_5(RemeshFormula):
    """L6_5 kernel."""
    def __init__(self):
        super(L6_5, self).__init__()
        self.shift = 3
        self.weights = [
            npw.asrealarray([-1006, 5533, -12285, 13785, -7829, 1803, -3, -5, 15, 4, -12, 0]) / 720.,
            npw.asrealarray([7042, -38731, 85995, -96495, 54803, -12620, 12, 60, -120, -54, 108, 0]) / 720.,
            npw.asrealarray([-21126, 116193, -257985, 289485, -164409, 37857, -15, -195, 195, 540, -540, 0]) / 720.,
            npw.asrealarray([35210, -193655, 429975, -482475, 274015, -63090, 0, 280, 0, -980, 0, 720]) / 720.,
            npw.asrealarray([-35210, 193655, -429975, 482475, -274015, 63085, 15, -195, -195, 540, 540, 0]) / 720.,
            npw.asrealarray([21126, -116193, 257985, -289485, 164409, -37848, -12, 60, 120, -54, -108, 0]) / 720.,
            npw.asrealarray([-7042, 38731, -85995, 96495, -54803, 12615, 3, -5, -15, 4, 12, 0]) / 720.,
            npw.asrealarray([1006, -5533, 12285, -13785, 7829, -1802, 0, 0, 0, 0, 0, 0]) / 720.,
            ]

class L6_6(RemeshFormula):
    """L6_6 kernel."""
    def __init__(self):
        super(L6_6, self).__init__()
        self.shift = 3
        self.weights = [
            npw.asrealarray([3604, -23426, 63866, -93577, 77815, -34869, 6587, 1, -3, -5, 15, 4, -12, 0]) / 720.,
            npw.asrealarray([-25228, 163982, -447062, 655039, -544705, 244083, -46109, -6, 12, 60, -120, -54, 108, 0]) / 720.,
            npw.asrealarray([75684, -491946, 1341186, -1965117, 1634115, -732249, 138327, 15, -15, -195, 195, 540, -540, 0]) / 720.,
            npw.asrealarray([-126140, 819910, -2235310, 3275195, -2723525, 1220415, -230545, -20, 0, 280, 0, -980, 0, 720]) / 720.,
            npw.asrealarray([126140, -819910, 2235310, -3275195, 2723525, -1220415, 230545, 15, 15, -195, -195, 540, 540, 0]) / 720.,
            npw.asrealarray([-75684, 491946, -1341186, 1965117, -1634115, 732249, -138327, -6, -12, 60, 120, -54, -108, 0]) / 720.,
            npw.asrealarray([25228, -163982, 447062, -655039, 544705, -244083, 46109, 1, 3, -5, -15, 4, 12, 0]) / 720.,
            npw.asrealarray([-3604, 23426, -63866, 93577, -77815, 34869, -6587, 0, 0, 0, 0, 0, 0, 0]) / 720.,
            ]


class L8_4(RemeshFormula):
    """L8_4 kernel."""
    def __init__(self):
        super(L8_4, self).__init__()
        self.shift = 4
        self.weights = [
            npw.asrealarray([-3569, 16061, -27454, 21126, -6125, 49, -196, -36, 144, 0]) / 40320.,
            npw.asrealarray([32121, -144548, 247074, -190092, 55125, -672, 2016, 512, -1536, 0]) / 40320.,
            npw.asrealarray([-128484, 578188, -988256, 760312, -221060, 4732, -9464, -4032, 8064, 0]) / 40320.,
            npw.asrealarray([299796, -1349096, 2305856, -1774136, 517580, -13664, 13664, 32256, -32256, 0]) / 40320.,
            npw.asrealarray([-449694, 2023630, -3458700, 2661540, -778806, 19110, 0, -57400, 0, 40320]) / 40320.,
            npw.asrealarray([449694, -2023616, 3458644, -2662016, 780430, -13664, -13664, 32256, 32256, 0]) / 40320.,
            npw.asrealarray([-299796, 1349068, -2305744, 1775032, -520660, 4732, 9464, -4032, -8064, 0]) / 40320.,
            npw.asrealarray([128484, -578168, 988176, -760872, 223020, -672, -2016, 512, 1536, 0]) / 40320.,
            npw.asrealarray([-32121, 144541, -247046, 190246, -55685, 49, 196, -36, -144, 0]) / 40320.,
            npw.asrealarray([3569, -16060, 27450, -21140, 6181, 0, 0, 0, 0, 0]) / 40320.,
            ]


def polynomial_optimisation():
    """Testing different python implementation of a polynomial expression.
    Use polynomial of degree 10 :
    10*x^10+9*x^9+8*x^8+7*x^7+6*x^6+5*x^5+4*x^4+3*x^3+2*x^2+x
    """
    def test_form(func, r, a, s, *args):
        tt = 0.
        for i in xrange(10):
            r[...] = 0.
            t = MPI.Wtime()
            r[...] = func(a, *args)
            tt += (MPI.Wtime() - t)
        print tt, s

    from hysop.mpi.main_var import MPI
    nb = 128
    a = npw.asrealarray(np.random.random((nb, nb, nb)))
    r = np.zeros_like(a)
    temp = np.zeros_like(a)
    lambda_p = lambda x: 1. + 2. * x + 3. * x ** 2 + 4.  *x ** 3 + 5. * x ** 4 + \
               6. * x ** 5 + 7. * x ** 6 + 8. * x ** 7 + 9. * x ** 8 + 10. * x ** 9 + \
               11. * x ** 10
    lambda_h = lambda x: (x * (x * (x * (x * (x * (x * (x * (x * (11. * x + 10.) + 9.) + \
                8.) + 7.) + 6.) + 5.) + 4.) + 3.) + 2.) + 1.
    coeffs = coeffs = npw.asrealarray(np.arange(11, 0, -1))

    def func_h(x, r):
        r[...] = coeffs[0]
        for c in coeffs[1:]:
            r[...] *= x
            r[...] += c

    def func_p(x, r, tmp):
        r[...] = 1.
        tmp[...] = x
        tmp[...] *= 2.
        r[...] += tmp
        tmp[...] = x ** 2
        tmp[...] *= 3.
        r[...] += tmp
        tmp[...] = x ** 3
        tmp[...] *= 4.
        r[...] += tmp
        tmp[...] = x ** 4
        tmp[...] *= 5.
        r[...] += tmp
        tmp[...] = x ** 5
        tmp[...] *= 6.
        r[...] += tmp
        tmp[...] = x ** 6
        tmp[...] *= 7.
        r[...] += tmp
        tmp[...] = x ** 7
        tmp[...] *= 8.
        r[...] += tmp
        tmp[...] = x ** 8
        tmp[...] *= 9.
        r[...] += tmp
        tmp[...] = x ** 9
        tmp[...] *= 10.
        r[...] += tmp
        tmp[...] = x ** 10
        tmp[...] *= 11.
        r[...] += tmp

    def func_p_bis(x, r, tmp):
        r[...] = 1.
        tmp[...] = x
        tmp[...] *= 2.
        r[...] += tmp
        tmp[...] = x
        tmp[...] *= x
        tmp[...] *= 3.
        r[...] += tmp
        tmp[...] = x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= 4.
        r[...] += tmp
        tmp[...] = x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= 5.
        r[...] += tmp
        tmp[...] = x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= 6.
        r[...] += tmp
        tmp[...] = x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= 7.
        r[...] += tmp
        tmp[...] = x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= 8.
        r[...] += tmp
        tmp[...] = x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= 9.
        r[...] += tmp
        tmp[...] = x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= 10.
        r[...] += tmp
        tmp[...] = x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= x
        tmp[...] *= 11.
        r[...] += tmp

    from numpy.polynomial.polynomial import polyval

    def np_polyval(x, r):
        r[...] = polyval(x, coeffs[::-1])

    assert lambda_h(1.) == 66.
    assert lambda_p(1.) == 66.
    single_val = npw.ones((1, ))
    single_val_r = np.zeros_like(single_val)
    single_val_tmp = np.zeros_like(single_val)
    func_p(single_val, single_val_r, single_val_tmp)
    assert single_val_r[0] == 66.
    single_val_r[0] = 0.
    func_p_bis(single_val, single_val_r, single_val_tmp)
    assert single_val_r[0] == 66.
    single_val_r[0] = 0.
    func_h(single_val, single_val_r)
    assert single_val_r[0] == 66.
    single_val_r[0] = 0.
    np_polyval(single_val, single_val_r)
    assert single_val_r[0] == 66.

    test_form(lambda_p, r, a, "Lambda base canonique")
    test_form(lambda_h, r, a, "Lambda Horner")
    test_form(func_p, r, a, "Function base canonique", r, temp)
    test_form(func_p_bis, r, a, "Function base canonique (bis)", r, temp)
    test_form(func_h, r, a, "Function Horner", r)
    test_form(np_polyval, r, a, "Numpy polyval", r)

    res_test = np.empty_like(a)
    res_test_coeff = np.empty_like(a)
    w_test = lambda y: (-12. + (4. + (15. + (-5. + (-3. + (1. + (6587. + (-34869. + \
        (77815. + (-93577. + (63866. + (-23426. + 3604. * y) * y) * y) * y) * y) * y) \
        * y) * y) * y) * y) * y) * y) * y / 720.
    res_test[...] = w_test(a)
    w_test_coeffs = npw.asrealarray([3604, -23426, 63866, -93577, 77815, -34869, 6587,
                                     1, -3, -5, 15, 4, -12, 0]) / 720.
    res_test_coeff[...] = w_test_coeffs[0]
    for c in w_test_coeffs[1:]:
        res_test_coeff[...] *= a
        res_test_coeff[...] += c

    print np.max(res_test - res_test_coeff)
    assert np.allclose(res_test, res_test_coeff)
