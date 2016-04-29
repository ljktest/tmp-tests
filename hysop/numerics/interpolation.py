"""
@file interpolation.py
"""
from hysop.constants import np, HYSOP_INTEGER, ORDER
from hysop.numerics.method import NumMethod


class Linear(NumMethod):
    """Linear interpolation of a field"""

    def __init__(self, tab, direction, topo, work, iwork):
        """
        @param var : numpy array to interpolate
        @param dx : space grid step
        @param origin : grid lower point coordinates
        @param work : Work arrays (floats)
        @param iwork : Work arrays (integers)
        work must be a list containing (1 element):
          - numpy float array like y[0]
        iwork must be a list containing (dim elements):
          - numpy integer array of shape like y[0]
          - numpy integer array of shape like y[0] (if in 2D and 3D)
          - numpy integer array of shape like y[0] (if in 3D)
        """
        NumMethod.__init__(self)
        self.name = 'LinearInterpolation'
        self.tab = tab
        self.topo = topo
        dimension = self.topo.domain.dimension
        self.work = work
        for iw in iwork:
            assert iw.dtype == HYSOP_INTEGER
        self.iwork = iwork
        assert len(self.work) == 1
        assert len(self.iwork) == dimension
        self.dir = direction
        if dimension == 3:
            if self.dir == 0:
                self._affect_working_arrays = self._affect_work_3D_X
            if self.dir == 1:
                self._affect_working_arrays = self._affect_work_3D_Y
            if self.dir == 2:
                self._affect_working_arrays = self._affect_work_3D_Z
        if dimension == 2:
            if self.dir == 0:
                self._affect_working_arrays = self._affect_work_2D_X
            if self.dir == 1:
                self._affect_working_arrays = self._affect_work_2D_Y
        if dimension == 1:
            if self.dir == 0:
                self._affect_working_arrays = self._affect_work_1D

    @staticmethod
    def getWorkLengths(nb_components=None, domain_dim=1):
        return 1, domain_dim

    def _affect_work_1D(self, resol):
        return (self.work[0], tuple(self.iwork))

    def _affect_work_2D_X(self, resol):
        self.iwork[1][...] = np.indices((resol[1],))[0][np.newaxis, :]
        return (self.work[0], tuple(self.iwork))

    def _affect_work_2D_Y(self, resol):
        self.iwork[0][...] = np.indices((resol[0],))[0][:, np.newaxis]
        return (self.work[0], tuple(self.iwork))

    def _affect_work_3D_X(self, resol):
        self.iwork[1][...] = np.indices((resol[1],))[0][np.newaxis,
                                                        :, np.newaxis]
        self.iwork[2][...] = np.indices((resol[2],))[0][np.newaxis,
                                                        np.newaxis, :]
        return (self.work[0], tuple(self.iwork))

    def _affect_work_3D_Y(self, resol):
        self.iwork[0][...] = np.indices((resol[0],))[0][:,
                                                        np.newaxis, np.newaxis]
        self.iwork[2][...] = np.indices((resol[2],))[0][np.newaxis,
                                                        np.newaxis, :]
        return (self.work[0], tuple(self.iwork))

    def _affect_work_3D_Z(self, resol):
        self.iwork[0][...] = np.indices((resol[0],))[0][:,
                                                        np.newaxis, np.newaxis]
        self.iwork[1][...] = np.indices((resol[1],))[0][np.newaxis,
                                                        :, np.newaxis]
        return (self.work[0], tuple(self.iwork))

    def __call__(self, t, y, result):
        """
        Computational core for interpolation.
        """
        origin = self.topo.domain.origin
        mesh = self.topo.mesh
        dx = mesh.space_step
        resolution = mesh.discretization.resolution
        x = y[0]
        res = result[0]
        i_y, index = self._affect_working_arrays(mesh.resolution)

        floor = res  # use res array as working array
        floor[...] = (x - origin[self.dir]) / dx[self.dir]
        i_y[...] = floor
        floor[...] = np.floor(floor)
        i_y[...] -= floor
        # use res as the result (no more uses to floor variable)

        index[self.dir][...] = np.asarray(
            floor, dtype=HYSOP_INTEGER, order=ORDER) % (resolution[self.dir] - 1)

        res[...] = self.tab[index] * (1. - i_y)

        index[self.dir][...] = (index[self.dir] + 1) \
            % (resolution[self.dir] - 1)

        res[...] += self.tab[index] * i_y

        return [res, ]
