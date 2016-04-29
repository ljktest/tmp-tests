from hysop.constants import XDIR, YDIR, ZDIR
import hysop.tools.numpywrappers as npw
import numpy as np


class Utils(object):

    i1 = [YDIR, ZDIR, XDIR]
    i2 = [ZDIR, XDIR, YDIR]
    gen_indices = zip(i1, i2)

    @staticmethod
    def sum_cross_product(x, y, sl, work):
        """
        Parameters
        ----------
        x : a tuple of arrays
            represents grid coordinates (like coords in mesh)
        y : list of numpy arrays
            represents a discrete field
        sl : list of slices
            mesh points indices (like topo.mesh.iCompute)
        work: numpy array
            temporary buffer

        Sum on a volume (defined by sl) of cross products of
        x with y at each grid point. Result in work.
        """
        current_dir = 0
        dim = len(y)
        assert work.size == y[0][sl].size
        res = npw.zeros(dim)
        for (i, j) in Utils.gen_indices:
            np.multiply(x[i], y[j][sl], work)
            res[current_dir] = npw.real_sum(work)
            np.multiply(x[j], y[i][sl], work)
            res[current_dir] -= npw.real_sum(work)
            current_dir += 1
        return res

    @staticmethod
    def sum_cross_product_2(x, y, ind, work):
        current_dir = 0
        res = npw.zeros(3)
        ilist = np.where(ind)
        nb = len(ilist[0])
        for (i, j) in Utils.gen_indices:
            work.flat[:nb] = x[i].flat[ilist[i]] * y[j][ind]\
                - x[j].flat[ilist[j]] * y[i][ind]
            res[current_dir] = npw.real_sum(work.flat[:nb])
            current_dir += 1
        return res

    @staticmethod
    def sum_cross_product_3(x, y, ind):
        """
        Integrate over the control box using python loops.
        ---> wrong way, seems to be really slower although
        it costs less in memory.
        Used only for tests (timing).
        """
        ilist = np.where(ind)
        res = npw.zeros(3)
        for(ix, iy, iz) in zip(ilist[0], ilist[YDIR], ilist[ZDIR]):
            res[XDIR] += x[YDIR][0, iy, 0] * y[ZDIR][ix, iy, iz]\
                - x[ZDIR][0, 0, iz] * y[YDIR][ix, iy, iz]
            res[YDIR] += x[ZDIR][0, 0, iz] * y[XDIR][ix, iy, iz]\
                - x[XDIR][ix, 0, 0] * y[ZDIR][ix, iy, iz]
            res[ZDIR] += x[XDIR][ix, 0, 0] * y[YDIR][ix, iy, iz]\
                - x[YDIR][0, iy, 0] * y[XDIR][ix, iy, iz]
        return res
