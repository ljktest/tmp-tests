# -*- coding: utf-8 -*-
"""
@file numpywrappers.py

Tools to build numpy arrays based on hysop setup (float type ...)
"""
from hysop.constants import HYSOP_REAL, ORDER, HYSOP_INTEGER,\
    HYSOP_DIM
import numpy as np
import scitools.filetable as ft
bool = np.bool


def zeros(shape, dtype=HYSOP_REAL):
    """
    Wrapper to numpy.zeros, force order to hysop.constants.ORDER
    """
    return np.zeros(shape, dtype=dtype, order=ORDER)


def ones(shape, dtype=HYSOP_REAL):
    """
    Wrapper to numpy.ones, force order to hysop.constants.ORDER
    """
    return np.ones(shape, dtype=dtype, order=ORDER)


def zeros_like(tab):
    """
    Wrapper to numpy.zeros_like, force order to hysop.constants.ORDER
    """
    return np.zeros_like(tab, dtype=tab.dtype, order=ORDER)

def ones_like(tab):
    """
    Wrapper to numpy.ones_like, force order to hysop.constants.ORDER
    """
    return np.ones_like(tab, dtype=tab.dtype, order=ORDER)


def reshape(tab, shape):
    """
    Wrapper to numpy.reshape, force order to hysop.constants.ORDER
    """
    return np.reshape(tab, shape, dtype=tab.dtype, order=ORDER)


def realempty(tab):
    """
    Wrapper to numpy.empty, force order to hysop.constants.ORDER
    """
    return np.empty(tab, dtype=HYSOP_REAL, order=ORDER)


def empty_like(tab):
    """
    Wrapper to numpy.empty_like, force order to hysop.constants.ORDER
    """
    return np.empty_like(tab, dtype=tab.dtype, order=ORDER)


def copy(tab):
    """
    Wrapper to numpy.copy, ensure the same ordering in copy.
    """
    return tab.copy(order='A')


def asarray(tab):
    """
    Wrapper to numpy.asarray, force order to hysop.constants.ORDER
    """
    return np.asarray(tab, order=ORDER, dtype=tab.dtype)


def asrealarray(tab):
    """
    Wrapper to numpy.asarray, force order to hysop.constants.ORDER
    and type to hysop.constants.HYSOP_REAL
    """
    return np.asarray(tab, order=ORDER, dtype=HYSOP_REAL)


def const_realarray(tab):
    """
    Wrapper to numpy.asarray, force order to hysop.constants.ORDER
    and type to hysop.constants.HYSOP_REAL.
    Forbid any later change in the content of the array.
    """
    tmp = np.asarray(tab, order=ORDER, dtype=HYSOP_REAL)
    tmp.flags.writeable = False
    return tmp


def const_dimarray(tab):
    """
    Wrapper to numpy.asarray, force order to hysop.constants.ORDER
    and type to hysop.constants.HYSOP_DIM.
    Forbid any later change in the content of the array.
    """
    tmp = np.asarray(tab, order=ORDER, dtype=HYSOP_DIM)
    tmp.flags.writeable = False
    return tmp


def asintarray(tab):
    """
    Wrapper to numpy.asarray, force order to hysop.constants.ORDER
    and type to hysop.constants.HYSOP_INTEGER.
    """
    return np.asarray(tab, order=ORDER, dtype=HYSOP_INTEGER)


def int_zeros(shape):
    """
    Wrapper to numpy.zeros, force order to hysop.constants.ORDER
    and type to hysop.constants.HYSOP_INTEGER.
    """
    return np.zeros(shape, order=ORDER, dtype=HYSOP_INTEGER)


def int_ones(shape):
    """
    Wrapper to numpy.ones, force order to hysop.constants.ORDER
    and type to hysop.constants.HYSOP_INTEGER.
    """
    return np.ones(shape, order=ORDER, dtype=HYSOP_INTEGER)


def asdimarray(tab):
    """
    Wrapper to numpy.asarray, force order to hysop.constants.ORDER
    and type to hysop.constants.HYSOP_DIM.
    """
    return np.asarray(tab, order=ORDER, dtype=HYSOP_DIM)


def asboolarray(tab):
    """
    Wrapper to numpy.asarray, force order to hysop.constants.ORDER
    and type to np.bool.
    """
    return np.asarray(tab, order=ORDER, dtype=np.bool)


def dim_ones(shape):
    """
    Wrapper to numpy.ones, force order to hysop.constants.ORDER
    and type to hysop.constants.HYSOP_INTEGER.
    """
    return np.ones(shape, order=ORDER, dtype=HYSOP_DIM)


def dim_zeros(shape):
    """
    Wrapper to numpy.ones, force order to hysop.constants.ORDER
    and type to hysop.constants.HYSOP_DIM.
    """
    return np.zeros(shape, order=ORDER, dtype=HYSOP_DIM)


def equal(a, b):
    msg = 'You try to compare two values of different '
    msg += 'types : ' + str(np.asarray(a).dtype) + 'and '
    msg += str(np.asarray(b).dtype) + '.'
    assert np.asarray(a).dtype == np.asarray(b).dtype, msg
    return np.equal(a, b)


def abs(tab):
    """
    Wrapper to numpy.abs, force order to hysop.constants.ORDER
    """
    return np.abs(tab, order=ORDER, dtype=tab.dtype)


def real_sum(tab):
    """
    Wrapper to numpy.sum, force type to hysop.constants.HYSOP_REAL.
    """
    return np.sum(tab, dtype=HYSOP_REAL)


def prod(tab, dtype=HYSOP_REAL):
    """
    Wrapper to numpy.prod
    """
    return np.prod(tab, dtype=dtype)


def add(a, b, c, dtype=HYSOP_REAL):
    """
    Wrapper to numpy.add
    """
    return np.add(a, b, c, dtype=dtype)


def writeToFile(fname, data, mode='a'):
    """
    write data (numpy array) to file fname
    """
    fout = open(fname, mode)
    ft.write(fout, data)
    fout.close()


def lock(tab):
    """
    Set tab as a non-writeable array
    """
    tab.flags.writeable = False


def unlock(tab):
    """
    set tab as a writeable array
    """
    tab.flags.writeable = True
