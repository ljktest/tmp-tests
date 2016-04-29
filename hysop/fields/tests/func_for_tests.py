"""A set of functions, used in tests
to initialize the fields
"""
import numpy as np
import math
sin = np.sin
cos = np.cos
pi = math.pi


# 3D scalar field, no vectorization
def func_scal_1(res, x, y, z, t):
    res[0][...] = x - 0.1 * y + 10. * z * z * t
    return res


# 2D scalar field, vectorization required
def func_scal_2(x, y, z, t):
    f = x - 0.1 * y + 10. * z * z * t
    return f


# 3D vector field, no vectorization
def func_vec_1(res, x, y, z, t):
    res[0][...] = x
    res[1][...] = 0.1 * y
    res[2][...] = 10. * z * z
    return res


# 2D vector field, no vectorization
def func_vec_2(x, y, z, t):
    f_x = x
    f_y = 0.1 * y
    f_z = 10. * z * z
    return f_x, f_y, f_z


# 3D vector field, no vectorization, extra params
def func_vec_3(res, x, y, z, t, theta):
    res[0][...] = x + theta
    res[1][...] = 0.1 * y
    res[2][...] = 10. * z * z
    return res


# 3D vector field, vectorization required
def func_vec_4(x, y, z, t, theta):
    f_x = x + theta
    f_y = 0.1 * y
    f_z = 10. * z * z
    return f_x, f_y, f_z


# 3D field, 4 components, no vectorization, extra params
def func_vec_5(res, x, y, z, t, theta):
    res[0][...] = x + theta
    res[1][...] = 0.1 * y
    res[2][...] = 10. * z * z
    res[3][...] = theta * z
    return res


# 2D field, 4 components, no vectorization, extra param
def func_vec_6(res, x, y, t, theta):
    res[0][...] = x + theta
    res[1][...] = 0.1 * y
    res[2][...] = 10. * y
    res[3][...] = theta * t
    return res


def v3d(res, x, y, z, t):
    res[0][...] = sin(x) + t
    res[1][...] = y
    res[2][...] = z
    return res


def v3dbis(res, x, y, z, t):
    res[0][...] = 0.
    res[1][...] = sin(y)
    res[2][...] = t
    return res


def v2d(res, x, y, t):
    res[0][...] = sin(x) + t
    res[1][...] = y
    return res


def v_time_2d(res, x, y, t):
    """For 2d vector field, all components depend on time
    """
    res[0][...] = sin(x) + t
    res[1][...] = y * t
    return res


def v_time_3d(res, x, y, z, t):
    """For 3d vector field, all components depend on time
    """
    res[0][...] = sin(x) + t
    res[1][...] = y * t
    res[2][...] = z * t ** 2
    return res


def v_TG(res, x, y, z, t):
    """Taylor Green velocity"""
    res[0][...] = sin(x) * cos(y) * cos(z)
    res[1][...] = - cos(x) * sin(y) * cos(z)
    res[2][...] = 0.
    return res


def w_TG(res, x, y, z, t): 
    """Taylor Green vorticity"""
    res[0][...] = - cos(x) * sin(y) * sin(z)
    res[1][...] = - sin(x) * cos(y) * sin(z)
    res[2][...] = 2. * sin(x) * sin(y) * cos(z)
    return res
