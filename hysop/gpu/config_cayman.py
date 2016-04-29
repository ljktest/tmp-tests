"""
@file config_cayman.py

OpenCL kernels configurations.
"""
from hysop.constants import np
FLOAT_GPU, DOUBLE_GPU = np.float32, np.float64

#build empty dictionaries
kernels_config = {}
kernels_config[2] = {FLOAT_GPU: {}, DOUBLE_GPU: {}}
kernels_config[3] = {FLOAT_GPU: {}, DOUBLE_GPU: {}}

# Copy kernel:
def copy_space_index_2d(size, t_dim, b_rows, vec):
    gwi = (int(size[0] / vec), int(b_rows * size[1] / t_dim), 1)
    lwi = (t_dim / vec, b_rows, 1)
    return gwi, lwi
def copy_space_index_3d(size, t_dim, b_rows, vec):
    gwi = (int(size[0] / vec), int(b_rows * size[1] / t_dim), int(size[2]))
    lwi = (t_dim / vec, b_rows, 1)
    return gwi, lwi
# Configs : sources, tile size, block rows, vector size, index space function
kernels_config[3][FLOAT_GPU]['copy'] = \
    ('kernels/copy.cl', 16, 8, 4, copy_space_index_3d)
kernels_config[3][DOUBLE_GPU]['copy'] = \
    ('kernels/copy_locMem.cl', 32, 8, 1, copy_space_index_3d)
kernels_config[2][FLOAT_GPU]['copy'] = \
    ('kernels/copy.cl', 16, 8, 2, copy_space_index_2d)
kernels_config[2][DOUBLE_GPU]['copy'] = \
    ('kernels/copy.cl', 32, 2, 2, copy_space_index_2d)

# Transpositions kernels:
# XY transposition
# Settings are taken from destination layout as current layout.
# gwi is computed form input layout (appears as transposed layout)
def xy_space_index_2d(size, t_dim, b_rows, vec):
    gwi = (int(size[1] / vec), int(b_rows * size[0] / t_dim), 1)
    lwi = (t_dim / vec, b_rows, 1)
    return gwi, lwi
def xy_space_index_3d(size, t_dim, b_rows, vec):
    gwi = (int(size[1] / vec), int(b_rows * size[0] / t_dim), int(size[2]))
    lwi = (t_dim / vec, b_rows, 1)
    return gwi, lwi
# Configs : sources, tile size, block rows, is padding, vector size,
#              index space function
kernels_config[3][FLOAT_GPU]['transpose_xy'] = \
    ('kernels/transpose_xy.cl', 16, 8, True, 2, xy_space_index_3d)
kernels_config[3][DOUBLE_GPU]['transpose_xy'] = \
    ('kernels/transpose_xy.cl', 32, 4, True, 4, xy_space_index_3d)
kernels_config[2][FLOAT_GPU]['transpose_xy'] = \
    ('kernels/transpose_xy.cl', 32, 8, True, 4, xy_space_index_2d)
kernels_config[2][DOUBLE_GPU]['transpose_xy'] = \
    ('kernels/transpose_xy.cl', 32, 2, True, 4, xy_space_index_2d)

# XZ transposition
# Settings are taken from destination layout as current layout.
# gwi is computed form input layout (appears as transposed layout)
def xz_space_index_3d(size, t_dim, b_rows, b_deph, vec):
    gwi = (int(size[2] / vec), int(b_rows * size[1] / t_dim), int(b_deph * size[0] / t_dim))
    lwi = (t_dim / vec, b_rows, b_deph)
    return gwi, lwi
# Configs : sources, tile size, block rows, block depth, is padding,
#              vector size, index space function
kernels_config[3][FLOAT_GPU]['transpose_xz'] = \
    ('kernels/transpose_xz.cl', 16, 4, 4, True, 1, xy_space_index_3d)
kernels_config[3][DOUBLE_GPU]['transpose_xz'] = \
    ('kernels/transpose_xz.cl', 8, 2, 2, False, 1, xy_space_index_3d)


def computational_kernels_index_space(size, vec):
    dim = len(size)
    if dim == 3:
        wi = 64
    if dim == 2:
        wi = 256
    # Change work-item regarding problem size
    if size[0] % wi > 0:
        if dim == 3:
            print "Warning : GPU best performances obtained for",
            print "problem sizes multiples of 64"
        else:
            print "Warning : GPU best performances obtained for",
            print "problem sizes multiples of 256"
    while(size[0] % wi > 0):
        wi = wi / 2
    # Change work-item regarding vector_width
    if wi * vec > size[0]:
        if size[0] % vec > 0:
            raise ValueError(
                "Resolution ({0}) must be a multiple of {1}".format(
                    size[0], vec))
        wi = size[0] // vec
    if dim == 3:
        gwi = (int(wi), int(size[1]), int(size[2]))
        lwi = (int(wi), 1, 1)
    else:
        gwi = (int(wi), int(size[1]))
        lwi = (int(wi), 1)
    return gwi, lwi

# Advection kernel
# Configs sources, is noBC, vector size, index space function
kernels_config[3][FLOAT_GPU]['advec'] = \
    (["common.cl",
      "advection/velocity_cache.cl", "advection/builtin_RKN.cl",
      "kernels/advection.cl"],
     False, 4, computational_kernels_index_space)
kernels_config[3][DOUBLE_GPU]['advec'] = \
    (["common.cl",
      "advection/velocity_cache.cl", "advection/builtin_RKN.cl",
      "kernels/advection.cl"],
     False, 2, computational_kernels_index_space)
kernels_config[2][FLOAT_GPU]['advec'] = \
    (["common.cl",
      "advection/velocity_cache.cl", "advection/builtin_RKN.cl",
      "kernels/advection.cl"],
     False, 4, computational_kernels_index_space)
kernels_config[2][DOUBLE_GPU]['advec'] = \
    (["common.cl",
      "advection/velocity_cache.cl", "advection/builtin_RKN_noVec.cl",
      "kernels/advection_noVec.cl"],
     False, 1, computational_kernels_index_space)

# Remeshing kernel
# Configs sources, is noBC, vector size, index space function
kernels_config[3][FLOAT_GPU]['remesh'] = \
    (["common.cl",
      "remeshing/weights_builtin.cl", "remeshing/private.cl",
      "kernels/remeshing.cl"],
     False, 4, computational_kernels_index_space)
kernels_config[3][DOUBLE_GPU]['remesh'] = \
    (["common.cl",
      "remeshing/weights_builtin.cl", "remeshing/private.cl",
      "kernels/remeshing.cl"],
     False, 4, computational_kernels_index_space)
kernels_config[2][FLOAT_GPU]['remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/basic.cl",
      "kernels/remeshing.cl"],
     True, 4, computational_kernels_index_space)
kernels_config[2][DOUBLE_GPU]['remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/basic.cl",
      "kernels/remeshing.cl"],
     True, 4, computational_kernels_index_space)

# Advection and remeshing kernel
# Configs sources, is noBC, vector size, index space function
kernels_config[3][FLOAT_GPU]['advec_and_remesh'] = \
    (["common.cl",
      "remeshing/weights_builtin.cl", "remeshing/private.cl",
      "advection/velocity_cache.cl","advection/builtin_RKN.cl",
      "kernels/advection_and_remeshing.cl"],
     False, 4, computational_kernels_index_space)
kernels_config[3][DOUBLE_GPU]['advec_and_remesh'] = \
    (["common.cl",
      "remeshing/weights_builtin.cl", "remeshing/private.cl",
      "advection/velocity_cache.cl", "advection/builtin_RKN.cl",
      "kernels/advection_and_remeshing.cl"],
     True, 4, computational_kernels_index_space)
kernels_config[2][FLOAT_GPU]['advec_and_remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/basic.cl",
      "advection/velocity_cache.cl", "advection/builtin_RKN.cl",
      "kernels/advection_and_remeshing.cl"],
     True, 8, computational_kernels_index_space)
kernels_config[2][DOUBLE_GPU]['advec_and_remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/basic.cl",
      "advection/velocity_cache.cl", "advection/builtin_RKN.cl",
      "kernels/advection_and_remeshing.cl"],
     True, 4, computational_kernels_index_space)



