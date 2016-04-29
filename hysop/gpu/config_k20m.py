"""
@file config_k20m.py

OpenCL kernels configurations.
"""
from hysop.constants import np
FLOAT_GPU, DOUBLE_GPU = np.float32, np.float64
MAX_GWI = (1024, 1024, 1024)


def _clamp_max(w, m):
    while w > m:
        w /= 2
    return int(w)


def check_max(t_gwi):
    return tuple([_clamp_max(w, m) for w, m in zip(t_gwi, MAX_GWI)])


#build empty dictionaries
kernels_config = {}
kernels_config[2] = {FLOAT_GPU: {}, DOUBLE_GPU: {}}
kernels_config[3] = {FLOAT_GPU: {}, DOUBLE_GPU: {}}

## Copy kernel is replaced by copy function from OpenCL API
# # Copy kernel:
# def copy_space_index_2d(size, t_dim, b_rows, vec):
#     gwi = (int(size[0] / vec), int(b_rows * size[1] / t_dim), 1)
#     lwi = (t_dim / vec, b_rows, 1)
#     return gwi, lwi
# def copy_space_index_3d(size, t_dim, b_rows, vec):
#     gwi = (int(size[0] / vec), int(b_rows * size[1] / t_dim), int(size[2]))
#     lwi = (t_dim / vec, b_rows, 1)
#     return gwi, lwi
# # Configs : sources, tile size, block rows, vector size, index space function
# kernels_config[3][FLOAT_GPU]['copy'] = \
#     ('kernels/copy_noVec.cl', 32, 8, 1, copy_space_index_3d)
# kernels_config[3][DOUBLE_GPU]['copy'] = \
#     ('kernels/copy_noVec.cl', 16, 16, 1, copy_space_index_3d)
# kernels_config[2][FLOAT_GPU]['copy'] = \
#     ('kernels/copy_noVec.cl', 32, 8, 1, copy_space_index_2d)
# kernels_config[2][DOUBLE_GPU]['copy'] = \
#     ('kernels/copy_noVec.cl', 16, 16, 1, copy_space_index_2d)

# Transpositions kernels:
# XY transposition
# Settings are taken from destination layout as current layout.
# gwi is computed form input layout (appears as transposed layout)
def xy_space_index_2d(size, t_dim, b_rows, vec):
    gwi = check_max((size[1] / vec, b_rows * size[0] / t_dim, 1))
    lwi = (t_dim / vec, b_rows, 1)
    blocs_nb = ((size[1] / vec) / lwi[0],
                (b_rows * size[0] / t_dim) / lwi[1], None)
    return gwi, lwi, blocs_nb
def xy_space_index_3d(size, t_dim, b_rows, vec):
    gwi = check_max((size[1] / vec, b_rows * size[0] / t_dim, size[2]))
    lwi = (t_dim / vec, b_rows, 1)
    block_nb = ((size[1] / vec) / lwi[0],
                (b_rows * size[0] / t_dim) / lwi[1], None)
    return gwi, lwi, block_nb
# Configs : sources, tile size, block rows, is padding, vector size,
#              index space function
kernels_config[3][FLOAT_GPU]['transpose_xy'] = \
    ('kernels/transpose_xy_noVec.cl', 32, 4, True, 1, xy_space_index_3d)
kernels_config[3][DOUBLE_GPU]['transpose_xy'] = \
    ('kernels/transpose_xy_noVec.cl', 32, 16, True, 1, xy_space_index_3d)
kernels_config[2][FLOAT_GPU]['transpose_xy'] = \
    ('kernels/transpose_xy_noVec.cl', 32, 2, True, 1, xy_space_index_2d)
kernels_config[2][DOUBLE_GPU]['transpose_xy'] = \
    ('kernels/transpose_xy_noVec.cl', 32, 8, True, 1, xy_space_index_2d)

# XZ transposition
# Settings are taken from destination layout as current layout.
# gwi is computed form input layout (appears as transposed layout)
def xz_space_index_3d(size, t_dim, b_rows, b_deph, vec):
    gwi = check_max((size[2] / vec, size[1], b_deph * size[0] / t_dim))
    lwi = (t_dim / vec, 1, b_deph)
    blocs_nb = ((size[2] / vec) / lwi[0], None,
                (b_deph * size[0] / t_dim) / lwi[2])
    return gwi, lwi, blocs_nb
# Configs : sources, tile size, block rows, is padding, vector size,
#              index space function
kernels_config[3][FLOAT_GPU]['transpose_xz'] = \
    ('kernels/transpose_xz_slice_noVec.cl', 32, 1, 8, True, 1, xz_space_index_3d)
kernels_config[3][DOUBLE_GPU]['transpose_xz'] = \
    ('kernels/transpose_xz_slice_noVec.cl', 32, 1, 8, True, 1, xz_space_index_3d)

def computational_kernels_index_space(wi, size, vec):
    # Change work-item regarding vector_width
    if wi * vec > size[0]:
        if size[0] % vec > 0:
            raise ValueError(
                "Resolution ({0}) must be a multiple of {1}".format(
                    size[0], vec))
        wi = size[0] // vec

    if len(size) == 3:
        gwi = (int(wi),
               _clamp_max(size[1], MAX_GWI[1]),
               _clamp_max(size[2], MAX_GWI[2]))
        lwi = (int(wi), 1, 1)
    else:
        gwi = (int(wi), _clamp_max(size[1], MAX_GWI[1]))
        lwi = (int(wi), 1)
    return gwi, lwi

def advection_index_space_3d(size, vec):
    wi = min(size[0] / 4, 1024)
    return computational_kernels_index_space(wi, size, vec)
def advection_index_space_2d_SP(size, vec):
    wi = min(size[0] / 8, 1024)
    return computational_kernels_index_space(wi, size, vec)
def advection_index_space_2d_DP(size, vec):
    wi = min(size[0] / 4, 1024)
    return computational_kernels_index_space(wi, size, vec)

def remeshing_index_space_3d(size, vec):
    wi = min(size[0] / 2, 1024)
    return computational_kernels_index_space(wi, size, vec)
def remeshing_index_space_2d(size, vec):
    wi = min(size[0] / 4, 1024)
    return computational_kernels_index_space(wi, size, vec)

def advection_and_remeshing_index_space(size, vec):
    wi = min(size[0] / 2, 1024)
    return computational_kernels_index_space(wi, size, vec)


# Advection kernel
# Configs sources, is noBC, vector size, index space function
kernels_config[3][FLOAT_GPU]['advec'] = \
    (["common.cl",
      "advection/velocity_cache_noVec.cl", "advection/builtin_RKN_noVec.cl",
      "kernels/advection_noVec.cl"],
     False, 1, advection_index_space_3d)
kernels_config[3][DOUBLE_GPU]['advec'] = \
    (["common.cl",
      "advection/velocity_cache_noVec.cl", "advection/builtin_RKN_noVec.cl",
      "kernels/advection_noVec.cl"],
     False, 1, advection_index_space_3d)
kernels_config[2][FLOAT_GPU]['advec'] = \
    (["common.cl",
      "advection/velocity_cache_noVec.cl", "advection/builtin_RKN_noVec.cl",
      "kernels/advection_noVec.cl"],
     False, 1, advection_index_space_2d_SP)
kernels_config[2][DOUBLE_GPU]['advec'] = \
    (["common.cl",
      "advection/velocity_cache_noVec.cl", "advection/builtin_RKN_noVec.cl",
      "kernels/advection_noVec.cl"],
     False, 1, advection_index_space_2d_DP)

# Remeshing kernel
# Configs sources, is noBC, vector size, index space function
kernels_config[3][FLOAT_GPU]['remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/basic_noVec.cl",
      "kernels/remeshing_noVec.cl"],
     False, 1, remeshing_index_space_3d)
kernels_config[3][DOUBLE_GPU]['remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/basic_noVec.cl",
      "kernels/remeshing_noVec.cl"],
     False, 1, remeshing_index_space_3d)
kernels_config[2][FLOAT_GPU]['remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/basic.cl",
      "kernels/remeshing.cl"],
     True, 2, remeshing_index_space_2d)
kernels_config[2][DOUBLE_GPU]['remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/basic.cl",
      "kernels/remeshing.cl"],
     True, 2, remeshing_index_space_2d)


# Advection and remeshing kernel
# Configs sources, is noBC, vector size, index space function
kernels_config[3][FLOAT_GPU]['advec_and_remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/private_noVec.cl",
      "advection/velocity_cache_noVec.cl", "advection/builtin_RKN_noVec.cl",
      "kernels/advection_and_remeshing_noVec.cl"],
     False, 1, advection_and_remeshing_index_space)
kernels_config[3][DOUBLE_GPU]['advec_and_remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/private_noVec.cl",
      "advection/velocity_cache_noVec.cl", "advection/builtin_RKN_noVec.cl",
      "kernels/advection_and_remeshing_noVec.cl"],
     False, 1, advection_and_remeshing_index_space)
kernels_config[2][FLOAT_GPU]['advec_and_remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/private_noVec.cl",
      "advection/velocity_cache_noVec.cl", "advection/builtin_RKN_noVec.cl",
      "kernels/advection_and_remeshing_noVec.cl"],
     False, 1, advection_and_remeshing_index_space)
kernels_config[2][DOUBLE_GPU]['advec_and_remesh'] = \
    (["common.cl",
      "remeshing/weights_noVec_builtin.cl", "remeshing/private_noVec.cl",
      "advection/velocity_cache_noVec.cl", "advection/builtin_RKN_noVec.cl",
      "kernels/advection_and_remeshing_noVec.cl"],
     False, 1, advection_and_remeshing_index_space)


def diffusion_space_index_3d(size, nb_part, tile):
    gwi = check_max((size[0], size[1] / nb_part))
    lwi = (tile, tile / nb_part)
    blocs_nb = (size[0] / tile, size[1] / tile)
    return gwi, lwi, blocs_nb


kernels_config[3][DOUBLE_GPU]['diffusion'] = \
    (["common.cl", "kernels/diffusion.cl"],
     16, 4, 1, diffusion_space_index_3d)


kernels_config[3][DOUBLE_GPU]['advec_comm'] = \
    (['common.cl', 'kernels/comm_advection_noVec.cl'],
     False, 1, advection_index_space_3d)
kernels_config[3][DOUBLE_GPU]['advec_MS_comm'] = \
    (['common.cl', "remeshing/weights_noVec_builtin.cl",
      'kernels/comm_MS_advection_noVec.cl'],
     False, 1, advection_index_space_3d)
kernels_config[3][DOUBLE_GPU]['remesh_comm'] = \
    (['common.cl', 'remeshing/weights_noVec.cl',
      'kernels/comm_remeshing_noVec.cl'],
     False, 1, remeshing_index_space_3d)
kernels_config[3][DOUBLE_GPU]['advec_and_remesh_comm'] = \
    (['common.cl', 'remeshing/weights_noVec.cl',
      'kernels/comm_advection_and_remeshing_noVec.cl'],
     False, 1, advection_and_remeshing_index_space)
kernels_config[3][DOUBLE_GPU]['advec_MS_and_remesh_comm'] = \
    (['common.cl', 'remeshing/weights_noVec.cl',
      'kernels/comm_advection_MS_and_remeshing_noVec.cl'],
     False, 1, advection_and_remeshing_index_space)


def fine_to_coarse_filter_index_space(size, stencil_width):
    wg = size[0] / (2 * stencil_width)
    return ((wg, size[1] / stencil_width, size[2] / stencil_width),
            (wg, 1, 1))


kernels_config[3][FLOAT_GPU]['fine_to_coarse_filter'] = \
    (["common.cl", 'remeshing/weights_noVec.cl',
      "kernels/fine_to_coarse_filter.cl"],
     1, fine_to_coarse_filter_index_space)
kernels_config[3][DOUBLE_GPU]['fine_to_coarse_filter'] = \
    (["common.cl", 'remeshing/weights_noVec.cl',
      "kernels/fine_to_coarse_filter.cl"],
     1, fine_to_coarse_filter_index_space)



def multiphase_baroclinic_index_space(size, tile):
    wg = (tile, tile, 1)
    ws = (int(size[0]), int(size[1]), 1)
    return ws, wg

kernels_config[3][FLOAT_GPU]['multiphase_baroclinic'] = \
    (["common.cl", "kernels/multiphase_baroclinic_rhs.cl"],
     8, 1, multiphase_baroclinic_index_space)
kernels_config[3][DOUBLE_GPU]['multiphase_baroclinic'] = \
    (["common.cl", "kernels/multiphase_baroclinic_rhs.cl"],
     8, 1, multiphase_baroclinic_index_space)
