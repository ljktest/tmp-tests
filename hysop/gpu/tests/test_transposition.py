"""
@file hysop.gpu.tests.test_transposition
Testing copy kernels.
"""
from hysop.gpu import cl
from hysop.constants import np
from hysop.gpu.tools import get_opencl_environment
from hysop.gpu.gpu_kernel import KernelLauncher
import hysop.tools.numpywrappers as npw


def _comparison(resolution, resolutionT,
                transpose_f, transpose_b,
                gwi, lwi, cl_env, axe=1):

    data_in = npw.asrealarray(np.random.random(resolution))
    data_out = npw.realempty(resolutionT)
    data_out2 = npw.realempty(resolution)
    data_gpu_in = cl.Buffer(cl_env.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=data_in.nbytes)
    data_gpu_out = cl.Buffer(cl_env.ctx,
                             cl.mem_flags.READ_WRITE,
                             size=data_out.nbytes)
    data_gpu_out2 = cl.Buffer(cl_env.ctx,
                              cl.mem_flags.READ_WRITE,
                              size=data_out2.nbytes)
    cl.enqueue_copy(cl_env.queue, data_gpu_in, data_in)
    cl.enqueue_copy(cl_env.queue, data_gpu_out, data_out)
    cl.enqueue_copy(cl_env.queue, data_gpu_out2, data_out2)
    cl_env.queue.finish()

    # gpu_out <- gpu_in.T
    transpose_f(data_gpu_in, data_gpu_out)
    cl_env.queue.finish()
    cl.enqueue_copy(cl_env.queue, data_out, data_gpu_out)
    cl_env.queue.finish()
    assert np.allclose(data_out, data_in.swapaxes(0, axe))

    # gpu_in <- gpu_out.T ( = gpu_in.T.T = gpu_in)
    transpose_b(data_gpu_out, data_gpu_out2)
    cl_env.queue.finish()
    cl.enqueue_copy(cl_env.queue, data_out2, data_gpu_out2)
    cl_env.queue.finish()
    assert np.allclose(data_out2, data_in)

    data_gpu_in.release()
    data_gpu_out.release()
    data_gpu_out2.release()


def test_transposition_xy2D():
    resolution = (256, 256)
    cl_env = get_opencl_environment()
    vec = 4
    src_transpose_xy = 'kernels/transpose_xy.cl'
    build_options = ""
    build_options += " -D NB_I=256 -D NB_II=256"
    build_options += " -D PADDING_XY=1"
    build_options += " -D TILE_DIM_XY=32 -D BLOCK_ROWS_XY=8"
    gwi = (int(resolution[0] / 4), int(resolution[1]) / 4)
    lwi = (8, 8)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0] / 4) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])

    # Build code
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy = KernelLauncher(
        prg.transpose_xy, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolution,
                init_transpose_xy, init_transpose_xy,
                gwi, lwi, cl_env)


def test_transposition_xy2D_noVec():
    resolution = (256, 256)
    cl_env = get_opencl_environment()
    src_transpose_xy = 'kernels/transpose_xy_noVec.cl'
    build_options = ""
    build_options += " -D NB_I=256 -D NB_II=256"
    build_options += " -D PADDING_XY=1"
    build_options += " -D TILE_DIM_XY=32 -D BLOCK_ROWS_XY=8"
    gwi = (int(resolution[0]), int(resolution[1]) / 4)
    lwi = (32, 8)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0]) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])

    # Build code
    prg = cl_env.build_src(src_transpose_xy, build_options)
    init_transpose_xy = KernelLauncher(
        prg.transpose_xy, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolution,
                init_transpose_xy, init_transpose_xy,
                gwi, lwi, cl_env)


def test_transposition_xy2D_rect():
    resolution = (512, 256)
    resolutionT = (256, 512)
    cl_env = get_opencl_environment()
    vec = 4
    src_transpose_xy = 'kernels/transpose_xy.cl'
    build_options = ""
    # Settings are taken from destination layout as current layout.
    # gwi is computed form input layout (appears as transposed layout)
    build_options += " -D NB_I=256 -D NB_II=512"
    build_options += " -D TILE_DIM_XY=32 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[0] / 4),
           int(resolution[1]) / 4)
    lwi = (8, 8)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0] / 4) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy_x = KernelLauncher(prg.transpose_xy,
                                         cl_env.queue,
                                         gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=512 -D NB_II=256"
    build_options += " -D TILE_DIM_XY=32 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[1] / 4),
           int(resolution[0]) / 4)
    lwi = (8, 8)
    build_options += " -D NB_GROUPS_I=" + str((resolution[1] / 4) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[0] / 4) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy_y = KernelLauncher(prg.transpose_xy,
                                         cl_env.queue,
                                         gwi, lwi)
    _comparison(resolution, resolutionT,
                init_transpose_xy_x, init_transpose_xy_y,
                gwi, lwi, cl_env)


def test_transposition_xy2D_noVec_rect():
    resolution = (512, 256)
    resolutionT = (256, 512)
    cl_env = get_opencl_environment()
    vec = 4
    src_transpose_xy = 'kernels/transpose_xy_noVec.cl'
    build_options = ""
    # Settings are taken from destination layout as current layout.
    # gwi is computed form input layout (appears as transposed layout)
    build_options += " -D NB_I=256 -D NB_II=512"
    build_options += " -D TILE_DIM_XY=32 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[0]),
           int(resolution[1]) / 4)
    lwi = (32, 8)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0]) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy_x = KernelLauncher(prg.transpose_xy,
                                         cl_env.queue,
                                         gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=512 -D NB_II=256"
    build_options += " -D TILE_DIM_XY=32 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[1]),
           int(resolution[0]) / 4)
    lwi = (32, 8)
    build_options += " -D NB_GROUPS_I=" + str((resolution[1]) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[0] / 4) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy_y = KernelLauncher(prg.transpose_xy,
                                         cl_env.queue,
                                         gwi, lwi)
    _comparison(resolution, resolutionT,
                init_transpose_xy_x, init_transpose_xy_y,
                gwi, lwi, cl_env)


def test_transposition_xy3D():
    resolution = (32, 32, 32)
    cl_env = get_opencl_environment()
    vec = 2
    src_transpose_xy = 'kernels/transpose_xy.cl'
    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XY=16 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[0] / 2),
           int(resolution[1] / 2),
           int(resolution[2]))
    lwi = (8, 8, 1)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0] / 2) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 2) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy = KernelLauncher(
        prg.transpose_xy, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolution,
                init_transpose_xy, init_transpose_xy,
                gwi, lwi, cl_env)


def test_transposition_xy3D_noVec():
    resolution = (32, 32, 32)
    cl_env = get_opencl_environment()
    vec = 2
    src_transpose_xy = 'kernels/transpose_xy_noVec.cl'
    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XY=16 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[0]),
           int(resolution[1] / 2),
           int(resolution[2]))
    lwi = (16, 8, 1)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0]) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 2) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy = KernelLauncher(
        prg.transpose_xy, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolution,
                init_transpose_xy, init_transpose_xy,
                gwi, lwi, cl_env)


def test_transposition_xy3D_rect():
    resolution = (32, 64, 32)
    resolutionT = (64, 32, 32)
    cl_env = get_opencl_environment()
    vec = 2
    src_transpose_xy = 'kernels/transpose_xy.cl'
    build_options = ""
    # Settings are taken from destination layout as current layout.
    # gwi is computed form input layout (appears as transposed layout)
    build_options += " -D NB_I=64 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XY=16 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[0] / 2),
           int(resolution[1] / 2),
           int(resolution[2]))
    lwi = (8, 8, 1)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0] / 2) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 2) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy_x = KernelLauncher(
        prg.transpose_xy, cl_env.queue, gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=64 -D NB_III=32"
    build_options += " -D TILE_DIM_XY=16 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[1] / 2),
           int(resolution[0] / 2),
           int(resolution[2]))
    lwi = (8, 8, 1)
    build_options += " -D NB_GROUPS_I=" + str((resolution[1] / 2) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[0] / 2) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy_y = KernelLauncher(
        prg.transpose_xy, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolutionT,
                init_transpose_xy_x, init_transpose_xy_y,
                gwi, lwi, cl_env)


def test_transposition_xy3D_noVec_rect():
    resolution = (32, 64, 32)
    resolutionT = (64, 32, 32)
    cl_env = get_opencl_environment()
    vec = 2
    src_transpose_xy = 'kernels/transpose_xy_noVec.cl'
    build_options = ""
    # Settings are taken from destination layout as current layout.
    # gwi is computed form input layout (appears as transposed layout)
    build_options += " -D NB_I=64 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XY=16 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[0]),
           int(resolution[1] / 2),
           int(resolution[2]))
    lwi = (16, 8, 1)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0]) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 2) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy_x = KernelLauncher(
        prg.transpose_xy, cl_env.queue, gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=64 -D NB_III=32"
    build_options += " -D TILE_DIM_XY=16 -D BLOCK_ROWS_XY=8 -D PADDING_XY=1"
    gwi = (int(resolution[1]),
           int(resolution[0] / 2),
           int(resolution[2]))
    lwi = (16, 8, 1)
    build_options += " -D NB_GROUPS_I=" + str((resolution[1]) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[0] / 2) / lwi[1])
    prg = cl_env.build_src(src_transpose_xy, build_options, vec)
    init_transpose_xy_y = KernelLauncher(
        prg.transpose_xy, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolutionT,
                init_transpose_xy_x, init_transpose_xy_y,
                gwi, lwi, cl_env)


def test_transposition_xz3D():
    resolution = (32, 32, 32)
    cl_env = get_opencl_environment()
    vec = 2
    src_transpose_xz = 'kernels/transpose_xz.cl'
    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=4"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int((resolution[0] / 2)),
           int(resolution[1] / 4),
           int(resolution[2] / 4))
    lwi = (8, 4, 4)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0] / 2) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])
    build_options += " -D NB_GROUPS_III=" + str((resolution[2] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolution,
                init_transpose_xz, init_transpose_xz,
                gwi, lwi, cl_env, axe=2)


def test_transposition_xz3D_noVec():
    resolution = (32, 32, 32)
    cl_env = get_opencl_environment()
    vec = 1
    src_transpose_xz = 'kernels/transpose_xz_noVec.cl'
    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=4"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[0]),
           int(resolution[1] / 4),
           int(resolution[2] / 4))
    lwi = (16, 4, 4)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0]) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])
    build_options += " -D NB_GROUPS_III=" + str((resolution[2] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolution,
                init_transpose_xz, init_transpose_xz,
                gwi, lwi, cl_env, axe=2)


def test_transposition_xz3D_rect():
    resolution = (32, 32, 64)
    resolutionT = (64, 32, 32)
    cl_env = get_opencl_environment()
    vec = 2
    src_transpose_xz = 'kernels/transpose_xz.cl'
    build_options = ""
    # Settings are taken from destination layout as current layout.
    # gwi is computed form input layout (appears as transposed layout)
    build_options += " -D NB_I=64 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=4"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int((resolution[0] / 2)),
           int(resolution[1] / 4),
           int(resolution[2] / 4))
    lwi = (8, 4, 4)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0] / 2) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])
    build_options += " -D NB_GROUPS_III=" + str((resolution[2] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz_x = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=64"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=4"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[2] / 2),
           int(resolution[1] / 4),
           int(resolution[0] / 4))
    lwi = (8, 4, 4)
    build_options += " -D NB_GROUPS_I=" + str((resolution[2] / 2) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])
    build_options += " -D NB_GROUPS_III=" + str((resolution[0] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz_z = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolutionT,
                init_transpose_xz_x, init_transpose_xz_z,
                gwi, lwi, cl_env, axe=2)


def test_transposition_xz3D_noVec_rect():
    resolution = (32, 32, 64)
    resolutionT = (64, 32, 32)
    cl_env = get_opencl_environment()
    vec = 1
    src_transpose_xz = 'kernels/transpose_xz_noVec.cl'
    build_options = ""
    # Settings are taken from destination layout as current layout.
    # gwi is computed form input layout (appears as transposed layout)
    build_options += " -D NB_I=64 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=4"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[0]),
           int(resolution[1] / 4),
           int(resolution[2] / 4))
    lwi = (16, 4, 4)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0]) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])
    build_options += " -D NB_GROUPS_III=" + str((resolution[2] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz_x = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=64"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=4"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[2]),
           int(resolution[1] / 4),
           int(resolution[0] / 4))
    lwi = (16, 4, 4)
    build_options += " -D NB_GROUPS_I=" + str((resolution[2]) / lwi[0])
    build_options += " -D NB_GROUPS_II=" + str((resolution[1] / 4) / lwi[1])
    build_options += " -D NB_GROUPS_III=" + str((resolution[0] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz_z = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolutionT,
                init_transpose_xz_x, init_transpose_xz_z,
                gwi, lwi, cl_env, axe=2)


def test_transposition_xz3Dslice():
    resolution = (32, 32, 32)
    cl_env = get_opencl_environment()
    vec = 2
    src_transpose_xz = 'kernels/transpose_xz_slice.cl'
    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=1"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[0] / 2),
           int(resolution[1]),
           int(resolution[2] / 4))
    lwi = (8, 1, 4)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0] / 2) / lwi[0])
    build_options += " -D NB_GROUPS_III=" + str((resolution[2] / 4) / lwi[2])

    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolution,
                init_transpose_xz, init_transpose_xz,
                gwi, lwi, cl_env, axe=2)

def test_transposition_xz3Dslice_noVec():
    resolution = (32, 32, 32)
    cl_env = get_opencl_environment()
    vec = 1
    src_transpose_xz = 'kernels/transpose_xz_slice_noVec.cl'
    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=1"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[0]),
           int(resolution[1]),
           int(resolution[2] / 4))
    lwi = (16, 1, 4)
    build_options += " -D NB_GROUPS_I=" + str(resolution[0] / lwi[0])
    build_options += " -D NB_GROUPS_III=" + str((resolution[2] / 4) / lwi[2])

    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolution,
                init_transpose_xz, init_transpose_xz,
                gwi, lwi, cl_env, axe=2)


def test_transposition_xz3Dslice_rect():
    resolution = (32, 32, 64)
    resolutionT = (64, 32, 32)
    cl_env = get_opencl_environment()
    vec = 2
    src_transpose_xz = 'kernels/transpose_xz_slice.cl'
    build_options = ""
    # Settings are taken from destination layout as current layout.
    # gwi is computed form input layout (appears as transposed layout)
    build_options += " -D NB_I=64 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=1"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[0] / 2),
           int(resolution[1]),
           int(resolution[2] / 4))
    lwi = (8, 1, 4)
    build_options += " -D NB_GROUPS_I=" + str((resolution[0] / 2) / lwi[0])
    build_options += " -D NB_GROUPS_III=" + str((resolution[2] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz_x = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=64"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=1"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[2] / 2),
           int(resolution[1]),
           int(resolution[0] / 4))
    lwi = (8, 1, 4)
    build_options += " -D NB_GROUPS_I=" + str((resolution[2] / 2) / lwi[0])
    build_options += " -D NB_GROUPS_III=" + str((resolution[0] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz_z = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolutionT,
                init_transpose_xz_x, init_transpose_xz_z,
                gwi, lwi, cl_env, axe=2)

def test_transposition_xz3Dslice_noVec_rect():
    resolution = (32, 32, 64)
    resolutionT = (64, 32, 32)
    cl_env = get_opencl_environment()
    vec = 1
    src_transpose_xz = 'kernels/transpose_xz_slice_noVec.cl'
    build_options = ""
    # Settings are taken from destination layout as current layout.
    # gwi is computed form input layout (appears as transposed layout)
    build_options += " -D NB_I=64 -D NB_II=32 -D NB_III=32"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=1"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[0]),
           int(resolution[1]),
           int(resolution[2] / 4))
    lwi = (16, 1, 4)
    build_options += " -D NB_GROUPS_I=" + str(resolution[0] / lwi[0])
    build_options += " -D NB_GROUPS_III=" + str((resolution[2] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz_x = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=32 -D NB_III=64"
    build_options += " -D TILE_DIM_XZ=16 -D BLOCK_ROWS_XZ=1"
    build_options += " -D BLOCK_DEPH_XZ=4 -D PADDING_XZ=1"
    gwi = (int(resolution[2]),
           int(resolution[1]),
           int(resolution[0] / 4))
    lwi = (16, 1, 4)
    build_options += " -D NB_GROUPS_I=" + str(resolution[2] / lwi[0])
    build_options += " -D NB_GROUPS_III=" + str((resolution[0] / 4) / lwi[2])
    prg = cl_env.build_src(src_transpose_xz, build_options, vec)
    init_transpose_xz_z = KernelLauncher(
        prg.transpose_xz, cl_env.queue, gwi, lwi)
    _comparison(resolution, resolutionT,
                init_transpose_xz_x, init_transpose_xz_z,
                gwi, lwi, cl_env, axe=2)

