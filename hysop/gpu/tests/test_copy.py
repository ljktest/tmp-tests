"""
@file hysop.gpu.tests.test_copy
Testing copy kernels.
"""
from hysop.gpu import cl
from hysop.constants import np
from hysop.gpu.tools import get_opencl_environment
from hysop.gpu.gpu_kernel import KernelLauncher
import hysop.tools.numpywrappers as npw


def test_copy2D():
    resolution = (256, 256)
    cl_env = get_opencl_environment()
    vec = 2
    src_copy = 'kernels/copy.cl'
    build_options = ""
    build_options += " -D NB_I=256 -D NB_II=256"
    build_options += " -D TILE_DIM_COPY=16"
    build_options += " -D BLOCK_ROWS_COPY=8"
    gwi = (int(resolution[0] / 2),
           int(resolution[1] / 2))
    lwi = (8, 8)
    prg = cl_env.build_src(src_copy, build_options, vec)
    copy = KernelLauncher(prg.copy, cl_env.queue, gwi, lwi)

    data_in = npw.asrealarray(np.random.random(resolution))
    data_out = npw.empty_like(data_in)
    data_gpu_in = cl.Buffer(cl_env.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=data_in.nbytes)
    data_gpu_out = cl.Buffer(cl_env.ctx,
                             cl.mem_flags.READ_WRITE,
                             size=data_out.nbytes)
    cl.enqueue_copy(cl_env.queue, data_gpu_in, data_in)
    cl_env.queue.finish()

    copy(data_gpu_in, data_gpu_out)
    cl_env.queue.finish()
    cl.enqueue_copy(cl_env.queue, data_out, data_gpu_out)
    cl_env.queue.finish()
    assert np.allclose(data_out, data_in)

    data_gpu_in.release()
    data_gpu_out.release()


def test_copy2D_rect():
    resolution = (256, 512)
    resolutionT = (512, 256)
    cl_env = get_opencl_environment()
    vec = 2
    src_copy = 'kernels/copy.cl'
    build_options = ""
    build_options += " -D NB_I=256 -D NB_II=512"
    build_options += " -D TILE_DIM_COPY=16"
    build_options += " -D BLOCK_ROWS_COPY=8"
    gwi = (int(resolution[0] / 2),
           int(resolution[1] / 2))
    lwi = (8, 8)
    prg = cl_env.build_src(src_copy, build_options, vec)
    copy_x = KernelLauncher(prg.copy, cl_env.queue, gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=512 -D NB_II=256"
    build_options += " -D TILE_DIM_COPY=16"
    build_options += " -D BLOCK_ROWS_COPY=8"
    gwi = (int(resolution[1] / 2),
           int(resolution[0] / 2))
    lwi = (8, 8)
    prg = cl_env.build_src(src_copy, build_options, vec)
    copy_y = KernelLauncher(prg.copy, cl_env.queue, gwi, lwi)

    data_in = npw.asrealarray(np.random.random(resolution))
    data_out = npw.empty_like(data_in)
    data_gpu_in = cl.Buffer(cl_env.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=data_in.nbytes)
    data_gpu_out = cl.Buffer(cl_env.ctx,
                             cl.mem_flags.READ_WRITE,
                             size=data_out.nbytes)
    cl.enqueue_copy(cl_env.queue, data_gpu_in, data_in)
    cl.enqueue_copy(cl_env.queue, data_gpu_out, data_out)
    cl_env.queue.finish()

    copy_x(data_gpu_in, data_gpu_out)
    cl_env.queue.finish()
    cl.enqueue_copy(cl_env.queue, data_out, data_gpu_out)
    cl_env.queue.finish()
    assert np.allclose(data_out, data_in)

    data_in = npw.asrealarray(np.random.random(resolutionT))
    data_out = npw.empty_like(data_in)
    cl.enqueue_copy(cl_env.queue, data_gpu_in, data_in)
    cl.enqueue_copy(cl_env.queue, data_gpu_out, data_out)
    cl_env.queue.finish()

    copy_y(data_gpu_in, data_gpu_out)
    cl_env.queue.finish()
    cl.enqueue_copy(cl_env.queue, data_out, data_gpu_out)
    cl_env.queue.finish()
    assert np.allclose(data_out, data_in)

    data_gpu_in.release()
    data_gpu_out.release()


def test_copy3D():
    resolution = (64, 64, 64)
    cl_env = get_opencl_environment()
    vec = 4
    src_copy = 'kernels/copy.cl'
    build_options = ""
    build_options += " -D NB_I=64 -D NB_II=64 -D NB_III=64"
    build_options += " -D TILE_DIM_COPY=16"
    build_options += " -D BLOCK_ROWS_COPY=8"
    gwi = (int(resolution[0] / 4),
           int(resolution[1] / 2),
           int(resolution[2]))
    lwi = (4, 8, 1)

    # Build code
    prg = cl_env.build_src(src_copy, build_options, vec)
    init_copy = KernelLauncher(prg.copy, cl_env.queue, gwi, lwi)

    data_in = npw.asrealarray(np.random.random(resolution))
    data_out = npw.empty_like(data_in)
    data_gpu_in = cl.Buffer(cl_env.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=data_in.nbytes)
    data_gpu_out = cl.Buffer(cl_env.ctx,
                             cl.mem_flags.READ_WRITE,
                             size=data_out.nbytes)
    cl.enqueue_copy(cl_env.queue, data_gpu_in, data_in)

    cl_env.queue.finish()
    init_copy(data_gpu_in, data_gpu_out)
    cl_env.queue.finish()
    cl.enqueue_copy(cl_env.queue, data_out, data_gpu_out)
    cl_env.queue.finish()
    assert np.allclose(data_out, data_in)

    data_gpu_in.release()
    data_gpu_out.release()


def test_copy3D_rect():
    resolution_x = (16, 32, 64)
    resolution_y = (32, 16, 64)
    resolution_z = (64, 16, 32)
    cl_env = get_opencl_environment()
    vec = 4
    src_copy = 'kernels/copy.cl'

    build_options = ""
    build_options += " -D NB_I=16 -D NB_II=32 -D NB_III=64"
    build_options += " -D TILE_DIM_COPY=16"
    build_options += " -D BLOCK_ROWS_COPY=8"
    gwi = (int(resolution_x[0] / 4),
           int(resolution_x[1] / 2),
           int(resolution_x[2]))
    lwi = (4, 8, 1)
    prg = cl_env.build_src(src_copy, build_options, vec)
    init_copy_x = KernelLauncher(prg.copy, cl_env.queue, gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=32 -D NB_II=16 -D NB_III=64"
    build_options += " -D TILE_DIM_COPY=16"
    build_options += " -D BLOCK_ROWS_COPY=8"
    gwi = (int(resolution_x[1] / 4),
           int(resolution_x[0] / 2),
           int(resolution_x[2]))
    lwi = (4, 8, 1)
    prg = cl_env.build_src(src_copy, build_options, vec)
    init_copy_y = KernelLauncher(prg.copy, cl_env.queue, gwi, lwi)

    build_options = ""
    build_options += " -D NB_I=64 -D NB_II=16 -D NB_III=32"
    build_options += " -D TILE_DIM_COPY=16"
    build_options += " -D BLOCK_ROWS_COPY=8"
    gwi = (int(resolution_x[2] / 4),
           int(resolution_x[0] / 2),
           int(resolution_x[1]))
    lwi = (4, 8, 1)
    prg = cl_env.build_src(src_copy, build_options, vec)
    init_copy_z = KernelLauncher(prg.copy, cl_env.queue, gwi, lwi)

    data_in = npw.asrealarray(np.random.random(resolution_x))
    data_out = np.empty_like(data_in)
    data_gpu_in = cl.Buffer(cl_env.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=data_in.nbytes)
    data_gpu_out = cl.Buffer(cl_env.ctx,
                             cl.mem_flags.READ_WRITE,
                             size=data_out.nbytes)
    cl.enqueue_copy(cl_env.queue, data_gpu_in, data_in)
    cl.enqueue_copy(cl_env.queue, data_gpu_out, data_out)

    cl_env.queue.finish()
    init_copy_x(data_gpu_in, data_gpu_out)
    cl_env.queue.finish()
    cl.enqueue_copy(cl_env.queue, data_out, data_gpu_out)
    cl_env.queue.finish()
    assert np.allclose(data_out, data_in)

    data_in = npw.asrealarray(np.random.random(resolution_y))
    data_out = npw.empty_like(data_in)
    cl.enqueue_copy(cl_env.queue, data_gpu_in, data_in)
    cl.enqueue_copy(cl_env.queue, data_gpu_out, data_out)
    cl_env.queue.finish()
    init_copy_y(data_gpu_in, data_gpu_out)
    cl_env.queue.finish()
    cl.enqueue_copy(cl_env.queue, data_out, data_gpu_out)
    cl_env.queue.finish()
    assert np.allclose(data_out, data_in)

    data_in = npw.asrealarray(np.random.random(resolution_z))
    data_out = npw.empty_like(data_in)
    cl.enqueue_copy(cl_env.queue, data_gpu_in, data_in)
    cl.enqueue_copy(cl_env.queue, data_gpu_out, data_out)
    cl_env.queue.finish()
    init_copy_z(data_gpu_in, data_gpu_out)
    cl_env.queue.finish()
    cl.enqueue_copy(cl_env.queue, data_out, data_gpu_out)
    cl_env.queue.finish()
    assert np.allclose(data_out, data_in)

    data_gpu_in.release()
    data_gpu_out.release()
