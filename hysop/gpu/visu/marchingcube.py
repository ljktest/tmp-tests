"""
@file marchingcube.py
Marching cube algorithm to compute isosurface of particle field
"""

# Math functions
from math import log

# System call
import ctypes

from hysop.constant import HYSOP_REAL
from hysop.gpu import cl
from hysop.gpu.tools import get_opencl_environment


class Marching_Cube(object):
    """
    Implement marching cube infrastructure for detecting
    isosurface on GPU
    """
    def __init__(self, size):
        """
        Build necessary data structure for holding
        hierarchical data
        """
        self._size_ = size
        self.buffers = []
        self.usr_src = "gpu-mc.cl"
        self._cl_env = get_opencl_environment(0, 0, 'gpu', HYSOP_REAL)
        self._create_cl_context_()

    def _create_cl_context_(self):
        """
        Initialize buffer pyramid storing image particle count
        """
        buffer_size = self._size_
        pitch_size = buffer_size * ctypes.c_ubyte
        shap = (buffer_size, buffer_size, buffer_size)
        pitc = (pitch_size, pitch_size)
        img_format = cl.get_supported_image_formats(
            self._cl_env.ctx, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE3D)[0]
        img_flag = cl.mem_flags.READ_WRITE
        self.buffers.append(cl.Image(self._cl_env.ctx, img_flag, img_format, shap))

        buffer_size /= 8
        shap = (buffer_size, buffer_size, buffer_size)
        self.buffers.append(cl.Image(self._cl_env.ctx, img_flag, img_format, shap))

        buffer_size /= 8
        shap = (buffer_size, buffer_size, buffer_size)
        pitch_size = buffer_size * ctypes.c_ubyte
        self.buffers.append(cl.Image(self._cl_env.ctx, img_flag, img_format, shap))

        buffer_size /= 8
        shap = (buffer_size, buffer_size, buffer_size)
        pitch_size = buffer_size * ctypes.c_ubyte
        #self.buffers.append(cl.Image(self._cl_env.ctx, img_flag, img_format, shap))

        buffer_size /= 8
        pitch_size = buffer_size * ctypes.c_ubyte
        shap = (buffer_size, buffer_size, buffer_size)
        #self.buffers.append(cl.Image(self._cl_env.ctx, img_flag, img_format, shap))

        for i in range(5,int(log(self._size_, 2))):
            buffer_size /= 8
            pitch_size = buffer_size * ctypes.c_ubyte
            shap = (buffer_size, buffer_size, buffer_size)
            #self.buffers.append(cl.Image(self._cl_env.ctx, img_flag, img_format, shap))

        # Add cube index
        buffer_size = self._size_ * self._size_ * self._size_
        pitch_size = buffer_size * ctypes.c_ubyte
        shap = (buffer_size, buffer_size, buffer_size)
        #self.buffers.append(cl.Image(self._cl_env.ctx, img_flag, img_format, shap))


        self.gpu_src = ""
        ## Build code
        #self.usr_src.
        options = "-D HP_SIZE=" + str(self._size_)
        self._cl_env.macros['**HP_SIZE**'] = self._size_
        self.prg = self._cl_env.build_src(self.usr_src, options)
        kernel_name = 'constructHPLevel' + self.field.name.split('_D')[0]
        self.numMethod = KernelLauncher(eval('self.prg.' + kernel_name),
                                        self.queue,
                                        self.gwi,
                                        self.lwi)
        kernel_name = 'classifyCubes' + self.field.name.split('_D')[0]
        kernel_name = 'traverseHP' + self.field.name.split('_D')[0]

if __name__ == "__main__":

    mc = Marching_Cube(256)

    print mc.gpu_src
