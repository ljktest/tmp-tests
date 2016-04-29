"""Everything concerning GPU in hysop.

OpenCL sources are located in the cl_src directory and organized as follows
  - kernels/
  Contains kernels src
    - advection.cl
    - remeshing.cl
    - advection_and_remeshing.cl
    - rendering.cl
  Functions used by kernels
  - advection/
    - builtin.cl
    - builtin_noVec.cl
  - remeshing/
    - basic.cl
    - basic_noVec.cl
    - private.cl
    - weights.cl
    - weights_builtin.cl
    - weights_noVec.cl
    - weights_noVec_builtin.cl
  - common.cl

Sources are parsed at build to handle several OpenCL features
see hysop.gpu.tools.parse_file

"""
import pyopencl
import pyopencl.tools
import pyopencl.array
## open cl underlying implementation
cl = pyopencl
## PyOpenCL tools
clTools = pyopencl.tools
## PyOpenCL arrays
clArray = pyopencl.array

import os
## GPU deflault sources
GPU_SRC = os.path.join(__path__[0], "cl_src", '')

## If use OpenCL profiling events to time computations
CL_PROFILE = False
