"""Python package dedicated to flow simulation using particular methods
on hybrid architectures (MPI-GPU)


"""
# Compilation flags
__MPI_ENABLED__ = "ON" is "ON"
__GPU_ENABLED__ = "ON" is "ON"
__FFTW_ENABLED__ = "ON" is "ON"
__SCALES_ENABLED__ = "ON" is "ON"
__VERBOSE__ = "OFF" in ["1", "3"]
__DEBUG__ = "OFF" in ["2", "3"]
__PROFILE__ = "OFF" in ["0", "1"]
__OPTIMIZE__ = "OFF" is "ON"

import os
from hysop.tools.sys_utils import SysUtils
# Box-type physical domain
from hysop.domain.box import Box
# Fields
from hysop.fields.continuous import Field
# Variable parameters
from hysop.fields.variable_parameter import VariableParameter
# Simulation parameters
from hysop.problem.simulation import Simulation
# Tools (io, mpi ...)
from hysop.tools.io_utils import IO, IOParams
from hysop.tools.parameters import MPIParams, Discretization
import hysop.mpi
# Problem
from hysop.problem.problem import Problem
# Solver
# import particular_solvers.basic
# ## #import particular_solvers.gpu
# ParticleSolver = particular_solvers.basic.ParticleSolver
# ## #GPUParticleSolver = particular_solvers.gpu.GPUParticleSolver
## from tools.explore_hardware import explore


__all__ = ['Box', 'Field', 'Discretization',
           'IOParams', 'Simulation', 'MPIParams', 'Problem', 'IO']

if SysUtils.is_interactive():
    # Set i/o default path to current directory
    # for interactive sessions
    # i.e. python interactive session or any call of ipython.
    defpath = os.path.join(os.getcwd(), 'interactive')
    IO.set_default_path(defpath)

default_path = IO.default_path()
msg_start = '\nStarting hysop version '
msg_start += str("2.0.0")
msg_io = '\nWarning : default path for all i/o is ' + default_path + '.\n'
msg_io += 'If you want to change this, use io.set_default_path function.\n'

# MPI
if __MPI_ENABLED__:
    if mpi.main_rank == 0:
        msg_start += ' with ' + str(mpi.main_size) + ' mpi process(es).'
        print msg_start
        print msg_io

else:
    print msg_start
    print msg_io

# OpenCL
__DEFAULT_PLATFORM_ID__ = 0
__DEFAULT_DEVICE_ID__ = 2


version = "1.0.0"

