"""Constant parameters required for the hysop package.

"""
from hysop import __DEBUG__, __PROFILE__
import numpy as np
import math
from hysop.mpi import MPI

PI = math.pi
# Set default type for real and integer numbers
HYSOP_REAL = np.float64
SIZEOF_HYSOP_REAL = int(HYSOP_REAL(1.).nbytes)
# type for array indices
HYSOP_INDEX = np.uint32
# type for integers
HYSOP_INTEGER = np.int32
# integer used for arrays dimensions
HYSOP_DIM = np.int16
# float type for MPI messages
HYSOP_MPI_REAL = MPI.DOUBLE
# int type for MPI messages
HYSOP_MPI_INTEGER = MPI.INT
## default array layout (fortran or C convention)
ORDER = 'F'
# to check array ordering with :
# assert tab.flags.f_contiguous is CHECK_F_CONT
if ORDER is 'F':
    CHECK_F_CONT = True
else:
    CHECK_F_CONT = False

## Default array layout for MPI
ORDERMPI = MPI.ORDER_F
## label for x direction
XDIR = 0
## label for y direction
YDIR = 1
## label for z direction
ZDIR = 2
## Tag for periodic boundary conditions
PERIODIC = 99
## Directions string
S_DIR = ["_X", "_Y", "_Z"]
## Stretching formulation (div(w:u))
CONSERVATIVE = 1
## Stretching formulation ([grad(u)][w])
GRADUW = 0
## Optimisation level for numerics methods
## default, no optim for numerics methods
## Optimisation level for time integrators
## No need to recompute the right-hand side, an initial guess
## must be given in input arguments.
WITH_GUESS = 1
## No need to recompute the right-hand side, an initial guess
## must be given in input arguments and we ensure that
## y is different from result arg.
NOALIAS = 2

## Default value for task id (mpi task)
DEFAULT_TASK_ID = 999


#define debug decorator:
def debugdecorator(f):
    if __DEBUG__:
    ## function f is being decorated
        def decorator(*args, **kw):
            # Print informations on decorated function
            if f.__name__ is '__new__':
                fullclassname = args[0].__mro__[0].__module__ + '.'
                fullclassname += args[0].__mro__[0].__name__
                print ('Instanciate :', fullclassname,)
                print (' (Inherits from : ',)
                print ([c.__name__ for c in args[0].__mro__[1:]], ')')
            else:
                print ('Call : ', f.__name__, 'in ', f.__code__.co_filename,)
                print (f.__code__.co_firstlineno)
            ## Calling f
            r = f(*args, **kw)
            if f.__name__ is '__new__':
                print ('       |-> : ', repr(r))
            return r
        return decorator
    else:
        #define empty debug decorator:
        return f

debug = debugdecorator

# redefine profile decorator
if __PROFILE__:
    from memory_profiler import profile
    prof = profile
else:
    def prof(f):
        # Nothing ...
        return f
