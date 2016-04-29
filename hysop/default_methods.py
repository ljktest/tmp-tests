"""
@file default_methods.py
Default parameter values for methods in operators.
"""
from hysop.methods_keys import TimeIntegrator, Interpolation, GhostUpdate,\
    Remesh, Support, Splitting, MultiScale, Formulation, SpaceDiscretisation, \
    dtCrit, Precision
from hysop.constants import HYSOP_REAL
from hysop.numerics.integrators.runge_kutta2 import RK2
from hysop.numerics.integrators.runge_kutta3 import RK3
from hysop.numerics.interpolation import Linear
from hysop.numerics.remeshing import L2_1, Linear as Rmsh_Linear
#from hysop.operator.discrete.stretching import Conservative


ADVECTION = {TimeIntegrator: RK2, Interpolation: Linear,
             Remesh: L2_1, Support: '', Splitting: 'o2', MultiScale: L2_1,
             Precision: HYSOP_REAL}

from hysop.numerics.finite_differences import FD_C_4, FD_C_2

DIFFERENTIAL = {SpaceDiscretisation: FD_C_4, GhostUpdate: True}

ADAPT_TIME_STEP = {TimeIntegrator: RK3, SpaceDiscretisation: FD_C_4,
                   dtCrit: 'vort'}

BAROCLINIC = {SpaceDiscretisation: FD_C_4}

MULTIPHASEBAROCLINIC = {SpaceDiscretisation: FD_C_4}

MULTIPHASEGRADP = {SpaceDiscretisation: FD_C_4}

DIFFUSION = {SpaceDiscretisation: 'fftw', GhostUpdate: True}

POISSON = {SpaceDiscretisation: 'fftw', GhostUpdate: True,
           Formulation: 'velocity'}

STRETCHING = {TimeIntegrator: RK3, Formulation: "Conservative",
              SpaceDiscretisation: FD_C_4}

FORCES = {SpaceDiscretisation: FD_C_2}

MULTIRESOLUTION_FILER = {Remesh: Rmsh_Linear, }
