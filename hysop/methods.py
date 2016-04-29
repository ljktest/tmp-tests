"""
@file methods.py
A list of numerical methods available in HySoP that may be used to set methods
in operators.
Usage:
method = {key: value, ...}
Keys must be one of the constants given in methods_keys.py.
Value is usually a class name
and sometimes a string. See details in each operator.

Example, the stretching case :
method = {TimeIntegrator: RK3, Formulation: Conservative,
          SpaceDiscretisation: FD_C_4}

Note FP: to avoid cycling, this file must never be imported
inside a HySoP module. It's only a review of all the methods
that can be imported by final user.
"""

import hysop.numerics.integrators.runge_kutta2 as runge_kutta2
RK2 = runge_kutta2.RK2
import hysop.numerics.integrators.runge_kutta3 as runge_kutta3
RK3 = runge_kutta3.RK3
import hysop.numerics.integrators.runge_kutta4 as runge_kutta4
RK4 = runge_kutta4.RK4
import hysop.numerics.integrators.euler as euler
Euler = euler.Euler

# Remesh
import hysop.numerics.remeshing as remesh
L2_1 = remesh.L2_1
L2_2 = remesh.L2_2
L2_3 = remesh.L2_3
L2_4 = remesh.L2_4
L4_2 = remesh.L4_2
L4_3 = remesh.L4_3
L4_4 = remesh.L4_4
L6_3 = remesh.L6_3
L6_4 = remesh.L6_4
L6_5 = remesh.L6_5
L6_6 = remesh.L6_6
L8_4 = remesh.L8_4
M8Prime = remesh.M8Prime
Rmsh_Linear = remesh.Linear
# A completer ...

# Interpolation
import hysop.numerics.interpolation as interpolation
Linear = interpolation.Linear

# Finite differences
import hysop.numerics.finite_differences as fd
FD_C_4 = fd.FD_C_4
FD_C_2 = fd.FD_C_2
FD2_C_2 = fd.FD2_C_2

# Stretching formulations
import hysop.operator.discrete.stretching as strd
Conservative = strd.Conservative
GradUW = strd.GradUW
