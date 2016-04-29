"""
@package hysop.numerics.integrators
ODE integrators.

\todo write a proper doc with the list of available solvers
and their param (optim and so on)

optim value :
- None : no work, no result, no guess
- LEVEL1 : result as input/output, no work, no guess
- LEVEL2 : result as input/output, work as input, no guess.
- WITH_GUESS : LEVEL1 + LEVEL2 + first eval of rhs in result.

Length of work list depends on integrator's type :

- RK4 : work must be at least of length 2*nb_components
- RK3 : work must be at least of length nb_components
- RK2 :
- Euler :

Work can be used to give extra parameters.

"""
