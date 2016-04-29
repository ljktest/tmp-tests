## @package hysop.operator.discrete
# Discrete operators classes.
#
# A DiscreteOperator is an object that represents the discretisation of a
# continuous operator for a specific given method and of its variables for
# some resolutions.
#
# Example: if we want to perform the advection of a scalar at
# velocity v using scales with M4 remesh, we define the following continuous
# operator :
#\code
# nbElem = [65, 65, 65]
# advec = Advection(velo, scal,
#                   resolutions={velo: nbElem,
#                                scal: nbElem},
#                   method = 'scales, p_M4',)
# ...
# advec.setup()
# ...
# advec.apply()
# \endcode
#
# setup call will result in the creation of a ScalesAdvection operator
# and apply will perform a call to scale's solver.
