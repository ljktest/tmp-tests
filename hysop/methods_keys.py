"""
@file methods_keys.py
A list of authorized keys that may be used to set methods
in operators.
Usage:
method = {key: value, ...}
Key must be one of the constants given below. Value is usually a class name
and sometimes a string. See details in each operator.

Example, the stretching case :
method = {TimeIntegrator: RK3, Formulation: Conservative,
          SpaceDiscretisation: FD_C_4}


"""
## Authorized keys for method.
## Time integrator scheme (see hysop.numerics.integrators for
## available names)
TimeIntegrator = 11111
## Remeshing scheme (hysop.numerics.remeshing)
Remesh = 22222
## Interpolation scheme (hysop.numerics.interpolation)
Interpolation = 33333
## Formulation (example in stretching : either Conservative or GradUW)
Formulation = 44444
## Space discretisation method
## (see for example hysop.numerics.finite_differences)
SpaceDiscretisation = 55555
## Scales method parameters
Scales = 66666
## Splitting method
Splitting = 77777
## Device (either GPU or CPU)
Support = 88888
## method[GhostUpdate] = True if the operator deals with
## its ghost points update
GhostUpdate = 99999
## List of criterions for adaptative time step computation
dtCrit = 0
## Multiscale method for multiscale advection
MultiScale = 12345
## Float precision for advection
Precision = 12346
## Extra arguments for discrete operators
ExtraArgs = 9876
