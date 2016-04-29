#
# Definition of the adaptative time step (for example) on the 3D domain:
# \code
# dt_adapt = Variable(dom3D, name="adaptative_timestep, data=[0.1]")
# ...
# \endcode
#
#
"""
@file variable_parameter.py

Tools to define parameters that do not depend on space but may
be used as input-output parameter during simulation.

Example : the time step.

\code
data = {'timestep': 0.1 }
dt = VariableParameter(data)

## An operator that can change dt
op = Adaptative_timestep(..., dt)
simu = Simulation(..., dt, ...)

op.apply(simu) ===> change simu.time_step

\endcode

A python function may be set to compute the parameter.
Example : if you want to define a time-dependent parameter
alpha, such that alpha = [sin(t), cos(t)], do :

\code
def myfunc(simu):
    return sin(simu.time), cos(simu.time)


alpha = VariableParameter(formula=myfunc, name='alpha')

simu = Simulation(tstart = 0., tend=1., timeStep=0.1)


print alpha
# --> return {'alpha': None}

alpha.update(simu)
print alpha
# --> return {'alpha': (0., 1.)}

simu.advance()
alpha.update(simu)
print alpha
# --> return {'alpha': (0.19866933079506122, 0.9800665778412416)}

"""


class VariableParameter(object):
    """
    Class to embed a user-defined parameter (a dictionnary indeed).

    VariableParameter has a member data which is mutable.
    """

    def __init__(self, data=None, name=None, formula=None):
        """
        Creates a dictionnary with data
        @param data: the data used as parameter
        @param name : optional name, used if data is not a dict.

        If data is a dictionnary, self.data = data, that's it.
        Else data will be added into self.data dict, with name as a key.
        """
        ## name of the variable
        self.name = name
        if self.name is None and formula is not None:
            self.name = formula.__name__
        ## Formula used to compute data (a python function)
        self.formula = formula
        if self.formula is None:
            self.formula = self._constant
        ## data values
        if isinstance(data, dict):
            msg = 'A dictionnary is used to initialize the variable.'
            msg += 'Name parameter will be ignored.'
            print (msg)
            self.data = data
            self.name = data.keys()[0]
        else:
            if self.name is None:
                msg = "Name arg is required when data is not a dict."
                raise AttributeError(msg)
            self.data = {self.name: data}

    def _constant(self, simu=None):
        return self.data[self.name]

    def update(self, simu=None):
        """
        Apply formula to compute data for
        a given simulation (current time ...)
        @param simu : a hysop.problem.simulation.Simulation
        """
        self.data[self.name] = self.formula(simu)

    def __getitem__(self, key):
        """ Access to the content of the data[key]
        @param key : requested key.
        @return component 'key' of the dict
        """
        return self.data[key]

    def __setitem__(self, key, value):
        """
        Set the data[key] = value
        """
        self.data[key] = value

    def __str__(self):
        return str(self.data)
