"""
Testing hysop.field.variable_parameter.Variable_parameter
"""
from hysop.fields.variable_parameter import VariableParameter
from hysop.problem.simulation import Simulation
from math import sin, cos
import numpy as np


def test_constant_var():
    uinf = 1.
    var = VariableParameter(data=uinf, name='uinf')
    assert var['uinf'] == 1.
    var['uinf'] *= 3
    assert var['uinf'] == 3.

    v2 = VariableParameter({'uinf': uinf})
    assert v2['uinf'] == 1.


def func(simu):
    time = simu.tk
    return np.asarray((sin(time), cos(time)))


def test_time_var():
    var = VariableParameter(formula=func)
    simu = Simulation(tinit=0., tend=1., timeStep=0.1)
    var.update(simu)
    assert np.allclose(var['func'], [0., 1.])
    assert var.name == 'func'
    simu.advance()
    var.update(simu)
    assert np.allclose(var['func'], [sin(0.1), cos(0.1)])
    var = VariableParameter(formula=func, name='nn')
    assert var.name == 'nn'
    assert var.formula is not None


def test_time_var2():
    var = VariableParameter(formula=func, name='toto', data={'alpha': 1.})
    simu = Simulation(tinit=0., tend=1., timeStep=0.1)
    assert var.name == 'alpha'
    assert np.allclose(var['alpha'], [1.])
    var.update(simu)
    assert np.allclose(var['alpha'], [sin(0.), cos(0.)])
    simu.advance()
    var.update(simu)
    assert np.allclose(var['alpha'], [sin(0.1), cos(0.1)])
