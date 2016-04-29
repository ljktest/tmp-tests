"""
@file navier_stokes.py
"""
from hysop.problem.problem import Problem
from hysop.operator.analytic import Analytic
from hysop.operator.advection import Advection
from hysop.operator.stretching import Stretching
from hysop.operator.poisson import Poisson
from hysop.operator.diffusion import Diffusion
from hysop.operator.penalization import Penalization


class NSProblem(Problem):
    """
    Navier Stokes problem description.
    """
    def __init__(self, operators, simulation,
                 dumpFreq=100, name=None):

        Problem.__init__(self, operators, simulation, dumpFreq, name)
        for op in operators:
            if isinstance(op, Advection):
                self.advection = op
            if isinstance(op, Stretching):
                self.stretch = op
            if isinstance(op, Diffusion):
                self.diffusion = op
            if isinstance(op, Poisson):
                self.poisson = op
            if isinstance(op, Penalization):
                self.penal = op
            if isinstance(op, Analytic):
                self.velocity = op
