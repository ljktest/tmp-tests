"""
@file transport.py
"""
from hysop.problem.problem import Problem
from hysop.operator.advection import Advection
from hysop.operator.analytic import Analytic


class TransportProblem(Problem):
    """
    Transport problem description.
    """
    def __init__(self, operators, simulation,
                 dumpFreq=100, name=None):
        super(TransportProblem, self).__init__(
            operators, simulation,
            dumpFreq=dumpFreq, name="TransportProblem")
        self.advection, self.velocity = None, None
        for op in self.operators:
            if isinstance(op, Advection):
                self.advection = op
            if isinstance(op, Analytic):
                self.velocity = op
        if self.advection is None:
            raise ValueError("Transport problem with no Advection operator")
