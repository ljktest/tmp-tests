"""
@file problem_with_GLRendering.py

Extends Problem description to handel real time rendering wit OpenGL.
"""
from hysop.constants import debug
from hysop.mpi import main_rank
from hysop.problem.problem import Problem


class ProblemGLRender(Problem):
    """
    For the GPU real-time rendering (i.e. use of an
    OpenGLRendering object), The loop over time-steps is passed to Qt4
    """

    @debug
    def __init__(self, operators, simulation,
                 dumpFreq=100, name=None):
        """
        Create a transport problem instance.

        @param operators : list of operators.
        @param simulation : a hysop.simulation.Simulation object
        to describe simulation parameters.
        @param name : an id for the problem
        @param dumpFreq : frequency of dump (i.e. saving to a file)
        for the problem; set dumpFreq = -1 for no dumps. Default = 100.
        """
        Problem.__init__(self, operators, simulation,
                         dumpFreq=dumpFreq,
                         name=name)
        self.gl_renderer = None

    @debug
    def setup(self):
        """
        Prepare operators (create topologies, allocate memories ...)
        """
        Problem.setup(self)
        for ope in self.operators:
            try:
                if ope.isGLRender:
                    self.gl_renderer = ope
                    ope.setMainLoop(self)
            except AttributeError:
                pass

    @debug
    def solve(self):
        """
        Solve problem.

        Performs simulations iterations by calling each
        operators of the list until timer ends.\n
        At end of time step, call an io step.\n
        Displays timings at simulation end.
        """
        if main_rank == 0:
            print ("\n\n Start solving ...")
        self.gl_renderer.startMainLoop()
