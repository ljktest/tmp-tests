"""
@file simulation.py

Description of the simulation parameters (time, iteration ...)
"""
import sys
eps = sys.float_info.epsilon


class Simulation(object):
    """
    Setup for simulation parameters.
    """

    def __init__(self, tinit=0.0, tend=1.0, nbIter=None, timeStep=None,
                 iterMax=1000):
        """
        Creates a Timer.
        @param tinit : Simulation starting time.
        @param tend : Simulation final time.
        @param nbIter : number of required iterations
        @param timeStep : default time step
        @param iterMax : maximum number of iterations (useful
        only if timeStep is modified. See adaptativeTimeStep operator).

        If both timeStep and nbIter are given, timeStep is not used.

        Notation:
        iteration number 'currentIteration'
        between tk and tkp1 = tk + timeStep
        """
        ## Simulation final time
        self.end = tend
        ## Starting time
        self.start = tinit
        ## Simulation current time
        self.time = tinit
        ## Is simulation is terminated
        self.isOver = False
        ## Iteration counter
        self.currentIteration = -1
        ## Number of iterations
        if nbIter is not None:
            self.nbIter = nbIter
            if nbIter is not None and timeStep is not None:
                print ('Warning : both nbIter and timeStep are given.\
                timeStep is ignored')
            self.timeStep = (self.end - self.start) / self.nbIter
        elif timeStep is not None:
            ## Simulation time step
            self.timeStep = timeStep
        else:
            raise ValueError('You must set nbIter or timeStep value.')
        self.iterMax = iterMax
        assert iterMax >= nbIter
        ## Starting time for the current iteration
        self.tk = tinit
        ## tk + dt
        self.tkp1 = tinit + self.timeStep
        self._lastStep = False
        assert self.end > self.start, \
            'Final time must be greater than initial time'
        assert (self.start + self.timeStep) <= self.end,\
            'start + step is bigger than end.'

        ## Internal tolerance for timer
        self.tol = eps

    def advance(self):
        """
        Proceed to next time.
        Advance time and iteration number.
        Compute a timestep for the incoming iteration (from a vairable and
        to reach the end of simulation)
        """
        # Increment iteration counter
        self.currentIteration += 1
        if self._lastStep:
            # The timestep was adjusted to reach end in the previous call
            # So now the simulation is over
            self.isOver = True
        else:
            if self.currentIteration < self.iterMax:
                # Advance time for the iteration just ended
                self.tk = self.tkp1
                self.tkp1 = self.tk + self.timeStep

                # Adjust last timestep to reach self.end
                if self.tkp1 > self.end:
                    self.timeStep = self.end - self.tk
                    if self.timeStep <= self.tol:
                        self.isOver = True
                    else:
                        self.tkp1 = self.end
                        self._lastStep = True
            else:
                # iteration number is reached
                self.isOver = True

        self.time = self.tkp1

    def updateTimeStep(self, newDt):
        """
        Update current time step.
        This function is usually called from Adapt_timestep operator.
        """
        self.timeStep = newDt

    def printState(self):
        """
        print current state
        """
        msg = "== Iteration : {0:3d}, from t = {1:6.5} to t = {2:6.5f} =="
        print (msg.format(self.currentIteration, self.tk, self.time))

    def __str__(self):
        s = "Simulation parameters : "
        s += "from " + str(self.start) + ' to ' + str(self.end)
        s += ', time step : ' + str(self.timeStep)
        s += ', current time : ' + str(self.time) + ', iteration number : '
        s += str(self.currentIteration) + ', max number of iterations : '
        s += str(self.iterMax)
        return s

    def initialize(self):
        self.tk = self.start
        self.tkp1 = self.start + self.timeStep
        self.time = self.tkp1
        self.isOver = False
        self.currentIteration = 0
        self._lastStep = False

    def finalize(self):
        """
        Useful for printers if printer.frequency is not 1
        """
        self.isOver = True
        self.currentIteration = -1
