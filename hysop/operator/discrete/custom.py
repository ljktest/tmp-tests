from hysop.operator.discrete.discrete import DiscreteOperator


class CustomOp(DiscreteOperator):
    def __init__(self, in_fields, out_fields, function, **kwds):
        super(CustomOp, self).__init__(**kwds)
        self.function = function
        self._in_fields = in_fields
        self._out_fields = out_fields

    def apply(self, simulation):
        self.function(simulation, self._in_fields, self._out_fields)


class CustomMonitor(DiscreteOperator):
    def __init__(self, function, res_shape=1, **kwds):
        super(CustomMonitor, self).__init__(**kwds)
        self.function = function
        self.res_shape = res_shape

    def apply(self, simulation=None):
        ite = simulation.currentIteration
        values = self.function(simulation,
                               self.variables)
        if self._writer is not None and self._writer.do_write(ite):
            self._writer.buffer[0, 0] = simulation.time
            self._writer.buffer[0, 1:] = values
            self._writer.write()
