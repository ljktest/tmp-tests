"""
@file redistribute.py
Abstract interface for data redistribution.
"""

from hysop.operator.continuous import Operator
from abc import ABCMeta, abstractmethod
from hysop.mpi.topology import Cartesian
from hysop.operator.computational import Computational


class Redistribute(Operator):
    """
    Bare interface to redistribute operators
    """

    __metaclass__ = ABCMeta

    def __init__(self, source, target, component=None,
                 run_till=None, **kwds):
        """
        @param source : topology or computational operator
        @param target : topology or computational operator
        @param component : which component must be distributed (default = all)
        @param run_till : a list of operators that must wait for the completion
        of this redistribute before any apply.
        """
        # Base class initialisation
        super(Redistribute, self).__init__(**kwds)

        # Object (may be an operator or a topology) which handles the
        # fields to be transfered
        self._source = source
        # Object (may an operator or a topology) which handles the fields
        # to be filled in from source.
        self._target = target

        self.component = component
        if component is None:
            # All components are considered
            self._range_components = lambda v: xrange(v.nb_components)
        else:
            # Only the given component is considered
            assert self.component >= 0, 'component value must be positive.'
            self._range_components = lambda v: (self.component)

        ## Bridge between topology of source and topology of target
        self.bridge = None
        # True if some MPI operations are running for the current operator.
        self._has_requests = False
        # Which operator must wait for this one before
        # any computation
        # Exp : run_till = op1 means that op1 will
        # wait for the end of this operator before
        # op1 starts its apply.
        if run_till is None:
            run_till = []

        assert isinstance(run_till, list)
        self._run_till = run_till

    @abstractmethod
    def setup(self, rwork=None, iwork=None):
        """
        Check/set the list of variables to be distributed

        What must be set at setup?
        ---> the list of continuous variables to be distributed
        ---> the bridge (one for all variables, which means
        that all vars must have the same topology in source
        and the same topology in target.
        ---> the list of discrete variables for source and
        for target.
        """
        assert self.domain is not None
        for v in self.variables:
            assert v.domain is self.domain
        super(Redistribute, self).setup(rwork, iwork)

    def _check_operator(self, op):
        """
        @param op : a computational operator
        - check if op is really a computational operator
        - discretize op
        - check if all required variables (if any) belong to op
        """
        assert isinstance(op, Computational)
        op.discretize()
        msg = 'The variables to be distributed '
        msg += 'do not belong to the input operator.'
        if len(self.variables) > 0:
            assert all(v in op.variables for v in self.variables), msg

    def _set_variables(self):
        """
        Check/set the list of variables proceed by the current operator.
        """
        # Set list of variables.
        # It depends on :
        # - the type of source/target : Cartesian, Computational or None
        # - the args variables : a list of variables or None
        # Possible cases:
        # - if source or target is None --> variables is required
        # - if source and target are Cartesian --> variables is required
        # - in all other cases, variables is optional.
        # If variables are not set at init,
        # they must be infered from source/target operators.
        has_var = len(self.variables) > 0
        vlist = (v for v in self.variables)

        if self._source is None or self._target is None:
            assert len(self.variables) > 0
            self.variables = [v for v in vlist]
        else:
            source_is_topo = isinstance(self._source, Cartesian)
            target_is_topo = isinstance(self._target, Cartesian)

            # both source and target are topologies. Variables required.
            if source_is_topo and target_is_topo:
                msg = 'Redistribute, a list of variables is required at init.'
                assert has_var, msg
                self.variables = [v for v in vlist]

            elif not source_is_topo and not target_is_topo:
                # both source and target are operators
                # --> intersection of their variables
                vsource = self._source.variables
                vtarget = self._target.variables
                if not has_var:
                    vlist = (v for v in vsource if v in vtarget)
                self.variables = [v for v in vlist]

            elif source_is_topo:
                # source = topo, target = operator
                vtarget = self._target.variables
                if not has_var:
                    vlist = (v for v in vtarget)
                self.variables = [v for v in vlist]

            else:
                # source = operator, target = topo
                vsource = self._source.variables
                if not has_var:
                    vlist = (v for v in vsource)
                self.variables = [v for v in vlist]

        assert len(self.variables) > 0

        # Variables is converted to a dict to be coherent with
        # computational operators ...
        self.variables = {key: None for key in self.variables}

        # All variables must have the same domain
        self.domain = self.variables.keys()[0].domain
        for v in self.variables:
            assert v.domain is self.domain

    def _set_topology(self, current):
        """
        @param current: a topology or a computational operator
        This function check if current is valid, fits with self.variables
        and get its topology to set self._topology.
        """
        if isinstance(current, Cartesian):
            result = current
            for v in self.variables:
                v.discretize(result)
        elif isinstance(current, Computational):
            self._check_operator(current)
            vref = self.variables.keys()[0]
            vcurrent = current.variables
            result = vcurrent[vref]
            # We ensure that all vars have
            # the same topo in target/target.
            for v in (v for v in self.variables if v is not vref):
                assert vcurrent[v] is result
        else:
            msg = "the source/target is neither an operator or a topology."
            raise AttributeError(msg)
        assert result.task_id() == self.domain.current_task()
        return result

    def computation_time(self):
        pass
