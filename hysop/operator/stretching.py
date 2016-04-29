# -*- coding: utf-8 -*-
"""Computation of the stretching term in Navier-Stokes.

See also
--------

* :ref:`stretching` in HySoP user guide.


"""
from hysop.constants import debug
from hysop.methods_keys import TimeIntegrator, Formulation, \
    SpaceDiscretisation
from hysop.numerics.finite_differences import FD_C_4
from hysop.operator.computational import Computational
from hysop.operator.continuous import opsetup
from hysop.operator.discrete.stretching import Conservative, GradUW
from hysop.operator.discrete.stretching import StretchingLinearized as StretchLinD
from hysop.numerics.integrators.euler import Euler
import hysop.numerics.differential_operations as diff_op



class Stretching(Computational):
    """
    """

    @debug
    def __init__(self, velocity, vorticity, **kwds):
        """
        Parameters
        -----------
        velocity, vorticity : :class:`~hysop.fields.continuous.Field`
        **kwds : extra parameters for base class

        Notes
        -----
        * The default formulation is the 'Conservative' one.
        * The default solving method is finite differences, 4th order, in space
          and Runge-Kutta 3 in time.

        """
        assert 'variables' not in kwds, 'variables parameter is useless.'
        super(Stretching, self).__init__(variables=[velocity, vorticity],
                                         **kwds)
        # velocity variable (vector)
        self.velocity = velocity
        # vorticity variable (vector)
        self.vorticity = vorticity
        # Numerical methods for time and space discretization

        if self.method is None:
            import hysop.default_methods as default
            self.method = default.STRETCHING
        assert Formulation in self.method.keys()
        assert SpaceDiscretisation in self.method.keys()
        assert TimeIntegrator in self.method.keys()

        # Formulation used for the stretching equation.
        # Default = conservative form.
        if self.method[Formulation] == "GradUW":
            self.formulation = GradUW
        else:
            self.formulation = Conservative

        self.input = [self.velocity, self.vorticity]
        self.output = [self.vorticity]

    def get_work_properties(self):
        """
        Get properties of internal work arrays. Must be call after discretize
        but before setup.

        Returns
        -------
        dictionnary
           keys = 'rwork' and 'iwork', values = list of shapes of internal
           arrays required by this operator (real arrays for rwork, integer
           arrays for iwork).

        Example
        -------

        >> works_prop = op.get_work_properties()
        >> print works_prop
        {'rwork': [(12, 12), (45, 12, 33)], 'iwork': None}

        means that the operator requires two real arrays of
        shape (12,12) and (45, 12, 33) and no integer arrays

        """
        if not self._is_discretized:
            msg = 'The operator must be discretized '
            msg += 'before any call to this function.'
            raise RuntimeError(msg)
        # Get fields local shape.
        vd = self.discreteFields[self.velocity]
        shape_v = vd[0][...].shape
        # Collect info from numerical methods
        # --> time-integrator required work space
        ti = self.method[TimeIntegrator]
        rwork_length = ti.getWorkLengths(3)
        # ---> differential operator work space
        if self.formulation is GradUW:
            rwork_length += diff_op.GradVxW.get_work_length()
        elif self.formulation is Conservative:
            rwork_length += diff_op.DivWV.get_work_length()
        res = {'rwork': [], 'iwork': None}
        # Set rwork shapes
        for _ in xrange(rwork_length):
            res['rwork'].append(shape_v)
        return res

    @debug
    def discretize(self):
        if self.method[SpaceDiscretisation] is FD_C_4:
            nbghosts = 2
        else:
            raise ValueError("Unknown method for space discretization of the\
                stretching operator.")

        super(Stretching, self)._standard_discretize(nbghosts)

    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        self.discrete_op =\
            self.formulation(velocity=self.discreteFields[self.velocity],
                             vorticity=self.discreteFields[self.vorticity],
                             method=self.method, rwork=rwork, iwork=iwork)
        self._is_uptodate = True


class StretchingLinearized(Stretching):
    """
    Solve the linearized stretching equation, i.e:
    \f{eqnarray*}
    \frac{\partial \omega}{\partial t} &=& (\om \cdot \nabla)u_b +
    (\om_b \cdot \nabla)u
    \f}
    """
    
    @debug
    def __init__(self, velocity_BF, vorticity_BF, **kwds):
        """
        Parameters
        -----------
        velocity_BF, vorticity_BF : base flow fields :
        class:`~hysop.fields.continuous.Field`
        **kwds : extra parameters for base class
            
        Notes
        -----
        * The default formulation is the 'Conservative' one.
        * The default solving method is finite differences, 4th order,
        in space and Runge-Kutta 3 in time.
            
        """
        super(StretchingLinearized, self).__init__(**kwds)
        self.variables[velocity_BF] = None
        self.variables[vorticity_BF] = None

        # Base flow velocity variable (vector)
        self.velocity_BF = velocity_BF
        # Base flow vorticity variable (vector)
        self.vorticity_BF = vorticity_BF
        
        # Usual stretching operator to compute the
        # first term of the linearized rhs: (w'.grad)ub
        self.usual_stretch = Stretching(self.velocity_BF, self.vorticity,
                                        discretization=self._discretization)

        self.input.append(self.velocity_BF)
        self.input.append(self.vorticity_BF)
   
    @debug
    def discretize(self):
        if self.method[SpaceDiscretisation] is FD_C_4:
            nbghosts = 2
        else:
            raise ValueError("Unknown method for space discretization of the\
                         stretching operator.")
        self.usual_stretch.discretize()
        super(StretchingLinearized, self)._standard_discretize(nbghosts)


    @debug
    @opsetup
    def setup(self, rwork=None, iwork=None):
        # Setup of the usual stretching operator
        self.usual_stretch.setup()

        # Setup of a second stretching operator (receiving 3 input variables)
        # to compute the 2nd term of the linearized rhs: (wb.grad)u'
        method_lin = self.method.copy()
        method_lin[TimeIntegrator] = Euler
        self.discrete_op =\
            StretchLinD(velocity=self.discreteFields[self.velocity],
                        vorticity=self.discreteFields[self.vorticity],
                        vorticity_BF=self.discreteFields[self.vorticity_BF],
                        usual_op=self.usual_stretch.discrete_op,
                        method=method_lin, rwork=rwork, iwork=iwork)
        self._is_uptodate = True


