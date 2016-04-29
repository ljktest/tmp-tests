"""
@file advectionDir.py

Advection of a field in a single direction.
"""
from hysop.constants import debug, S_DIR
from hysop.methods_keys import Support, MultiScale, \
    TimeIntegrator, Interpolation
from hysop.numerics.remeshing import L2_1, L4_2, L4_4, Remeshing, Linear
from hysop.operator.computational import Computational
# To get default method values for advection:
import hysop.default_methods as default
import numpy as np
from hysop.operator.continuous import opsetup, opapply


class AdvectionDir(Computational):
    """
    Advection of a field,
    \f{eqnarray*}
    X = Op(X,velocity)
    \f} for :
    \f{eqnarray*}
    \frac{\partial{X}}{\partial{t}} + velocity.\nabla X = 0
    \f}
    Note : we assume incompressible flow.

    Computations are performed in a given direction.
    """

    @debug
    def __init__(self, velocity, direction, advected_fields=None,
                 name_suffix='', **kwds):
        ## Get the other arguments to pass to the discrete operators
        self._kwds = kwds.copy()
        self._kwds.pop('discretization')
        ## Transport velocity
        self.velocity = velocity
        if 'variables' in kwds:
            self._kwds.pop('variables')
            kw = kwds.copy()
            kw['variables'] = kwds['variables'].copy()
            # In that case, variables must contains only the advected fields
            # with their discretization param.
            # Velocity must always be given outside variables, with its
            # own discretization.
            assert advected_fields is None, 'too many input arguments.'
            self.advected_fields = kwds['variables'].keys()
            kw['variables'][self.velocity] = kwds['discretization']
            kw.pop('discretization')
            super(AdvectionDir, self).__init__(**kw)
        else:
            v = [self.velocity]
            if isinstance(advected_fields, list):
                self.advected_fields = advected_fields
            else:
                self.advected_fields = [advected_fields]
            v += self.advected_fields
            super(AdvectionDir, self).__init__(variables=v, **kwds)

        # Set default method, if required
        if self.method is None:
            self.method = default.ADVECTION
        self.output = self.advected_fields
        self.input = [var for var in self.variables]

        from hysop.methods_keys import Splitting
        if Splitting not in self.method.keys():
            self.method[Splitting] = 'o2'
        self.name += name_suffix + S_DIR[direction]

        ## direction to advect
        self.direction = direction

        ## Fields on particles
        self.particle_fields = None

        ## Positions of the particles
        self.particle_positions = None

    @debug
    def discretize(self):
        if self._is_discretized:
            return

        build_topos = self._check_variables()

        # Check if multiscale is available
        if not self._single_topo:
            if self.method[Support].find('gpu') < 0:
                raise ValueError("Multiscale advection is not yet supported "
                                 "in pure Python, use Scales or GPU.")

        ## Default topology cutdir for parallel advection
        cutdir = [False] * self.domain.dimension
        cutdir[-1] = True

        if self._single_topo:
            # One topo for all fields ...
            self.method[MultiScale] = None
            if build_topos:
                topo = self.domain.create_topology(
                    discretization=self._discretization, cutdir=cutdir)
                for v in self.variables:
                    self.variables[v] = topo
            else:
                # Topo is already built, just check if it is 1D
                topo = self.variables.values()[0]
                msg = str(topo.cutdir) + ' != ' + str(cutdir)
                assert (topo.cutdir == cutdir).all(), msg

        else:
            # ... or one topo for each field.
            for v in self.variables:
                if build_topos[v]:
                    topo = self.domain.create_topology(
                        discretization=self.variables[v], cutdir=cutdir)
                    self.variables[v] = topo
                    build_topos[v] = False
            # compute velocity minimal ghost layer size
            self._check_ghost_layer(build_topos)

        # All topos are built, we can discretize fields.
        self._discretize_vars()

        self._is_discretized = True

    def _check_ghost_layer(self, build_topos):
        """
        Only meaningful if fields have different resolutions.
        Check/set interpolation method for multiscale and
        set ghost layer size, if required.
        """
        # Set method to default if unknown
        if MultiScale not in self.method.keys():
            self.method[MultiScale] = L2_1

        mscale = self.method[MultiScale]
        if mscale == Linear:
            min_ghosts = 1
        elif mscale == L2_1:
            min_ghosts = 2
        elif mscale == L4_2 or mscale == L4_4:
            min_ghosts = 3
        else:
            raise ValueError("Unknown multiscale method")

        # Topo or resolution associated with velocity
        discr_v = self.variables[self.velocity]
        if build_topos[self.velocity]:
            # discr_v = Discretization
            ghosts_v = discr_v.ghosts
        else:
            # discr_v = Cartesian
            ghosts_v = discr_v.ghosts()
        msg = 'Ghost layer required for velocity. Size min = '
        msg += str(min_ghosts) + " (" + str(ghosts_v) + " given)"
        assert (ghosts_v >= min_ghosts).all(), msg

    @opsetup
    def setup(self, rwork=None, iwork=None):
        # select discretization of the advected fields
        advected_discrete_fields = [self.discreteFields[v]
                                    for v in self.variables
                                    if v is not self.velocity]
        # GPU advection ...
        if self.method[Support].find('gpu') >= 0:
            topo_shape = advected_discrete_fields[0].topology.shape
            if topo_shape[self.direction] == 1:
                from hysop.gpu.gpu_particle_advection \
                    import GPUParticleAdvection as advec
            else:
                from hysop.gpu.multi_gpu_particle_advection \
                    import MultiGPUParticleAdvection as advec
        else:
            # pure-python advection
            from hysop.operator.discrete.particle_advection \
                import ParticleAdvection as advec

        self.discrete_op = advec(
            velocity=self.discreteFields[self.velocity],
            fields_on_grid=advected_discrete_fields,
            direction=self.direction,
            rwork=rwork, iwork=iwork,
            **self._kwds)

        self._is_uptodate = True

    def get_work_properties(self):
        """
        Work vector for advection in one dir :

        [ interp , part_positions, fields_on_particles]
        interp part is used also for remesh and time-integrator.
        """
        if not self._is_discretized:
            msg = 'The operator must be discretized '
            msg += 'before any call to this function.'
            raise RuntimeError(msg)
        dimension = self.domain.dimension
        res = {'rwork': [], 'iwork': []}
        if self.method[Support].find('gpu') < 0:
            tiw = self.method[TimeIntegrator].getWorkLengths(1)
            iw, iiw = \
                self.method[Interpolation].getWorkLengths(domain_dim=dimension)
            rw, riw = Remeshing.getWorkLengths(domain_dim=dimension)
            iwl = max(iiw, riw)
            rw = max(tiw + iw, rw)
        else:
            # For GPU version, no need of numerics works
            iwl, rw = 0, 0
        # Shape of reference comes from fields, not from velocity
        advected_discrete_fields = [self.discreteFields[v]
                                    for v in self.variables
                                    if v is not self.velocity]
        memshape = advected_discrete_fields[0].topology.mesh.resolution
        rw += np.sum([f.nb_components for f in self.advected_fields])
        if self.method[Support].find('gpu') < 0 or \
           self.method[Support].find('gpu_2k') >= 0:
            rw += 1  # positions
        for i in xrange(rw):
            res['rwork'].append(memshape)
        for i in xrange(iwl):
            res['iwork'].append(memshape)
        return res

    @debug
    @opapply
    def apply(self, simulation=None, dtCoeff=1.0, split_id=0, old_dir=0):
        """
        Apply this operator to its variables.
        @param simulation : object that describes the simulation
        parameters (time, time step, iteration number ...), see
        hysop.problem.simulation.Simulation for details.
        """
        self.discrete_op.apply(simulation,
                                    dtCoeff, split_id, old_dir)
