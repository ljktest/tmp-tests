"""
@file gpu_discrete.py

Contains class for discrete fields on GPU.
"""
from hysop import __VERBOSE__
from hysop.constants import ORDER, np,\
    debug, HYSOP_REAL, S_DIR
from hysop.fields.discrete import DiscreteField
from hysop.gpu import cl, CL_PROFILE
from hysop.gpu.gpu_kernel import KernelLauncher, KernelListLauncher
from hysop.tools.profiler import FProfiler

fromLayoutMgrFunc_3D_seq = [
    lambda a, shape: a.reshape(shape, order=ORDER)[...],
    lambda a, shape: a.reshape(shape, order=ORDER).swapaxes(0, 1)[...],
    lambda a, shape: a.reshape(
        shape, order=ORDER).swapaxes(0, 2).swapaxes(0, 1)[...]
    ]
shapeFunc_3D_seq = [
    lambda shape: (shape[0], shape[1], shape[2]),
    lambda shape: (shape[1], shape[0], shape[2]),
    lambda shape: (shape[2], shape[0], shape[1]),
    ]
toLayoutMgrFunc_3D_seq = [
    lambda a: a.ravel(order=ORDER)[...],
    lambda a: a.swapaxes(0, 1).ravel(order=ORDER)[...],
    lambda a: a.swapaxes(0, 1).swapaxes(0, 2).ravel(order=ORDER)[...]
    ]
fromLayoutMgrFunc_3D = [
    lambda a, shape: a.reshape(shape, order=ORDER)[...],
    lambda a, shape: a.reshape(shape, order=ORDER).swapaxes(0, 1)[...],
    lambda a, shape: a.reshape(shape, order=ORDER).swapaxes(0, 2)[...]
    ]
shapeFunc_3D = [
    lambda shape: (shape[0], shape[1], shape[2]),
    lambda shape: (shape[1], shape[0], shape[2]),
    lambda shape: (shape[2], shape[1], shape[0]),
    ]
toLayoutMgrFunc_3D = [
    lambda a: a.ravel(order=ORDER)[...],
    lambda a: a.swapaxes(0, 1).ravel(order=ORDER)[...],
    lambda a: a.swapaxes(0, 2).ravel(order=ORDER)[...]
    ]
fromLayoutMgrFunc_2D = [
    lambda a, shape: a.reshape(shape, order=ORDER)[...],
    lambda a, shape: a.reshape(shape, order=ORDER).swapaxes(0, 1)[...]
    ]
shapeFunc_2D = [
    lambda shape: (shape[0], shape[1]),
    lambda shape: (shape[1], shape[0])
    ]
toLayoutMgrFunc_2D = [
    lambda a: a.ravel(order=ORDER)[...],
    lambda a: a.swapaxes(0, 1).ravel(order=ORDER)[...]
    ]


class GPUDiscreteField(DiscreteField):
    """
    GPU Discrete vector field implementation.
    Allocates OpenCL device memory for the field.
    """
    def __init__(self, cl_env, topology=None, is_vector=False, name="?",
                 precision=HYSOP_REAL, layout=True, simple_layout=False):
        """
        Constructor.
        @param queue : OpenCL queue
        @param precision : Floating point precision
        @param parent : Continuous field.
        @param topology : Topology informations
        @param name : Field name
        @param idFromParent : Index in the parent's discrete fields
        @param layout : Boolean indicating if components are arranged in memory
        Defaut : all components are considered in the same way.
        @param simple_layout : Boolean indicating if in the Z direction,
        layout is ZYX (simple) or ZXY.
        @see hysop.fields.vector.VectorField.__init__
        """
        super(GPUDiscreteField, self).__init__(topology, is_vector, name)
        ## OpenCL environment
        self.cl_env = cl_env
        ## Precision for the field
        self.precision = precision
        ## Memory used
        self.mem_size = 0
        ## Initialization OpenCL kernel as KernelLauncher
        self.init_kernel = None
        self._isReleased = False
        ## OpenCL Buffer pointer
        self.gpu_data = [None] * self.nb_components
        ## Is the device allocations are performed
        self.gpu_allocated = False
        ## OpenCL Events list modifying this field
        self.events = []

        # Get the process number involved in this field discretisation
        # By default, all mpi process are take, otherwise, user create and
        # gives his own topologies.
        if topology is None:
            from hysop.mpi.main_var import main_rank
            self._rank = main_rank
        else:
            self._rank = topology.rank

        ## Data layout is direction dependant
        self.layout = layout
        ## Layout for the Z direction
        self.simple_layout = simple_layout
        ## Layout and shape managers
        if self.domain.dimension == 3:
            if self.simple_layout:
                self._shapeFunc = shapeFunc_3D
                self._fromLayoutMgrFunc = fromLayoutMgrFunc_3D
                self._toLayoutMgrFunc = toLayoutMgrFunc_3D
            else:
                self._shapeFunc = shapeFunc_3D_seq
                self._fromLayoutMgrFunc = fromLayoutMgrFunc_3D_seq
                self._toLayoutMgrFunc = toLayoutMgrFunc_3D_seq
        else:
            self._shapeFunc = shapeFunc_2D
            self._fromLayoutMgrFunc = fromLayoutMgrFunc_2D
            self._toLayoutMgrFunc = toLayoutMgrFunc_2D

        self.profiler += FProfiler("Transfer_toHost")
        self.profiler += FProfiler("Transfer_toDevice")
        ## Transfer size counter (to device)
        self.to_dev_size = 0.
        ## Transfer size counter (to host)
        self.to_host_size = 0.

        ## Temporary cpu buffer to change data layout between cpu ang gpu
        self.host_data_pinned = [None, ] * self.nb_components

    def allocate(self):
        """Device memory allocations no batch."""
        if not self.gpu_allocated:
            evt = [None, ] * self.nb_components
            for d in xrange(self.nb_components):
                self.data[d] = np.asarray(self.data[d],
                                          dtype=self.precision, order=ORDER)
                self.gpu_data[d] = self.cl_env.global_allocation(self.data[d])
                self.mem_size += self.gpu_data[d].size
                self.host_data_pinned[d], evt[d] = cl.enqueue_map_buffer(
                    self.cl_env.queue,
                    self.gpu_data[d],
                    offset=0, shape=(int(np.prod(self.data[0].shape)), ),
                    flags=cl.map_flags.READ | cl.map_flags.WRITE,
                    dtype=HYSOP_REAL, is_blocking=False, order=ORDER)
            for d in xrange(self.nb_components):
                evt[d].wait()
            self.gpu_allocated = True
            if __VERBOSE__:
                print self.name, self.mem_size, "Bytes (",
                print self.mem_size / (1024 ** 2), "MB)"

    @classmethod
    def fromField(cls, cl_env, vfield, precision=HYSOP_REAL,
                  layout=True, simple_layout=False):
        """
        Contructor from a discrete vector field.
        Mutates the given VectorField to a GPUVectorField.
        @param cls : Class of the class method (GPUVectorField)
        @param queue : OpenCL queue
        @param vfield : VectorField
        @param precision : Floating point precision
        @param layout : Boolean indicating if components are arranged in memory
        @param simple_layout : Boolean indicating if in the Z direction,
        layout is ZYX (simple) or ZXY.
        """
        if not isinstance(vfield, GPUDiscreteField):
            vfield.__class__ = cls
            GPUDiscreteField.__init__(
                vfield, cl_env,
                vfield.topology, vfield.nb_components > 1, vfield.name,
                precision, layout, simple_layout)

    def setInitializationKernel(self, kernel):
        """
        Set the initialization kernel
        @param kernel : KernelLauncher to use for initialize field.
        """
        self.init_kernel = kernel

    @debug
    def dump(self, filename):
        """
        @remark Synchronized OpenCL calls (waiting for event(s) completion)
        """
        self.toHost()
        self.wait()
        DiscreteField.dump(self, filename)

    @debug
    def load(self, filename, fieldname=None):
        """
        @remark Synchronized OpenCL calls (waiting for event(s) completion)
        """
        DiscreteField.load(self, filename, fieldname)
        self.toDevice()

    @debug
    def initialize(self, formula=None, vectorize_formula=False, time=0.,
                   *args):
        """
        GPU data initialization.
        Performs the initialization from different ways if device not already
        contains up-to-date data:
          - with an OpenCL kernel,
          - with a python formula (as VectorField) and the copy data to device.
        @param formula : Formula to use.
        @param args : formula extra parameters
        @remark Synchronized OpenCL calls (waiting for event(s) completion)
        """
        t = self.precision(time)
        if __VERBOSE__:
            print "{" + str(self._rank) + "}", "Initialize", self.name
        isGPUKernel = isinstance(formula, KernelLauncher) \
            or isinstance(formula, KernelListLauncher)
        if not isGPUKernel and self.init_kernel is None:
            DiscreteField.initialize(self, formula, False, time, *args)
            for d in xrange(self.nb_components):
                self.data[d] = np.asarray(
                    self.data[d],
                    dtype=self.precision, order=ORDER)
            self.toDevice()
        else:
            if isGPUKernel:
                self.init_kernel = formula
            coord_min = np.ones(4, dtype=self.precision)
            mesh_size = np.ones(4, dtype=self.precision)
            coord_min[:self.dimension] = np.asarray(
                self.topology.mesh.origin,
                dtype=self.precision)
            mesh_size[:self.dimension] = np.asarray(
                self.topology.mesh.space_step,
                dtype=self.precision)
            if self.nb_components == 2:
                evt = self.init_kernel(self.gpu_data[0],
                                       self.gpu_data[1],
                                       coord_min, mesh_size, t,
                                       *args,
                                       wait_for=self.events)
            elif self.nb_components == 3:
                evt = self.init_kernel(self.gpu_data[0],
                                       self.gpu_data[1],
                                       self.gpu_data[2],
                                       coord_min, mesh_size, t,
                                       *args,
                                       wait_for=self.events)
            else:
                evt = self.init_kernel(self.gpu_data[0],
                                       coord_min, mesh_size, t,
                                       *args,
                                       wait_for=self.events)
            self.events.append(evt)

    def finalize(self):
        if not self._isReleased:
            if __VERBOSE__:
                print "deallocate :", self.name,
                print " (" + str(self.mem_size / (1024. ** 2)) + " MBytes)"
            self.wait()
            for d in xrange(self.nb_components):
                self.host_data_pinned[d].base.release(self.cl_env.queue)
                self.cl_env.global_deallocation(self.gpu_data[d])
            self._isReleased = True

    def get_profiling_info(self):
        if self.init_kernel is not None:
            for p in self.init_kernel.profile:
                self.profiler += p

    def toDevice(self, component=None, layoutDir=None):
        """
        Host to device method.
        @param component : Component to consider (Default : all components)
        @param layoutDir : layout to use
        If the field have a layout per component, layoutDir is unused. Other
        fields can be transfered with a given layout.

        Performs a direct OpenCL copy from numpy arrays
        to OpenCL Buffers.\n
        Arrange memory on device so that vector components are
        contiguous in the direction of the component, if layout flag is True.\n
        Example : A 3D vector field F(x,y,z) is made up of 3
        OpenCL Buffers Fx, Fy, Fz. The memory layout is :
        - Fx : x-major ordering. On device,
        Fx[i + j*WIDTH + k*WIDTH*HEIGHT] access to Fx(i,j,k)
        - Fy : y-major ordering. On device,
        Fy[i + j*WIDTH + k*WIDTH*HEIGHT] access to Fy(j,i,k)
        - Fz : z-major ordering. On device,
        Fz[i + j*WIDTH + k*WIDTH*HEIGHT] access to Fz(k,i,j)
        """
        if component is None:
            range_components = xrange(self.nb_components)
            evt = [None] * self.nb_components
        else:
            range_components = [component]
            evt = [None]
        self.wait()
        mem_transfered = 0
        for d_id, d in enumerate(range_components):
            if self.layout:
                layoutDir = d
            if layoutDir is None:
                layoutDir = 0
            if __VERBOSE__:
                print "{" + str(self._rank) + "}", "host->device :", \
                    self.name, S_DIR[d], layoutDir
            self.host_data_pinned[d][...] = \
                self._toLayoutMgrFunc[layoutDir](self.data[d])
            evt[d_id] = cl.enqueue_copy(
                self.cl_env.queue, self.gpu_data[d], self.host_data_pinned[d],
                is_blocking=False)
            mem_transfered += self.gpu_data[d].size
        for e in evt:
            self.events.append(e)
        time = 0.
        self.to_dev_size += mem_transfered / (1024. ** 3)
        if CL_PROFILE:
            for e in evt:
                if e is not None:
                    e.wait()
                    time += (e.profile.end - e.profile.start) * 1e-9
            self.profiler['Transfer_toDevice'] += time
        if __VERBOSE__ and CL_PROFILE:
            print self.mem_size, "Bytes transfered at ",
            print "{0:.3f} GBytes/sec".format(
                mem_transfered / (time * 1024 ** 3))

    def toHost(self, component=None, layoutDir=None):
        """
        Device to host method.
        @param component : Component to consider (Default : all components)
        @param layoutDir : layout to use
        If the field have a layout per component, layoutDir is unused. Other
        fields can be transfered with a given layout.

        Performs a direct OpenCL copy from OpenCL Buffers
        to numpy arrays.\n
        As memory layout, if set, is arranged on device, not only a
        copy is performed but also transpositions to have numpy
        arrays consistent to each other.
        """
        self.wait()
        if component is None:
            range_components = xrange(self.nb_components)
            evt = [None] * self.nb_components
        else:
            range_components = [component]
            evt = [None]

        mem_transfered = 0
        for d_id, d in enumerate(range_components):
            if self.layout:
                layoutDir = d
            if layoutDir is None:
                layoutDir = 0
            if __VERBOSE__:
                print "{" + str(self._rank) + "}", "device->host :", \
                    self.name, S_DIR[d], layoutDir
            evt[d_id] = cl.enqueue_copy(self.cl_env.queue,
                                        self.host_data_pinned[d],
                                        self.gpu_data[d],
                                        wait_for=self.events,
                                        is_blocking=False)
            mem_transfered += self.gpu_data[d].size
        for d_id, d in enumerate(range_components):
            shape = self._shapeFunc[layoutDir](self.data[d].shape)
            evt[d_id].wait()
            self.data[d][...] = self._fromLayoutMgrFunc[layoutDir](
                self.host_data_pinned[d], shape)
        for e in evt:
            self.events.append(e)
        time = 0.
        self.to_host_size += mem_transfered / (1024. ** 3)
        if CL_PROFILE:
            for e in evt:
                if e is not None:
                    e.wait()
                    time += (e.profile.end - e.profile.start) * 1e-9
            self.profiler['Transfer_toHost'] += time
        if __VERBOSE__:
            if CL_PROFILE:
                print self.mem_size, "Bytes transfered at ",
                print "{0:.3f} GBytes/sec".format(
                    mem_transfered / (time * 1024 ** 3))

    def wait(self):
        """
        Waiting for all events completion in the field list.
        Resets the events list.
        """
        if __VERBOSE__:
            print "{" + str(self._rank) + "}", "Wait events :", self.name
        for e in self.events:
            e.wait()
        self.events = []

    def clean_events(self):
        """
        Waiting for all events completion in the field list.
        Resets the events list.
        """
        if __VERBOSE__:
            print "{" + str(self._rank) + "}", "Clean events :", \
                self.name, len(self.events)
        c = cl.command_execution_status.COMPLETE
        for e in self.events:
            e.wait()
        self.events = [e for e in self.events
                       if e.command_execution_status != c]
