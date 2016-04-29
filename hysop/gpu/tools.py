"""
@file tools.py

Tools for gpu management.
"""
from hysop import __VERBOSE__, __DEFAULT_PLATFORM_ID__, __DEFAULT_DEVICE_ID__
from hysop.constants import np, HYSOP_REAL, ORDER
from hysop.gpu import cl, clTools, GPU_SRC, CL_PROFILE
import hysop.tools.numpywrappers as npw
import re
import mpi4py.MPI as mpi
FLOAT_GPU, DOUBLE_GPU = np.float32, np.float64

## Global variable handling an OpenCL Environment instance
__cl_env = None


class OpenCLEnvironment(object):
    """OpenCL environment informations and useful functions.
    """

    def __init__(self, platform_id, device_id, device_type,
                 precision, gl_sharing=False, comm=None):
        """Create environment.
        @param platform_id : OpenCL platform id
        @param device_id : OpenCL device id
        @param device_type : OpenCL device type
        @param precision : Recquired precision
        @param resolution : Global resolution
        @param gl_sharing : Flag to build a OpenGL shared OpenCL context
        """
        self._platform_id = platform_id
        self._device_id = device_id
        self._device_type = device_type
        self._gl_sharing = gl_sharing
        ## OpenCL platform
        self.platform = self._get_platform(platform_id)
        ## OpenCL device
        self.device = self._get_device(self.platform, device_id, device_type)
        ## Device available memory
        self.available_mem = self.device.global_mem_size
        ## OpenCL context
        self.ctx = self._get_context(self.device, gl_sharing)
        ## OpenCL queue
        self.queue = self._get_queue(self.ctx)

        ## MPI sub-communicator for all processes attached to the same device
        if comm is None:
            from hysop.mpi.main_var import main_comm
        else:
            main_comm = comm
        # Splitting the mpi communicator by the device id is not enough:
        # the id of the first gpu of each node is 0
        # We build color from the processor name and the id
        import hashlib
        # The md5 sum of the proc name is tuncated to obtain an integer
        # for fortran (32bit)
        hash_name = hashlib.md5(mpi.Get_processor_name()).hexdigest()[-7:]
        self.gpu_comm = main_comm.Split(
            color=int(hash_name, 16) + device_id,
            key=main_comm.Get_rank())

        ## Floating point precision
        self.precision = precision
        if self.precision is FLOAT_GPU:
            self.prec_size = 4
        elif self.precision is DOUBLE_GPU:
            self.prec_size = 8
        self.default_build_opts = ""
        if CL_PROFILE and self.device.vendor.find('NVIDIA') >= 0:
            self.default_build_opts += " -cl-nv-verbose"
        self.default_build_opts += " -Werror" + self._get_precision_opts()

        ## Kernels configuration dictionary
        if self.device.name == "Cayman":
            from config_cayman import kernels_config as kernel_cfg
        elif self.device.name == "Tesla K20m" or \
                self.device.name == "Tesla K20Xm":
            from config_k20m import kernels_config as kernel_cfg
        else:
            print "/!\\ Get a defautl kernels config for", self.device.name
            from config_default import kernels_config as kernel_cfg
        self.kernels_config = kernel_cfg
        self._locMem_Buffers = {}

    def modify(self, platform_id, device_id, device_type,
               precision, gl_sharing=False):
        """
        Modify OpenCL environment parameters.
        @param platform_id : OpenCL platform id
        @param device_id : OpenCL device id
        @param device_type : OpenCL device type
        @param precision : Recquired precision
        @param resolution : Global resolution
        @param gl_sharing : Flag to build a OpenGL shared OpenCL context
        """
        platform_changed, device_changed = False, False
        if not platform_id == self._platform_id:
            print ("platform changed")
            self._platform_id = platform_id
            self.platform = self._get_platform(platform_id)
            platform_changed = True
        if platform_changed or not (device_id is self._device_id
                                    and device_type == self._device_type):
            print ("device changed")
            self._device_id = device_id
            self._device_type = device_type
            self.device = self._get_device(self.platform,
                                           device_id, device_type)
            self.available_mem = self.device.global_mem_size
            device_changed = True
        if platform_changed or device_changed or \
                (not self._gl_sharing and not gl_sharing is self._gl_sharing):
            if self._gl_sharing and not gl_sharing:
                print ("Warning: Loosing Gl shared context.")
            self._gl_sharing = gl_sharing
            self.ctx = self._get_context(self.device, gl_sharing)
            self.queue = self._get_queue(self.ctx)
        if not self.precision is precision and not precision is None:
            if not self.precision is None:
                print ("Warning, GPU precision is overrided from",)
                print (self.precision, 'to', precision)
            self.precision = precision
            self.default_build_opts = ""
            if CL_PROFILE and self.device.vendor.find('NVIDIA') >= 0:
                self.default_build_opts += " -cl-nv-verbose"
            self.default_build_opts += "-Werror" + self._get_precision_opts()

    def _get_platform(self, platform_id):
        """
        Get an OpenCL platform.
        @param platform_id : OpenCL platform id
        @return OpenCL platform
        """
        try:
            # OpenCL platform
            platform = cl.get_platforms()[platform_id]
        except IndexError:
            print ("  Incorrect platform_id :", platform_id, ".",)
            print (" Only ", len(cl.get_platforms()), " available.",)
            print (" Getting default platform. ")
            platform = cl.get_platforms()[0]
        if __VERBOSE__:
            print ("  Platform   ")
            print ("  - Name       :", platform.name)
            print ("  - Version    :", platform.version)
        return platform

    def _get_device(self, platform, device_id, device_type):
        """
        Get an OpenCL device.
        @param platform : OpenCL platform
        @param device_id : OpenCL device id
        @param device_type : OpenCL device type
        @return OpenCL device

        Try to use given parameters and in case of fails, use pyopencl context
        creation function.
        """
        display = False
        try:
            if device_type is not None:
                device = platform.get_devices(
                    eval("cl.device_type." + str(device_type.upper()))
                    )[device_id]
            else:
                device = platform.get_devices()[device_id]
        except cl.RuntimeError as e:
            print ("RuntimeError:", e)
            device = cl.create_some_context().devices[0]
            display = True
        except AttributeError as e:
            print ("AttributeError:", e)
            device = cl.create_some_context().devices[0]
            display = True
        except IndexError:
            print ("  Incorrect device_id :", device_id, ".",)
            print (" Only ", len(platform.get_devices()), " available.",)
            if device_type is not None:
                print (" Getting first device of type " +
                       str(device_type.upper()))
            else:
                print (" Getting first device of the platform")
            device = platform.get_devices()[0]
            display = True
        if device_type is not None:
            assert device_type.upper() == cl.device_type.to_string(device.type)
        if display or __VERBOSE__:
            print ("  Device")
            print ("  - id                :", device_id)
            print ("  - Name                :",)
            print (device.name)
            print ("  - Type                :",)
            print cl.device_type.to_string(device.type)
            print ("  - C Version           :",)
            print (device.opencl_c_version)
            print ("  - Global mem size     :",)
            print device.global_mem_size / (1024 ** 3), "GB"
        return device

    def _get_context(self, device, gl_sharing):
        """
        Get an OpenCL context.
        @param device : OpenCL device
        @param gl_sharing : Flag to build a OpenGL shared OpenCL context
        @return OpenCL context

        """
        props = None
        if gl_sharing:
            from pyopencl.tools import get_gl_sharing_context_properties
            import sys
            if sys.platform == "darwin":
                props = get_gl_sharing_context_properties()
            else:
                # Some OSs prefer clCreateContextFromType, some prefer
                # clCreateContext. Try both.
                props = \
                    [(cl.context_properties.PLATFORM, self.platform)] \
                    + get_gl_sharing_context_properties()
            ctx = cl.Context(properties=props, devices=[device])
        else:
            ctx = cl.Context([device])
        if __VERBOSE__:
            print " Context:"
            if not props is None:
                print "  - properties           :", props
        return ctx

    def _get_queue(self, ctx):
        """
        Get OpenCL queue from context
        @param ctx : OpenCL context
        @return OpenCL queue
        """
        props = None
        if CL_PROFILE:
            props = cl.command_queue_properties.PROFILING_ENABLE
            queue = cl.CommandQueue(ctx, properties=props)
        else:
            queue = cl.CommandQueue(ctx)
        if __VERBOSE__:
            print " Queue"
            if not props is None:
                print "  - properties           :", props
            print "==="
        return queue

    def create_other_queue(self):
        return self._get_queue(self.ctx)

    def get_WorkItems(self, resolution, vector_width=1):
        """
        Set the optimal work-item number and OpenCL space index.
        @param resolution : Problem resolution
        @param vector_width : OpenCL vector types width
        @return work-item number, global space index and local space index

        Use 64 work-items in 3D and 256 in 2D.
        \todo Use Both the number from device capability
        The problem must be a multiple of and greater
        than work-item number * vector_width
        """
        # Optimal work item number
        if len(resolution) == 3:
            workItemNumber = 64 if min(resolution) >= 64 \
                else min(resolution)
        else:
            workItemNumber = 256 if min(resolution) >= 256 \
                else min(resolution)
        # Change work-item regarding problem size
        if resolution[0] % workItemNumber > 0:
            if len(resolution) == 3:
                print "Warning : GPU best performances obtained for",
                print "problem sizes multiples of 64"
            else:
                print "Warning : GPU best performances obtained for",
                print "problem sizes multiples of 256"
        while(resolution[0] % workItemNumber > 0):
            workItemNumber = workItemNumber / 2
        # Change work-item regarding vector_width
        if workItemNumber * vector_width > resolution[0]:
            if resolution[0] % vector_width > 0:
                raise ValueError(
                    "Resolution ({0}) must be a multiple of {1}".format(
                        resolution[0], vector_width))
            workItemNumber = resolution[0] // vector_width
        if len(resolution) == 3:
            gwi = (int(workItemNumber),
                   int(resolution[1]), int(resolution[2]))
            lwi = (int(workItemNumber), 1, 1)
        else:
            gwi = (int(workItemNumber),
                   int(resolution[1]))
            lwi = (int(workItemNumber), 1)
        return workItemNumber, gwi, lwi

    def _get_precision_opts(self):
        """Check if device is capable regarding given precision
        @return build options regarding precision recquired
        """
        opts = ""
        # Precision supported
        if __VERBOSE__:
            print " Precision capability  ",
        fp32_rounding_flag = True
        if self.precision is FLOAT_GPU:
            opts += " -cl-single-precision-constant"
            prec = "single"
        else:
            if self.device.double_fp_config <= 0:
                raise ValueError("Double Precision is not supported by device")
            prec = "double"
        if __VERBOSE__:
            print "for " + prec + " Precision: "
        for v in ['DENORM', 'INF_NAN',
                  'ROUND_TO_NEAREST', 'ROUND_TO_ZERO', 'ROUND_TO_INF',
                  'FMA', 'CORRECTLY_ROUNDED_DIVIDE_SQRT', 'SOFT_FLOAT']:
            try:
                if eval('(self.device.' + prec + '_fp_config &' +
                        ' cl.device_fp_config.' +
                        v + ') == cl.device_fp_config.' + v):
                    if __VERBOSE__:
                        print v
                else:
                    if v is 'CORRECTLY_ROUNDED_DIVIDE_SQRT':
                        fp32_rounding_flag = False
            except AttributeError as ae:
                if v is 'CORRECTLY_ROUNDED_DIVIDE_SQRT':
                    fp32_rounding_flag = False
                if __VERBOSE__:
                    print v, 'is not supported in OpenCL C 1.2.\n',
                    print '   Exception catched : ', ae
        if fp32_rounding_flag:
            opts += " -cl-fp32-correctly-rounded-divide-sqrt"
        return opts

    def build_src(self, files, options="", vector_width=4,
                  nb_remesh_components=1, macros=None):
        """
        Build OpenCL sources
        @param files: Source files
        @param options : Compiler options to use for buildind
        @param vector_width : OpenCL vector type width
        @param nb_remesh_components : number of remeshed components
        @return OpenCL binaries

        Parse the sources to handle single and double precision.
        """
        gpu_src = ""
        if cl.device_type.to_string(self.device.type) == 'GPU' and \
                self.precision is DOUBLE_GPU:
            gpu_src += "#pragma OPENCL EXTENSION cl_khr_fp64: enable \n"
        if isinstance(files, list):
            file_list = files
        else:
            file_list = [files]
        if __VERBOSE__:
            print "=== Kernel sources compiling ==="
            for sf in file_list:
                print "   - ", sf
        for sf in file_list:
            try:
                f = open(sf, 'r')
            except IOError as ioe:
                if ioe.errno == 2:
                    f = open(GPU_SRC + sf, 'r')
                else:
                    raise ioe
            gpu_src += "".join(
                self.parse_file(f, vector_width, nb_remesh_components))
            f.close()
            #print gpu_src
        if macros is not None:
            for k in macros:
                gpu_src = gpu_src.replace(k, str(macros[k]))
        if self.precision is FLOAT_GPU:
            # Rexexp to add 'f' suffix to float constants
            # Match 1.2, 1.234, 1.2e3, 1.2E-05
            float_replace = re.compile(r'(?P<float>\d\.\d+((e|E)-?\d+)?)')
            prg = cl.Program(
                self.ctx,
                float_replace.sub(r'\g<float>f', gpu_src))
        else:
            prg = cl.Program(self.ctx, gpu_src.replace('float', 'double'))
        # OpenCL program
        try:
            build = prg.build(self.default_build_opts + options)
        except Exception, e:
            print "Build files : "
            for sf in file_list:
                print "   - ", sf
            print "Build options : ", self.default_build_opts + options
            print "Vectorization : ", vector_width
            raise e
        if __VERBOSE__:
            #print options
            print "Build options : ",
            print build.get_build_info(
                self.device, cl.program_build_info.OPTIONS)
            print "Compiler status : ",
            print build.get_build_info(
                self.device, cl.program_build_info.STATUS)
            print "Compiler log : ",
            print build.get_build_info(self.device,
                                       cl.program_build_info.LOG)
            print "===\n"
        elif CL_PROFILE:
            print "Build files: " + str(file_list)
            print "With build options: " + self.default_build_opts + options
            print "Compiler output : " + build.get_build_info(
                self.device, cl.program_build_info.LOG)
        return build

    def parse_file(self, f, n=8, nb_remesh_components=1):
        """
        Parse a file containing OpenCL sources.
        @param f : source file
        @param n : vector width
        @param nb_remesh_components : number of remeshed components
        @return parsed sources as string.

        - <code>__N__</code> is expanded as an integer corresponding to a
        vector with
        - <code>__NN__</code>, instruction is duplicated to operate on each
        vector component:
          - if line ends with '<code>;</code>', the whole instruciton is
          duplicated.
          - if line ends with '<code>,</code>' and contains
          '<code>(float__N__)(</code>', the float element is duplicated
        - Remeshing fields components are expanded as follows :
          All code between '<code>__RCOMPONENT_S__</code>' and
          '<code>__RCOMPONENT_E__</code>' flags are duplicated n times with n
          the number of components to compute. In this duplicated code, the
          flag '<code>__ID__</code>' is replaced by index of a range of lenght
          the number of components. A flag '<code>__RCOMPONENT_S__P__</code>'
          may be used and the duplicated elements are separated by ',' (for
          function parameters expanding).

        Examples with a 4-width vector:\n
        \code
        float__N__ x;           ->  float4 x;

        x.s__NN__ = 1.0f;       ->  x.s0 = 1.0f;
                                    x.s1 = 1.0f;
                                    x.s2 = 1.0f;
                                    x.s3 = 1.0f;

        x = (int__N__)(\__NN__,  ->  x = (int4)(0,
                       );                      1,
                                               2,
                                               3,
                                               );
        \endcode

        Examples with a 2 components expansion:\n
        __RCOMP_P __global const float* var__ID__,
        -> __global const float* var0,__global const float* var1,

        __RCOMP_I var__ID__[i] = 0.0;
        -> var0[i] = 0.0;var1[i] = 0.0;

        aFunction(__RCOMP_P var__ID__, __RCOMP_P other__ID__);
        -> aFunction(var0, var1, other0, other1);
        \endcode
        """
        src = ""
        # replacement for floatN elements
        vec_floatn = re.compile(r'\(float__N__\)\(')
        vec_nn = re.compile('__NN__')
        vec_n = re.compile('__N__')
        for l in f.readlines():
            # Expand floatN items
            if vec_floatn.search(l) and vec_nn.search(l) and \
                    l[-2] == ',':
                sl = l.split("(float__N__)(")
                l = sl[0] + "(float" + str(n) + ")("
                el = sl[1].rsplit(',', 1)[0]
                for i in xrange(n):
                    l += vec_nn.sub(str(i), el) + ','
                l = l[:-1] + '\n'
            # Expand floatN elements access
            elif vec_nn.search(l) and l[-2] == ';':
                el = ""
                for i in xrange(n):
                    el += vec_nn.sub(str(i), l)
                l = el
            # Replace vector length
            src += vec_n.sub(str(n), l)

        # Replacement for remeshed components
        re_instr = re.compile(r'__RCOMP_I([\w\s\.,()\[\]+*/=-]+;)')
        # __RCOMP_I ...;

        def repl_instruction(m):
            return ''.join(
                [m.group(1).replace('__ID__', str(i))
                 for i in xrange(nb_remesh_components)])
        # __RCOMP_P ..., ou __RCOMP_P ...)
        re_param = re.compile(r'__RCOMP_P([\w\s\.\[\]+*/=-]+(?=,|\)))')

        def repl_parameter(m):
            return ', '.join(
                [m.group(1).replace('__ID__', str(i))
                 for i in xrange(nb_remesh_components)])

        src = re_instr.sub(repl_instruction, src)
        src = re_param.sub(repl_parameter, src)
        return src

    def global_allocation(self, array):
        clBuff = cl.Buffer(self.ctx,
                           cl.mem_flags.ALLOC_HOST_PTR, size=array.nbytes)
        # Touch the buffer on device to performs the allocation
        # Transfers a single element in device (the precision no matters here)
        e = np.zeros((1,), dtype=np.float64)
        cl.enqueue_copy(self.queue, clBuff, e,
                        buffer_origin=(0, 0, 0), host_origin=(0, 0, 0),
                        region=(e.nbytes,)).wait()
        self.available_mem -= clBuff.size
        return clBuff

    def global_deallocation(self, cl_mem):
        self.available_mem += cl_mem.size
        cl_mem.release()

    # def LocalMemAllocator(self, sizes_list, type_list=None):
    #     """
    #     Allocates spaces in device local memory.
    #     @param sizes_list : list of sizes.
    #     @param type_list : list of corresponding types
    #     It returns a list of buffers of given size (one per size specified in
    #     in the list) and the size of new buffers.
    #     @remark : Buffers are stored and could be reused.
    #     @remark : it assumes that all returned buffers are different
    #     """
    #     new_alloc = 0
    #     if type_list is None:
    #         type_list = [HYSOP_REAL] * len(sizes_list)
    #     buff_list = []  # Returned list
    #     keys_list = []
    #     for s, t in zip(sizes_list, type_list):
    #         keys_list.append(int(t(0).nbytes * s))

    #     for size, key, t in zip(sizes_list, keys_list, type_list):
    #         buff = None
    #         try:
    #             # List of existing buffers not already in the list
    #             avail_buff = [b for b in self._locMem_Buffers[key]
    #                           if b not in buff_list]
    #             if len(avail_buff) > 0:
    #                 # adding the first buffer
    #                 buff = avail_buff[0]
    #             else:
    #                 # Allocate a new buffer
    #                 buff = cl.LocalMemory(int(t(0).nbytes * size))
    #                 new_alloc += buff.size
    #                 self._locMem_Buffers[key].append(buff)
    #         except KeyError:
    #             # Allocate a fist buffer of given size
    #             buff = cl.LocalMemory(int(t(0).nbytes * size))
    #             new_alloc += buff.size
    #             self._locMem_Buffers[key] = [buff]
    #         buff_list.append(buff)
    #     return buff_list, new_alloc


def get_opengl_shared_environment(platform_id=None,
                                  device_id=None,
                                  device_type=None, precision=HYSOP_REAL,
                                  comm=None):
    """
    Get an OpenCL environment with OpenGL shared enable.

    @param platform_id :OpenCL platform id
    @param device_id : OpenCL device id
    @param device_type : OpenCL device type
    @param precision : Required precision
    @return OpenCL platform, device, context and queue

    The context is obtained with gl-shared properties depending on the OS.
    """
    if platform_id is None:
        platform_id = __DEFAULT_PLATFORM_ID__
    if device_id is None:
        device_id = __DEFAULT_DEVICE_ID__
    global __cl_env
    if __cl_env is None:
        __cl_env = OpenCLEnvironment(platform_id, device_id, device_type,
                                     precision, gl_sharing=True, comm=comm)
    else:
        __cl_env.modify(platform_id, device_id, device_type,
                        precision, gl_sharing=True)
    return __cl_env


def get_opencl_environment(platform_id=None,
                           device_id=None,
                           device_type=None, precision=HYSOP_REAL,
                           comm=None):
    """
    Get an OpenCL environment.

    @param platform_id :OpenCL platform id
    @param device_id : OpenCL device id
    @param device_type : OpenCL device type
    @param precision : Required precision
    @return OpenCL platform, device, context and queue

    """
    if platform_id is None:
        platform_id = __DEFAULT_PLATFORM_ID__
    if device_id is None:
        device_id = __DEFAULT_DEVICE_ID__
    global __cl_env
    if __cl_env is None:
        __cl_env = OpenCLEnvironment(platform_id, device_id, device_type,
                                     precision, comm=comm)
    else:
        __cl_env.modify(platform_id, device_id, device_type,
                        precision)
    return __cl_env


def explore():
    """Print environment details"""
    print "OpenCL exploration : "
    platforms = cl.get_platforms()
    platforms_info = ["name", "version", "vendor", "profile", "extensions"]
    devices_info = ["name",
                    "version",
                    "vendor",
                    "profile",
                    "extensions",
                    "available",
                    "type",
                    "compiler_available",
                    "double_fp_config",
                    "single_fp_config",
                    "global_mem_size",
                    "global_mem_cache_type",
                    "global_mem_cache_size",
                    "global_mem_cacheline_size",
                    "local_mem_size",
                    "local_mem_type",
                    "max_clock_frequency",
                    "max_compute_units",
                    "max_constant_buffer_size",
                    "max_mem_alloc_size",
                    "max_work_group_size",
                    "max_work_item_dimensions",
                    "max_work_item_sizes",
                    "preferred_vector_width_double",
                    "preferred_vector_width_float",
                    "preferred_vector_width_int"]
    for pltfm in platforms:
        print "Platform:", pltfm.name
        for pltfm_info in platforms_info:
            print "  |-", pltfm_info, ':', eval("pltfm." + pltfm_info)
        devices = pltfm.get_devices()
        for dvc in devices:
            print "  |- Device:", dvc.name
            for dvc_info in devices_info:
                print "    |-", dvc_info, ':', eval("dvc." + dvc_info)
