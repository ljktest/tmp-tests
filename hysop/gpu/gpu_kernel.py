"""
@file gpu_kernel.py
"""
from hysop.constants import debug, S_DIR
from hysop import __VERBOSE__
from hysop.gpu import cl, CL_PROFILE
from hysop.tools.profiler import FProfiler

class KernelListLauncher(object):
    """
    OpenCL kernel list launcher.

    Manage launching of OpenCL kernels as a list.
    """
    @debug
    def __init__(self, kernel, queue, gsize, lsize=None):
        """
        Create a kernel list launcher.
        @param kernel : kernel list.
        @param queue : OpenCL command queue.
        @param gsize : OpenCL global size index.
        @param lsize : OpenCL local size index.
        """
        ## OpenCL Kernel list
        self.kernel = kernel
        #print [k.function_name for k in self.kernel]
        ## OpenCL command queue
        self.queue = queue
        ## OpenCL global size index.
        self.global_size = gsize
        ## OpenCL local size index.
        self.local_size = lsize
        if CL_PROFILE:
            if len(self.kernel) == 1:
                try:
                    self.profile = [FProfiler("OpenCL_" + k.function_name)
                                    for k in self.kernel]
                except AttributeError:
                    self.profile = [FProfiler("OpenCL_" + k.__name__)
                                    for k in self.kernel]
            else:
                self.profile = [
                    FProfiler("OpenCL_" + k.function_name + S_DIR[d])
                    for d, k in enumerate(self.kernel)]
        else:
            self.profile = []

    @debug
    def __call__(self, d, *args, **kwargs):
        """
        Launch a kernel.

        OpenCL global size and local sizes are not given in
        args. Class member are used.

        @param d : kernel index in kernel list.
        @param args : kernel arguments.
        @return OpenCL Event
        """
        return KernelListLauncher.launch_sizes_in_args(
            self, d, self.global_size[d], self.local_size[d], *args, **kwargs)

    @debug
    def launch_sizes_in_args(self, d, *args, **kwargs):
        """
        Launch a kernel.

        Opencl global and local sizes are given in args.

        @param d : kernel index in kernel list.
        @param args : kernel arguments.
        @return OpenCL Event.
        """
        if __VERBOSE__:
            try:
                print "OpenCL kernel:", self.kernel[d].function_name, d, args[0], args[1]
                #print d, args[0], args[1], args, kwargs
            except AttributeError:
                print "OpenCL kernel:", self.kernel[d].__name__
        evt = self.kernel[d](self.queue, *args, **kwargs)
        if CL_PROFILE:
            evt.wait()
            self.profile[d] += (evt.profile.end - evt.profile.start) * 1e-9
        return evt

    def function_name(self, d=None):
        """Prints OpenCL Kernels function names informations"""
        if d is not None:
            return self.kernel[d].get_info(cl.kernel_info.FUNCTION_NAME)
        else:
            return [k.get_info(cl.kernel_info.FUNCTION_NAME)
                    for k in self.kernel]


class KernelLauncher(KernelListLauncher):
    """
    OpenCL kernel launcher.

    Manage launching of one OpenCL kernel as a KernelListLauncher
    with a list of one kernel.
    """
    @debug
    def __init__(self, kernel, queue, gsize=None, lsize=None):
        """
        Create a KernelLauncher.

        Create a KernelListLauncher with a list of one kernel.

        @param kernel : kernel.
        @param queue : OpenCL command queue.
        @param gsize : OpenCL global size index.
        @param lsize : OpenCL local size index.
        """
        KernelListLauncher.__init__(self, [kernel], queue, [gsize], [lsize])

    @debug
    def launch_sizes_in_args(self, *args, **kwargs):
        """
        Launch the kernel.

        Opencl global and local sizes are given in args.

        @param args : kernel arguments.
        @return OpenCL Event.
        """
        return KernelListLauncher.launch_sizes_in_args(
            self, 0, *args, **kwargs)

    @debug
    def __call__(self, *args, **kwargs):
        """
        Launch the kernel.

        OpenCL global size and local sizes are not given in args.
        Class member are used.

        @param args : kernel arguments.
        @return OpenCL Event
        """
        return KernelListLauncher.__call__(self, 0, *args, **kwargs)

    def function_name(self):
        """Prints OpenCL Kernel function name informations"""
        res = KernelListLauncher.function_name(self, 0)
        return res
