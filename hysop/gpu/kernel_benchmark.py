"""
@file kernel_benchmark.py

Package for benchmarking OpenCL kernels.
"""
from hysop.gpu import cl
from hysop.constants import np, HYSOP_REAL
import pickle


class BenchmarkSuite(object):
    """Benchark suite management"""

    def __init__(self, sizes, name,
                 kernels, configs, versions, setupFunction,
                 test=False, true_res=None, arg_to_test=0,
                 inputs=None, file_name="Benchmarks_data",
                 precision=HYSOP_REAL, nb_run=20):
        """
        Creates a benchmak suite, that consists in a list of Benchmark.

        @param sizes : list of different problem sizes to benchmark.
        @param name : name of the kernel to benchmark.
        @param kernels : list of tuples containing kernel versions
        (kernel sources file, OpenCL kernel name ).
        @param configs : dictionary of configurations.
        keys are kernel OpenCL name,
        values are tuples containing (kernel parameters vectorization
        and identifier in last position).
        @param versions : kernel versions to bench (used as dictionaries keys)
        @param setupFunction : Function that returns building options and
        kernel arguments (assuming arrays are numpy arrays) depending on a
        given config, size and input dictionary.
        @param test : by default no results tests are performed
        @param true_res : function to compute true results
        @param arg_to_test : index of kernel arguments that contains result
        @param inputs : input data
        @param file_name : name of file to store results in
        @param precision : Floating point precision for kernels
        @param nb_run : Launch number to compute an average

        On creation, data are loaded from a serialized version of timings
        in the file represented by file_name parameter.
        If no such file, a new database is created.
        """
        self.pickle_file_name = file_name
        if precision == HYSOP_REAL:
            self.pickle_file_name += '_DP'
        else:
            self.pickle_file_name += '_SP'
        self.sizes = sizes
        self.versions = versions
        self.kernels_files = kernels
        self.configs = configs
        if inputs is None:
            self.inputs = {}
        else:
            self.inputs = inputs
        self.test = test
        self.compute_true_res = true_res
        self.arg_to_test = arg_to_test
        self.kernel_name = name
        self.precision = precision
        self.nb_run = nb_run
        self.setupFunction = setupFunction

        if not self.test:
            try:
                print 'Loading form pickled file ...',
                self.timings = pickle.load(
                    open(self.pickle_file_name + '.pickle', 'r'))
                print 'Done.'
            except IOError:
                print 'No such file : ', self.pickle_file_name + '.pickle'
                print 'start new database'
                self.timings = {}
        else:
            assert not true_res is None
            assert arg_to_test >= 2
            self.timings = {}
        self.complete_timings()

    def complete_timings(self):
        """
        Manage dictionary structure of timings.

        Add all new keys in dictionaries.
        """
        if self.kernel_name not in self.timings.keys():
            self.timings[self.kernel_name] = {}
        for v in self.versions:
            if not v in self.timings[self.kernel_name].keys():
                self.timings[self.kernel_name][v] = {}
            for c in self.configs[v]:
                if not c[-1] in self.timings[self.kernel_name][v].keys():
                    self.timings[self.kernel_name][v][c[-1]] = {}

    def launch(self):
        """
        Performs the benchmark for all kernel versions and all configs.

        If test flag is set to True, results is compared to the true result
        and timings are not saved.
        Else, timings are added to timings dictionary and then serialized in
        a file. A text version is also writed.
        """
        if self.test:
            self.true_res = {}
            self.compute_true_res(self.sizes, self.true_res, self.inputs)
        for v in self.versions:
            conf_list = self.configs[v]
            for conf in conf_list:
                b = Benchmark(
                    self.kernels_files[v][0], self.kernels_files[v][1],
                    self.sizes, conf, self.setupFunction,
                    inputs=self.inputs,
                    precision=self.precision, nb_run=self.nb_run)
                print self.kernel_name, v, conf[-1]
                if self.test:
                    b.test(self.true_res, self.arg_to_test)
                else:
                    b.launch()
                    [self.timings[self.kernel_name][v][conf[-1]].__setitem__(
                        t[0], t[1]) for t in b.timings.items()]
                    pickle.dump(
                        self.timings, open(
                            self.pickle_file_name + '.pickle', 'w'), 0)
                    self.write_file()

    def write_file(self):
        """
        Write a text version of database.

        Two outputs are created :
        @li full : kernels versions and configs are given in columns and sizes
        in rows.
        @li hist : all data is given in rows to enable gnuplot to plot
        histograms.
        """
        f = open(self.pickle_file_name + '_full.dat', 'w')
        #build size set
        sizes_set = set()
        config_set = set()
        cols_lists = {}
        for k in self.timings.keys():
            for v in self.timings[k].keys():
                cols_lists[v] = []
                for c in self.timings[k][v].keys():
                    for s in self.timings[k][v][c]:
                        sizes_set.add(s)
                        config_set.add(c)
        f.write("size dim ")
        i = 0
        for k in sorted(self.timings.keys()):
            for v in sorted(self.timings[k].keys()):
                for c in sorted(self.timings[k][v].keys()):
                    f.write(v + '_' + c + ' ')
                    cols_lists[v].append(i)
                    i += 1
        f.write("\n")
        for s in sorted(sizes_set):
            f.write(str(s[0]) + " " + str(len(s)) + " ")
            for k in sorted(self.timings.keys()):
                for v in sorted(self.timings[k].keys()):
                    for c in sorted(self.timings[k][v].keys()):
                        try:
                            f.write(str(self.timings[k][v][c][s]) + " ")
                        except KeyError as ke:
                            if ke.message is s:
                                f.write("- ")
                            else:
                                raise ke
            f.write("\n")
        for k in sorted(self.timings.keys()):
            for v in sorted(self.timings[k].keys()):
                f.write('#' + v + '=')
                for i in cols_lists[v]:
                    f.write(str(i) + ' ')
                f.write('\n')
        f.close()
        f = open(self.pickle_file_name + '_hist.dat', 'w')
        f.write("#kernel_nb=" + str(len(self.timings.keys())) + "\n")
        f.write("#kernel_names=")
        for k in sorted(self.timings.keys()):
            f.write(k + " ")
        f.write("\n")
        f.write("#version_nb=")
        for k in sorted(self.timings.keys()):
            f.write(str(len(self.timings[k].keys())) + " ")
        f.write("\n")
        f.write("#config_nb=" + str(len(config_set)) + "\n")
        for i, s in enumerate(sorted(sizes_set)):
            f.write("#Block_{0}_{1}={2}\n".format(s[0], len(s), i))
        for s in sorted(sizes_set):
            for c in sorted(config_set):
                for k in sorted(self.timings.keys()):
                    for v in sorted(self.timings[k].keys()):
                        f.write(str(s[0]) + " " + str(len(s)) + " ")
                        f.write(k + " ")
                        f.write(v + " ")
                        f.write(c + " ")
                        #print c
                        # Compute work-item number from configuration string:
                        # If config, start with 'wi=N', work-item number is set to N
                        # Else, it assume a configuration matching 'A[xB]+[_fn]?'
                        # It replace 'x' by '*' and divide by n. String is evaluated as python instruction
                        if c[0:3] == 'wi=':
                            cse = c.split('=')[1].split('_')[0]
                        else:
                            cs = c.replace(
                                'Nx', str(s[0])).replace('x', '*').split('_')
                            cse = cs[0] + '/' + cs[1][1] if len(cs) == 2 else cs[0]
                        #print cse
                        f.write(str(eval(cse)) + ' ')
                        try:
                            f.write(str(self.timings[k][v][c][s]) + "\n")
                        except:
                            f.write('-\n')
            f.write("\n")
        f.close()


def find_min(filename, kernel=None, version=None, config=None, size=None):
    d = pickle.load(open(filename, 'r'))
    cc = {}
    kl = d.keys() if kernel is None else [kernel] \
        if isinstance(kernel, str) else kernel
    for k in kl:
        vl = d[k].keys() if version is None else [version] \
            if isinstance(version, str) else version
        for v in vl:
            cl = d[k][v].keys() if config is None else [config] \
                if isinstance(config, str) else config
            for c in cl:
                sl = d[k][v][c].keys() if size is None else [size] \
                    if isinstance(size, tuple) else size
                for s in sl:
                    cc[k + '_' + v + '_' + c] = d[k][v][c][s]
    print cc[min(cc, key=cc.get)], min(cc, key=cc.get)


class Benchmark(object):
    """Benchmark management"""

    def __init__(self, cl_env, kernel_file, kernel_name, sizes,
                 config, setupFunction,
                 nb_run=20, inputs=None):
        """
        Creates a benchmark for a given source kernel_file, kernel for
        different problem sizes.

        @param kernel_file : kernels source file
        @param kernel_name : name of the kernel to benchmark as a string
        @param sizes : list of different problem sizes to launch kernel
        @param config : list of kernel parameters
        @param setupFunction : Function that returns building options and
        kernel arguments (assuming arrays are numpy arrays) depending on a
        given config, size and input dictionary.
        @param nb_run : number of launches to average time (default = 20)
        @param inputs : input data
        @param precision : Floating point precision for kernels
        """
        self.cl_env = cl_env
        self.platform = self.cl_env.platform
        self.device = self.cl_env.device
        self.ctx = self.cl_env.ctx
        self.queue = self.cl_env.queue
        ## OpenCL Source kernel_file
        self.kernel_file = kernel_file
        ## Kernel name
        self.kernel = kernel_name
        ## Compiled programs
        self.prg = {}
        ## Kernel timings
        self.timings = {}
        ## Kernel arguments
        self.kernel_args = {}
        ## Run number
        self.nb_run = nb_run
        ## Problems sizes
        self.sizes = sizes
        ## Problems inputs
        self.inputs = inputs
        ## Function to test size
        self.setupFunction = setupFunction

        self.setup = {}
        if self.kernel_file is not None:
            for size in self.sizes:
                self.setup[size] = self.setupFunction(config, size, self.inputs)
                if self.setup[size] is not None:
                    toDelete = False
                    #print np.prod(self.setup[size][1][1]), "WI (",self.device.max_work_group_size ," max )"
                    if np.prod(self.setup[size][1][1]) > self.device.max_work_group_size:
                        toDelete = True
                    global_mem_used=0
                    for arg in self.setup[size][1]:
                        if isinstance(arg, np.ndarray) and \
                                len(arg.shape) > 1:
                            #print "Alloc : ", arg.nbytes, "Bytes (", self.device.max_mem_alloc_size, "max)"
                            if arg.nbytes > self.device.max_mem_alloc_size:
                                toDelete = True
                            global_mem_used += arg.nbytes
                            #print "Total Alloc : ", global_mem_used, "Bytes (", self.device.global_mem_size, "max)"
                    if global_mem_used > self.device.global_mem_size:
                        toDelete = True
                        #print "Local Alloc : ", self.setup[size][2], "Bytes (", self.device.local_mem_size, "max)"
                    if self.setup[size][2] > self.device.local_mem_size:
                        toDelete = True
                    if toDelete:
                        self.setup[size] = None
        if self.kernel_file is not None:
            print kernel_file
            for size in self.sizes:
                if self.setup[size] is not None:
                    print self.setup[size][0]
                    self.prg[size] = self.cl_env.build_src(
                        kernel_file, **self.setup[size][0])

    def test(self, true_res, ind_res):
        """
        Testing result validity against a given 'true_res'.

        @param true_res : expected result
        @param ind_res : kernel argument index containig result
        """
        print "Testing : "
        for size in self.sizes:
            if self.setup[size] is not None:
                kernel = eval('self.prg[size].' + self.kernel)
                kernelArgs = self.setup[size][1]
                clkernelArgs = [None]*len(self.setup[size][1])
                res = np.empty_like(kernelArgs[ind_res])
                mem_used = 0
                for i in xrange(len(kernelArgs)):
                    if isinstance(kernelArgs[i], np.ndarray):
                        print "buffer", kernelArgs[i].shape
                        buff = cl.Buffer(
                            self.ctx, cl.mem_flags.READ_WRITE,
                            size=kernelArgs[i].nbytes)
                        cl.enqueue_copy(self.queue, buff, kernelArgs[i])
                        mem_used += kernelArgs[i].nbytes
                        clkernelArgs[i] = buff
                    else:
                        clkernelArgs[i] = kernelArgs[i]
                print "Memory used : {0:.5f} GiB ({1:.2f}%)".format(
                    mem_used / (1024. ** 3),
                    100. * mem_used / (self.device.global_mem_size * 1.),
                    mem_used / (1024. ** 3))
                self.queue.finish()
                print size, clkernelArgs[0:2],
                kernel(self.queue, *tuple(clkernelArgs))
                self.queue.finish()
                cl.enqueue_copy(self.queue, res, clkernelArgs[ind_res])
                self.queue.finish()
                for i in xrange(len(kernelArgs)):
                    if isinstance(clkernelArgs[i], cl.Buffer):
                        print "Released", clkernelArgs[i]
                        clkernelArgs[i].release()
                try:
                    # if len(res.shape) == 3:
                    #     res = res[:size[0], :size[1], :size[2]]
                    # else:
                    #     res = res[:size[0], :size[1]]
                    if np.float64 == HYSOP_REAL:
                        exp = 15
                    else:
                        exp = 6
                    print res.shape, size, true_res[size].shape
                    np.testing.assert_array_almost_equal(
                        res, true_res[size], decimal=exp)
                    print 'Ok'
                except AssertionError as ae:
                    print 'Fail'
                    print res.shape, true_res[size].shape, res - true_res[size]
                    i = 0
                    nb_bad_values = \
                        res[np.where(np.abs(res - true_res[size]) >=
                            eval('1e-' + str(exp - i)))].shape[0]
                    print nb_bad_values,
                    print "bad elements ~ 1e-{0}".format(exp - i)
                    while nb_bad_values > 0:
                        i += 1
                        nb_bad_values = \
                            res[np.where(np.abs(res - true_res[size]) >=
                                         eval('1e-' + str(exp - i)))].shape[0]
                        print nb_bad_values,
                        print "bad elements ~ 1e-{0}".format(exp - i)
                        if i == 4:
                            raise ae

    def launch(self, d=True):
        """
        Perform kernel benchmark.

        Kernels are run nb_run times plus one. The first run is not
        taken in timing average.
        """
        if d:
            print "\nRunning : "
        for size in self.sizes:
            if d:
                print size,
            if self.setup[size] is not None:
                kernel = eval('self.prg[size].' + self.kernel)
                kernelArgs = self.setup[size][1]
                mem_used = 0
                for i in xrange(len(kernelArgs)):
                    if isinstance(kernelArgs[i], np.ndarray):
                        buff = cl.Buffer(
                            self.ctx, cl.mem_flags.READ_WRITE,
                            size=kernelArgs[i].nbytes)
                        cl.enqueue_copy(self.queue, buff, kernelArgs[i])
                        mem_used += kernelArgs[i].nbytes
                        kernelArgs[i] = buff
                if d:
                    print "Memory used : {0:.5f} GiB ({1:.2f}%)".format(
                        mem_used / (1024. ** 3),
                        100. * mem_used / (self.device.global_mem_size * 1.)),
                self.queue.finish()
                if d:
                    print kernelArgs[0:2],
                evt = kernel(self.queue, *tuple(kernelArgs))
                self.queue.finish()
                evts = []
                for i in xrange(self.nb_run):
                    evt = kernel(self.queue, *tuple(kernelArgs))
                    evts.append(evt)
                self.queue.finish()
                time = 0.
                for evt in evts:
                    time += (evt.profile.end - evt.profile.start) * 1e-9
                self.timings[size] = time / self.nb_run
                self.kernel_args[size] = kernelArgs[0:2]
                if d:
                    print self.timings[size]
                for i in xrange(len(kernelArgs)):
                    if isinstance(kernelArgs[i], cl.Buffer):
                        kernelArgs[i].release()
            else:
                print "Incompatible sizes"
