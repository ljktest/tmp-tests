"""I/O operators

.. currentmodule:: hysop.operator.hdf_io

* :class:`~HDF_Writer` : operator to write fields into an hdf file
* :class:`~HDF_Reader` : operator to read fields from an hdf file
* :class:`~HDF_IO` abstract interface for hdf io classes

"""
from hysop.constants import S_DIR, debug, HYSOP_REAL
from hysop.operator.computational import Computational
from hysop.operator.continuous import opapply, opsetup
import hysop.tools.numpywrappers as npw
from hysop.tools.io_utils import IO, IOParams, XMF
from abc import ABCMeta, abstractmethod

try:
    import h5py
except ImportError as h5py_error:
    h5py = None
    msg_err = 'Warning, h5py not found, you may not be able to'
    msg_err += ' use hdf5 I/O functionnalities.'
    print msg_err

from hysop.tools.profiler import profile


class HDF_IO(Computational):
    """Abstract interface to read/write from/to hdf files, for
    hysop fields.
    """

    __metaclass__ = ABCMeta

    def __init__(self, var_names=None, subset=None, **kwds):
        """Read/write some fields data from/into hdf/xmdf files.
        Parallel io.


        Parameters
        ----------
        var_names : a dictionnary, optional
            keys = :class:`~hysop.fields.continuous.Field`,
            values = string, field name. See notes below.
        subset : :class:`~hysop.domain.subset.Subset`, optional
            a subset of the domain, on which data are read or written,
            default=the whole domain.

        Notes
        -----
        Dataset in hdf files are identified with names. In hysop, when
        writing a file, default dataset name is
        'continuous_field.name + topo.id + component direction', but
        this might be changed thanks to var_names argument.
        For example, if variables=[velo, vorti], and if hdf file contains
        'vel_1_X, vel_1_Y, vel_1_Z, dat_2_X, dat_2_Y, dat_2_Z' keys, then
        use :
        var_names = {velo: 'vel', vorti:'dat'} if you want to read vel/dat
        into velo/vorti.

        """
        super(HDF_IO, self).__init__(**kwds)

        if h5py is None:
            print ('You try to use HDF5 reader but h5py module ',)
            print ('has not been found on your system.', )
            raise h5py_error

        self.input = self.variables
        self.output = self.variables

        # If no filename is given, set it to
        # the concatenation of variables'names.
        if self.io_params is None:
            # if name is not set, name = concatenation of fields names,
            # like v1_v2.
            # WARNING FP: we must sort names (alph. order), else
            # it seems the order may change from one mpi process
            # to another.
            name = ''
            names = []
            for var in self.input:
                names.append(var.name)
            names.sort()
            for nn in names:
                name += nn + '_'
            name = name[:-1]
            self.io_params = IOParams(name, fileformat=IO.HDF5)
        else:
            assert self.io_params.fileformat is IO.HDF5
        # Set a subset of the original domain
        self.subset = subset

        # Dictionnary of names to search in hdf file. May be None.
        # It will be checked during setup.
        self.var_names = var_names

        # Local topology, that MUST be common to all variables.
        self.topology = None
        self._slices = None
        self._global_resolution = None
        self._sl = None
        # Dictionnary of discrete fields. Key = name in hdf file,
        # Value = discrete field
        self.dataset = {}
        # Get hdf file name. Depends on read/write process. Must be
        # defined in HDF_READER or _WRITER init.
        self._get_filename = lambda i=None: None
        # File Object that holds hdf file
        self._hdf_file = None

    def discretize(self):
        super(HDF_IO, self)._standard_discretize()
        assert self._single_topo, 'Multi-resolution case is not allowed.'
        self.topology = self.variables.values()[0]

        # Discretize the subset, if required
        if self.subset is not None:
            self.subset.discretize(self.topology)
            refmesh = self.subset.mesh[self.topology]
        else:
            refmesh = self.topology.mesh
        # Global resolution for hdf5 output (warning : this must
        # be the whole domain resolution, not the subset resolution)
        self._global_resolution = list(refmesh.global_resolution())
        self._slices = refmesh.iCompute
        if refmesh.on_proc:
            sl = list(refmesh.position)
        else:
            sl = [slice(0, 0) for _ in xrange(self.domain.dimension)]
        # Reverse order, to fit with xdmf req.
        self._global_resolution.reverse()
        sl.reverse()
        self._sl = tuple(sl)

    @opsetup
    def setup(self, rwork=None, iwork=None):
        # No list of hdf dataset names provided by user ...
        if self.var_names is None:
            # Get field names and initialize dataset dict.
            for df in self.discreteFields.values():
                for d in xrange(df.nb_components):
                    name = df.name + S_DIR[d]
                    self.dataset[name] = df.data[d]
        else:
            for var in self.var_names:
                # Discrete field associated to var
                var_d = var.discretize(self.topology)
                for d in xrange(var_d.nb_components):
                    name = self.var_names[var] + S_DIR[d]
                    self.dataset[name] = var_d.data[d]

    def open_hdf(self, count, mode):
        filename = self._get_filename(count)
        if self.topology.size == 1:
            self._hdf_file = h5py.File(filename, mode)
            compression = 'gzip'
        else:
            self._hdf_file = h5py.File(filename, mode, driver='mpio',
                                       comm=self.topology.comm)
            compression = None

        return compression

    @abstractmethod
    @opapply
    def apply(self, simulation=None):
        """
        Abstract interface to read/write process
        """


class HDF_Writer(HDF_IO):
    """
    Print field(s) values on a given topo, in HDF5 format.
    """
    def __init__(self, xmfalways=True, **kwds):
        """
        Write some fields data into hdf/xmdf files.
        Parallel writings.

        Parameters
        ----------
        xmfalways : boolean, optional
            true if xmf output must be done every time
            an hdf5 file is created (i.e. at each time step),
            default=True
        kwds : base class arguments
        """
        super(HDF_Writer, self).__init__(**kwds)

        # count the number of calls
        self._count = 0

        if xmfalways:
            self.step = self._step_HDF5_XMF
            self.finalize = lambda: 1
        else:
            self.step = self._step_HDF5
            self.finalize = self.createXMFFile
        self._xdmf_data_files = []
        # filename = prefix_N, N = counter value
        self._get_filename = self._input_fname
        # In xdmf file, it is not possible to have two times
        # the same time value. So we save the last value of
        # time where xdmf has been written, to raise an exception
        # if that happens.
        self._last_written_time = None

    def _input_fname(self, i):
        """Set output file name for current iteration"""
        msg = 'count < 0, simu must be initialized.'
        assert i >= 0, msg
        return self.io_params.filename + "_{0:05d}".format(i) + '.h5'

    @debug
    @profile
    @opapply
    def apply(self, simulation=None):
        if simulation is None:
            raise ValueError("Missing simulation value for monitoring.")
        ite = simulation.currentIteration
        if ite == -1 or ite % self.io_params.frequency == 0:
            self.step(simulation)
            self._count += 1

    def createXMFFile(self):
        """Create and fill the xdmf file
        """

        if self._mpis.rank == self.io_params.io_leader:
            f = open(self.io_params.filename + '.xmf', 'w')
            f.write("<?xml version=\"1.0\" ?>\n")
            f.write("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n")
            f.write("<Xdmf Version=\"2.0\">\n")
            f.write(" <Domain>\n")
            f.write("  <Grid Name=\"CellTime\" GridType=\"Collection\" ")
            f.write("CollectionType=\"Temporal\">\n")
            ds_names = self.dataset.keys()
            for i, t in self._xdmf_data_files:
                f.write(XMF.write_grid_attributes(
                    self.topology, ds_names, i, t, self._get_filename(i),
                    self.subset))
            f.write("  </Grid>\n")
            f.write(" </Domain>\n")
            f.write("</Xdmf>\n")
            f.close()

#    def finalize(self):
#        self.createXMFFile()

    def _step_HDF5(self, simu):
        """Write an h5 file with data on each mpi process.
        """
        # Remarks:
        # - force np.float64, ParaView seems unable to read float32
        # - writing compressed hdf5 files (gzip compression seems the best)
        # - gzip compression does not work in parallel.

        # Get 'current' filename. It depends on the number
        # of the current output (count) and on the current process
        # rank.
        self._count = simu.currentIteration
        compression = self.open_hdf(self._count, mode='w')
        # Get the names of output variables and create the corresponding
        # datasets
        for name in self.dataset:
            ds = self._hdf_file.create_dataset(name,
                                               self._global_resolution,
                                               dtype=HYSOP_REAL,
                                               compression=compression)
            # In parallel, each proc must write at the right place
            # of the dataset --> use self._slices.
            ds[self._sl] = npw.asrealarray(self.dataset[name][self._slices].T)
        # Collect datas required to write the xdmf file
        # --> add tuples (counter, time).
        msg = 'You cannot write two hdf files for the same '
        msg += '(time, var) set. '
        msg += 'If you want to save a field two times for '
        msg += 'a single time value, please use two hdf_writer operators.'
        assert simu.time != self._last_written_time, msg
        self._xdmf_data_files.append((self._count, simu.time))
        self._last_written_time = simu.time
        self._hdf_file.close()

    def _step_HDF5_XMF(self, simu):
        self._step_HDF5(simu)
        self.createXMFFile()


class HDF_Reader(HDF_IO):
    """
    Parallel reading of hdf/xdmf files to fill some fields in.
    """
    def __init__(self, restart=None, **kwds):
        """Read some fields data from hdf/xmdf files.
        Parallel readings.

        Parameters
        ----------
        restart : int, optional
            number of a specific iteration to be read, default=None,
            i.e. read first iteration.
        kwds : base class arguments

        Notes: restart corresponds to the number which appears in
        the hdf file name, corresponding to the number of the
        iteration where writing occured.
        See examples in tests_hdf_io.py
        """
        super(HDF_Reader, self).__init__(**kwds)
        self.restart = restart
        if self.restart is not None:
            # filename = prefix_N, N = counter value
            self._get_filename = lambda i=self.restart: \
                self.io_params.filename + "_{0:05d}".format(i) + '.h5'
        else:
            self._get_filename = lambda i=None: self.io_params.filename

    @debug
    @profile
    @opapply
    def apply(self, simulation=None):
        # Read HDF file
        self.open_hdf(count=self.restart, mode='r')

        # Get the list of dataset names available in the hdf file
        dsnames = self.dataset_names()
        # And check if required dataset (from self.dataset)
        # are all in this list.
        msg = 'You try to read a dataset not present in hdf file : '
        for name in self.dataset:
            #            for d in xrange(self.dataset[name].nb_components):
            assert name in dsnames, msg + name
            # Read data
            self.dataset[name][self._slices] = self._hdf_file[name][self._sl].T

        self._hdf_file.close()
        # Set to None to check if it is closed in finalize
        self._hdf_file = None
        # Do we need a CPU->GPU transfer here?

    def dataset_names(self):
        """Return the list of available names for datasets in
        the required file.
        """
        assert self._is_discretized, 'a call to discretize() is required.'
        if self._hdf_file is None:
            self.open_hdf(count=self.restart, mode='r')
        return self._hdf_file.keys()

    def finalize(self):
        if self._hdf_file is not None:
            self._hdf_file.close()
