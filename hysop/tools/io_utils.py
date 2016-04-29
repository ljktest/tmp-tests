"""Tools related to i/o in HySoP.

.. currentmodule hysop.tools.io_utils

* :class:`~IO`
* :class:`~IOParams`
* :class:`~Writer`
* :class:`~XMF`, tools to prepare/write xmf files.

"""
import os
import scitools.filetable as ft
import hysop.tools.numpywrappers as npw
from inspect import getouterframes, currentframe
import hysop.mpi as mpi
from re import findall
from hysop.tools.parameters import MPIParams
from collections import namedtuple
from hysop.constants import S_DIR
import h5py


class IO(object):
    """
    Static class with utilities to set/find the place where i/o files
    will be read/written.
    """

    _default_path = None

    HDF5 = 998
    """HDF5 format id"""

    ASCII = 997
    """ascii format id"""

    @staticmethod
    def default_path():
        """Get the current default path used for io.

        Returns
        -------
        string
            the default value of the current i/o path. If not set,
            the basename of the top exec file is used.

        Examples
        --------

        If 'mpirun -np 4 python MyExe.py' is run,
        results will be saved in ./MyExe/p4.
        The number after 'p' will be the number of mpi processus
        set for the simulation.
        """
        # Memo for getouterframe usage:
        #    frame, filename, line_number, function_name, lines, index =\
        # inspect.getouterframes(inspect.currentframe())[-1]
        # Warning FP : the behavior of python and ipython is different for
        # this command. Moreover, output differs when called from interactive
        # session.

        if IO._default_path is not None:
            # which must be the case for any interactive session
            # due to call of set_default_path in hysop.__init__.py.
            return IO._default_path

        a = getouterframes(currentframe())
        #ind = -1
        interactive_path = './interactive/p' + str(mpi.main_size)
        interactive_path = os.path.abspath(interactive_path)
        # --- ipython ---
        #from hysop.tools.sys_utils import SysUtils
        # if SysUtils.in_ipython():
        #     # Note FP: because of set_default_path call
        #     # in __init__.py, this condition must never happen.
        #     # But we keep the code below, just in case ...

        #     # list of files (fullpath) which contain the callers
        #     sublist = [i[1] for i in a]
        #     # look for ipython in callers ...
        #     # If found, keep index of the file just before
        #     # first occurence of ipython, i.e. the name
        #     # of the 'main' file
        #     for val in sublist:
        #         ll = findall('ipython', val, IGNORECASE)
        #         if len(ll) > 0:
        #             ind = sublist.index(val) - 1
        #             break

        #     if ind > -1:
        #         # -- interactive ipython but call with execfile--
        #         if len(findall('io_utils', a[ind][1])) > 0:
        #             return interactive_path
        #         a = a[ind]
        #     else:
        #         # -- interactive ipython without execfile call --
        #         return interactive_path

        # else:
        # -- python --
        # if test session, set default path to interactive_path
        for fname in a:
            cond1 = len(findall('py.test', fname[1])) > 0
            cond2 = len(findall('pytest', fname[1])) > 0
            if cond1 or cond2:
                IO._default_path = interactive_path
                return interactive_path

        # else (not in tests)
        # get name of the file which contains the last caller
        # and check if not None
        a = a[-1]
        if a[-1] is None:
            # interactive python
            IO._default_path = interactive_path
            return interactive_path
        # Finally (if not interactive, not in tests ...)
        # use root name of the file which contains
        # the last caller
        apath = os.path.abspath(os.path.dirname(a[1]))
        sphinxbuild = findall('sphinx-build', a[1])
        if len(sphinxbuild) > 0:
            a = a[1]
        else:
            a = os.path.basename(a[1]).split('.')[-2]
            if a.find('__init__') != -1:
                IO._default_path = interactive_path
                return interactive_path
        a = os.path.join(apath, a)
        return os.path.join(a, 'p' + str(mpi.main_size))

    @staticmethod
    def check_dir(filename, io_rank=0, comm=None):
        """Check if the directory of 'filename' exists and creates it if not.

        Parameters
        -----------
        filename : string
            file name with full or relative path
        io_rank : int
            processus rank that does the check.
        comm : mpi communicator
            the mpi communicator that does the check.
        """
        # Create output dir if required
        if comm is None:
            comm = mpi.main_comm
        if comm.Get_rank() == io_rank:
            d = os.path.dirname(filename)
            if not os.path.exists(d):
                os.makedirs(d)

    @staticmethod
    def set_default_path(pathdir):
        """Set a new default path for hysop i/o.

        Parameters
        -----------
        pathdir : string
            the new path

        Notes
        ------
        pN will be add to path name, N being the number of MPI process
        used for the simulation.

        """
        IO._default_path = pathdir
        IO._default_path = os.path.join(IO._default_path,
                                        'p' + str(mpi.main_size))
        IO.check_dir(IO._default_path)

    @staticmethod
    def set_datasetname(field_name, topo, direction=None):
        """Return the dataset name of a given continuous field,
        saved for a given topology
        """
        val = field_name + '_' + str(topo.get_id())
        if direction is not None:
            val += S_DIR[direction]
        return val

    @staticmethod
    def get_datasetnames(filename):
        """Return the list of dataset names present
        in hdf input file

        Parameters
        ----------
        filename : string
            hdf file

        Returns
        -------
            a list of strings
        """
        hdf_file = h5py.File(filename, 'r')
        keys = hdf_file.keys()
        hdf_file.close()
        return keys


class IOParams(namedtuple("IOParams", ['filename', 'filepath',
                                       'frequency', 'fileformat',
                                       'io_leader'])):
    """
    A struct to handle I/O files parameters

    Parameters
    -----------
    filename : string
        name of the file (absolute or relative path)
    filepath : string
        location of the file
    frequency : int
        frequency of output or input (e.g. every N times steps)
    fileformat : int
        format of the file. See notes for available format. Default=HDF5.
    io_leader : int
        rank of the mpi process dealing with the io. Default is 0.

    See examples in hysop.operator.hdf_io

    Notes
    -----
    Format parameter must be one of the following :
      - :class:`~IO.HDF5`
      - :class:`~IO.ASCII`

    """
    def __new__(cls, filename, filepath=None, frequency=1,
                fileformat=None, io_leader=0):

        # Filename is absolute path, filepath arg is ignored.
        if os.path.isabs(filename):
            filepath = os.path.dirname(filename)

        else:
            if filepath is not None:
                filename = os.path.join(filepath, filename)
                filepath = os.path.abspath(os.path.dirname(filename))
            else:
                filepath = os.path.dirname(filename)
                if filepath == '':
                    # Get default output path
                    filepath = IO.default_path()
                    filename = os.path.join(filepath, filename)
                else:
                    filepath = os.path.abspath(filepath)
                    filename = os.path.join(filepath,
                                            os.path.basename(filename))
        if fileformat is None:
            fileformat = IO.HDF5

        IO.check_dir(filename)
        return super(IOParams, cls).__new__(cls, filename, filepath,
                                            frequency, fileformat, io_leader)


class Writer(object):
    """
    To write data from a 2D numpy array into an ascii file.

    Examples
    --------

    >>> from hysop.tools.io_utils import IOParams, IO
    >>> params = IOParams(filename='r.dat', fileformat=IO.ASCII)
    >>> wr = Writer(params, buffshape=(1, 2))
    >>> ite = 3 # current iteration number
    >>> if wr.do_write(ite):
    ...    wr.buffer[...] = 3.
    ...    wr.write()
    >>> wr.finalize()

    result : buffer is written into r.dat
    """
    def __init__(self, io_params, buffshape=None, mpi_params=None,
                 safe_io=True):
        """

        Parameters
        ----------
        io_params : hysop.tools.io_utils.IOParams
            setup for file ouput (name, location ...)
        buffshape : tuple
            2D numpy.array.shape like tuple, shape of the output/input buffer.
        mpi_params : hysop.tools.parameters.MPIParams
            mpi setup (comm that owns the writer)
        safe_io : boolean
            True --> open/close file everytime data are written.
            False --> open at init and close during finalize.
            Cost less but if simu crashes, data are lost.

        """
        # Absolute path + name for i/o file
        # Note that if filename contains absolute path
        # filepath is ignored
        msg = 'wrong type for io_params arg.'
        assert isinstance(io_params, IOParams), msg
        assert io_params.fileformat == IO.ASCII
        self.io_params = io_params

        # A reference communicator, just to identify a
        # process rank for io.
        if mpi_params is None:
            mpi_params = MPIParams()
        else:
            msg = 'wrong type for mpi_params arg.'
            assert isinstance(mpi_params, MPIParams), msg
        self._mpis = mpi_params

        # check if output dir exists, create it if not.
        IO.check_dir(self.io_params.filename, self.io_params.io_leader,
                     self._mpis.comm)

        # Shape of the output buffer (must be a 2D numpy array)
        if buffshape is None:
            buffshape = (1, 1)
        self._buffshape = buffshape
        assert len(self._buffshape) == 2,\
            '2D shape required : set arg buffshape as a 2D tuple: (x,y)'

        # The buffer (numpy array) that will be printed to a file
        self.buffer = npw.zeros(self._buffshape)
        " buffer used to save printed data"

        # Defines how often
        # output file is written :
        # True --> open/close file everytime
        # data are written.
        # False --> open at init and close
        # during finalize. Cost less but if simu
        # crashes, data are lost.
        if safe_io:
            self.write = self._fullwrite
        else:
            self.write = self._partialwrite

        # Force synchro to be sure that all output dirs
        # have been created.
        self._mpis.comm.barrier()
        if self._mpis.rank == self.io_params.io_leader:
            self._file = open(self.io_params.filename, 'w')

    def do_write(self, ite):
        """Returns true if output is required
        for iteration ite

        Parameters
        ----------
        ite : int
            current iteration number

        """
        num = ite + 1  # Number of iterations done
        rk = self._mpis.rank
        return rk == self.io_params.io_leader and \
            (num % self.io_params.frequency) == 0

    def _fullwrite(self):
        """open, write and close"""
        self._file = open(self.io_params.filename, 'a')
        ft.write(self._file, self.buffer)
        self._file.close()

    def _partialwrite(self):
        """just write, no open, nor close"""
        ft.write(self._file, self.buffer)

    def finalize(self):
        """close, if required"""
        if self._mpis.rank == self.io_params.io_leader:
            if not self._file.closed:
                self._file.close()

    def __str__(self):
        if self._mpis.rank == self.io_params.io_leader:
            s = ' === Writer === \n'
            s += ' - filename = ' + self.io_params.filename
            s += '\n - buffshape = ' + str(self._buffshape)
            s += '\n - frequ = ' + str(self.io_params.frequency)
            return s


class XMF(object):
    """Static class - Tools to prepare and write xmf file
    """

    @staticmethod
    def _list_format(l):
        """Format a list to the xml output.
        Removes the '[]()' and replace ',' with ' ' in default str.

        Parameters
        ----------
        l : list to format

        """
        buff = str(l).replace(',', ' ').replace('[', '')
        return buff.replace(']', '').replace('(', '').replace(')', '')

    @staticmethod
    def write_grid_attributes(topo, dataset_names, ite,
                              time, filename, subset=None):
        """
        Write XDMF header into a file

        Parameters
        -----------
        topo : :class:`hysop.mpi.topology.Cartesian`
             used as reference to define local and global meshes in xdmf file.
        dataset_names : list
            all datasets names
        ite : int
            iteration number
        time : double
            current time
        filename : string
            name of the hdf file which contains datas for the current process.
        subset : :class:`hysop.domain.subsets.Subset`, optional
            to define a grid only on this subset.
            If None, grid on the whole domain (from topo)

        Returns:
        --------
        string
            the xml-like header.

        """
        # The header (xml-like), saved in a string.
        xml_grid = ""
        dimension = topo.domain.dimension
        if dimension == 2:
            topo_type = "2DCORECTMesh"
            geo_type = "ORIGIN_DXDY"
        elif dimension == 3:
            topo_type = "3DCORECTMesh"
            geo_type = "ORIGIN_DXDYDZ"
        xml_grid += "   <Grid Name=\"Iteration {0:03d}\"".format(ite)
        xml_grid += " GridType=\"Uniform\">\n"
        xml_grid += "    <Time Value=\"{0}\" />\n".format(time)
        xml_grid += "    <Topology TopologyType=\"" + str(topo_type) + "\""
        xml_grid += " NumberOfElements=\""

        # Check substet to find the required grid resolution
        if subset is not None:
            resolution = list(subset.mesh[topo].global_resolution())
            origin = list(subset.real_orig[topo])
        else:
            resolution = list(topo.mesh.global_resolution())
            origin = list(topo.domain.origin)
        resolution.reverse()
        origin.reverse()
        xml_grid += XMF._list_format(resolution) + " \"/>\n"
        xml_grid += "    <Geometry GeometryType=\"" + geo_type + "\">\n"
        xml_grid += "     <DataItem Dimensions=\"" + str(dimension) + " \""
        xml_grid += " NumberType=\"Float\" Precision=\"8\" Format=\"XML\">\n"
        xml_grid += "     " + XMF._list_format(origin) + "\n"
        xml_grid += "     </DataItem>\n"
        xml_grid += "     <DataItem Dimensions=\"" + str(dimension) + " \""
        xml_grid += " NumberType=\"Float\" Precision=\"8\" Format=\"XML\">\n"
        step = list(topo.mesh.space_step)
        step.reverse()
        xml_grid += "     " + XMF._list_format(step) + "\n"
        xml_grid += "     </DataItem>\n"
        xml_grid += "    </Geometry>\n"
        # Append dataset parameters
        for name in dataset_names:
            xml_grid += "    <Attribute Name=\""
            xml_grid += name + "\""
            xml_grid += " AttributeType=\"Scalar\" Center=\"Node\">\n"
            xml_grid += "     <DataItem Dimensions=\""
            xml_grid += XMF._list_format(resolution) + " \""
            xml_grid += " NumberType=\"Float\" Precision=\"8\" Format=\"HDF\""
            xml_grid += " Compression=\"Raw\">\n"  #
            xml_grid += "      " + filename.split('/')[-1]
            xml_grid += ":/" + name
            xml_grid += "\n     </DataItem>\n"
            xml_grid += "    </Attribute>\n"
        xml_grid += "   </Grid>\n"
        return xml_grid
