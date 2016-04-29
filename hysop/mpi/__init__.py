"""Hysop interface to the mpi implementation.

It contains :

* mpi basic variables (main communicator, rank, size ...)
* :class:`hysop.mpi.topology.Cartesian` : mpi process distribution + local mesh


This package is used to hide the underlying mpi interface
in order to make any change of this interface, if required, easiest.

At the time we use mpi4py : http://mpi4py.scipy.org


"""

# Everything concerning the chosen mpi implementation is hidden in main_var
# Why? --> to avoid that things like mpi4py. ... spread everywhere in the
# soft so to ease a change of this implementation (if needed).
from hysop.mpi import main_var

MPI = main_var.MPI
"""MPI underlying implementation
"""

main_comm = main_var.main_comm
"""Main communicator (copy of comm_world)
"""

main_rank = main_var.main_rank
"""Rank of the current process in the main communicator
"""

main_size = main_var.main_size
"""Size of the main communicator
"""

Wtime = MPI.Wtime
"""Timer for MPI calls"""
