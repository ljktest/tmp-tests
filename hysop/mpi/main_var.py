"""
@file main_var.py

Global parameters related to mpi, for hysop package.

"""

# Set the underlying implementation of MPI
# mpi4py must be hidden from user
import mpi4py.MPI
MPI = mpi4py.MPI

# Create hysop main communicator from COMM_WORLD
main_comm = MPI.COMM_WORLD.Dup()
main_rank = main_comm.Get_rank()
main_size = main_comm.Get_size()
