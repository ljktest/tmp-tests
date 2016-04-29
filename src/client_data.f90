!> Some global parameters and variables
module client_data

  use MPI, only : MPI_DOUBLE_PRECISION
  use, intrinsic :: iso_c_binding ! required for fftw
  implicit none

  !> kind for real variables (simple or double precision)
  integer, parameter :: mk = kind(1.0d0) ! double precision
  !> kind for real variables in mpi routines
  integer, parameter :: mpi_mk = MPI_DOUBLE_PRECISION
  !> Problem dimension (model, required for ppm to work properly)
  integer, parameter :: dime = 2
  !> Real dimension
  integer, parameter :: dim3 = 3
  !> Pi constant
  real(mk), parameter :: pi = 4.0*atan(1.0_mk)
  !> MPI main communicator
  integer :: main_comm
  !> Rank of the mpi current process
  integer :: rank ! current mpi-processus rank
  !> Total number of mpi process
  integer :: nbprocs
  !>  trick to identify coordinates in a more user-friendly way
  integer,parameter :: c_X=1,c_Y=2,c_Z=3
  !> to activate (or not) screen output
  logical,parameter :: verbose = .True.
  !> i (sqrt(-1) ...)
  complex(C_DOUBLE_COMPLEX), parameter :: Icmplx = cmplx(0._mk,1._mk, kind=mk)
  !> tolerance used to compute error
  real(mk), parameter :: tolerance = 1e-12

end module client_data
