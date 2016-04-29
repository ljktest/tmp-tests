!> Poisson solvers based on fftw .
!
!! These routines are just used for tests, since their equivalent is implemented in python interface,
!! with direct call to functions from fft2d and fft3d modules.
module poisson

  use fft2d
  use fft3d
  use client_data
  use mpi

  implicit none

contains

  subroutine initPoissonSolver(pbdim,resolution,lengths,parent_comm)

    integer, intent(in) :: pbdim
    integer, dimension(pbdim),intent(in) :: resolution
    real(mk),dimension(pbdim), intent(in) :: lengths
    integer, intent(in)                 :: parent_comm

    integer :: ierr

    ! Duplicate parent_comm
    call mpi_comm_dup(parent_comm, main_comm, ierr)

    if(pbdim == 2) then
       call init_r2c_2d(resolution,lengths)
    else if(pbdim == 3) then
       call init_r2c_3d(resolution,lengths)
    end if

  end subroutine initPoissonSolver

  subroutine solvePoisson2D(omega,velocity_x,velocity_y, ghosts_w, ghosts_v)

    real(mk),dimension(:,:),intent(in) :: omega
    real(mk),dimension(:,:),intent(inout) :: velocity_x,velocity_y
    integer, dimension(2), intent(in) :: ghosts_w, ghosts_v

    !! Compute fftw forward transform
    !! Omega is used to initialize the fftw buffer for input field.
    call r2c_scalar_2d(omega, ghosts_w)

    call filter_poisson_2d()

    call c2r_2d(velocity_x,velocity_y, ghosts_v)

  end subroutine solvePoisson2D

  subroutine solvePoisson3D(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z, ghosts_w, ghosts_v)

    real(mk),dimension(:,:,:),intent(in) :: omega_x,omega_y,omega_z
    real(mk),dimension(:,:,:),intent(inout) :: velocity_x,velocity_y,velocity_z
    integer, dimension(3), intent(in) :: ghosts_w, ghosts_v
    real(mk) :: start
    !! Compute fftw forward transform
    !! Omega is used to initialize the fftw buffer for input field.

    start = MPI_WTIME()
    call r2c_3d(omega_x,omega_y,omega_z, ghosts_w)
    print *, "time for r2c : ", MPI_WTIME() - start

    start = MPI_WTIME()
    call filter_poisson_3d()
    print *, "time for filter : ", MPI_WTIME() - start

    start = MPI_WTIME()
    call c2r_3d(velocity_x,velocity_y,velocity_z, ghosts_v)
    print *, "time for c2 : ", MPI_WTIME() - start

  end subroutine solvePoisson3D
  subroutine solvePoisson3D_many(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z)

    real(mk),dimension(:,:,:),intent(in) :: omega_x,omega_y,omega_z
    real(mk),dimension(:,:,:),intent(inout) :: velocity_x,velocity_y,velocity_z
    real(mk) :: start
    !! Compute fftw forward transform
    !! Omega is used to initialize the fftw buffer for input field.

    print *, "--------------------------- Start solve 3d many case ..."

    start = MPI_WTIME()
    call r2c_3d_many(omega_x,omega_y,omega_z)
    print *, "time for r2cmany : ", MPI_WTIME() - start

    start = MPI_WTIME()
    call filter_poisson_3d_many()
    print *, "time for filtermany : ", MPI_WTIME() - start

    start = MPI_WTIME()
    call c2r_3d_many(velocity_x,velocity_y,velocity_z)
    print *, "time for c2rmany : ", MPI_WTIME() - start

  end subroutine solvePoisson3D_many


  subroutine initPoissonSolverC(pbdim,resolution,lengths,parent_comm)

    integer, intent(in) :: pbdim
    integer, dimension(pbdim),intent(in) :: resolution
    real(mk),dimension(pbdim), intent(in) :: lengths
    integer, intent(in)                 :: parent_comm

    integer :: ierr

    ! Duplicate parent_comm
    call mpi_comm_dup(parent_comm, main_comm, ierr)

    if(pbdim == 2) then
       call init_c2c_2d(resolution,lengths)
    elseif(pbdim == 3) then
       call init_c2c_3d(resolution,lengths)
    endif

  end subroutine initPoissonSolverC

  subroutine solvePoisson2DC(omega,velocity_x,velocity_y)

    complex(mk),dimension(:,:) :: omega,velocity_x,velocity_y

    !! Compute fftw forward transform
    !! Omega is used to initialize the fftw buffer for input field.
    call c2c_2d(omega,velocity_x,velocity_y)

  end subroutine solvePoisson2DC

  subroutine solvePoisson3DC(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z)

    complex(mk),dimension(:,:,:),intent(in) :: omega_x,omega_y,omega_z
    complex(mk),dimension(:,:,:),intent(inout) :: velocity_x,velocity_y,velocity_z

    !! Compute fftw forward transform
    !! Omega is used to initialize the fftw buffer for input field.
    call c2c_3d(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z)

  end subroutine solvePoisson3DC

  subroutine cleanPoissonSolver3D()
    integer ::info
    call MPI_BARRIER(main_comm,info)
    call cleanFFTW_3d()

  end subroutine cleanPoissonSolver3D
  subroutine cleanPoissonSolver2D()
    integer ::info
    call MPI_BARRIER(main_comm,info)
    call cleanFFTW_2d()

  end subroutine cleanPoissonSolver2D


end module poisson


