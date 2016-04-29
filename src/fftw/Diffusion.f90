!> Diffusion solvers based on fftw .
!! These routines are just used for tests, since their equivalent is implemented in python interface,
!! with direct call to functions from fft2d and fft3d modules.
module diffusion

  use fft2d
  use fft3d

  use client_data

  implicit none

contains

!!$  subroutine solveDiffusion2D(nudt,velocity_x,velocity_y,omega)
!!$
!!$    real(mk), intent(in) :: nudt
!!$    real(mk),dimension(:,:),intent(in) :: velocity_x,velocity_y
!!$    real(mk),dimension(:,:),intent(inout) :: omega
!!$
!!$    !! Compute fftw forward transform
!!$    !! Omega is used to initialize the fftw buffer for input field.
!!$    call r2c_diffusion_2d(velocity_x,velocity_y)
!!$
!!$    call filter_diffusion_2d(nudt)
!!$
!!$    call c2r_diffusion_2d(omega)
!!$
!!$  end subroutine solveDiffusion2D

  subroutine solveDiffusion3D(nudt,velocity_x,velocity_y,velocity_z,omega_x,omega_y,omega_z, ghosts_v, ghosts_w)
    real(mk), intent(in) :: nudt
    real(mk),dimension(:,:,:),intent(inout) :: omega_x,omega_y,omega_z
    real(mk),dimension(:,:,:),intent(in) :: velocity_x,velocity_y,velocity_z
    integer, dimension(3), intent(in) :: ghosts_w, ghosts_v

    !! Compute fftw forward transform
    !! Omega is used to initialize the fftw buffer for input field.

    call r2c_3d(velocity_x,velocity_y,velocity_z, ghosts_v)

    call filter_curl_diffusion_3d(nudt)

    call c2r_3d(omega_x,omega_y,omega_z, ghosts_w)

  end subroutine solveDiffusion3D


  subroutine initDiffusionSolverC(pbdim,resolution,lengths,parent_comm)

    integer, intent(in) :: pbdim
    integer, dimension(pbdim),intent(in) :: resolution
    real(mk),dimension(pbdim), intent(in) :: lengths
    integer, intent(in)                 :: parent_comm

    integer :: ierr

    ! Duplicate parent_comm
    call mpi_comm_dup(parent_comm, main_comm, ierr)

    if(pbdim == 2) then
       !!call init_c2c_diffusion_2d(resolution,lengths)
       print * , "WARNING, diffusion not yet implemented in 2D"
    elseif(pbdim == 3) then
       call init_c2c_3d(resolution,lengths)
    endif

  end subroutine initDiffusionSolverC

!!$  subroutine solveDiffusion2DC(nudt,velocity_x,velocity_y,omega)
!!$
!!$    real(mk), intent(in) :: nudt
!!$    complex(mk),dimension(:,:) :: omega,velocity_x,velocity_y
!!$
!!$    !! Compute fftw forward transform
!!$    !! Omega is used to initialize the fftw buffer for input field.
!!$    call c2c_2d(omega,velocity_x,velocity_y)
!!$
!!$  end subroutine solveDiffusion2DC
!!$
!!$  subroutine solveDiffusion3DC(nudt, omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z)
!!$
!!$    complex(mk),dimension(:,:,:),intent(in) :: omega_x,omega_y,omega_z
!!$    complex(mk),dimension(:,:,:),intent(inout) :: velocity_x,velocity_y,velocity_z
!!$
!!$    !! Compute fftw forward transform
!!$    !! Omega is used to initialize the fftw buffer for input field.
!!$    call c2c_3d(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z)
!!$
!!$  end subroutine solveDiffusion3DC

  subroutine cleanDiffusionSolver3D()
    integer ::info
    call MPI_BARRIER(main_comm,info)
    call cleanFFTW_3d()

  end subroutine cleanDiffusionSolver3D
  subroutine cleanDiffusionSolver2D()
    integer ::info
    call MPI_BARRIER(main_comm,info)
    call cleanFFTW_2d()

  end subroutine cleanDiffusionSolver2D


end module diffusion


