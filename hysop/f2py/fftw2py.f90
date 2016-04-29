!> @file fftw2py.f90
!! Fortran to python interface file.

!> Interface to mpi-fftw (fortran) utilities
module fftw2py

  use client_data
  use hysopparam
  !> 2d case
   use fft2d
  !> 3d case
  use fft3d
  use mpi
  implicit none

contains

  !> Initialisation of fftw context : create plans and memory buffers
  !! @param[in] resolution global resolution of the discrete domain
  !! @param[in] lengths width of each side of the domain
  !! @param[in] comm MPI communicator
  !! @param[out] datashape local dimension of the input/output field
  !! @param[out] offset absolute index of the first component of the local field
  subroutine init_fftw_solver(resolution,lengths,comm,datashape,offset,dim,fftw_type_real)

    integer, intent(in) :: dim
    integer, dimension(dim),intent(in) :: resolution
    real(pk),dimension(dim), intent(in) :: lengths
    integer(ik), dimension(dim), intent(out) :: datashape
    integer(ik), dimension(dim), intent(out) :: offset
    integer, intent(in)                 :: comm
    logical, optional :: fftw_type_real
    !f2py optional :: dim=len(resolution)
    !f2py intent(hide) dim
    !f2py logical optional, intent(in) :: fftw_type_real = 1

    integer :: ierr

    ! Duplicate comm into client_data::main_comm (used later in fft2d and fft3d)
    call mpi_comm_dup(comm, main_comm, ierr)

    if(fftw_type_real) then
       if(dim == 2) then
          !print*, "Init fftw/poisson solver for a 2d problem"
          call init_r2c_2d(resolution,lengths)
       else
          !print*, "Init fftw/poisson solver for a 3d problem"
          call init_r2c_3d(resolution,lengths)
       end if
    else
       if(dim == 2) then
          !print*, "Init fftw/poisson solver for a 2d problem"
          call init_c2c_2d(resolution,lengths)
       else
          !print*, "Init fftw/poisson solver for a 3d problem"
          call init_c2c_3d(resolution,lengths)
       end if
    end if

    if(dim==2) then
       call getParamatersTopologyFFTW2d(datashape,offset)
    else
       call getParamatersTopologyFFTW3d(datashape,offset)
    end if
  end subroutine init_fftw_solver


    !> Initialisation of fftw context : create plans and memory buffers
  !! @param[in] resolution global resolution of the discrete domain
  !! @param[in] lengths width of each side of the domain
  !! @param[in] comm MPI communicator
  !! @param[out] datashape local dimension of the input/output field
  !! @param[out] offset absolute index of the first component of the local field
  subroutine init_fftw_solver_scalar(resolution,lengths,comm,datashape,offset,dim,fftw_type_real)

    integer, intent(in) :: dim
    integer, dimension(dim),intent(in) :: resolution
    real(pk),dimension(dim), intent(in) :: lengths
    integer(ik), dimension(dim), intent(out) :: datashape
    integer(ik), dimension(dim), intent(out) :: offset
    integer, intent(in)                 :: comm
    logical, optional :: fftw_type_real
    !f2py optional :: dim=len(resolution)
    !f2py intent(hide) dim
    !f2py logical optional, intent(in) :: fftw_type_real = 1

    integer :: ierr

    ! Duplicate comm into client_data::main_comm (used later in fft2d and fft3d)
    call mpi_comm_dup(comm, main_comm, ierr)
    
    !print*, "Init fftw/poisson solver for a 3d problem"
    call init_r2c_scalar_3d(resolution,lengths)
    
    call getParamatersTopologyFFTW3d(datashape,offset)
    
  end subroutine init_fftw_solver_scalar

  !> Free memory allocated for fftw-related objects (plans and buffers)
  subroutine clean_fftw_solver(dim)

    integer, intent(in) :: dim
    if(dim == 2) then
       call cleanFFTW_2d()
    else
       call cleanFFTW_3d()
    end if
  end subroutine clean_fftw_solver

  !> Solve
  !! \f[ \nabla (\nabla \times velocity) = - \omega \f]
  !! velocity being a 2D vector field and omega a 2D scalar field.
  subroutine solve_poisson_2d(omega,velocity_x,velocity_y, ghosts_vort, ghosts_velo)
    real(pk),dimension(:,:),intent(in):: omega
    real(pk),dimension(size(omega,1),size(omega,2)),intent(out) :: velocity_x,velocity_y
    integer, dimension(2), intent(in) :: ghosts_vort, ghosts_velo
    !f2py intent(in,out) :: velocity_x,velocity_y
    
    call r2c_scalar_2d(omega, ghosts_vort)

    call filter_poisson_2d()

    call c2r_2d(velocity_x,velocity_y, ghosts_velo)
    !!print *, "fortran resolution time : ", MPI_WTime() - start

  end subroutine solve_poisson_2d

  !> Solve
  !! \f{eqnarray*} \frac{\partial \omega}{\partial t} &=& \nu \Delta \omega \f}
  !! omega being a 2D scalar field.
  subroutine solve_diffusion_2d(nudt, omega, ghosts_vort)
    real(pk), intent(in) :: nudt
    real(pk),dimension(:,:),intent(inout):: omega
    integer, dimension(2), intent(in) :: ghosts_vort
    !f2py intent(in,out) :: omega

    call r2c_scalar_2d(omega, ghosts_vort)

    call filter_diffusion_2d(nudt)

    call c2r_scalar_2d(omega, ghosts_vort)

  end subroutine solve_diffusion_2d

  !> Solve
  !! \f{eqnarray*} \Delta \psi &=& - \omega \\ velocity = \nabla\times\psi \f}
  !! velocity and omega being 3D vector fields.
  subroutine solve_poisson_3d(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z, ghosts_vort, ghosts_velo)
    real(pk),dimension(:,:,:),intent(in):: omega_x,omega_y,omega_z
    real(pk),dimension(size(omega_x,1),size(omega_y,2),size(omega_z,3)),intent(out) :: velocity_x,velocity_y,velocity_z
    integer, dimension(3), intent(in) :: ghosts_vort, ghosts_velo
    real(pk) :: start
    !f2py intent(in,out) :: velocity_x,velocity_y,velocity_z
    start = MPI_WTime()
    call r2c_3d(omega_x,omega_y,omega_z, ghosts_vort)

    call filter_poisson_3d()

    call c2r_3d(velocity_x,velocity_y,velocity_z, ghosts_velo)
    !!print *, "fortran resolution time : ", MPI_WTime() - start

  end subroutine solve_poisson_3d

  !> Solve
  !! \f{eqnarray*} \Delta \psi &=& - \omega \\ velocity = \nabla\times\psi \f}
  !! velocity being a 2D complex vector field and omega a 2D complex scalar field.
  subroutine solve_poisson_2d_c(omega,velocity_x,velocity_y)
    complex(pk),dimension(:,:),intent(in):: omega
    complex(pk),dimension(size(omega,1),size(omega,2)),intent(out) :: velocity_x,velocity_y
    !f2py intent(in,out) :: velocity_x,velocity_y

    call c2c_2d(omega,velocity_x,velocity_y)

  end subroutine solve_poisson_2d_c

  !> Solve
  !!  \f{eqnarray*} \Delta \psi &=& - \omega \\ velocity = \nabla\times\psi \f}
  !! velocity and omega being 3D complex vector fields.
  subroutine solve_poisson_3d_c(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_Z)
    complex(pk),dimension(:,:,:),intent(in):: omega_x,omega_y,omega_z
    complex(pk),dimension(size(omega_x,1),size(omega_y,2),size(omega_z,3)),intent(out) :: velocity_x,velocity_y,velocity_z
    !f2py intent(in,out) :: velocity_x,velocity_y,velocity_z

    call c2c_3d(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z)

  end subroutine solve_poisson_3d_c

  !> Solve
  !! \f{eqnarray*} \omega &=& \nabla \times v \\ \frac{\partial \omega}{\partial t} &=& \nu \Delta \omega \f}
  !! velocity and omega being 3D vector fields.
  subroutine solve_curl_diffusion_3d(nudt,velocity_x,velocity_y,velocity_z,omega_x,omega_y,omega_z, ghosts_velo, ghosts_vort)
    real(pk), intent(in) :: nudt
    real(pk),dimension(:,:,:),intent(in):: velocity_x,velocity_y,velocity_z
    real(pk),dimension(size(velocity_x,1),size(velocity_y,2),size(velocity_z,3)),intent(out) :: omega_x,omega_y,omega_z
    integer, dimension(3), intent(in) :: ghosts_vort, ghosts_velo
    !f2py intent(in,out) :: omega_x,omega_y,omega_z

    call r2c_3d(velocity_x,velocity_y,velocity_z, ghosts_velo)

    call filter_curl_diffusion_3d(nudt)

    call c2r_3d(omega_x,omega_y,omega_z, ghosts_vort)

  end subroutine solve_curl_diffusion_3d

  !> Solve
  !! \f{eqnarray*} \frac{\partial \omega}{\partial t} &=& \nu \Delta \omega \f}
  !! omega being 3D vector field.
  subroutine solve_diffusion_3d(nudt,omega_x,omega_y,omega_z, ghosts)
    real(pk), intent(in) :: nudt
    real(pk),dimension(:,:,:),intent(inout):: omega_x,omega_y,omega_z
    integer, dimension(3), intent(in) :: ghosts
    !f2py intent(in,out) :: omega_x,omega_y,omega_z

    call r2c_3d(omega_x,omega_y,omega_z, ghosts)

    call filter_diffusion_3d(nudt)

    call c2r_3d(omega_x,omega_y,omega_z, ghosts)

  end subroutine solve_diffusion_3d

  !> Perform solenoidal projection to ensure divergence free vorticity field
  !! \f{eqnarray*} \omega ' &=& \omega - \nabla\pi \f}
  !! omega being a 3D vector field.
  subroutine projection_om_3d(omega_x,omega_y,omega_z, ghosts)
    real(pk),dimension(:,:,:),intent(inout):: omega_x,omega_y,omega_z
   integer, dimension(3), intent(in) :: ghosts
    !f2py intent(in,out) :: omega_x,omega_y,omega_z

    call r2c_3d(omega_x,omega_y,omega_z, ghosts)

    call filter_projection_om_3d()

    call c2r_3d(omega_x,omega_y,omega_z, ghosts)

  end subroutine projection_om_3d

  !> Projects vorticity values from fine to coarse grid :
  !! @param[in] dxf, dyf, dzf: grid filter size = domainLength/(CoarseRes-1)
  !! in the following, omega is the 3D vorticity vector field.
  subroutine multires_om_3d(dxf, dyf, dzf, omega_x,omega_y,omega_z, ghosts)
    real(pk), intent(in) :: dxf, dyf, dzf
    real(pk),dimension(:,:,:),intent(inout):: omega_x,omega_y,omega_z
    integer, dimension(3), intent(in) :: ghosts

    !f2py intent(in,out) :: omega_x,omega_y,omega_z

    call r2c_3d(omega_x,omega_y,omega_z, ghosts)

    call filter_multires_om_3d(dxf, dyf, dzf)

    call c2r_3d(omega_x,omega_y,omega_z, ghosts)

  end subroutine multires_om_3d

  !> Compute the pressure from the velocity field, solving a Poisson equation.
  !! \f{eqnarray*} \Delta p ' &=& rhs \f}
  !! with rhs depending on the first derivatives of the velocity field
  !! @param[in, out] pressure
  !! in the following, pressure is used as inout parameter. It must contains the rhs of poisson equation.
  subroutine pressure_3d(pressure, ghosts)
    integer, dimension(3), intent(in) :: ghosts
    real(pk),dimension(:,:,:),intent(inout):: pressure
    !f2py intent(in,out) :: pressure

    call r2c_scalar_3d(pressure, ghosts)

    call filter_pressure_3d()

    call c2r_scalar_3d(pressure, ghosts)

  end subroutine pressure_3d

  !> Solve
  !! \f{eqnarray*} \omega &=& \nabla \times v
  !! velocity and omega being 3D vector fields.
  subroutine solve_curl_3d(velocity_x,velocity_y,velocity_z,omega_x,omega_y,omega_z, ghosts_velo, ghosts_vort)
    real(pk),dimension(:,:,:),intent(in):: velocity_x,velocity_y,velocity_z
    real(pk),dimension(size(velocity_x,1),size(velocity_y,2),size(velocity_z,3)),intent(out) :: omega_x,omega_y,omega_z
    integer, dimension(3), intent(in) :: ghosts_velo, ghosts_vort
    !f2py intent(in,out) :: omega_x,omega_y,omega_z

    call r2c_3d(velocity_x,velocity_y,velocity_z, ghosts_velo)

    call filter_curl_3d()

    call c2r_3d(omega_x,omega_y,omega_z, ghosts_vort)

  end subroutine solve_curl_3d


  !> Solve
  !! \f{eqnarray*} \omega &=& \nabla \times v
  !! velocity and omega being 2D vector and scalar fields.
  subroutine solve_curl_2d(velocity_x,velocity_y, omega_z, ghosts_velo, ghosts_vort)
    real(pk), dimension(:,:), intent(in):: velocity_x,velocity_y
    real(pk), dimension(size(velocity_x,1), size(velocity_x,2)), intent(out) :: omega_z
    integer, dimension(2), intent(in) :: ghosts_velo, ghosts_vort
    !f2py intent(in,out) :: omega_z

    call r2c_2d(velocity_x,velocity_y, ghosts_velo)

    call filter_curl_2d()

    call c2r_scalar_2d(omega_z, ghosts_vort)

  end subroutine solve_curl_2d

  !> Compute spectrum of a scalar field
  !! @param[in] field
  !! @param[out] spectrum
  subroutine spectrum_3d(field, spectrum, wavelengths, ghosts, length)
    real(pk),dimension(:,:,:),intent(in):: field
    integer, dimension(3), intent(in) :: ghosts
    real(pk),dimension(:), intent(inout) :: spectrum
    real(pk),dimension(:), intent(inout) :: wavelengths
    real(pk),intent(in) :: length
    !f2py intent(in) :: field
    !f2py intent(inout) :: spectrum
    !f2py intent(inout) :: wavelengths

    call r2c_3d_scal(field, ghosts)

    call filter_spectrum_3d(spectrum, wavelengths, length)

  end subroutine spectrum_3d

end module fftw2py
