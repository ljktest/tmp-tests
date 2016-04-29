!> Fast Fourier Transform routines (Fortran, based on fftw) to solve 3d Poisson and diffusion problems.
!!
!! This module provides :
!! \li 1 - fftw routines for the "complex to complex" case : solves the problem for
!! complex input/output. The names of these routines contain "c2c".
!! \li 2 - fftw routines for the "real to complex" case : solves the problem for
!!  input fields which are real. The names of these routines contain "r2c".
!! \li 3 - fftw routines for the "real to complex" case : solves the problem for
!! input fields which are real and using the "many" interface of the fftw.
!! It means that transforms are applied to the 3 input fields at the same time.
!! Names of these routines contain "many".
!!
!! Obviously, all the above cases should lead to the same results. By default
!! case 2 must be chosen (if input is real). Case 1 and 3 are more or less
!! dedicated to tests and validation.
module fft3d

  use client_data
  use mpi
  implicit none
  include 'fftw3-mpi.f03'

  private

  public :: init_r2c_3d,init_c2c_3d,init_r2c_scalar_3d, r2c_3d,r2c_scalar_3d,c2c_3d,c2r_3d,c2r_scalar_3d,cleanFFTW_3d,&
       getParamatersTopologyFFTW3d,filter_poisson_3d,filter_curl_diffusion_3d, &
       init_r2c_3d_many, r2c_3d_many, c2r_3d_many, filter_diffusion_3d_many,&
       filter_poisson_3d_many, filter_diffusion_3d, filter_curl_3d, filter_projection_om_3d,&
       filter_multires_om_3d, filter_pressure_3d, r2c_3d_scal, filter_spectrum_3d

  !> plan for fftw "c2c" forward or r2c transform
  type(C_PTR) :: plan_forward1, plan_forward2, plan_forward3
  !> plan for fftw "c2c" backward or c2r transform
  type(C_PTR) :: plan_backward1, plan_backward2, plan_backward3
  !> memory buffer for fftw (input and output buffer will point to this location)
  type(C_PTR) :: cbuffer1
  !> second memory buffer for fftw
  type(C_PTR) :: cbuffer2
  !> third memory buffer for fftw
  type(C_PTR) :: cbuffer3
  !! Note Franck : check if local declarations of datain/out works and improve perfs.
  !> Field (complex values) for fftw input
  complex(C_DOUBLE_COMPLEX), pointer :: datain1(:,:,:)=>NULL(), datain2(:,:,:)=>NULL(), datain3(:,:,:)=>NULL()
  !> Field (real values) for fftw input (these are only pointers to the cbuffers)
  real(C_DOUBLE), pointer :: rdatain1(:,:,:)=>NULL() ,rdatain2(:,:,:)=>NULL() ,rdatain3(:,:,:)=>NULL()
  !> Field (real values) for fftw input in the fftw-many case
  real(C_DOUBLE), pointer :: rdatain_many(:,:,:,:)=>NULL()
  !> Field (complex values) for fftw (forward) output
  complex(C_DOUBLE_COMPLEX), pointer :: dataout1(:,:,:)=>NULL() ,dataout2(:,:,:)=>NULL() ,dataout3(:,:,:)=>NULL()
  !> Field (complex values) for fftw (forward) output in the fftw-many case
  complex(C_DOUBLE_COMPLEX), pointer :: dataout_many(:,:,:,:)=>NULL()
  !> GLOBAL number of points in each direction on which fft is applied (--> corresponds to "real" resolution - 1)
  integer(C_INTPTR_T),pointer :: fft_resolution(:)=>NULL()
  !> LOCAL number of points for fft
  integer(c_INTPTR_T),dimension(3) :: local_resolution
  !> Offset in the direction of distribution
  integer(c_INTPTR_T),dimension(3) :: local_offset
  !> wave numbers for fft in x direction
  real(C_DOUBLE), pointer :: kx(:)
  !> wave numbers for fft in y direction
  real(C_DOUBLE), pointer :: ky(:)
  !> wave numbers for fft in z direction
  real(C_DOUBLE), pointer :: kz(:)
  !> log file for fftw
  character(len=20),parameter :: filename ="hysopfftw.log"
  !> normalization factor
  real(C_DOUBLE) :: normFFT
  !> true if we use fftw-many routines
  logical :: manycase
  !> true if all the allocation stuff for global variables has been done.
  logical :: is3DUpToDate = .false.

contains
  !========================================================================
  !   Complex to complex transforms
  !========================================================================

  !> Initialisation of the fftw context for complex to complex transforms (forward and backward)
  !! @param[in] resolution global domain resolution
  !! @param[in] lengths width of each side of the domain
  subroutine init_c2c_3d(resolution,lengths)

    integer, dimension(3), intent(in) :: resolution
    real(mk),dimension(3), intent(in) :: lengths

    !! Size of the local memory required for fftw (cbuffer)
    integer(C_INTPTR_T) :: alloc_local

    if(is3DUpToDate) return

    ! init fftw mpi context
    call fftw_mpi_init()

    if(rank==0) open(unit=21,file=filename,form="formatted")

    ! set fft resolution
    allocate(fft_resolution(3))
    fft_resolution(:) = resolution(:)-1

    ! compute "optimal" size (according to fftw) for local data (warning : dimension reversal)
    alloc_local = fftw_mpi_local_size_3d_transposed(fft_resolution(c_Z),fft_resolution(c_Y),fft_resolution(c_X),main_comm,&
         local_resolution(c_Z),local_offset(c_Z),local_resolution(c_Y),local_offset(c_Y));

    ! Set a default value for c_X components.
    local_offset(c_X) = 0
    local_resolution(c_X) = fft_resolution(c_X)

    ! allocate local buffer (used to save datain/dataout ==> in-place transform!!)
    cbuffer1 = fftw_alloc_complex(alloc_local)
    ! link datain and dataout to cbuffer, setting the right dimensions for each
    call c_f_pointer(cbuffer1, datain1, [fft_resolution(c_X),fft_resolution(c_Y),local_resolution(c_Z)])
    call c_f_pointer(cbuffer1, dataout1, [fft_resolution(c_X),fft_resolution(c_Z),local_resolution(c_Y)])

    ! second buffer used for backward transform. Used to copy dataout into dataout2 (input for backward transform and filter)
    ! and to save (in-place) the transform of the second component of the velocity
    cbuffer2 = fftw_alloc_complex(alloc_local)
    call c_f_pointer(cbuffer2, datain2, [fft_resolution(c_X),fft_resolution(c_Y),local_resolution(c_Z)])
    call c_f_pointer(cbuffer2, dataout2, [fft_resolution(c_X),fft_resolution(c_Z),local_resolution(c_Y)])

    ! second buffer used for backward transform. Used to copy dataout into dataout2 (input for backward transform and filter)
    ! and to save (in-place) the transform of the second component of the velocity
    cbuffer3 = fftw_alloc_complex(alloc_local)
    call c_f_pointer(cbuffer3, datain3, [fft_resolution(c_X),fft_resolution(c_Y),local_resolution(c_Z)])
    call c_f_pointer(cbuffer3, dataout3, [fft_resolution(c_X),fft_resolution(c_Z),local_resolution(c_Y)])

    !   create MPI plan for in-place forward/backward DFT (note dimension reversal)
    plan_forward1 = fftw_mpi_plan_dft_3d(fft_resolution(c_Z), fft_resolution(c_Y), fft_resolution(c_X),datain1,dataout1,&
         main_comm,FFTW_FORWARD,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward1 = fftw_mpi_plan_dft_3d(fft_resolution(c_Z),fft_resolution(c_Y),fft_resolution(c_X),dataout1,datain1,&
         main_comm,FFTW_BACKWARD,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))
    plan_forward2 = fftw_mpi_plan_dft_3d(fft_resolution(c_Z), fft_resolution(c_Y), fft_resolution(c_X),datain2,dataout2,&
         main_comm,FFTW_FORWARD,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward2 = fftw_mpi_plan_dft_3d(fft_resolution(c_Z),fft_resolution(c_Y),fft_resolution(c_X),dataout2,datain2,&
         main_comm,FFTW_BACKWARD,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))
    plan_forward3 = fftw_mpi_plan_dft_3d(fft_resolution(c_Z), fft_resolution(c_Y), fft_resolution(c_X),datain3,dataout3,&
         main_comm,FFTW_FORWARD,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward3 = fftw_mpi_plan_dft_3d(fft_resolution(c_Z),fft_resolution(c_Y),fft_resolution(c_X),dataout3,datain3,&
         main_comm,FFTW_BACKWARD,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))

    call computeKx(lengths(c_X))
    call computeKy(lengths(c_Y))
    call computeKz(lengths(c_Z))

    !! call fft3d_diagnostics(alloc_local)

    normFFT = 1./(fft_resolution(c_X)*fft_resolution(c_Y)*fft_resolution(c_Z))
    manycase = .false.

    is3DUpToDate = .true.

  end subroutine init_c2c_3d

  !> Solve poisson problem for complex input and output vector fields
  !!  @param[in] omega_x 3d scalar field, x-component of the input vector field
  !!  @param[in] omega_y 3d scalar field, y-component of the input vector field
  !!  @param[in] omega_z 3d scalar field, z-component of the input vector field
  !!  @param[in] velocity_x 3d scalar field, x-component of the output vector field
  !!  @param[in] velocity_y 3d scalar field, y-component of the output vector field
  !!  @param[in] velocity_z 3d scalar field, z-component of the output vector field
  subroutine c2c_3d(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z)
    complex(mk),dimension(:,:,:) :: velocity_x,velocity_y,velocity_z
    complex(mk),dimension(:,:,:),intent(in) :: omega_x,omega_y,omega_z

    integer(C_INTPTR_T) :: i,j,k
    ! Copy input data into the fftw buffer
    do k = 1, local_resolution(c_Z)
       do j = 1, fft_resolution(c_Y)
          do i = 1, fft_resolution(c_X)
             datain1(i,j,k) = omega_x(i,j,k)
             datain2(i,j,k) = omega_y(i,j,k)
             datain3(i,j,k) = omega_z(i,j,k)
          end do
       end do
    end do
    ! compute transform (as many times as desired)
    call fftw_mpi_execute_dft(plan_forward1, datain1, dataout1)
    call fftw_mpi_execute_dft(plan_forward2, datain2, dataout2)
    call fftw_mpi_execute_dft(plan_forward3, datain3, dataout3)

    ! apply poisson filter
    call filter_poisson_3d()

    ! inverse transform to retrieve velocity
    call fftw_mpi_execute_dft(plan_backward1, dataout1,datain1)
    call fftw_mpi_execute_dft(plan_backward2,dataout2,datain2)
    call fftw_mpi_execute_dft(plan_backward3,dataout3,datain3)
    do k =1, local_resolution(c_Z)
       do j = 1, fft_resolution(c_Y)
          do i = 1, fft_resolution(c_X)
             velocity_x(i,j,k) = datain1(i,j,k)*normFFT
             velocity_y(i,j,k) = datain2(i,j,k)*normFFT
             velocity_z(i,j,k) = datain3(i,j,k)*normFFT
          end do
       end do
    end do

  end subroutine c2c_3d

  !========================================================================
  !  Real to complex transforms
  !========================================================================

  !> Initialisation of the fftw context for real to complex transforms (forward and backward)
  !! @param[in] resolution global domain resolution
  !! @param[in] lengths width of each side of the domain
  subroutine init_r2c_3d(resolution,lengths)

    integer, dimension(3), intent(in) :: resolution
    real(mk),dimension(3), intent(in) :: lengths
    !! Size of the local memory required for fftw (cbuffer)
    integer(C_INTPTR_T) :: alloc_local,halfLength

    if(is3DUpToDate) return

    ! init fftw mpi context
    call fftw_mpi_init()

    if(rank==0) open(unit=21,file=filename,form="formatted")
    allocate(fft_resolution(3))
    fft_resolution(:) = resolution(:)-1
    halfLength = fft_resolution(c_X)/2+1

    ! compute "optimal" size (according to fftw) for local data (warning : dimension reversal)
    alloc_local = fftw_mpi_local_size_3d_transposed(fft_resolution(c_Z),fft_resolution(c_Y),halfLength,&
         main_comm,local_resolution(c_Z),local_offset(c_Z),local_resolution(c_Y),local_offset(c_Y));

    ! init c_X part. This is required to compute kx with the same function in 2d and 3d cases.
    local_offset(c_X) = 0
    local_resolution(c_X) = halfLength

    ! allocate local buffer (used to save datain/dataout ==> in-place transform!!)
    cbuffer1 = fftw_alloc_complex(alloc_local)
    cbuffer2 = fftw_alloc_complex(alloc_local)
    cbuffer3 = fftw_alloc_complex(alloc_local)

    ! link rdatain and dataout to cbuffer, setting the right dimensions for each
    call c_f_pointer(cbuffer1, rdatain1, [2*halfLength,fft_resolution(c_Y),local_resolution(c_Z)])
    call c_f_pointer(cbuffer1, dataout1, [halfLength, fft_resolution(c_Z), local_resolution(c_Y)])
    call c_f_pointer(cbuffer2, rdatain2, [2*halfLength,fft_resolution(c_Y),local_resolution(c_Z)])
    call c_f_pointer(cbuffer2, dataout2, [halfLength, fft_resolution(c_Z), local_resolution(c_Y)])
    call c_f_pointer(cbuffer3, rdatain3, [2*halfLength,fft_resolution(c_Y),local_resolution(c_Z)])
    call c_f_pointer(cbuffer3, dataout3, [halfLength, fft_resolution(c_Z), local_resolution(c_Y)])

    rdatain1 = 0.0
    rdatain2 = 0.0
    rdatain3 = 0.0

    !   create MPI plans for in-place forward/backward DFT (note dimension reversal)
   plan_forward1 = fftw_mpi_plan_dft_r2c_3d(fft_resolution(c_Z),fft_resolution(c_Y), fft_resolution(c_X), rdatain1, dataout1, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward1 = fftw_mpi_plan_dft_c2r_3d(fft_resolution(c_Z),fft_resolution(c_Y), fft_resolution(c_X), dataout1, rdatain1, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))
    plan_forward2 = fftw_mpi_plan_dft_r2c_3d(fft_resolution(c_Z),fft_resolution(c_Y), fft_resolution(c_X), rdatain2, dataout2, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward2 = fftw_mpi_plan_dft_c2r_3d(fft_resolution(c_Z),fft_resolution(c_Y), fft_resolution(c_X), dataout2, rdatain2, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))
    plan_forward3 = fftw_mpi_plan_dft_r2c_3d(fft_resolution(c_Z),fft_resolution(c_Y), fft_resolution(c_X), rdatain3, dataout3, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward3 = fftw_mpi_plan_dft_c2r_3d(fft_resolution(c_Z),fft_resolution(c_Y), fft_resolution(c_X), dataout3, rdatain3, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))

    call computeKx(lengths(c_X))
    call computeKy(lengths(c_Y))
    call computeKz(lengths(c_Z))

    normFFT = 1./(fft_resolution(c_X)*fft_resolution(c_Y)*fft_resolution(c_Z))
    !! call fft3d_diagnostics(alloc_local)
    manycase = .false.
    is3DUpToDate = .true.

  end subroutine init_r2c_3d

  !> Initialisation of the fftw context for real to complex transforms (forward and backward)
  !! @param[in] resolution global domain resolution
  !! @param[in] lengths width of each side of the domain
  subroutine init_r2c_scalar_3d(resolution,lengths)

    integer, dimension(3), intent(in) :: resolution
    real(mk),dimension(3), intent(in) :: lengths
    !! Size of the local memory required for fftw (cbuffer)
    integer(C_INTPTR_T) :: alloc_local,halfLength

    if(is3DUpToDate) return

    ! init fftw mpi context
    call fftw_mpi_init()

    if(rank==0) open(unit=21,file=filename,form="formatted")
    allocate(fft_resolution(3))
    fft_resolution(:) = resolution(:)-1
    halfLength = fft_resolution(c_X)/2+1

    ! compute "optimal" size (according to fftw) for local data (warning : dimension reversal)
    alloc_local = fftw_mpi_local_size_3d_transposed(fft_resolution(c_Z),fft_resolution(c_Y),halfLength,&
         main_comm,local_resolution(c_Z),local_offset(c_Z),local_resolution(c_Y),local_offset(c_Y));

    ! init c_X part. This is required to compute kx with the same function in 2d and 3d cases.
    local_offset(c_X) = 0
    local_resolution(c_X) = halfLength

    ! allocate local buffer (used to save datain/dataout ==> in-place transform!!)
    cbuffer1 = fftw_alloc_complex(alloc_local)

    ! link rdatain and dataout to cbuffer, setting the right dimensions for each
    call c_f_pointer(cbuffer1, rdatain1, [2*halfLength,fft_resolution(c_Y),local_resolution(c_Z)])
    call c_f_pointer(cbuffer1, dataout1, [halfLength, fft_resolution(c_Z), local_resolution(c_Y)])

    rdatain1 = 0.0

    !   create MPI plans for in-place forward/backward DFT (note dimension reversal)
    plan_forward1 = fftw_mpi_plan_dft_r2c_3d(fft_resolution(c_Z),fft_resolution(c_Y), fft_resolution(c_X), rdatain1, dataout1, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward1 = fftw_mpi_plan_dft_c2r_3d(fft_resolution(c_Z),fft_resolution(c_Y), fft_resolution(c_X), dataout1, rdatain1, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))
    
    call computeKx(lengths(c_X))
    call computeKy(lengths(c_Y))
    call computeKz(lengths(c_Z))

    normFFT = 1./(fft_resolution(c_X)*fft_resolution(c_Y)*fft_resolution(c_Z))
    !! call fft3d_diagnostics(alloc_local)
    manycase = .false.
    is3DUpToDate = .true.

  end subroutine init_r2c_scalar_3d
  
  !> forward transform - The result is saved in local buffers
  !!  @param[in] omega_x 3d scalar field, x-component of the input vector field
  !!  @param[in] omega_y 3d scalar field, y-component of the input vector field
  !!  @param[in] omega_z 3d scalar field, z-component of the input vector field
  !!  @param[in] ghosts, number of points in the ghost layer of input fields.
  subroutine r2c_3d(omega_x,omega_y,omega_z, ghosts)

    real(mk),dimension(:,:,:),intent(in) :: omega_x,omega_y,omega_z
    integer, dimension(3), intent(in) :: ghosts
    !real(mk) :: start
    integer(C_INTPTR_T) :: i,j,k, ig, jg, kg

    ! ig, jg, kg are used to take into account
    ! ghost points in input fields

    ! init
    do k =1, local_resolution(c_Z)
       kg = k + ghosts(c_Z)
       do j = 1, fft_resolution(c_Y)
          jg = j + ghosts(c_Y)
          do i = 1, fft_resolution(c_X)
             ig = i + ghosts(c_X)
             rdatain1(i,j,k) = omega_x(ig,jg,kg)
             rdatain2(i,j,k) = omega_y(ig,jg,kg)
             rdatain3(i,j,k) = omega_z(ig,jg,kg)
          end do
       end do
    end do

    ! compute transforms for each component
    !start = MPI_WTIME()
    call fftw_mpi_execute_dft_r2c(plan_forward1, rdatain1, dataout1)
    call fftw_mpi_execute_dft_r2c(plan_forward2, rdatain2, dataout2)
    call fftw_mpi_execute_dft_r2c(plan_forward3, rdatain3, dataout3)
    !!print *, "r2c time = ", MPI_WTIME() - start

  end subroutine r2c_3d

  !> Backward fftw transform
  !!  @param[in,out] velocity_x 3d scalar field, x-component of the output vector field
  !!  @param[in,out] velocity_y 3d scalar field, y-component of the output vector field
  !!  @param[in,out] velocity_z 3d scalar field, z-component of the output vector field
  !!  @param[in] ghosts, number of points in the ghost layer of in/out velocity field.
  subroutine c2r_3d(velocity_x,velocity_y,velocity_z, ghosts)
    real(mk),dimension(:,:,:),intent(inout) :: velocity_x,velocity_y,velocity_z
    integer, dimension(3), intent(in) :: ghosts
    real(mk) :: start
    integer(C_INTPTR_T) :: i,j,k, ig, jg, kg
    start = MPI_WTIME()
    call fftw_mpi_execute_dft_c2r(plan_backward1,dataout1,rdatain1)
    call fftw_mpi_execute_dft_c2r(plan_backward2,dataout2,rdatain2)
    call fftw_mpi_execute_dft_c2r(plan_backward3,dataout3,rdatain3)
!!    print *, "c2r time : ", MPI_WTIME() -start
    ! copy back to velocity and normalisation
    do k =1, local_resolution(c_Z)
       kg = k + ghosts(c_Z)
       do j = 1, fft_resolution(c_Y)
          jg = j + ghosts(c_Y)
          do i = 1, fft_resolution(c_X)
             ig = i + ghosts(c_X)
             velocity_x(ig,jg,kg) = rdatain1(i,j,k)*normFFT
             velocity_y(ig,jg,kg) = rdatain2(i,j,k)*normFFT
             velocity_z(ig,jg,kg) = rdatain3(i,j,k)*normFFT
          end do
       end do
    end do

  end subroutine c2r_3d

  !> forward transform - The result is saved in a local buffer
  !!  @param[in] omega 3d scalar field, x-component of the input vector field
  !!  @param[in] ghosts, number of points in the ghost layer of input field.
  subroutine r2c_scalar_3d(scalar, ghosts)

    real(mk),dimension(:,:,:),intent(in) :: scalar
    integer, dimension(3), intent(in) :: ghosts
    real(mk) :: start
    integer(C_INTPTR_T) :: i,j,k, ig, jg, kg

    ! ig, jg, kg are used to take into account
    ! ghost points in input fields

    ! init
    do k =1, local_resolution(c_Z)
       kg = k + ghosts(c_Z)
       do j = 1, fft_resolution(c_Y)
          jg = j + ghosts(c_Y)
          do i = 1, fft_resolution(c_X)
             ig = i + ghosts(c_X)
             rdatain1(i,j,k) = scalar(ig,jg,kg)
          end do
       end do
    end do

    ! compute transforms for each component
    start = MPI_WTIME()
    call fftw_mpi_execute_dft_r2c(plan_forward1, rdatain1, dataout1)
    !!print *, "r2c time = ", MPI_WTIME() - start

  end subroutine r2c_scalar_3d

  !> Backward fftw transform
  !!  @param[in,out] scalar 3d scalar field
  !!  @param[in] ghosts, number of points in the ghost layer of in/out scalar field.
  subroutine c2r_scalar_3d(scalar, ghosts)
    real(mk),dimension(:,:,:),intent(inout) :: scalar
    integer, dimension(3), intent(in) :: ghosts
    real(mk) :: start
    integer(C_INTPTR_T) :: i,j,k, ig, jg, kg
    start = MPI_WTIME()
    call fftw_mpi_execute_dft_c2r(plan_backward1,dataout1,rdatain1)
!!    print *, "c2r time : ", MPI_WTIME() -start
    ! copy back to velocity and normalisation
    do k =1, local_resolution(c_Z)
       kg = k + ghosts(c_Z)
       do j = 1, fft_resolution(c_Y)
          jg = j + ghosts(c_Y)
          do i = 1, fft_resolution(c_X)
             ig = i + ghosts(c_X)
             scalar(ig,jg,kg) = rdatain1(i,j,k)*normFFT
          end do
       end do
    end do

  end subroutine c2r_scalar_3d

  !========================================================================
  !  Real to complex transforms based on "many" fftw routines
  !========================================================================

  !> Initialisation of the fftw context for real to complex transforms (forward and backward)
  !! @param[in] resolution global domain resolution
  !! @param[in] lengths width of each side of the domain
  subroutine init_r2c_3d_many(resolution,lengths)

    integer, dimension(3), intent(in) :: resolution
    real(mk),dimension(3), intent(in) :: lengths
    !! Size of the local memory required for fftw (cbuffer)
    integer(C_INTPTR_T) :: alloc_local,halfLength,howmany, blocksize
    integer(C_INTPTR_T),dimension(3) :: n

    ! init fftw mpi context
    call fftw_mpi_init()
    blocksize = FFTW_MPI_DEFAULT_BLOCK
    if(rank==0) open(unit=21,file=filename,form="formatted")
    allocate(fft_resolution(3))
    fft_resolution(:) = resolution(:)-1
    halfLength = fft_resolution(c_X)/2+1
    n(1) = fft_resolution(c_Z)
    n(2) = fft_resolution(c_Y)
    n(3) = halfLength
    howmany = 3
    ! compute "optimal" size (according to fftw) for local data (warning : dimension reversal)
    alloc_local = fftw_mpi_local_size_many_transposed(3,n,howmany,blocksize,blocksize,&
         main_comm,local_resolution(c_Z),local_offset(c_Z),local_resolution(c_Y),local_offset(c_Y));

    ! init c_X part. This is required to compute kx with the same function in 2d and 3d cases.
    local_offset(c_X) = 0
    local_resolution(c_X) = halfLength

    ! allocate local buffer (used to save datain/dataout ==> in-place transform!!)
    cbuffer1 = fftw_alloc_complex(alloc_local)

    ! link rdatain and dataout to cbuffer, setting the right dimensions for each
    call c_f_pointer(cbuffer1, rdatain_many, [howmany,2*halfLength,fft_resolution(c_Y),local_resolution(c_Z)])
    call c_f_pointer(cbuffer1, dataout_many, [howmany,halfLength, fft_resolution(c_Z), local_resolution(c_Y)])

    !   create MPI plans for in-place forward/backward DFT (note dimension reversal)
    n(3) = fft_resolution(c_X)

    plan_forward1 = fftw_mpi_plan_many_dft_r2c(3,n,howmany,blocksize,blocksize, rdatain_many, dataout_many, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward1 = fftw_mpi_plan_many_dft_c2r(3,n,howmany,blocksize,blocksize, dataout_many, rdatain_many, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))
    call computeKx(lengths(c_X))
    call computeKy(lengths(c_Y))
    call computeKz(lengths(c_Z))

    normFFT = 1./(fft_resolution(c_X)*fft_resolution(c_Y)*fft_resolution(c_Z))
    !! call fft3d_diagnostics(alloc_local,1)
    manycase = .true.

    is3DUpToDate = .true.

  end subroutine init_r2c_3d_many

  !> forward transform - The result is saved in local buffers
  !!  @param[in] omega_x 3d scalar field, x-component of the input vector field
  !!  @param[in] omega_y 3d scalar field, y-component of the input vector field
  !!  @param[in] omega_z 3d scalar field, z-component of the input vector field
  !! @param input data
  subroutine r2c_3d_many(omega_x,omega_y,omega_z)

    real(mk),dimension(:,:,:),intent(in) :: omega_x,omega_y,omega_z
    real(mk) :: start
    integer(C_INTPTR_T) :: i,j,k

    ! init
    do k =1, local_resolution(c_Z)
       do j = 1, fft_resolution(c_Y)
          do i = 1, fft_resolution(c_X)
             rdatain_many(1,i,j,k) = omega_x(i,j,k)
             rdatain_many(2,i,j,k) = omega_y(i,j,k)
             rdatain_many(3,i,j,k) = omega_z(i,j,k)
          end do
       end do
    end do

    ! compute transform (as many times as desired)
    start = MPI_WTIME()
    call fftw_mpi_execute_dft_r2c(plan_forward1, rdatain_many, dataout_many)
!!    print *, "r2c time = ", MPI_WTIME() - start

  end subroutine r2c_3d_many

  !> Backward fftw transform
  !!  @param[in,out] velocity_x 3d scalar field, x-component of the output vector field
  !!  @param[in,out] velocity_y 3d scalar field, y-component of the output vector field
  !!  @param[in,out] velocity_z 3d scalar field, z-component of the output vector field
  subroutine c2r_3d_many(velocity_x,velocity_y,velocity_z)
    real(mk),dimension(:,:,:),intent(inout) :: velocity_x,velocity_y,velocity_z
    real(mk) :: start
    integer(C_INTPTR_T) :: i,j,k

    start = MPI_WTIME()
    call fftw_mpi_execute_dft_c2r(plan_backward1,dataout_many,rdatain_many)
!!    print *, "c2r time : ", MPI_WTIME() -start
    do k =1, local_resolution(c_Z)
       do j = 1, fft_resolution(c_Y)
          do i = 1, fft_resolution(c_X)
             velocity_x(i,j,k) = rdatain_many(1,i,j,k)*normFFT
             velocity_y(i,j,k) = rdatain_many(2,i,j,k)*normFFT
             velocity_z(i,j,k) = rdatain_many(2,i,j,k)*normFFT
          end do
       end do
    end do

  end subroutine c2r_3d_many

  !========================================================================
  ! Common (r2c, c2c) subroutines
  !========================================================================

  !> Computation of frequencies coeff, over the distributed direction in the real/complex case
  !> @param lengths size of the domain
  subroutine computeKx(length)

    real(mk),intent(in) :: length

    !! Local loops indices
    integer(C_INTPTR_T) :: i

    !! Compute filter coefficients
    allocate(kx(local_resolution(c_X)))
    do i = 1, fft_resolution(c_X)/2+1
       kx(i) = 2.*pi/length*(i-1)
    end do
    !! write(*,'(a,i5,a,i5,i5)') '[',rank,'] kx size', size(kx),i
    do i = fft_resolution(c_X)/2+2,local_resolution(c_X)
       kx(i) = 2.*pi/length*(i-fft_resolution(c_X)-1)
    end do
  end subroutine computeKx

  !> Computation of frequencies coeff, over distributed direction(s)
  !> @param lengths size of the domain
  subroutine computeKy(length)
    real(C_DOUBLE), intent(in) :: length

    !! Local loops indices
    integer(C_INTPTR_T) :: i
    allocate(ky(local_resolution(c_Y)))

    !! y frequencies (distributed over proc)
    !! If we deal with positive frequencies only
    if(local_offset(c_Y)+local_resolution(c_Y) <= fft_resolution(c_Y)/2+1 ) then
       do i = 1,local_resolution(c_Y)
          ky(i) =  2.*pi/length*(local_offset(c_Y)+i-1)
       end do
    else
       !! else if we deal with negative frequencies only
       if(local_offset(c_Y)+1 > fft_resolution(c_Y)/2+1 ) then
          do i = 1,local_resolution(c_Y)
             ky(i) =  2.*pi/length*(local_offset(c_Y)+i-1-fft_resolution(c_Y))
          end do
          !! final case : start positive freq, end in negative ones
       else
          do i = local_offset(c_Y)+1, fft_resolution(c_Y)/2+1 !! global index
             ky(i-local_offset(c_Y)) =  2.*pi/length*(i-1)
          end do
          do i = fft_resolution(c_Y)/2+2,local_resolution(c_Y)+local_offset(c_Y)
             ky(i-local_offset(c_Y)) =  2.*pi/length*(i-1-fft_resolution(c_Y))
          end do
       end if
    end if

  end subroutine computeKy

  !> Computation of frequencies coeff, over non-distributed direction(s)
  !> @param length size of the domain
  subroutine computeKz(length)
    real(mk),intent(in) :: length

    !! Local loops indices
    integer(C_INTPTR_T) :: i
    allocate(kz(fft_resolution(c_Z)))
    do i = 1, fft_resolution(c_Z)/2+1
       kz(i) = 2.*pi/length*(i-1)
    end do
    do i = fft_resolution(c_Z)/2+2,fft_resolution(c_Z)
       kz(i) = 2.*pi/length*(i-fft_resolution(c_Z)-1)
    end do

  end subroutine computeKz

  !> Solve Poisson problem in the Fourier space :
  !! \f{eqnarray*} \Delta \psi &=& - \omega \\ v = \nabla\times\psi \f}
  subroutine filter_poisson_3d()

    integer(C_INTPTR_T) :: i,j,k
    complex(C_DOUBLE_COMPLEX) :: coeff
    complex(C_DOUBLE_COMPLEX) :: buffer1,buffer2

    ! Set first coeff (check for "all freq = 0" case)
    if(local_offset(c_Y) == 0) then
       dataout1(1,1,1) = 0.0
       dataout2(1,1,1) = 0.0
       dataout3(1,1,1) = 0.0
    else
       coeff = Icmplx/(ky(1)**2)
       buffer1 = dataout1(1,1,1)
       dataout1(1,1,1) = coeff*ky(1)*dataout3(1,1,1)
       dataout2(1,1,1) = 0.0
       dataout3(1,1,1) = -coeff*ky(1)*buffer1
    endif

    ! !! mind the transpose -> index inversion between y and z
    do i = 2, local_resolution(c_X)
       coeff = Icmplx/(kx(i)**2+ky(1)**2)
       buffer1 = dataout1(i,1,1)
       buffer2 = dataout2(i,1,1)
       dataout1(i,1,1) = coeff*ky(1)*dataout3(i,1,1)
       dataout2(i,1,1) = -coeff*kx(i)*dataout3(i,1,1)
       dataout3(i,1,1) = coeff*(kx(i)*buffer2-ky(1)*buffer1)
    end do

    ! !! mind the transpose -> index inversion between y and z
    do k = 2, fft_resolution(c_Z)
       do i = 1, local_resolution(c_X)
          coeff = Icmplx/(kx(i)**2+ky(1)**2+kz(k)**2)
          buffer1 = dataout1(i,k,1)
          buffer2 = dataout2(i,k,1)
          dataout1(i,k,1) = coeff*(ky(1)*dataout3(i,k,1)-kz(k)*dataout2(i,k,1))
          dataout2(i,k,1) = coeff*(kz(k)*buffer1-kx(i)*dataout3(i,k,1))
          dataout3(i,k,1) = coeff*(kx(i)*buffer2-ky(1)*buffer1)
       end do
    end do

    ! !! mind the transpose -> index inversion between y and z
    do j = 2,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             coeff = Icmplx/(kx(i)**2+ky(j)**2+kz(k)**2)
             buffer1 = dataout1(i,k,j)
             buffer2 = dataout2(i,k,j)
             dataout1(i,k,j) = coeff*(ky(j)*dataout3(i,k,j)-kz(k)*dataout2(i,k,j))
             dataout2(i,k,j) = coeff*(kz(k)*buffer1-kx(i)*dataout3(i,k,j))
             dataout3(i,k,j) = coeff*(kx(i)*buffer2-ky(j)*buffer1)
          end do
       end do
    end do

  end subroutine filter_poisson_3d

  !> Solve diffusion problem in the Fourier space :
  !! \f{eqnarray*} \omega &=& \nabla \times v \\ \frac{\partial \omega}{\partial t} &=& \nu \Delta \omega \f}
  !! @param[in] nudt \f$ \nu\times dt\f$, diffusion coefficient times current time step
  subroutine filter_curl_diffusion_3d(nudt)

    real(C_DOUBLE), intent(in) :: nudt
    integer(C_INTPTR_T) :: i,j,k
    complex(C_DOUBLE_COMPLEX) :: coeff
    complex(C_DOUBLE_COMPLEX) :: buffer1,buffer2

    !! mind the transpose -> index inversion between y and z
    do j = 1,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             coeff = Icmplx/(1. + nudt * (kx(i)**2+ky(j)**2+kz(k)**2))
             buffer1 = dataout1(i,k,j)
             buffer2 = dataout2(i,k,j)
             dataout1(i,k,j) = coeff*(ky(j)*dataout3(i,k,j)-kz(k)*dataout2(i,k,j))
             dataout2(i,k,j) = coeff*(kz(k)*buffer1-kx(i)*dataout3(i,k,j))
             dataout3(i,k,j) = coeff*(kx(i)*buffer2-ky(j)*buffer1)
          end do
       end do
    end do

  end subroutine filter_curl_diffusion_3d

  !> Solve diffusion problem in the Fourier space :
  !! \f{eqnarray*} \frac{\partial \omega}{\partial t} &=& \nu \Delta \omega \f}
  !! @param[in] nudt \f$ \nu\times dt\f$, diffusion coefficient times current time step
  subroutine filter_diffusion_3d(nudt)

    real(C_DOUBLE), intent(in) :: nudt
    integer(C_INTPTR_T) :: i,j,k
    complex(C_DOUBLE_COMPLEX) :: coeff

    !! mind the transpose -> index inversion between y and z
    do j = 1,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             coeff = 1./(1. + nudt * (kx(i)**2+ky(j)**2+kz(k)**2))
             dataout1(i,k,j) = coeff*dataout1(i,k,j)
             dataout2(i,k,j) = coeff*dataout2(i,k,j)
             dataout3(i,k,j) = coeff*dataout3(i,k,j)
          end do
       end do
    end do

  end subroutine filter_diffusion_3d

  !> Solve curl problem in the Fourier space :
  !! \f{eqnarray*} \omega &=& \nabla \times v
  subroutine filter_curl_3d()

    integer(C_INTPTR_T) :: i,j,k
    complex(C_DOUBLE_COMPLEX) :: coeff
    complex(C_DOUBLE_COMPLEX) :: buffer1,buffer2

    !! mind the transpose -> index inversion between y and z
    do j = 1,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             coeff = Icmplx
             buffer1 = dataout1(i,k,j)
             buffer2 = dataout2(i,k,j)
             dataout1(i,k,j) = coeff*(ky(j)*dataout3(i,k,j)-kz(k)*dataout2(i,k,j))
             dataout2(i,k,j) = coeff*(kz(k)*buffer1-kx(i)*dataout3(i,k,j))
             dataout3(i,k,j) = coeff*(kx(i)*buffer2-ky(j)*buffer1)
          end do
       end do
    end do

  end subroutine filter_curl_3d

  !> Perform solenoidal projection to ensure divergence free vorticity field
  !! \f{eqnarray*} \omega ' &=& \omega - \nabla\pi \f}
  subroutine filter_projection_om_3d()

    integer(C_INTPTR_T) :: i,j,k
    complex(C_DOUBLE_COMPLEX) :: coeff
    complex(C_DOUBLE_COMPLEX) :: buffer1,buffer2,buffer3

    ! Set first coeff (check for "all freq = 0" case)
    if(local_offset(c_Y) == 0) then
       dataout1(1,1,1) = dataout1(1,1,1)
       dataout2(1,1,1) = dataout2(1,1,1)
       dataout3(1,1,1) = dataout3(1,1,1)
    else
       coeff = 1./(ky(1)**2)
       buffer2 = dataout2(1,1,1)
       dataout1(1,1,1) = dataout1(1,1,1)
       dataout2(1,1,1) = dataout2(1,1,1) - coeff*ky(1)*(ky(1)*buffer2)
       dataout3(1,1,1) = dataout3(1,1,1)
    endif

    ! !! mind the transpose -> index inversion between y and z
    do i = 2, local_resolution(c_X)
       coeff = 1./(kx(i)**2+ky(1)**2+kz(1)**2)
       buffer1 = dataout1(i,1,1)
       buffer2 = dataout2(i,1,1)
       buffer3 = dataout3(i,1,1)
       dataout1(i,1,1) = dataout1(i,1,1) - coeff*kx(i)*(kx(i)*buffer1+ky(1)*buffer2+kz(1)*buffer3)
       dataout2(i,1,1) = dataout2(i,1,1) - coeff*ky(1)*(kx(i)*buffer1+ky(1)*buffer2+kz(1)*buffer3)
       dataout3(i,1,1) = dataout3(i,1,1) - coeff*kz(1)*(kx(i)*buffer1+ky(1)*buffer2+kz(1)*buffer3)
    end do

    ! !! mind the transpose -> index inversion between y and z
    do k = 2, fft_resolution(c_Z)
       do i = 1, local_resolution(c_X)
          coeff = 1./(kx(i)**2+ky(1)**2+kz(k)**2)
          buffer1 = dataout1(i,k,1)
          buffer2 = dataout2(i,k,1)
          buffer3 = dataout3(i,k,1)
          dataout1(i,k,1) = dataout1(i,k,1) - coeff*kx(i)*(kx(i)*buffer1+ky(1)*buffer2+kz(k)*buffer3)
          dataout2(i,k,1) = dataout2(i,k,1) - coeff*ky(1)*(kx(i)*buffer1+ky(1)*buffer2+kz(k)*buffer3)
          dataout3(i,k,1) = dataout3(i,k,1) - coeff*kz(k)*(kx(i)*buffer1+ky(1)*buffer2+kz(k)*buffer3)
       end do
    end do

    ! !! mind the transpose -> index inversion between y and z
    do j = 2,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             coeff = 1./(kx(i)**2+ky(j)**2+kz(k)**2)
             buffer1 = dataout1(i,k,j)
             buffer2 = dataout2(i,k,j)
             buffer3 = dataout3(i,k,j)
             dataout1(i,k,j) = dataout1(i,k,j) - coeff*kx(i)*(kx(i)*buffer1+ky(j)*buffer2+kz(k)*buffer3)
             dataout2(i,k,j) = dataout2(i,k,j) - coeff*ky(j)*(kx(i)*buffer1+ky(j)*buffer2+kz(k)*buffer3)
             dataout3(i,k,j) = dataout3(i,k,j) - coeff*kz(k)*(kx(i)*buffer1+ky(j)*buffer2+kz(k)*buffer3)
          end do
       end do
    end do

  end subroutine filter_projection_om_3d

  !> Projects vorticity values from fine to coarse grid :
  !> the smallest modes of vorticity are nullified
  !! @param[in] dxf, dyf, dzf: grid filter size = domainLength/(CoarseRes-1)
  subroutine filter_multires_om_3d(dxf, dyf, dzf)

    real(C_DOUBLE), intent(in) :: dxf, dyf, dzf
    integer(C_INTPTR_T) :: i,j,k
    real(C_DOUBLE) :: kxc, kyc, kzc

    kxc = pi / dxf
    kyc = pi / dyf
    kzc = pi / dzf

    !! mind the transpose -> index inversion between y and z
    do j = 2,local_resolution(c_Y)
       do k = 2, fft_resolution(c_Z)
          do i = 2, local_resolution(c_X)
             if ((abs(kx(i))>kxc) .and. (abs(ky(j))>kyc) .and. (abs(kz(k))>kzc)) then
                dataout1(i,k,j) = 0.
                dataout2(i,k,j) = 0.
                dataout3(i,k,j) = 0.
             endif
          end do
       end do
    end do

  end subroutine filter_multires_om_3d

  !> Solve the Poisson problem allowing to recover
  !! pressure from velocity in the Fourier space
  subroutine filter_pressure_3d()
    integer(C_INTPTR_T) :: i,j,k
    complex(C_DOUBLE_COMPLEX) :: coeff

    ! Set first coeff (check for "all freq = 0" case)
    if(local_offset(c_Y) == 0) then
       dataout1(1,1,1) = 0.0
    else
       coeff = -1./(ky(1)**2)
       dataout1(1,1,1) = coeff*dataout1(1,1,1)
    endif

    ! !! mind the transpose -> index inversion between y and z
    do i = 2, local_resolution(c_X)
       coeff = -1./(kx(i)**2+ky(1)**2)
       dataout1(i,1,1) = coeff*dataout1(i,1,1)
    end do

    ! !! mind the transpose -> index inversion between y and z
    do k = 2, fft_resolution(c_Z)
       do i = 1, local_resolution(c_X)
          coeff = -1./(kx(i)**2+ky(1)**2+kz(k)**2)
          dataout1(i,k,1) = coeff*dataout1(i,k,1)
       end do
    end do

    ! !! mind the transpose -> index inversion between y and z
    do j = 2,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             coeff = -1./(kx(i)**2+ky(j)**2+kz(k)**2)
             dataout1(i,k,j) = coeff*dataout1(i,k,j)
          end do
       end do
    end do
  end subroutine filter_pressure_3d

  !> Solve Poisson problem in the Fourier space :
  !! \f{eqnarray*} \Delta \psi &=& - \omega \\ v &=& \nabla\times\psi \f}
  subroutine filter_poisson_3d_many()

    integer(C_INTPTR_T) :: i,j,k
    complex(C_DOUBLE_COMPLEX) :: coeff
    complex(C_DOUBLE_COMPLEX) :: buffer1,buffer2

    ! Set first coeff (check for "all freq = 0" case)
    if(local_offset(c_Y) == 0) then
       dataout_many(:,1,1,1) = 0.0
    else
       coeff = Icmplx/(ky(1)**2)
       buffer1 = dataout_many(1,1,1,1)
       dataout_many(1,1,1,1) = coeff*ky(1)*dataout_many(3,1,1,1)
       dataout_many(2,1,1,1) = 0.0
       dataout_many(3,1,1,1) = -coeff*ky(1)*buffer1
    endif

    ! !! mind the transpose -> index inversion between y and z
    do i = 2, local_resolution(c_X)
       coeff = Icmplx/(kx(i)**2+ky(1)**2)
       buffer1 = dataout_many(1,i,1,1)
       buffer2 = dataout_many(2,i,1,1)
       dataout_many(1,i,1,1) = coeff*ky(1)*dataout_many(3,i,1,1)
       dataout_many(2,i,1,1) = -coeff*kx(i)*dataout_many(3,i,1,1)
       dataout_many(3,i,1,1) = coeff*(kx(i)*buffer2-ky(1)*buffer1)
    end do

    ! !! mind the transpose -> index inversion between y and z
    do k = 2, fft_resolution(c_Z)
       do i = 1, local_resolution(c_X)
          coeff = Icmplx/(kx(i)**2+ky(1)**2+kz(k)**2)
          buffer1 = dataout_many(1,i,k,1)
          buffer2 = dataout_many(2,i,k,1)
          dataout_many(1,i,k,1) = coeff*(ky(1)*dataout_many(3,i,k,1)-kz(k)*dataout_many(2,i,k,1))
          dataout_many(2,i,k,1) = coeff*(kz(k)*buffer1-kx(i)*dataout_many(3,i,k,1))
          dataout_many(3,i,k,1) = coeff*(kx(i)*buffer2-ky(1)*buffer1)
       end do
    end do

    ! !! mind the transpose -> index inversion between y and z
    do j = 2,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             coeff = Icmplx/(kx(i)**2+ky(j)**2+kz(k)**2)
             buffer1 = dataout_many(1,i,k,j)
             buffer2 = dataout_many(2,i,k,j)
             dataout_many(1,i,k,j) = coeff*(ky(j)*dataout_many(3,i,k,j)-kz(k)*dataout_many(2,i,k,j))
             dataout_many(2,i,k,j) = coeff*(kz(k)*buffer1-kx(i)*dataout_many(3,i,k,j))
             dataout_many(3,i,k,j) = coeff*(kx(i)*buffer2-ky(j)*buffer1)
          end do
       end do
    end do

  end subroutine filter_poisson_3d_many

  !> Solve diffusion problem in the Fourier space :
  !! \f{eqnarray*} \omega &=& \nabla \times v \\ \frac{\partial \omega}{\partial t} &=& \nu \Delta \omega \f}
  !! @param[in] nudt \f$ \nu\times dt\f$, diffusion coefficient times current time step
  subroutine filter_diffusion_3d_many(nudt)

    real(C_DOUBLE), intent(in) :: nudt
    integer(C_INTPTR_T) :: i,j,k
    complex(C_DOUBLE_COMPLEX) :: coeff
    complex(C_DOUBLE_COMPLEX) :: buffer1,buffer2

    !! mind the transpose -> index inversion between y and z
    do j = 1,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             coeff = Icmplx/(1. + nudt * kx(i)**2+ky(j)**2+kz(k)**2)
             buffer1 = dataout_many(1,i,k,j)
             buffer2 = dataout_many(2,i,k,j)
             dataout_many(1,i,k,j) = coeff*(ky(j)*dataout_many(3,i,k,j)-kz(k)*dataout_many(2,i,k,j))
             dataout_many(2,i,k,j) = coeff*(kz(k)*buffer1-kx(i)*dataout_many(3,i,k,j))
             dataout_many(3,i,k,j) = coeff*(kx(i)*buffer2-ky(j)*buffer1)
          end do
       end do
    end do

  end subroutine filter_diffusion_3d_many

  !> Clean fftw context (free memory, plans ...)
  subroutine cleanFFTW_3d()
    call fftw_destroy_plan(plan_forward1)
    call fftw_destroy_plan(plan_backward1)
    if(.not.manycase) then
       call fftw_destroy_plan(plan_forward2)
       call fftw_destroy_plan(plan_backward2)
       call fftw_destroy_plan(plan_forward3)
       call fftw_destroy_plan(plan_backward3)
       call fftw_free(cbuffer2)
       call fftw_free(cbuffer3)
    endif
    call fftw_free(cbuffer1)
    call fftw_mpi_cleanup()
    deallocate(fft_resolution)
    deallocate(kx,ky,kz)
    if(rank==0) close(21)
  end subroutine cleanFFTW_3d

  !> some information about memory alloc, arrays sizes and so on
  subroutine fft3d_diagnostics(nbelem,howmany)
    integer(C_INTPTR_T), intent(in) :: nbelem
    ! number of buffers used for fftw
    integer, optional,intent(in) :: howmany
    complex(C_DOUBLE_COMPLEX) :: memoryAllocated

    integer :: nbFields
    if(present(howmany)) then
       nbFields = howmany
    else
       nbFields = 3
    end if
    memoryAllocated = real(nbelem*sizeof(memoryAllocated),mk)*1e-6
    write(*,'(a,i5,a,i12,f10.2)') '[',rank,'] size of each buffer (elements / memory in MB):', &
         nbelem, memoryAllocated
    write(*,'(a,i5,a,3i12)') '[',rank,'] size of kx,y,z vectors (number of elements):', &
         size(kx), size(ky),size(kz)
    write(*,'(a,i5,a,6i5)') '[',rank,'] local resolution and offset :', local_resolution, local_offset
    memoryAllocated = nbFields*memoryAllocated + real(sizeof(kx) + sizeof(ky) + sizeof(kz), mk)*1e-6
    write(*,'(a,i5,a,f10.2)') '[',rank,'] Total memory used for fftw buffers (MB):', memoryAllocated

  end subroutine fft3d_diagnostics

  !> Get local size of input and output field on fftw topology
  !! @param datashape local shape of the input field for the fftw process
  !! @param offset index of the first component of the local field (in each dir) in the global set of indices
  subroutine getParamatersTopologyFFTW3d(datashape,offset)
    integer(C_INTPTR_T), intent(out),dimension(3) :: datashape
    integer(C_INTPTR_T), intent(out),dimension(3) :: offset
    integer(C_INTPTR_T) :: zero = 0
    datashape = (/fft_resolution(c_X), fft_resolution(c_Y), local_resolution(c_Z)/)
    offset = (/zero, zero, local_offset(c_Z)/)

  end subroutine getParamatersTopologyFFTW3d


  !> forward transform - The result is saved in local buffers
  !!  @param[in] field 3d scalar field, x-component of the input vector field
  !!  @param[in] ghosts, number of points in the ghost layer of input fields.
  subroutine r2c_3d_scal(field, ghosts)

    real(mk),dimension(:,:,:),intent(in) :: field
    integer, dimension(3), intent(in) :: ghosts
    !real(8) :: start
    integer(C_INTPTR_T) :: i,j,k, ig, jg, kg

    ! ig, jg, kg are used to take into account
    ! ghost points in input fields
    ! init
    do k =1, local_resolution(c_Z)
       kg = k + ghosts(c_Z)
       do j = 1, fft_resolution(c_Y)
          jg = j + ghosts(c_Y)
          do i = 1, fft_resolution(c_X)
             ig = i + ghosts(c_X)
             rdatain1(i,j,k) = field(ig,jg,kg)
          end do
       end do
    end do

    ! compute transforms for each component
    !start = MPI_WTIME()
    call fftw_mpi_execute_dft_r2c(plan_forward1, rdatain1, dataout1)
    !!print *, "r2c time = ", MPI_WTIME() - start

  end subroutine r2c_3d_scal

  !> Compute spectrum of the given data
  subroutine filter_spectrum_3d(spectrum,wavelengths,length)

    real(mk),dimension(:),intent(inout) :: spectrum
    real(mk),dimension(:),intent(inout) :: wavelengths
    real(mk),intent(in) :: length
    integer(C_INTPTR_T) :: i,j,k
    real(mk) :: c
    real(mk) :: kk,dk,kc,eps
    integer(C_INTPTR_T) :: ik
    spectrum = 0
    dk = 2.0_mk*pi/length
    kc = pi*fft_resolution(c_X)/length
    eps=kc/1000000.0_mk

    !! mind the transpose -> index inversion between y and z
    c = 1._mk/real(fft_resolution(c_Z)*fft_resolution(c_Y)*fft_resolution(c_X),mk)
    c = c * c
    do j = 1,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             kk=sqrt(kx(i)**2+ky(j)**2+kz(k)**2)
             if ((kk.gt.eps).and.(kk.le.kc)) then
                ik=1+int(kk/dk+0.5_mk)
                spectrum(ik) = spectrum(ik) + real(dataout1(i,j,k)*conjg(dataout1(i,j,k)), mk) * c
             end if
          end do
       end do
    end do
    wavelengths(:) = kx

  end subroutine filter_spectrum_3d

end module fft3d
