!> Fast Fourier Transform routines (Fortran, based on fftw)
!! to solve 2d Poisson and diffusion problems.
!!
!! This module provides :
!! \li 1 - fftw routines for the "complex to complex" case :
!! solves the problem for
!! complex input/output. The names of these routines contain "c2c".
!! \li 2 - fftw routines for the "real to complex" case :
!!  solves the problem for
!!  input fields which are real. The names of these routines contain "r2c".
!! \li 3 - fftw routines for the "real to complex" case :
!!  solves the problem for
!! input fields which are real and using the "many" interface of the fftw.
!! It means that transforms are applied to the 2 input fields at the same time.
!! Names of these routines contain "many".
!!
!! Obviously, all the above cases should lead to the same results. By default
!! case 2 must be chosen (if input is real). Case 1 and 3 are more or less
!! dedicated to tests and validation.
module fft2d

  use client_data

  implicit none
  include 'fftw3-mpi.f03'

  private

  public :: init_c2c_2d,init_r2c_2d, r2c_scalar_2d, c2c_2d,c2r_2d,c2r_scalar_2d,&
       r2c_2d,cleanFFTW_2d, filter_poisson_2d, filter_curl_2d, getParamatersTopologyFFTW2d, &
       filter_diffusion_2d, init_r2c_2dBIS


  !> plan for fftw "c2c" forward or r2c transform
  type(C_PTR) :: plan_forward1, plan_forward2
  !> plan for fftw "c2c" backward or c2r transform
  type(C_PTR) :: plan_backward1,plan_backward2
  !> memory buffer for fftw
  !! (input and output buffer will point to this location)
  type(C_PTR) :: cbuffer1
  !> second memory buffer for fftw (used for backward transform)
  type(C_PTR) :: cbuffer2
  !! Note Franck : check if local declarations of datain/out works and improve perfs.
  !> Field (complex values) for fftw input
  complex(C_DOUBLE_COMPLEX), pointer :: datain1(:,:),datain2(:,:)
  !> Field (real values) for fftw input
  real(C_DOUBLE), pointer :: rdatain1(:,:)
  !> Field (complex values) for fftw (forward) output
  complex(C_DOUBLE_COMPLEX), pointer :: dataout1(:,:)
  !> Field (real values) for fftw output
  real(C_DOUBLE), pointer :: rdatain2(:,:)
  !> Field (complex values) for fftw (forward) output
  complex(C_DOUBLE_COMPLEX), pointer :: dataout2(:,:)
  !> GLOBAL number of points in each direction
  integer(C_INTPTR_T),pointer :: fft_resolution(:)
  !> LOCAL resolution
  integer(c_INTPTR_T),dimension(2) :: local_resolution
  !> Offset in the direction of distribution
  integer(c_INTPTR_T),dimension(2) :: local_offset
  !> wave numbers for fft in x direction
  real(C_DOUBLE), pointer :: kx(:)
  !> wave numbers for fft in y direction
  real(C_DOUBLE), pointer :: ky(:)
  !> log file for fftw
  character(len=20),parameter :: filename ="hysopfftw.log"
  !> normalization factor
  real(C_DOUBLE) :: normFFT
  !> true if all the allocation stuff for global variables has been done.
  logical :: is2DUpToDate = .false.

contains
  !========================================================================
  !   Complex to complex transforms
  !========================================================================

  !> Initialisation of the fftw context for complex
  !! to complex transforms (forward and backward)
  !! @param[in] resolution global domain resolution
  subroutine init_c2c_2d(resolution,lengths)

    !! global domain resolution
    integer, dimension(2), intent(in) :: resolution
    real(mk),dimension(2), intent(in) :: lengths

    !! Size of the local memory required for fftw (cbuffer)
    integer(C_INTPTR_T) :: alloc_local

    if(is2DUpToDate) return

    ! init fftw mpi context
    call fftw_mpi_init()

    if(rank==0) open(unit=21,file=filename,form="formatted")

    ! set fft resolution
    allocate(fft_resolution(2))
    fft_resolution = resolution-1

    ! compute "optimal" size (according to fftw) for local date
    ! (warning : dimension reversal)
    alloc_local = fftw_mpi_local_size_2d_transposed(fft_resolution(c_Y), &
         fft_resolution(c_X),main_comm, local_resolution(c_Y), &
         local_offset(c_Y), local_resolution(c_X),local_offset(c_X));

    ! allocate local buffer (used to save datain/dataout1
    ! ==> in-place transform!!)
    cbuffer1 = fftw_alloc_complex(alloc_local)
    ! link datain and dataout1 to cbuffer, setting the right dimensions
    call c_f_pointer(cbuffer1, datain1, &
         [fft_resolution(c_X),local_resolution(c_Y)])
    call c_f_pointer(cbuffer1, dataout1, &
         [fft_resolution(c_Y),local_resolution(c_X)])

    ! second buffer used for backward transform. Used to copy dataout1
    ! into dataout2 (input for backward transform and filter)
    ! and to save (in-place) the transform of the second component
    ! of the velocity
    cbuffer2 = fftw_alloc_complex(alloc_local)
    call c_f_pointer(cbuffer2, datain2,&
         [fft_resolution(c_X),local_resolution(c_Y)])
    call c_f_pointer(cbuffer2, dataout2, [fft_resolution(c_Y),local_resolution(c_X)])

    !   create MPI plan for in-place forward/backward DFT (note dimension reversal)
    plan_forward1 = fftw_mpi_plan_dft_2d(fft_resolution(c_Y), fft_resolution(c_X),datain1,dataout1,&
         main_comm,FFTW_FORWARD,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward1 = fftw_mpi_plan_dft_2d(fft_resolution(c_Y),fft_resolution(c_X),dataout1,datain1,&
         main_comm,FFTW_BACKWARD,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))
    plan_backward2 = fftw_mpi_plan_dft_2d(fft_resolution(c_Y),fft_resolution(c_X),dataout2,datain2,&
         main_comm,FFTW_BACKWARD,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))

    call computeKxC(lengths(c_X))
    call computeKy(lengths(c_Y))
    normFFT =  1./(fft_resolution(c_X)*fft_resolution(c_Y))
    !! call fft2d_diagnostics(alloc_local)

    is2DUpToDate = .true.

!!$    write(*,'(a,i5,a,16f10.4)') 'kx[',rank,'] ', kx
!!$    write(*,'(a,i5,a,16f10.4)') 'ky[',rank,'] ', ky
!!$
  end subroutine init_c2c_2d

  !> Execute fftw forward transform, according to the pre-defined plans.
  subroutine c2c_2d(inputData,velocity_x,velocity_y)
    complex(mk),dimension(:,:) :: velocity_x,velocity_y
    complex(mk),dimension(:,:),intent(in) :: inputData

    integer(C_INTPTR_T) :: i, j

    do j = 1, local_resolution(c_Y)
       do i = 1, fft_resolution(c_X)
          datain1(i, j) = inputData(i,j)
       end do
    end do

    ! compute transform (as many times as desired)
    call fftw_mpi_execute_dft(plan_forward1, datain1, dataout1)

!!$    do i = 1, fft_resolution(c_Y)
!!$       write(*,'(a,i5,a,16f10.4)') 'out[',rank,'] ', dataout1(i,1:local_resolution(c_X))
!!$    end do
!!$
    call filter_poisson_2d()

    call fftw_mpi_execute_dft(plan_backward1, dataout1, datain1)
    call fftw_mpi_execute_dft(plan_backward2,dataout2,datain2)
    do j = 1, local_resolution(c_Y)
       do i = 1, fft_resolution(c_X)
          velocity_x(i,j) = datain1(i,j)*normFFT
          velocity_y(i,j) = datain2(i,j)*normFFT
       end do
    end do

!!$    do i = 1, fft_resolution(c_X)
!!$       write(*,'(a,i5,a,16f10.4)') 'vxx[',rank,'] ', velocity_x(i,1:local_resolution(c_Y))
!!$    end do
!!$    write(*,'(a,i5,a)') '[',rank,'] ==============================='
!!$    do i = 1, fft_resolution(c_X)
!!$       write(*,'(a,i5,a,16f10.4)') 'vyy[',rank,'] ', velocity_y(i,1:local_resolution(c_Y))
!!$    end do
!!$    write(*,'(a,i5,a)') '[',rank,'] ==============================='
!!$
  end subroutine c2c_2d

  !========================================================================
  !  Real to complex transforms
  !========================================================================

  !> Initialisation of the fftw context for real to complex transforms (forward and backward)
  !! @param[in] resolution global domain resolution
  subroutine init_r2c_2d(resolution,lengths)

    integer, dimension(2), intent(in) :: resolution
    real(mk),dimension(2), intent(in) :: lengths
    !! Size of the local memory required for fftw (cbuffer)
    integer(C_INTPTR_T) :: alloc_local,halfLength

    if(is2DUpToDate) return

    ! init fftw mpi context
    call fftw_mpi_init()

    if(rank==0) open(unit=21,file=filename,form="formatted")

    allocate(fft_resolution(2))
    fft_resolution(:) = resolution(:)-1
    halfLength = fft_resolution(c_X)/2+1
    ! allocate local buffer (used to save datain/dataout1 ==> in-place transform!!)
    alloc_local = fftw_mpi_local_size_2d_transposed(fft_resolution(c_Y),halfLength,main_comm,local_resolution(c_Y),&
         local_offset(c_Y),local_resolution(c_X),local_offset(c_X));

    ! allocate local buffer (used to save datain/dataout1 ==> in-place transform!!)
    cbuffer1 = fftw_alloc_complex(alloc_local)

    ! link rdatain1 and dataout1 to cbuffer, setting the right dimensions for each
    call c_f_pointer(cbuffer1, rdatain1, [2*halfLength,local_resolution(c_Y)])
    call c_f_pointer(cbuffer1, dataout1, [fft_resolution(c_Y),local_resolution(c_X)])

    ! second buffer used for backward transform. Used to copy dataout1 into dataout2 (input for backward transform and filter)
    ! and to save (in-place) the transform of the second component of the velocity
    cbuffer2 = fftw_alloc_complex(alloc_local)

    call c_f_pointer(cbuffer2, rdatain2, [2*halfLength,local_resolution(c_Y)])
    call c_f_pointer(cbuffer2, dataout2, [fft_resolution(c_Y),local_resolution(c_X)])

    !   create MPI plans for in-place forward/backward DFT (note dimension reversal)
    plan_forward1 = fftw_mpi_plan_dft_r2c_2d(fft_resolution(c_Y), fft_resolution(c_X), rdatain1, dataout1, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward1 = fftw_mpi_plan_dft_c2r_2d(fft_resolution(c_Y), fft_resolution(c_X), dataout1, rdatain1, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))
    plan_backward2 = fftw_mpi_plan_dft_c2r_2d(fft_resolution(c_Y), fft_resolution(c_X), dataout2, rdatain2, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))

    call computeKx(lengths(c_X))
    call computeKy(lengths(c_Y))
    normFFT = 1./(fft_resolution(c_X)*fft_resolution(c_Y))
    !! call fft2d_diagnostics(alloc_local)
!!$
!!$    write(*,'(a,i5,a,16f10.4)') 'kx[',rank,'] ', kx
!!$    write(*,'(a,i5,a,16f10.4)') 'ky[',rank,'] ', ky
!!$
    is2DUpToDate = .true.

  end subroutine init_r2c_2d


  !> forward transform - The result is saved in local buffers
  !! @param input data
  subroutine r2c_scalar_2d(inputData, ghosts)

    real(mk),dimension(:,:), intent(in) :: inputData
    integer, dimension(2), intent(in) :: ghosts

    integer(C_INTPTR_T) :: i, j, ig, jg
    ! init
    do j = 1, local_resolution(c_Y)
       jg = j + ghosts(c_Y)
       do i = 1, fft_resolution(c_X)
          ig = i + ghosts(c_X)
          rdatain1(i, j) = inputData(ig,jg)
       end do
    end do

!!$    do i = 1, fft_resolution(c_X)
!!$       write(*,'(a,i5,a,16f10.4)') 'rr[',rank,'] ', rdatain1(i,1:local_resolution(c_Y))
!!$    end do
!!$
    ! compute transform (as many times as desired)
    call fftw_mpi_execute_dft_r2c(plan_forward1, rdatain1, dataout1)

!!$    do i = 1, fft_resolution(c_Y)
!!$       write(*,'(a,i5,a,16f10.4)') 'aaaa[',rank,'] ', dataout1(i,1:local_resolution(c_X))
!!$    end do

  end subroutine r2c_scalar_2d

  !> forward transform - The result is saved in local buffers
  !! @param[in] input data
  subroutine r2c_2d(input_x, input_y, ghosts)

    real(mk),dimension(:,:), intent(in) :: input_x, input_y
    integer, dimension(2), intent(in) :: ghosts

    integer(C_INTPTR_T) :: i, j, ig, jg
    ! init
    do j = 1, local_resolution(c_Y)
       jg = j + ghosts(c_Y)
       do i = 1, fft_resolution(c_X)
          ig = i + ghosts(c_X)
          rdatain1(i, j) = input_x(ig,jg)
          rdatain2(i, j) = input_y(ig,jg)    
       end do
    end do

    ! compute transform (as many times as desired)
    call fftw_mpi_execute_dft_r2c(plan_forward1, rdatain1, dataout1)
    call fftw_mpi_execute_dft_r2c(plan_forward2, rdatain2, dataout2)

  end subroutine r2c_2d

  !> Backward transform
  subroutine c2r_2d(velocity_x,velocity_y, ghosts)
    real(mk),dimension(:,:),intent(inout) :: velocity_x,velocity_y
    integer, dimension(2), intent(in) :: ghosts
    integer(C_INTPTR_T) :: i, j, ig, jg

    call fftw_mpi_execute_dft_c2r(plan_backward1,dataout1,rdatain1)
    call fftw_mpi_execute_dft_c2r(plan_backward2,dataout2,rdatain2)
    do j = 1, local_resolution(c_Y)
       jg = j + ghosts(c_Y)
       do i = 1, fft_resolution(c_X)
          ig = i + ghosts(c_X)
          velocity_x(ig,jg) = rdatain1(i,j)*normFFT
          velocity_y(ig,jg) = rdatain2(i,j)*normFFT
       end do
    end do

!!$
!!$    do i = 1, fft_resolution(c_X)
!!$       write(*,'(a,i5,a,16f10.4)') 'xx[',rank,'] ', velocity_x(i,1:local_resolution(c_Y))
!!$    end do
!!$    write(*,'(a,i5,a)') '[',rank,'] ==============================='
!!$    do i = 1, fft_resolution(c_X)
!!$       write(*,'(a,i5,a,16f10.4)') 'yy[',rank,'] ', velocity_y(i,1:local_resolution(c_Y))
!!$    end do
!!$    write(*,'(a,i5,a)') '[',rank,'] ==============================='


  end subroutine c2r_2d

  !> Backward transform for scalar field
  subroutine c2r_scalar_2d(omega, ghosts)
    real(mk),dimension(:,:),intent(inout) :: omega
    integer, dimension(2), intent(in) :: ghosts
    integer(C_INTPTR_T) :: i, j, ig, jg

    call fftw_mpi_execute_dft_c2r(plan_backward1,dataout1,rdatain1)
    do j = 1, local_resolution(c_Y)
       jg = j + ghosts(c_Y)
       do i = 1, fft_resolution(c_X)
          ig = i + ghosts(c_X)
          omega(ig,jg) = rdatain1(i,j)*normFFT
       end do
    end do

!!$
!!$    do i = 1, fft_resolution(c_X)
!!$       write(*,'(a,i5,a,16f10.4)') 'xx[',rank,'] ', velocity_x(i,1:local_resolution(c_Y))
!!$    end do
!!$    write(*,'(a,i5,a)') '[',rank,'] ==============================='
!!$    do i = 1, fft_resolution(c_X)
!!$       write(*,'(a,i5,a,16f10.4)') 'yy[',rank,'] ', velocity_y(i,1:local_resolution(c_Y))
!!$    end do
!!$    write(*,'(a,i5,a)') '[',rank,'] ==============================='


  end subroutine c2r_scalar_2d


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

    do i = local_offset(c_X)+1,local_offset(c_X)+local_resolution(c_X) !! global index
       kx(i-local_offset(c_X)) =  2.*pi/length*(i-1)
    end do

  end subroutine computeKx

  !> Computation of frequencies coeff, over the distributed direction in the complex/complex case
  !> @param lengths size of the domain
  subroutine computeKxC(length)

    real(mk),intent(in) :: length

    !! Local loops indices
    integer(C_INTPTR_T) :: i

    !! Compute filter coefficients
    allocate(kx(local_resolution(c_X)))

    !! x frequencies (distributed over proc)
    !! If we deal with positive frequencies only
    if(local_offset(c_X)+local_resolution(c_X) <= fft_resolution(c_X)/2+1 ) then
       do i = 1,local_resolution(c_X)
          kx(i) =  2.*pi/length*(local_offset(c_X)+i-1)
       end do

    else
       !! else if we deal with negative frequencies only
       if(local_offset(c_X)+1 > fft_resolution(c_X)/2+1 ) then
          do i = 1,local_resolution(c_X)
             kx(i) =  2.*pi/length*(local_offset(c_X)+i-1-fft_resolution(c_X))
          end do
          !! final case : start positive freq, end in negative ones
       else
          do i = local_offset(c_X)+1, fft_resolution(c_X)/2+1 !! global index
             kx(i-local_offset(c_X)) =  2.*pi/length*(i-1)
          end do
          do i = fft_resolution(c_X)/2+2,local_resolution(c_X)+local_offset(c_X)
             kx(i-local_offset(c_X)) =  2.*pi/length*(i-1-fft_resolution(c_X))
          end do
       end if
    end if

  end subroutine computeKxC

  !> Computation of frequencies coeff, over non-distributed direction(s)
  !> @param lengths size of the domain
  subroutine computeKy(length)
    real(mk), intent(in) :: length

    !! Local loops indices
    integer(C_INTPTR_T) :: i
    allocate(ky(fft_resolution(c_Y)))

    do i = 1, fft_resolution(c_Y)/2+1
       ky(i) = 2.*pi/length*(i-1)
    end do
    do i = fft_resolution(c_Y)/2+2,fft_resolution(c_Y)
       ky(i) = 2.*pi/length*(i-fft_resolution(c_Y)-1)
    end do

  end subroutine computeKy

  subroutine filter_poisson_2d()

    integer(C_INTPTR_T) :: i, j
    complex(C_DOUBLE_COMPLEX) :: coeff
    if(local_offset(c_X)==0) then
       if(local_offset(c_Y) == 0) then
          dataout1(1,1) = 0.0
          dataout2(1,1) = 0.0
       else
          coeff = Icmplx/(kx(1)**2+ky(1)**2)
          dataout2(1,1) = -coeff*kx(1)*dataout1(1,1)
          dataout1(1,1) = coeff*ky(1)*dataout1(1,1)
       endif

       do j = 2, fft_resolution(c_Y)
          coeff = Icmplx/(kx(1)**2+ky(j)**2)
          dataout2(j,1) = -coeff*kx(1)*dataout1(j,1)
          dataout1(j,1) = coeff*ky(j)*dataout1(j,1)
       end do
       do i = 2,local_resolution(c_X)
          do j = 1, fft_resolution(c_Y)
             coeff = Icmplx/(kx(i)**2+ky(j)**2)
             dataout2(j,i) = -coeff*kx(i)*dataout1(j,i)
             dataout1(j,i) = coeff*ky(j)*dataout1(j,i)
          end do
       end do
    else
       do i = 1,local_resolution(c_X)
          do j = 1, fft_resolution(c_Y)
             coeff = Icmplx/(kx(i)**2+ky(j)**2)
             dataout2(j,i) = -coeff*kx(i)*dataout1(j,i)
             dataout1(j,i) = coeff*ky(j)*dataout1(j,i)
          end do
       end do
    end if

!!$    do i = 1,local_resolution(c_X)
!!$       write(*,'(a,i5,a,16f10.4)') 'xx[',rank,'] ', dataout1(1:fft_resolution(c_Y),i)
!!$    end do
!!$
!!$    write(*,'(a,i5,a)') '[',rank,'] ==============================='
!!$    do i = 1,local_resolution(c_X)
!!$       write(*,'(a,i5,a,16f10.4)') 'yy[',rank,'] ', dataout2(1:fft_resolution(c_Y),i)
!!$    end do
!!$    write(*,'(a,i5,a)') '[',rank,'] ==============================='

  end subroutine filter_poisson_2d

  subroutine filter_diffusion_2d(nudt)

    real(C_DOUBLE), intent(in) :: nudt
    integer(C_INTPTR_T) :: i, j
    complex(C_DOUBLE_COMPLEX) :: coeff

    do i = 1,local_resolution(c_X)
       do j = 1, fft_resolution(c_Y)
          coeff = 1./(1. + nudt * (kx(i)**2+ky(j)**2))
          dataout1(j,i) = coeff*dataout1(j,i)
       end do
    end do

  end subroutine filter_diffusion_2d

  !> Clean fftw context (free memory, plans ...)
  subroutine cleanFFTW_2d()
    call fftw_destroy_plan(plan_forward1)
    call fftw_destroy_plan(plan_backward1)
    !call fftw_destroy_plan(plan_forward2)
    !call fftw_destroy_plan(plan_backward2)
    call fftw_free(cbuffer1)
    call fftw_free(cbuffer2)
    call fftw_mpi_cleanup()
    deallocate(fft_resolution)
    if(rank==0) close(21)
  end subroutine cleanFFTW_2d

  !> Solve curl problem in the Fourier space :
  !! \f{eqnarray*} \omega &=& \nabla \times v
  subroutine filter_curl_2d()

    integer(C_INTPTR_T) :: i,j,k
    complex(C_DOUBLE_COMPLEX) :: coeff

    !! mind the transpose -> index inversion between y and z
    do j = 1,local_resolution(c_Y)
       do k = 1, fft_resolution(c_Z)
          do i = 1, local_resolution(c_X)
             coeff = Icmplx
             dataout1(j,i) = coeff*(kx(i)*dataout2(j,i) - ky(j)*dataout1(j,i))
          end do
       end do
    end do

  end subroutine filter_curl_2d

  subroutine fft2d_diagnostics(nbelem)
    integer(C_INTPTR_T), intent(in) :: nbelem
    complex(C_DOUBLE_COMPLEX) :: memoryAllocated
    memoryAllocated = real(nbelem*sizeof(memoryAllocated),mk)*1e-6
    write(*,'(a,i5,a,i12,f10.2)') '[',rank,'] size of each buffer (elements / memory in MB):', &
         nbelem, memoryAllocated
    write(*,'(a,i5,a,2i12)') '[',rank,'] size of kx,y,z vectors (number of elements):', &
         size(kx), size(ky)
    write(*,'(a,i5,a,4i5)') '[',rank,'] local resolution and offset :', local_resolution, local_offset
    memoryAllocated = 2*memoryAllocated + real(sizeof(kx) + sizeof(ky), mk)*1e-6
    write(*,'(a,i5,a,f10.2)') '[',rank,'] Total memory used for fftw buffers (MB):', memoryAllocated

  end subroutine fft2d_diagnostics

  !> Get local size of input and output field on fftw topology
  !! @param datashape local shape of the input field for the fftw process
  !! @param offset index of the first component of the local field (in each dir) in the global set of indices
  subroutine getParamatersTopologyFFTW2d(datashape,offset)
    integer(C_INTPTR_T), intent(out),dimension(2) :: datashape
    integer(C_INTPTR_T), intent(out),dimension(2) :: offset
    integer(C_INTPTR_T) :: offsetx = 0
    datashape = (/fft_resolution(c_X), local_resolution(c_Y)/)
    offset = (/ offsetx, local_offset(c_Y)/)

  end subroutine getParamatersTopologyFFTW2d
  !> Initialisation of the fftw context for real to complex transforms (forward and backward)
  !! @param[in] resolution global domain resolution
  subroutine init_r2c_2dBIS(resolution,lengths)

    integer, dimension(2), intent(in) :: resolution
    real(mk),dimension(2), intent(in) :: lengths
    !! Size of the local memory required for fftw (cbuffer)
    integer(C_INTPTR_T) :: alloc_local,halfLength,howmany
    integer(C_INTPTR_T), dimension(2) :: n

    !> Field (real values) for fftw input
    real(C_DOUBLE), pointer :: rdatain1Many(:,:,:)

    ! init fftw mpi context
    call fftw_mpi_init()
    howmany = 1
    if(rank==0) open(unit=21,file=filename,form="formatted")

    allocate(fft_resolution(2))
    fft_resolution(:) = resolution(:)-1
    halfLength = fft_resolution(c_X)/2+1
    n(1) = fft_resolution(2)
    n(2) = halfLength
    ! allocate local buffer (used to save datain/dataout1 ==> in-place transform!!)
    alloc_local = fftw_mpi_local_size_many_transposed(2,n,howmany,FFTW_MPI_DEFAULT_BLOCK,&
         FFTW_MPI_DEFAULT_BLOCK,main_comm,local_resolution(c_Y),&
         local_offset(c_Y),local_resolution(c_X),local_offset(c_X));

    ! allocate local buffer (used to save datain/dataout1 ==> in-place transform!!)
    cbuffer1 = fftw_alloc_complex(alloc_local)

    ! link rdatain1 and dataout1 to cbuffer, setting the right dimensions for each
    call c_f_pointer(cbuffer1, rdatain1Many, [howmany,2*halfLength,local_resolution(c_Y)])
    call c_f_pointer(cbuffer1, dataout1, [fft_resolution(c_Y),local_resolution(c_X)])

    ! second buffer used for backward transform. Used to copy dataout1 into dataout2 (input for backward transform and filter)
    ! and to save (in-place) the transform of the second component of the velocity
    cbuffer2 = fftw_alloc_complex(alloc_local)

    call c_f_pointer(cbuffer2, rdatain1Many, [howmany,2*halfLength,local_resolution(c_Y)])
    call c_f_pointer(cbuffer2, dataout2, [fft_resolution(c_Y),local_resolution(c_X)])

    !   create MPI plans for in-place forward/backward DFT (note dimension reversal)
    plan_forward1 = fftw_mpi_plan_dft_r2c_2d(fft_resolution(c_Y), fft_resolution(c_X), rdatain1Many, dataout1, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_OUT))
    plan_backward1 = fftw_mpi_plan_dft_c2r_2d(fft_resolution(c_Y), fft_resolution(c_X), dataout1, rdatain1, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))
    plan_backward2 = fftw_mpi_plan_dft_c2r_2d(fft_resolution(c_Y), fft_resolution(c_X), dataout2, rdatain2, &
         main_comm,ior(FFTW_MEASURE,FFTW_MPI_TRANSPOSED_IN))

    call computeKx(lengths(c_X))
    call computeKy(lengths(c_Y))
    normFFT = 1./(fft_resolution(c_X)*fft_resolution(c_Y))
    !! call fft2d_diagnostics(alloc_local)

  end subroutine init_r2c_2dBIS

end module fft2d
