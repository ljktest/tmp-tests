!> routines to compute curl, cross prod ...
!! WARNING : many of the following routines are out-of-date with
!! remaining "ppm" type call of field shapes.
!! \todo : clean everything and move to python.
module vectorcalculus

  use client_data
  !use client_topology, only:nsublist
  use mpi
  implicit none

  ! temp to avoid ppm dependence
  integer, parameter,private :: nsublist = 1

contains

  !> compute \f[ fieldout = \nabla \times fieldin \f] using 4th order finite differences
  subroutine curlDF4(fieldin,fieldout,resolution,step)
    !> input field
    real(mk), dimension(:,:,:,:,:), pointer :: fieldin
    !> output field
    real(mk), dimension(:,:,:,:,:), pointer :: fieldout
    !> the local resolution
    integer, dimension(dim3),intent(in) :: resolution
    !> size of mesh step in each dir
    real(mk), dimension(dim3),intent(in) :: step

    real(mk) :: facx, facy, facz
    integer :: i,j,k

    fieldout = 0.0

    facx = 1.0/(12.0*step(c_X))
    facy = 1.0/(12.0*step(c_Y))
    facz = 1.0/(12.0*step(c_Z))

    do k=1,resolution(c_Z)
       do j=1,resolution(c_Y)
          do i=1,resolution(c_X)
             fieldout(c_X,i,j,k,nsublist) = -facz*(&
                  -fieldin(c_Y,i,j,k+2,nsublist)+8.0*fieldin(c_Y,i,j,k+1,nsublist)-8.0&
                  *fieldin(c_Y,i,j,k-1,nsublist)+fieldin(c_Y,i,j,k-2,nsublist)&
                  )  + facy*(&
                  -fieldin(c_Z,i,j+2,k,nsublist)+8.0*fieldin(c_Z,i,j+1,k,nsublist)-8.0&
                  *fieldin(c_Z,i,j-1,k,nsublist)+fieldin(c_Z,i,j-2,k,nsublist)&
                  )
             fieldout(c_Y,i,j,k,nsublist) = -facx*(&
                  -fieldin(c_Z,i+2,j,k,nsublist)+8.0*fieldin(c_Z,i+1,j,k,nsublist)-8.0&
                  *fieldin(c_Z,i-1,j,k,nsublist)+fieldin(c_Z,i-2,j,k,nsublist)&
                  ) + facz*( &
                  -fieldin(c_X,i,j,k+2,nsublist)+8.0*(fieldin(c_X,i,j,k+1,nsublist)-&
                  fieldin(c_X,i,j,k-1,nsublist))+fieldin(c_X,i,j,k-2,nsublist))
             fieldout(c_Z,i,j,k,nsublist) = -facy*(&
                  -fieldin(c_X,i,j+2,k,nsublist)+8.0*fieldin(c_X,i,j+1,k,nsublist)-8.0&
                  *fieldin(c_X,i,j-1,k,nsublist)+fieldin(c_X,i,j-2,k,nsublist)&
                  ) +facx*(&
                  -fieldin(c_Y,i+2,j,k,nsublist)+8.0*fieldin(c_Y,i+1,j,k,nsublist)-8.0&
                  *fieldin(c_Y,i-1,j,k,nsublist)+fieldin(c_Y,i-2,j,k,nsublist))
          enddo
       enddo
    enddo

  end subroutine curlDF4

  !> Computes strech and diffusion terms. This is a copy of Adrien's code.
  subroutine computeRHS(velocity,vorticity,rhs,resolution,step,nu)

    !> Velocity field
    real(mk), dimension(:,:,:,:,:), pointer :: velocity
    !> vorticity field
    real(mk), dimension(:,:,:,:,:), pointer :: vorticity
    !> rhs, output
    real(mk), dimension(:,:,:,:,:), pointer :: rhs
    !> local mesh resolution
    integer,dimension(3),intent(in) :: resolution
    !> mesh step sizes
    real(mk),  dimension(3),intent(in) :: step
    real(mk), intent(in) :: nu

    integer :: i,j,k
    real(mk), dimension(3) :: tx, ty, tz, stretch,diffusion
    real(mk) :: facx,facy,facz, facx2,facy2,facz2

    facx=1./(12.*step(c_X))
    facy=1./(12.*step(c_Y))
    facz=1./(12.*step(c_Z))
    facx2=nu/(12.*step(c_X)**2)
    facy2=nu/(12.*step(c_Y)**2)
    facz2=nu/(12.*step(c_Z)**2)

    rhs = 0.0
    do k=1,resolution(c_Z)
       do j=1,resolution(c_Y)
          do i=1,resolution(c_X)
             !stretch
             !------
             tx(c_X)= &
                  vorticity(c_X,i-2,j,k,nsublist)*velocity(c_X,i-2,j,k,nsublist)  - 8.*&
                  vorticity(c_X,i-1,j,k,nsublist)*velocity(c_X,i-1,j,k,nsublist)  + 8.*&
                  vorticity(c_X,i+1,j,k,nsublist)*velocity(c_X,i+1,j,k,nsublist)  -  &
                  vorticity(c_X,i+2,j,k,nsublist)*velocity(c_X,i+2,j,k,nsublist)

             tx(c_Y)= &
                  vorticity(c_X,i-2,j,k,nsublist)*velocity(c_Y,i-2,j,k,nsublist)  - 8.*&
                  vorticity(c_X,i-1,j,k,nsublist)*velocity(c_Y,i-1,j,k,nsublist)  + 8.*&
                  vorticity(c_X,i+1,j,k,nsublist)*velocity(c_Y,i+1,j,k,nsublist)  - &
                  vorticity(c_X,i+2,j,k,nsublist)*velocity(c_Y,i+2,j,k,nsublist)

             tx(c_Z)=&
                  vorticity(c_X,i-2,j,k,nsublist)*velocity(c_Z,i-2,j,k,nsublist)  - 8.*&
                  vorticity(c_X,i-1,j,k,nsublist)*velocity(c_Z,i-1,j,k,nsublist)  + 8.*&
                  vorticity(c_X,i+1,j,k,nsublist)*velocity(c_Z,i+1,j,k,nsublist)  - &
                  vorticity(c_X,i+2,j,k,nsublist)*velocity(c_Z,i+2,j,k,nsublist)

             ty(c_X)=&
                  vorticity(c_Y,i,j-2,k,nsublist)*velocity(c_X,i,j-2,k,nsublist) - 8.*&
                  vorticity(c_Y,i,j-1,k,nsublist)*velocity(c_X,i,j-1,k,nsublist) + 8.*&
                  vorticity(c_Y,i,j+1,k,nsublist)*velocity(c_X,i,j+1,k,nsublist) - &
                  vorticity(c_Y,i,j+2,k,nsublist)*velocity(c_X,i,j+2,k,nsublist)

             ty(c_Y)=&
                  vorticity(c_Y,i,j-2,k,nsublist)*velocity(c_Y,i,j-2,k,nsublist) - 8.*&
                  vorticity(c_Y,i,j-1,k,nsublist)*velocity(c_Y,i,j-1,k,nsublist) + 8.*&
                  vorticity(c_Y,i,j+1,k,nsublist)*velocity(c_Y,i,j+1,k,nsublist) - &
                  vorticity(c_Y,i,j+2,k,nsublist)*velocity(c_Y,i,j+2,k,nsublist)

             ty(c_Z)=&
                  vorticity(c_Y,i,j-2,k,nsublist)*velocity(c_Z,i,j-2,k,nsublist) - 8.*&
                  vorticity(c_Y,i,j-1,k,nsublist)*velocity(c_Z,i,j-1,k,nsublist) + 8.*&
                  vorticity(c_Y,i,j+1,k,nsublist)*velocity(c_Z,i,j+1,k,nsublist) - &
                  vorticity(c_Y,i,j+2,k,nsublist)*velocity(c_Z,i,j+2,k,nsublist)

             tz(c_X)=&
                  vorticity(c_Z,i,j,k-2,nsublist)*velocity(c_X,i,j,k-2,nsublist) - 8.*&
                  vorticity(c_Z,i,j,k-1,nsublist)*velocity(c_X,i,j,k-1,nsublist) + 8.*&
                  vorticity(c_Z,i,j,k+1,nsublist)*velocity(c_X,i,j,k+1,nsublist) - &
                  vorticity(c_Z,i,j,k+2,nsublist)*velocity(c_X,i,j,k+2,nsublist)

             tz(c_Y)=&
                  vorticity(c_Z,i,j,k-2,nsublist)*velocity(c_Y,i,j,k-2,nsublist) - 8.*&
                  vorticity(c_Z,i,j,k-1,nsublist)*velocity(c_Y,i,j,k-1,nsublist) + 8.*&
                  vorticity(c_Z,i,j,k+1,nsublist)*velocity(c_Y,i,j,k+1,nsublist) - &
                  vorticity(c_Z,i,j,k+2,nsublist)*velocity(c_Y,i,j,k+2,nsublist)

             tz(c_Z)=&
                  vorticity(c_Z,i,j,k-2,nsublist)*velocity(c_Z,i,j,k-2,nsublist) - 8.*&
                  vorticity(c_Z,i,j,k-1,nsublist)*velocity(c_Z,i,j,k-1,nsublist) + 8.*&
                  vorticity(c_Z,i,j,k+1,nsublist)*velocity(c_Z,i,j,k+1,nsublist) -  &
                  vorticity(c_Z,i,j,k+2,nsublist)*velocity(c_Z,i,j,k+2,nsublist)

             stretch = facx*tx+facy*ty+facz*tz

             !diffusion
             !----------
             tx(c_X)= - &
                  vorticity(c_X,i+2,j,k,nsublist) + 16.*&
                  vorticity(c_X,i+1,j,k,nsublist) - 30.*&
                  vorticity(c_X,i,j,k,nsublist)   + 16.*&
                  vorticity(c_X,i-1,j,k,nsublist) - &
                  vorticity(c_X,i-2,j,k,nsublist)

             tx(c_Y)= - &
                  vorticity(c_Y,i+2,j,k,nsublist) + 16.*&
                  vorticity(c_Y,i+1,j,k,nsublist) - 30.*&
                  vorticity(c_Y,i,j,k,nsublist)   + 16.*&
                  vorticity(c_Y,i-1,j,k,nsublist) - &
                  vorticity(c_Y,i-2,j,k,nsublist)

             tx(c_Z) = - &
                  vorticity(c_Z,i+2,j,k,nsublist) + 16.*&
                  vorticity(c_Z,i+1,j,k,nsublist) - 30.*&
                  vorticity(c_Z,i,j,k,nsublist)   + 16.*&
                  vorticity(c_Z,i-1,j,k,nsublist) - &
                  vorticity(c_Z,i-2,j,k,nsublist)

             ty(c_X)= - &
                  vorticity(c_X,i,j+2,k,nsublist) + 16.*&
                  vorticity(c_X,i,j+1,k,nsublist) - 30.*&
                  vorticity(c_X,i,j,k,nsublist)   + 16.*&
                  vorticity(c_X,i,j-1,k,nsublist) - &
                  vorticity(c_X,i,j-2,k,nsublist)

             ty(c_Y)= - &
                  vorticity(c_Y,i,j+2,k,nsublist) + 16.*&
                  vorticity(c_Y,i,j+1,k,nsublist) - 30.*&
                  vorticity(c_Y,i,j,k,nsublist)   + 16.*&
                  vorticity(c_Y,i,j-1,k,nsublist) - &
                  vorticity(c_Y,i,j-2,k,nsublist)

             ty(c_Z)= - &
                  vorticity(c_Z,i,j+2,k,nsublist) + 16.*&
                  vorticity(c_Z,i,j+1,k,nsublist) - 30.*&
                  vorticity(c_Z,i,j,k,nsublist)   + 16.*&
                  vorticity(c_Z,i,j-1,k,nsublist) - &
                  vorticity(c_Z,i,j-2,k,nsublist)

             tz(c_X)= - &
                  vorticity(c_X,i,j,k+2,nsublist) + 16.*&
                  vorticity(c_X,i,j,k+1,nsublist) - 30.*&
                  vorticity(c_X,i,j,k,nsublist)   + 16.*&
                  vorticity(c_X,i,j,k-1,nsublist) - &
                  vorticity(c_X,i,j,k-2,nsublist)

             tz(c_Y)= - &
                  vorticity(c_Y,i,j,k+2,nsublist) + 16.*&
                  vorticity(c_Y,i,j,k+1,nsublist) - 30.*&
                  vorticity(c_Y,i,j,k,nsublist)   + 16.*&
                  vorticity(c_Y,i,j,k-1,nsublist) - &
                  vorticity(c_Y,i,j,k-2,nsublist)

             tz(c_Z)=- &
                  vorticity(c_Z,i,j,k+2,nsublist) + 16.*&
                  vorticity(c_Z,i,j,k+1,nsublist) - 30.*&
                  vorticity(c_Z,i,j,k,nsublist)   + 16.*&
                  vorticity(c_Z,i,j,k-1,nsublist) - &
                  vorticity(c_Z,i,j,k-2,nsublist)

             diffusion = facx2*tx + facy2*ty + facz2*tz
             rhs(:,i,j,k,nsublist) = stretch + diffusion

          end do
       end do
    end do
  end subroutine computeRHS

  !> Computes strech.
  subroutine computeStretch(velocity,vorticity,stretch,resolution,step)

    !> Velocity field
    real(mk), dimension(:,:,:,:), pointer :: velocity
    !> vorticity field
    real(mk), dimension(:,:,:,:), pointer :: vorticity
    !> rhs, output
    real(mk), dimension(:,:,:,:), pointer :: stretch
    !> local mesh resolution
    integer,dimension(3),intent(in) :: resolution
    !> mesh step sizes
    real(mk),  dimension(3),intent(in) :: step

    integer :: i,j,k
    real(mk), dimension(3) :: tx, ty, tz
    real(mk) :: facx,facy,facz

    facx=1./(12.*step(c_X))
    facy=1./(12.*step(c_Y))
    facz=1./(12.*step(c_Z))

    do k=1,resolution(c_Z)
       do j=1,resolution(c_Y)
          do i=1,resolution(c_X)
             !stretch
             !------
             tx(c_X)= &
                  vorticity(c_X,i-2,j,k)*velocity(c_X,i-2,j,k)  - 8.*&
                  vorticity(c_X,i-1,j,k)*velocity(c_X,i-1,j,k)  + 8.*&
                  vorticity(c_X,i+1,j,k)*velocity(c_X,i+1,j,k)  -  &
                  vorticity(c_X,i+2,j,k)*velocity(c_X,i+2,j,k)

             tx(c_Y)= &
                  vorticity(c_X,i-2,j,k)*velocity(c_Y,i-2,j,k)  - 8.*&
                  vorticity(c_X,i-1,j,k)*velocity(c_Y,i-1,j,k)  + 8.*&
                  vorticity(c_X,i+1,j,k)*velocity(c_Y,i+1,j,k)  - &
                  vorticity(c_X,i+2,j,k)*velocity(c_Y,i+2,j,k)

             tx(c_Z)=&
                  vorticity(c_X,i-2,j,k)*velocity(c_Z,i-2,j,k)  - 8.*&
                  vorticity(c_X,i-1,j,k)*velocity(c_Z,i-1,j,k)  + 8.*&
                  vorticity(c_X,i+1,j,k)*velocity(c_Z,i+1,j,k)  - &
                  vorticity(c_X,i+2,j,k)*velocity(c_Z,i+2,j,k)

             ty(c_X)=&
                  vorticity(c_Y,i,j-2,k)*velocity(c_X,i,j-2,k) - 8.*&
                  vorticity(c_Y,i,j-1,k)*velocity(c_X,i,j-1,k) + 8.*&
                  vorticity(c_Y,i,j+1,k)*velocity(c_X,i,j+1,k) - &
                  vorticity(c_Y,i,j+2,k)*velocity(c_X,i,j+2,k)

             ty(c_Y)=&
                  vorticity(c_Y,i,j-2,k)*velocity(c_Y,i,j-2,k) - 8.*&
                  vorticity(c_Y,i,j-1,k)*velocity(c_Y,i,j-1,k) + 8.*&
                  vorticity(c_Y,i,j+1,k)*velocity(c_Y,i,j+1,k) - &
                  vorticity(c_Y,i,j+2,k)*velocity(c_Y,i,j+2,k)

             ty(c_Z)=&
                  vorticity(c_Y,i,j-2,k)*velocity(c_Z,i,j-2,k) - 8.*&
                  vorticity(c_Y,i,j-1,k)*velocity(c_Z,i,j-1,k) + 8.*&
                  vorticity(c_Y,i,j+1,k)*velocity(c_Z,i,j+1,k) - &
                  vorticity(c_Y,i,j+2,k)*velocity(c_Z,i,j+2,k)

             tz(c_X)=&
                  vorticity(c_Z,i,j,k-2)*velocity(c_X,i,j,k-2) - 8.*&
                  vorticity(c_Z,i,j,k-1)*velocity(c_X,i,j,k-1) + 8.*&
                  vorticity(c_Z,i,j,k+1)*velocity(c_X,i,j,k+1) - &
                  vorticity(c_Z,i,j,k+2)*velocity(c_X,i,j,k+2)

             tz(c_Y)=&
                  vorticity(c_Z,i,j,k-2)*velocity(c_Y,i,j,k-2) - 8.*&
                  vorticity(c_Z,i,j,k-1)*velocity(c_Y,i,j,k-1) + 8.*&
                  vorticity(c_Z,i,j,k+1)*velocity(c_Y,i,j,k+1) - &
                  vorticity(c_Z,i,j,k+2)*velocity(c_Y,i,j,k+2)

             tz(c_Z)=&
                  vorticity(c_Z,i,j,k-2)*velocity(c_Z,i,j,k-2) - 8.*&
                  vorticity(c_Z,i,j,k-1)*velocity(c_Z,i,j,k-1) + 8.*&
                  vorticity(c_Z,i,j,k+1)*velocity(c_Z,i,j,k+1) -  &
                  vorticity(c_Z,i,j,k+2)*velocity(c_Z,i,j,k+2)

             stretch(:,i,j,k) = facx*tx+facy*ty+facz*tz

          end do
       end do
    end do
  end subroutine computeStretch


  function iscloseto_3d(x,y,tol,step,error)

    logical :: iscloseto_3d
    real(mk),dimension(:,:,:), intent(in) :: x,y
    real(mk),intent(in) :: tol
    real(mk), dimension(3),intent(in) :: step
    real(mk), intent(out) :: error
    real(mk) :: res
    integer :: info
    real(mk) :: h3
    res = sum(abs(x-y)**2)
    iscloseto_3d = .false.
    call MPI_AllReduce(res,error,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,info)
    h3 = product(step(:))
    error = sqrt(h3*error)
    if(error < tol) iscloseto_3d = .True.

  end function iscloseto_3d

  function iscloseto_2d(x,y,tol,step,error)

    logical :: iscloseto_2d
    real(mk),dimension(:,:), intent(in) :: x,y
    real(mk),intent(in) :: tol
    real(mk), dimension(2),intent(in) :: step
    real(mk), intent(out) :: error
    real(mk) :: res
    integer :: info
    real(mk) :: h3
    res = sum(abs(x-y)**2)
    iscloseto_2d = .false.
    call MPI_AllReduce(res,error,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,info)
    h3 = product(step(:))
    error = sqrt(h3*error)
    if(error < tol) iscloseto_2d = .True.

  end function iscloseto_2d
  function iscloseto_3dc(x,y,tol,step,error)

    logical :: iscloseto_3dc
    complex(mk),dimension(:,:,:), intent(in) :: x,y
    real(mk),intent(in) :: tol
    real(mk), dimension(3),intent(in) :: step
    real(mk), intent(out) :: error
    real(mk) :: res
    integer :: info
    real(mk) :: h3
    res = sum(abs(real(x,mk)-real(y,mk))**2)
    res = res + sum(abs(aimag(x)-aimag(y))**2)
    iscloseto_3dc = .false.
    call MPI_AllReduce(res,error,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,info)
    h3 = product(step(:))
    error = sqrt(h3*error)
    if(error < tol) iscloseto_3dc = .True.

  end function iscloseto_3dc

  function iscloseto_2dc(x,y,tol,step,error)

    logical :: iscloseto_2dc
    complex(mk),dimension(:,:), intent(in) :: x,y
    real(mk),intent(in) :: tol
    real(mk), dimension(2),intent(in) :: step
    real(mk), intent(out) :: error
    real(mk) :: res
    integer :: info
    real(mk) :: h3
    res = sum(abs(real(x,mk)-real(y,mk))**2)
    res = res + sum(abs(aimag(x)-aimag(y))**2)
    iscloseto_2dc = .false.
    call MPI_AllReduce(res,error,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,info)
    h3 = product(step(:))
    error = sqrt(h3*error)
    if(error < tol) iscloseto_2dc = .True.

  end function iscloseto_2dc

!!$  !> compute euclidian norm of a 3D real field
!!$  function norm2_f3d(field,resolution,step)
!!$
!!$    real(mk), dimension(3) :: norm2
!!$    real(mk), dimension(:,:,:), pointer :: field
!!$    !> the local resolution
!!$    integer, dimension(3),intent(in) :: resolution
!!$    !> size of mesh step in each dir
!!$    real(mk), dimension(3),intent(in) :: step
!!$    real(mk),dimension(3) :: buffer
!!$    real(mk) :: h3
!!$    integer :: i,info
!!$    integer,dimension(dim3) :: nn
!!$    h3 = product(step(:))
!!$    nn = max(1,resolution-1)
!!$    buffer = 0.0
!!$    do i = 1,3
!!$       buffer(i) = sum(abs(field(1:nn(c_X),1:nn(c_Y),1:nn(c_Z))(:,:,:))**2)
!!$    end do
!!$    ! Norm is computed only on proc 0
!!$    call MPI_Reduce(buffer,norm2,dime,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,info)
!!$    if(rank == 0) norm2 = sqrt(h3*norm2)
!!$  end function norm2
!!$
!!$  function norm22d(field,resolution,step)
!!$    real(mk), dimension(2) :: norm22d
!!$    real(mk), dimension(:,:), pointer :: field
!!$    !> the local resolution
!!$    integer, dimension(2),intent(in) :: resolution
!!$    !> size of mesh step in each dir
!!$    real(mk), dimension(2),intent(in) :: step
!!$    real(mk),dimension(2) :: buffer
!!$    real(mk) :: h2
!!$    integer :: i,info
!!$    integer,dimension(2) :: nn
!!$    h2 = product(step(:))
!!$    nn = max(1,resolution-1)
!!$    buffer = 0.0
!!$    do i = 1,2
!!$       buffer(i) = sum(abs(field(1:nn(c_X),1:nn(c_Y)))**2)
!!$    end do
!!$    ! Norm is computed only on proc 0
!!$    call MPI_Reduce(buffer,norm22d,dime,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,info)
!!$    if(rank == 0) norm22d = sqrt(h2*norm22d)
!!$
!!$  end function norm22d
!!$
!!$  function norm22dC(field,resolution,step)
!!$    real(mk), dimension(2) :: norm22d
!!$    complex(mk), dimension(:,:), pointer :: field
!!$    !> the local resolution
!!$    integer, dimension(2),intent(in) :: resolution
!!$    !> size of mesh step in each dir
!!$    real(mk), dimension(2),intent(in) :: step
!!$    real(mk),dimension(2) :: buffer
!!$    real(mk) :: h2
!!$    integer :: i,info
!!$    integer,dimension(2) :: nn
!!$    h2 = product(step(:))
!!$    nn = max(1,resolution-1)
!!$    buffer = 0.0
!!$    do i = 1,2
!!$       buffer(i) = sum(abs(field(1:nn(c_X),1:nn(c_Y)))**2)
!!$    end do
!!$    ! Norm is computed only on proc 0
!!$    call MPI_Reduce(buffer,norm22d,dime,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,info)
!!$    if(rank == 0) norm22d = sqrt(h2*norm22d)
!!$
!!$  end function norm22dC
!!$
!!$  function normInf(field,resolution)
!!$    real(mk), dimension(dime) :: normInf
!!$    real(mk), dimension(:,:,:,:), pointer :: field
!!$    !> the local resolution
!!$    integer, dimension(dim3),intent(in) :: resolution
!!$    !> size of mesh step in each dir
!!$    real(mk),dimension(dime) :: buffer
!!$    integer :: i,info
!!$    do i = 1,dime
!!$       buffer(i) = maxval(abs(field(i,1:resolution(c_X),1:resolution(c_Y),1:resolution(c_Z))))
!!$    end do
!!$    ! Norm is computed only on proc 0
!!$    call MPI_Reduce(buffer,normInf,dime,MPI_DOUBLE_PRECISION,MPI_MAX,0,MPI_COMM_WORLD,info)
!!$
!!$  end function normInf

  function cross_prod(v1,v2)
    real(mk), dimension(3), intent(in) :: v1
    real(mk), dimension(3), intent(in) :: v2
    real(mk), dimension(3) :: cross_prod

    cross_prod(c_X) = v1(c_Y)*v2(c_Z)-v1(c_Z)*v2(c_Y)
    cross_prod(c_Y) = v1(c_Z)*v2(c_X)-v1(c_X)*v2(c_Z)
    cross_prod(c_Z) = v1(c_X)*v2(c_Y)-v1(c_Y)*v2(c_X)

  end function cross_prod

  !! compute nabla(vect) = [ d/dx vect,d/dy vect, d/dz vect]
  !! at point i,j,k of the grid using central finite difference scheme, order 2
  function nabla(vect,i,j,k,step)
    real(mk), dimension(3) :: nabla
    !> Vector to be differentiate
    real(mk), dimension(:,:,:),intent(in),pointer:: vect
    !> indices of the considered grid point
    integer, intent(in) :: i,j,k
    !> mesh step size
    real(mk),dimension(3),intent(in) :: step

    ! --- nabla vect_component ---
    nabla(c_X) = (vect(i+1,j,k) - vect(i-1,j,k))/(2.*step(c_X)) ! d/dx u_component
    nabla(c_Y) = (vect(i,j+1,k) - vect(i,j-1,k))/(2.*step(c_Y)) ! d/dy u_component
    nabla(c_Z) = (vect(i,j,k+1) - vect(i,j,k-1))/(2.*step(c_Z)) ! d/dz u_component
  end function nabla

  !> Differentiate a vector with respect to x (FD, centered)
  function diffX(vect,i,j,k,step)
    real(mk) :: diffX
    !> The vector to be differentiate
    real(mk),dimension(:,:,:),intent(in),pointer::vect
    !> indices of the considered grid point
    integer,intent(in)::i,j,k
    !> step size in x direction
    real(mk),intent(in)::step
    diffX = (vect(i+1,j,k)-vect(i-1,j,k))/(2.*step)

  end function diffX

  !> Differentiate a vector with respect to y
  function diffY(vect,i,j,k,step)
    real(mk) :: diffY
    !> The vector to be differentiate
    real(mk),dimension(:,:,:),intent(in),pointer::vect
    !> indices of the considered grid point
    integer,intent(in)::i,j,k
    !> step size in y direction
    real(mk),intent(in)::step

    diffY = (vect(i,j+1,k)-vect(i,j-1,k))/(2.*step)
  end function diffY

  !> Differentiate a vector with respect to z
  function diffZ(vect,i,j,k,step)
    real(mk) :: diffZ
    !> The vector to be differentiate
    real(mk),dimension(:,:,:),intent(in),pointer::vect
    !> indices of the considered grid point
    integer,intent(in)::i,j,k
    !> step size in z direction
    real(mk),intent(in)::step

    diffZ = (vect(i,j,k+1)-vect(i,j,k-1))/(2.*step)
  end function diffZ

  !> Compute the laplacian of a 3d field
  function laplacian(vect,i,j,k,step)
    real(mk) :: laplacian
    !> The vector to be differentiate
    real(mk),dimension(:,:,:),intent(in),pointer::vect
    !> indices of the considered grid point
    integer,intent(in)::i,j,k
    !> mesh step size
    real(mk),dimension(3),intent(in) :: step

    laplacian = (vect(i+1,j,k) + vect(i-1,j,k) - 2.*vect(i,j,k))/(step(c_X)**2) &
         + (vect(i,j+1,k) + vect(i,j-1,k)- 2.*vect(i,j,k))/(step(c_Y)**2) &
         + (vect(i,j,k+1) + vect(i,j,k-1) - 2.*vect(i,j,k))/(step(c_Z)**2)

  end function laplacian

end module vectorcalculus
