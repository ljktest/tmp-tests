!USEFORTEST interpolation
!USEFORTEST advec
!> @addtogroup part
!! @{
!------------------------------------------------------------------------------
!
! MODULE: advec_common_velo
!
!
! DESCRIPTION:
!> The module ``advec_common_velo'' gather function and subroutines used to interpolate
!! velocity at particle position which are not specific to a direction
!! @details
!! This module gathers functions and routines used to advec scalar which are not
!! specific to a direction. This is a parallel implementation using MPI and
!! the cartesien topology it provides. It also contains the variables common to
!! the solver along each direction and other generic variables used for the
!! advection based on the particle method.
!!
!! Except for testing purpose, this module is not supposed to be used by the
!! main code but only by the other advection module. More precisly, an final user
!! must only used the generic "advec" module wich contain all the interface to
!! solve the advection equation with the particle method, and to choose the
!! remeshing formula, the dimensionnal splitting and everything else. Except for
!! testing purpose, the other advection modules have only to include
!! "advec_common".
!!
!! The module "test_advec" can be used in order to validate the procedures
!! embedded in this module.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module Interpolation_velo

    use precision_tools
    use cart_topology
    use mpi, only: MPI_STATUS_SIZE, MPI_STATUSES_IGNORE
    implicit none

    public


    ! ===== Public procedures =====
    !----- To interpolate velocity -----

    ! ===== Interpolation formula =====
    ! For newer version of GCC (>= 4.7) replace these subroutine by function
    ! which return the weight (array) with position as input argument. It works
    ! with gcc 4.7 and later and IBM compiler, and intel too.
    !> Generic subroutine (pointer intialize to the choosen formula)
    procedure(weight_M4), pointer,  public :: get_weight => null()
    !> Specific interpolation formula
    public :: weight_M4, weight_Mprime4, weight_Lambda4_4, weight_linear

    ! ===== Private variables =====
    character(len=4), protected :: interpol = 'Mp4'
    integer, protected :: stencil_size = 4
    integer, protected :: stencil_g = 1
    integer, protected :: stencil_d = 2


contains

! ===== Public procedure =====

! ============================================================
! ====================     Initialisation ====================
! ============================================================
!> To choose interpolation formula
subroutine interpol_init(formula, verbose)

    character(len=*), optional, intent(in)  ::  formula
    logical, optional, intent(in)           ::  verbose

    logical :: verbosity

    if(present(formula)) then
      interpol = formula
    else
      interpol = 'Mp4'
    end if

    verbosity = .false.
    if(present(verbose)) verbosity = verbose

    select case(trim(interpol))
    case('lin')
      stencil_size = 2
      stencil_d = 1
      stencil_g = 0
      get_weight => weight_linear
      if ((cart_rank==0).and.(verbosity)) &
        & write(*,'(6x,a)') '============= Interpolation = linear  ==========='
    case('L4_4')
      stencil_size = 6
      stencil_d = 3
      stencil_g = 2
      get_weight => weight_Lambda4_4
      if ((cart_rank==0).and.(verbosity)) &
        & write(*,'(6x,a)') '============= Interpolation = Lambda 4,4 ==========='
    case('M4')
      stencil_size = 4
      stencil_d = 2
      stencil_g = 1
      get_weight => weight_M4
      if ((cart_rank==0).and.(verbosity)) &
        & write(*,'(6x,a)') '============= Interpolation = M 4 ==========='
    case default
      stencil_size = 4
      stencil_d = 2
      stencil_g = 1
      get_weight => weight_Mprime4
      if ((cart_rank==0).and.(verbosity)) &
        & write(*,'(6x,a)') '============= Interpolation = Mprime 4 ==========='
    end select

end subroutine interpol_init

! ==========================================================================================
! ====================     Interpolation of each velocity component     ====================
! ==========================================================================================
! Except for test purpose, only these brick must

! For advection solver
subroutine Interpol_2D_3D_vect(dx_f, dx_c, Vx, Vy, Vz, Vx_c, Vx_f, Vy_c, Vy_f, Vz_c, Vz_f)

  real(WP), dimension(3), intent(in)        :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: Vx, Vy, Vz
  real(WP), dimension(:,:,:),intent(inout)  :: Vx_c, Vy_c, Vz_c
  real(WP), dimension(:,:,:),intent(inout)  :: Vx_f, Vy_f, Vz_f

  call Interpol_2D_vect(dx_f, dx_c, Vx, Vy, Vz, Vx_c, Vy_c, Vz_c)

  call Inter_FirstDir_no_com(Vx_c, dx_c(1), Vx_f, dx_f(1))

  call Inter_FirstDir_com(2, Vy_c, dx_c(2), Vy_f, dx_f(2))
  call Inter_FirstDir_com(3, Vz_c, dx_c(3), Vz_f, dx_f(3))

end subroutine Interpol_2D_3D_vect



!> Interpolate each componnent of a vector along a transverse direction.
!!    @param[in]        dx_c        = space step on the coarse grid (for last direction)
!!    @param[in]        dx_f        = space step on the fine grid (for last direction)
!!    @param[in]        Vx          = vector component along X
!!    @param[in]        Vy          = vector component along Y
!!    @param[in]        Vz          = vector component along Z
!!    @param[out]       InterX      = interpolation ov Vx along Y and Z
!!    @param[out]       InterY      = interpolation ov VY along X and Z
!!    @param[out]       InterZ      = interpolation ov VZ along X and Y
subroutine Interpol_2D_vect(dx_f, dx_c, Vx, Vy, Vz, InterX, InterY, InterZ)

  real(WP), dimension(3), intent(in)        :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: Vx, Vy, Vz
  real(WP), dimension(:,:,:),intent(inout)  :: InterX, InterY, InterZ
  ! Local variable
  real(WP), dimension(2)                    :: d_f, d_c

  ! For Vx, interpolation along Y and Z
  call Inter_YZ(Vx, dx_c(2:3), InterX, dx_f(2:3))
  ! For Vy, interpolation along Z (with communications) then along X (no communication required)
  d_c = (/dx_c(1), dx_c(3)/)
  d_f = (/dx_f(1), dx_f(3)/)
  call Inter_XZ_permut(Vy, d_c, InterY, d_f)
  ! For Vz, interpolation along Y (with communications) then along X (no communication required)
  call Inter_XY_permut(Vz, d_c(1:2), InterZ, d_f(1:2))

end subroutine Interpol_2D_vect

!> 3D interpolation of a field to a finer grid - no transpositions.
!!    @param[in]        V_coarse    = velocity to interpolate along the last direction
!!    @param[in]        dx_c        = space step on the coarse grid (for last direction)
!!    @param[in]        V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for last direction)
subroutine Interpol_3D(V_coarse, dx_c, V_fine, dx_f)

  real(WP), dimension(3), intent(in)        :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local variable
  real(WP), dimension(size(V_fine,1),size(V_coarse,2),size(V_coarse,3))    :: V_middle ! to save result of interpolation along X

  ! Interpolate along X
  call Inter_FirstDir_no_com(V_coarse, dx_c(1), V_middle, dx_f(1))

  ! And then along Y and Z
  call Inter_YZ(V_middle, dx_c(2:3), V_fine, dx_f(2:3))

end subroutine Interpol_3D

! ========================================================================
! ====================        2D interpolation        ====================
! ========================================================================

!> Interpolate a field (ordonnate along X,Y,Z) along X and Y-axis
!!    @param[in]        V_coarse    = velocity to interpolate along the last direction
!!    @param[in]        dx_c        = space step on the coarse grid (for last direction)
!!    @param[in]        V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for last direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along the third
!! direction.
subroutine Inter_XY(V_coarse, dx_c, V_fine, dx_f)

  

  ! Input/Output
  real(WP), dimension(2), intent(in)        :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local
  real(WP), dimension(size(V_coarse,1),size(V_coarse,3),size(V_coarse,2))  :: V_permut ! permutation required for first interpolation
  real(WP), dimension(size(V_coarse,1),size(V_fine,2),size(V_coarse,3))    :: V_middle ! to save result of interpolation along Z + permutation
  integer :: ind  ! loop indice

  ! Check field sizes
  if(.not.(size(V_fine,3)==size(V_coarse,3))) then
    write(*,'(a)') '[ERROR] Interpolation along XY : V_coarse and V_fine does not have the same resolution along Z axis'
    stop
  end if

  ! Permutation to prepare first interpolation
  do ind = 1, size(V_coarse,3)
    V_permut(:,ind,:) = V_coarse(:,:,ind)
  end do

  ! Interpolation along last direction = Y-direction + permutation to re-order indices
  call Inter_LastDir_Permut_com(2, V_permut, dx_c(2), V_middle, dx_f(2))

  ! Interpolation along X = first direction
  call Inter_FirstDir_no_com(V_middle, dx_c(1), V_fine, dx_f(1))

end subroutine Inter_XY


!> Interpolate a field (ordannate along X,Y,Z) along X and Y-axis + permutation
!! in order to get a field sotred along (Z,X,Y)
!!    @param[in]        V_coarse    = velocity to interpolate along the last direction
!!    @param[in]        dx_c        = space step on the coarse grid (for last direction)
!!    @param[in]        V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for last direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along the third
!! direction.
subroutine Inter_XY_permut(V_coarse, dx_c, V_fine, dx_f)

  

  ! Input/Output
  real(WP), dimension(2), intent(in)        :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local
  real(WP), dimension(size(V_coarse,1),size(V_coarse,3),size(V_coarse,2))  :: V_permut ! permutation required for first interpolation
  real(WP), dimension(size(V_coarse,1),size(V_coarse,3),size(V_fine,3))    :: V_middle ! to save result of interpolation along Z + permutation
  integer :: ind  ! loop indice

  ! Check field sizes
  if(.not.(size(V_fine,1)==size(V_coarse,3))) then
    write(*,'(a)') '[ERROR] Interpolation along XY : V_coarse and V_fine does not have the same resolution along Z axis'
    stop
  end if

  ! Permutation to prepare first interpolation
  do ind = 1, size(V_coarse,3)
    V_permut(:,ind,:) = V_coarse(:,:,ind)
  end do

  ! Interpolation along last direction = Y-direction
  call Inter_LastDir_com(2, V_permut, dx_c(2), V_middle, dx_f(2))

  ! Interpolation along X = first direction  + permutation to re-order indices
  call Inter_FirstDir_Permut_no_com(V_middle, dx_c(1), V_fine, dx_f(1))

end subroutine Inter_XY_permut


!> Interpolate a field (ordannate along X,Y,Z) along Y and Z-axis
!!    @param[in]        V_coarse    = velocity to interpolate along Y and Z directions
!!    @param[in]        dx_c        = space step on the coarse grid (for second and last directions)
!!    @param[in,out]    V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for second and last direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along the first direction.
subroutine Inter_YZ(V_coarse, dx_c, V_fine, dx_f)

  

  ! Input/Output
  real(WP), dimension(2), intent(in)        :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local
  real(WP), dimension(size(V_coarse,1),size(V_fine,3),size(V_coarse,2))    :: V_middle ! to save result of interpolation along Z + permutation


  ! Check if array have the right size
  if(.not.(size(V_fine,1)==size(V_coarse,1))) then
    write(*,'(a)') '[ERROR] Interpolation along YZ : V_coarse and V_fine does not have the same resolution along first direction'
    stop
  end if

  ! Interpolation along Z + permutation between Y and Z
  call Inter_LastDir_Permut_com(3, V_coarse, dx_c(2), V_middle, dx_f(2))

  ! Interpolation along Y(=third direction thanks to previous permutation) + permutation between Y and Z
  call Inter_LastDir_Permut_com(2, V_middle, dx_c(1), V_fine, dx_f(1))

end subroutine Inter_YZ


!> Interpolate a field (ordannate along X,Y,Z) along X and Z-axis
!!    @param[in]        V_coarse    = velocity to interpolate along X and Z directions
!!    @param[in]        dx_c        = space step on the coarse grid (for first and last directions)
!!    @param[in,out]    V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for first and last direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along the second direction.
!! direction.
subroutine Inter_XZ(V_coarse, dx_c, V_fine, dx_f)

  

  ! Input/Output
  real(WP), dimension(2), intent(in)        :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local
  real(WP), dimension(size(V_coarse,1),size(V_coarse,2),size(V_fine,3))    :: V_middle ! to save result of interpolation along Z + permutation


  ! Check if array have the right size
  if(.not.(size(V_fine,2)==size(V_coarse,2))) then
    write(*,'(a)') '[ERROR] Interpolation along XZ : V_coarse and V_fine does not have the same resolution along first direction'
    stop
  end if

  ! Interpolation along Z
  call Inter_LastDir_com(3, V_coarse, dx_c(2), V_middle, dx_f(2))

  ! Interpolation along X
  call Inter_FirstDir_no_com(V_middle, dx_c(1), V_fine, dx_f(1))

end subroutine Inter_XZ


!> Interpolate a field (ordannate along X,Y,Z) along X and Z-axis and get a
!! field stored in function of (Y,X,Z)
!!    @param[in]        V_coarse    = velocity to interpolate along X and Z directions
!!    @param[in]        dx_c        = space step on the coarse grid (for first and last directions)
!!    @param[in,out]    V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for first and last direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along the X-axis.
subroutine Inter_XZ_permut(V_coarse, dx_c, V_fine, dx_f)

  

  ! Input/Output
  real(WP), dimension(2), intent(in)        :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local
  real(WP), dimension(size(V_coarse,1),size(V_coarse,2),size(V_fine,3))    :: V_middle ! to save result of interpolation along Z + permutation


  ! Check if array have the right size
  if(.not.(size(V_fine,1)==size(V_coarse,2))) then
    write(*,'(a)') '[ERROR] Interpolation along XZ_permut : V_coarse and V_fine does not have the same resolution along first direction'
    stop
  end if

  ! Interpolation along Z
  call Inter_LastDir_com(3, V_coarse, dx_c(2), V_middle, dx_f(2))

  ! Interpolation along X
  call Inter_FirstDir_Permut_no_com(V_middle, dx_c(1), V_fine, dx_f(1))

end subroutine Inter_XZ_permut

! =================================================================================
! ====================   Elementary brick = 1D interpolation   ====================
! =================================================================================
! Do not use directly this, except for test purpose. If you want to use it,
! check the input size (not checked here, because already tested in function
! wich call them, when needed)

!> Interpolate a field along the last direction - with communication : V_fine(i,j,k) = interpolation(V_coarse(i,j,k_interpolation))
!!    @param[in]        dir         = last directions (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        V_coarse    = velocity to interpolate along the last direction
!!    @param[in]        dx_c        = space step on the coarse grid (for last direction)
!!    @param[in,out]    V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for last direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along second and third
!! directions.
subroutine Inter_LastDir_com(dir, V_coarse, dx_c, V_fine, dx_f)

  

  ! Input/Output
  integer, intent(in)                       :: dir
  real(WP), intent(in)                      :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local variable
  real(WP), dimension(:,:,:), allocatable   :: V_beg, V_end ! ghost values of velocity
  real(WP), dimension(stencil_size)         :: weight       ! interpolation weight
  integer               :: i,ind,i_bis, V_ind   ! some loop indices
  integer               :: ind_max, ind_min, ind_limit, ind_limit_2
  real(WP)              :: pos
  integer               :: N_coarse, N_fine     ! number of grid points
  integer               :: com_pos              ! to deal with multiple communications - if (stencil size)/2 > local size of coarse grid
                                                ! = position where ghost values are recieved in V_beg and  V_end
  integer, dimension(2) :: com_nb               ! number of communcation (if (stencil size)/2 > local size of coarse grid)
  integer, dimension(2) :: com_size             ! size of mpi communication for ghost points
  integer               :: ierr                 ! mpi error code
  integer, dimension(:),allocatable   :: beg_request  ! mpi communication request (handle) of nonblocking receive
  integer, dimension(:),allocatable   :: end_request  ! mpi communication request (handle) of nonblocking receive
  real(WP), parameter :: eps = 1e-6

  ! Initialisation
  com_size(1) = size(V_coarse,1)*size(V_coarse,2)
  N_coarse = size(V_coarse,3)
  N_fine = size(V_fine,3)
  ! ind_max = max(indice ind on fine grid as V_fine(i) can be computed without communication)
  !         = max{ind : V_ind=floor[(ind-1)*dx_f/dx_c]+1  <=  (N_coarse-stencil_d)            }
  !         = max{ind : V_ind=floor[(ind-1)*dx_f/dx_c]    <=  (N_coarse-stencil_d-1)          }
  !         = max{ind : pos  =      (ind-1)*dx_f/dx_c     <   (N_coarse-stencil_d-1)+1        }
  !         = max{ind :             (ind-1)               <   (N_coarse-stencil_d)*dx_c/dx_f  }
  !         = max{ind : ind < [(N_coarse-stencil_d)*(dx_c/dx_f)]+1}
  ! Define real_max = [(N_coarse-stencil_d)*(dx_c/dx_f)] as a real. One gets:
  ! ind_max = max{ind integer as : ind < real_max+1} and thus
  ! ind_max = real_max            if real_max is an integer
  !           floor(real_max+1)   else
  ! ie ind_max = ceiling(real_max)
  !ind_max = ceiling((N_coarse-stencil_d)*dx_c/dx_f)
  ind_max = ceiling(((N_coarse-stencil_d)*dx_c/dx_f)-eps)  ! To avoid numerical error and thus segmentation fault
  ! ind_min = min(indice ind on fine grid as V_fine(i) can be computed without communication)
  !         = min{ind : V_ind=floor[(ind-1)*dx_f/dx_c]+1 - stencil_g > 0}
  !         = min{ind : V_ind=floor[(ind-1)*dx_f/dx_c] >  stencil_g -1  }
  !         = min{ind : V_ind=floor[(ind-1)*dx_f/dx_c] >= stencil_g     }
  !         = min{ind :             (ind-1)*dx_f/dx_c  >= stencil_g     }
  !         = min{ind : pos=        (ind-1)*dx_f       >= stencil_g*dx_c}
  !         = min{ind : ind >= (stencil_g*dx_c/dx_f) + 1}
  !         = ceiling[(stencil_g*dx_c/dx_f) + 1]
  ind_min = ceiling((stencil_g)*dx_c/dx_f)+1 ! here numerical truncature can not lead to seg. fault

  ! ==== Communication ====
  if(stencil_g>0) then
    allocate(V_beg(size(V_coarse,1),size(V_coarse,2),stencil_g))
    com_nb(1) = ceiling(real(stencil_g, WP)/N_coarse) ! number of required communication to get ghost
    allocate(beg_request(com_nb(1)))
    com_pos = stencil_g+1              ! i = 1 + missing (or remainding) ghost lines
    com_size(2) = com_size(1)*N_coarse
    ! Except for last communication, send all local coarse data.
    ! Note that it happen if local coarse grid containt less than (stencil_size/2)
      ! points along current direction (ie if coarse grid is very coarse)
    do ind = 1, com_nb(1)-1
      com_pos = com_pos - N_coarse  ! = 1 + missing ghost lines after this step
      ! Communication
      call Mpi_Irecv(V_beg(1,1,com_pos),com_size(2),MPI_REAL_WP, &
        & neighbors(dir,-ind), 100+ind, D_comm(dir), beg_request(ind), ierr)
      call Mpi_Send(V_coarse(1,1,1),com_size(2),MPI_REAL_WP, &
        & neighbors(dir,ind), 100+ind, D_comm(dir), ierr)
    end do
    ! Last communication to complete "right" ghost (begining points)
    ! We use that missing ghost lines = com_pos - 1
    com_size(2) = com_size(1)*(com_pos-1)
    call Mpi_Irecv(V_beg(1,1,1),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,-com_nb(1)), 1, D_comm(dir), beg_request(com_nb(1)), ierr)
    call Mpi_Send(V_coarse(1,1,N_coarse-com_pos+2),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,com_nb(1)), 1, D_comm(dir), ierr)
  end if

  if(stencil_d>0) then
    allocate(V_end(size(V_coarse,1),size(V_coarse,2),stencil_d))
    com_nb(2) = ceiling(real(stencil_d, WP)/N_coarse) ! number of required communication to get ghost
    allocate(end_request(com_nb(2)))
    com_pos = 1   ! Reception from next processus is done in position 1
    com_size(2) = com_size(1)*N_coarse
    ! Except for last communication, send all local coarse data.
    ! Note that it happen if local coarse grid containt less than (stencil_size/2)
      ! points along current direction (ie if coarse grid is very coarse)
    do ind = 1, com_nb(2)-1
      ! Communication
      call Mpi_Irecv(V_end(1,1,com_pos),com_size(2),MPI_REAL_WP, &
        & neighbors(dir,ind), 200+ind, D_comm(dir), end_request(ind), ierr)
      call Mpi_Send(V_coarse(1,1,1),com_size(2),MPI_REAL_WP, &
        & neighbors(dir,-ind), 200+ind, D_comm(dir), ierr)
      ! next com_pos = (ind*N_coarse)+1 = com_pos + N_coarse
      com_pos = com_pos + N_coarse
      end do
    ! Last step
    ! Note that: missing ghost lines = stencil_d - (com_nb-1)*N_coarse
    com_size(2) = com_size(1)*(stencil_d-((com_nb(2)-1)*N_coarse))
    ! Perform communication
    call Mpi_Irecv(V_end(1,1,com_pos),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,com_nb(2)), 2, D_comm(dir), end_request(com_nb(2)), ierr)
    ! Send data
    call Mpi_Send(V_coarse(1,1,1),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,-com_nb(2)), 2, D_comm(dir), ierr)
  end if

  ! ==== Interpolation ====
  ! -- For middle points --
  do ind = ind_min, ind_max
    pos = (ind-1)*dx_f/dx_c
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    V_ind = V_ind - stencil_g
    V_fine(:,:,ind) = weight(1)*V_coarse(:,:,V_ind)
    do i = 1, (stencil_size - 1)
      V_fine(:,:,ind) = V_fine(:,:,ind) + weight(i+1)*V_coarse(:,:,V_ind+i)
    end do
  end do
  ! -- Wait for communication completion before dealing with the end --
  if(stencil_g>0) then
    call mpi_waitall(com_nb(1),beg_request, MPI_STATUSES_IGNORE, ierr)
    deallocate(beg_request)
  end if
  if(stencil_d>0) then
    call mpi_waitall(com_nb(2),end_request, MPI_STATUSES_IGNORE, ierr)
    deallocate(end_request)
  end if
  ! -- For begining --
  ! Use that interpolation formula are exact - no computation for the first point of each line
  V_fine(:,:,1) = V_coarse(:,:,1)
  ! For other first points
  do ind = 2, min(ind_min-1, N_fine)  ! Be carful, in some massively parrallel context, ind_min could bigger than N_fine +1
    pos = (ind-1)*dx_f/dx_c
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    !V_ind = V_ind - stencil_g
    !V_fine(:,:,ind) = weight(1)*V_beg(:,:,V_ind+stencil_g) ! Array start from 1
    V_fine(:,:,ind) = weight(1)*V_beg(:,:,V_ind) ! Array start from 1
    ind_limit = stencil_g - V_ind + 1
    do i = 2, ind_limit
      !V_fine(:,:,ind) = V_fine(:,:,ind) + weight(i)*V_beg(:,:,V_ind+stencil_g+i)
      V_fine(:,:,ind) = V_fine(:,:,ind) + weight(i)*V_beg(:,:,V_ind-1+i) ! first point in V_beg stands for 1-stencil_g position
    end do
    ! If N_coarse < stencil_size, last interpolation points does not belong to local domain
      ! but to domain of processus of coordinnates = (mine+1) inside the mpi-topology.
      ! Then we search in V_end for values at last interpolation point,
    ind_limit_2 = min(stencil_size, N_coarse+ind_limit) ! for very coarse grid, stencil size could be bigger than N_coarse
    do i_bis = ind_limit+1, ind_limit_2
      ! We look for first local value of V_coarse at position (:,:,1) ! (array starts at 1)
      V_fine(:,:,ind) = V_fine(:,:,ind) + weight(i_bis)*V_coarse(:,:,i_bis-ind_limit)
    end do
    ! Values in V_end
    do i_bis = ind_limit_2+1, stencil_size
      V_fine(:,:,ind) = V_fine(:,:,ind) + weight(i_bis)*V_end(:,:,i_bis-ind_limit_2)
    end do
  end do
  ! -- For point of at the end of a line along the current direction --
  ! If ind_max<= ind_min (ie if stencil_size>N_coarse), computation are already done for first point
  do ind = max(ind_max+1,ind_min), N_fine
    pos = (ind-1)*dx_f/dx_c
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    !V_ind = V_ind - stencil_g
    V_ind = V_ind - stencil_g-1
    V_fine(:,:,ind) = weight(1)*V_coarse(:,:,V_ind+1)
    ind_limit = min((stencil_size),N_coarse-V_ind)
    do i = 2, ind_limit
      V_fine(:,:,ind) = V_fine(:,:,ind) + weight(i)*V_coarse(:,:,V_ind+i)
    end do
    V_ind = V_ind - N_coarse
    do i_bis = ind_limit+1, stencil_size
      V_fine(:,:,ind) = V_fine(:,:,ind) + weight(i_bis)*V_end(:,:,i_bis+V_ind)
    end do
  end do

  ! Free memory
  if(stencil_d>0) deallocate(V_end)
  if(stencil_g>0) deallocate(V_beg)

end subroutine Inter_LastDir_com


!> Interpolate a field along the last direction and permut second and third directions - with communication : V_fine(i,j,k) = interpolation(V_coarse(i,j,k_interpolation))
!!    @param[in]        dir         = last directions (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        V_coarse    = velocity to interpolate along the last direction
!!    @param[in]        dx_c        = space step on the coarse grid (for last direction)
!!    @param[in]        V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for last direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along second and third
!! direction.
subroutine Inter_LastDir_Permut_com(dir, V_coarse, dx_c, V_fine, dx_f)

  

  ! Input/Output
  integer, intent(in)                       :: dir
  real(WP), intent(in)                      :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local variable
  real(WP), dimension(:,:,:), allocatable   :: V_beg, V_end ! ghost values of velocity
  real(WP), dimension(stencil_size)         :: weight       ! interpolation weight
  integer               :: i,ind,i_bis, V_ind   ! some loop indices
  integer               :: ind_max, ind_min, ind_limit, ind_limit_2
  real(WP)              :: pos
  integer               :: N_coarse, N_fine     ! number of grid points
  integer               :: com_pos              ! to deal with multiple communications - if (stencil size)/2 > local size of coarse grid
                                                ! = position where ghost values are recieved in V_beg and  V_end
  integer, dimension(2) :: com_nb               ! number of communcation (if (stencil size)/2 > local size of coarse grid)
  integer, dimension(2) :: com_size             ! size of mpi communication for ghost points
  integer               :: ierr                 ! mpi error code
  integer, dimension(:),allocatable   :: beg_request  ! mpi communication request (handle) of nonblocking receive
  integer, dimension(:),allocatable   :: end_request  ! mpi communication request (handle) of nonblocking receive

  ! Initialisation
  com_size(1) = size(V_coarse,1)*size(V_coarse,2)
  N_coarse = size(V_coarse,3)
  N_fine = size(V_fine,2)
  ind_max = ceiling(((N_coarse-stencil_d)*dx_c/dx_f)-1e-6)  ! To avoid numerical error and thus segmentation fault
  ind_min = ceiling((stencil_g)*dx_c/dx_f)+1

  ! ==== Communication ====
  if(stencil_g>0) then
    allocate(V_beg(size(V_coarse,1),size(V_coarse,2),stencil_g))
    com_nb(1) = ceiling(real(stencil_g, WP)/N_coarse) ! number of required communication to get ghost
    allocate(beg_request(com_nb(1)))
    com_pos = stencil_g+1              ! i = 1 + missing (or remainding) ghost lines
    com_size(2) = com_size(1)*N_coarse
    ! Except for last communication, send all local coarse data.
    ! Note that it happen if local coarse grid containt less than (stencil_size/2)
      ! points along current direction (ie if coarse grid is very coarse)
    do ind = 1, com_nb(1)-1
      com_pos = com_pos - N_coarse  ! = 1 + missing ghost lines after this step
      ! Communication
      call Mpi_Irecv(V_beg(1,1,com_pos),com_size(2),MPI_REAL_WP, &
        & neighbors(dir,-ind), 100+ind, D_comm(dir), beg_request(ind), ierr)
      call Mpi_Send(V_coarse(1,1,1),com_size(2),MPI_REAL_WP, &
        & neighbors(dir,ind), 100+ind, D_comm(dir), ierr)
    end do
    ! Last communication to complete "right" ghost (begining points)
    ! We use that missing ghost lines = com_pos - 1
    com_size(2) = com_size(1)*(com_pos-1)
    call Mpi_Irecv(V_beg(1,1,1),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,-com_nb(1)), 1, D_comm(dir), beg_request(com_nb(1)), ierr)
    call Mpi_Send(V_coarse(1,1,N_coarse-com_pos+2),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,com_nb(1)), 1, D_comm(dir), ierr)
  end if

  if(stencil_d>0) then
    allocate(V_end(size(V_coarse,1),size(V_coarse,2),stencil_d))
    com_nb(2) = ceiling(real(stencil_d, WP)/N_coarse) ! number of required communication to get ghost
    allocate(end_request(com_nb(2)))
    com_pos = 1   ! Reception from next processus is done in position 1
    com_size(2) = com_size(1)*N_coarse
    ! Except for last communication, send all local coarse data.
    ! Note that it happen if local coarse grid containt less than (stencil_size/2)
      ! points along current direction (ie if coarse grid is very coarse)
    do ind = 1, com_nb(2)-1
      ! Communication
      call Mpi_Irecv(V_end(1,1,com_pos),com_size(2),MPI_REAL_WP, &
        & neighbors(dir,ind), 200+ind, D_comm(dir), end_request(ind), ierr)
      call Mpi_Send(V_coarse(1,1,1),com_size(2),MPI_REAL_WP, &
        & neighbors(dir,-ind), 200+ind, D_comm(dir), ierr)
      ! next com_pos = (ind*N_coarse)+1 = com_pos + N_coarse
      com_pos = com_pos + N_coarse
      end do
    ! Last step
    ! Note that: missing ghost lines = stencil_d - (com_nb-1)*N_coarse
    com_size(2) = com_size(1)*(stencil_d-((com_nb(2)-1)*N_coarse))
    ! Perform communication
    call Mpi_Irecv(V_end(1,1,com_pos),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,com_nb(2)), 2, D_comm(dir), end_request(com_nb(2)), ierr)
    ! Send data
    call Mpi_Send(V_coarse(1,1,1),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,-com_nb(2)), 2, D_comm(dir), ierr)
  end if

  ! ==== Interpolation ====
  ! -- For middle points --
  do ind = ind_min, ind_max
    pos = (ind-1)*dx_f/dx_c
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    V_ind = V_ind - stencil_g
    V_fine(:,ind,:) = weight(1)*V_coarse(:,:,V_ind)
    do i = 1, (stencil_size - 1)
      V_fine(:,ind,:) = V_fine(:,ind,:) + weight(i+1)*V_coarse(:,:,V_ind+i)
    end do
  end do

  ! -- Wait for communication completion before dealing with the end --
  if(stencil_g>0) then
    call mpi_waitall(com_nb(1),beg_request, MPI_STATUSES_IGNORE, ierr)
    deallocate(beg_request)
  end if
  if(stencil_d>0) then
    call mpi_waitall(com_nb(2),end_request, MPI_STATUSES_IGNORE, ierr)
    deallocate(end_request)
  end if

  ! -- For begining --
  ! Use that interpolation formula are exact
  V_fine(:,1,:) = V_coarse(:,:,1)
  ! For other first points
  do ind = 2, min(ind_min-1, N_fine)  ! Be carful, in some massively parrallel context, ind_min could bigger than N_fine +1
    pos = (ind-1)*(dx_f/dx_c)
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    V_fine(:,ind,:) = weight(1)*V_beg(:,:,V_ind) ! Array start from 1
    ind_limit = stencil_g - V_ind + 1
    do i = 2, ind_limit
      V_fine(:,ind,:) = V_fine(:,ind,:) + weight(i)*V_beg(:,:,V_ind-1+i) ! first point in V_beg stands for 1-stencil_g position
    end do
    ind_limit_2 = min(stencil_size, N_coarse+ind_limit) ! for very coarse grid, stencil size could be bigger than N_coarse
    do i_bis = ind_limit+1, ind_limit_2
      V_fine(:,ind,:) = V_fine(:,ind,:) + weight(i_bis)*V_coarse(:,:,i_bis-ind_limit)
    end do
    do i_bis = ind_limit_2+1, stencil_size
      V_fine(:,ind,:) = V_fine(:,ind,:) + weight(i_bis)*V_end(:,:,i_bis-ind_limit_2)
    end do
  end do
  ! -- For point of at the end of a line along the current direction --
  ! If ind_max<= ind_min (ie if stencil_size>N_coarse), computation are already done for first point
  do ind = max(ind_max+1,ind_min), N_fine
    pos = (ind-1)*(dx_f/dx_c)
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    !V_ind = V_ind - stencil_g
    V_ind = V_ind - stencil_g -1
    V_fine(:,ind,:) = weight(1)*V_coarse(:,:,V_ind+1)
    ind_limit = min((stencil_size),N_coarse-V_ind)
    do i = 2, ind_limit
      V_fine(:,ind,:) = V_fine(:,ind,:) + weight(i)*V_coarse(:,:,V_ind+i)
    end do
    V_ind = V_ind - N_coarse
    do i_bis = ind_limit+1, stencil_size
      V_fine(:,ind,:) = V_fine(:,ind,:) + weight(i_bis)*V_end(:,:,i_bis+V_ind)
    end do
  end do

  ! Free memory
  if(stencil_d>0) deallocate(V_end)
  if(stencil_g>0) deallocate(V_beg)

end subroutine Inter_LastDir_Permut_com


!> Interpolate a field along the first direction - no communication : V_fine(i,j,k) = interpolation(V_coarse(i_interpolation,j,k))
!!    @param[in]        V_coarse    = velocity to interpolate along the first direction
!!    @param[in]        dx_c        = space step on the coarse grid (for first direction)
!!    @param[in]        V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for first direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along second and last
!! direction.
subroutine Inter_FirstDir_no_com(V_coarse, dx_c, V_fine, dx_f)

  ! Input/Output
  real(WP), intent(in)                      :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local variable
  real(WP), dimension(stencil_size)         :: weight       ! interpolation weight
  integer               :: N_coarse, N_fine                 ! number of grid points
  integer               :: i, ind, V_ind                    ! some loop indices
  real(WP)              :: pos

V_Fine = 0.0_WP

  ! ==== Initialisation ====
  N_coarse = size(V_coarse,1)
  N_fine = size(V_fine,1)

  ! ==== Interpolation ====
  ! Use periodicity for boundaries
  do ind = 1, N_fine
    pos = (ind-1)*(dx_f/dx_c)
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    V_ind = V_ind - stencil_g
    V_fine(ind,:,:) = weight(1)*V_coarse(modulo(V_ind-1,N_coarse)+1,:,:)
    do i = 1, (stencil_size - 1)
      V_fine(ind,:,:) = V_fine(ind,:,:) + weight(i+1)*V_coarse(modulo(V_ind+i-1,N_coarse)+1,:,:)
    end do
  end do

end subroutine Inter_FirstDir_no_com


!> Interpolate a field along the first direction and permute first and second direction - no communication : V_fine(j,i,k) = interpolation(V_coarse(i_interpolation,j,k))
!!    @param[in]        V_coarse    = velocity to interpolate along the first direction
!!    @param[in]        dx_c        = space step on the coarse grid (for first direction)
!!    @param[in]        V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for first direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along the directions
!! without interpolation.
subroutine Inter_FirstDir_Permut_no_com(V_coarse, dx_c, V_fine, dx_f)

  ! Input/Output
  real(WP), intent(in)                      :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local variable
  real(WP), dimension(stencil_size)         :: weight       ! interpolation weight
  integer               :: N_coarse, N_fine                 ! number of grid points
  integer               :: i, ind, V_ind                    ! some loop indices
  integer               :: i1, i2                           ! for permutation along the two first direction
  real(WP)              :: pos, V_current

  ! ==== Initialisation ====
  N_coarse = size(V_coarse,1)
  N_fine = size(V_fine,2)

  ! ==== Interpolation ====
  ! Use periodicity for boundaries
  do ind = 1, N_fine
    pos = (ind-1)*dx_f/dx_c
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    V_ind = V_ind - stencil_g
    do i2 = 1, size(V_coarse,3)
      do i1 = 1, size(V_coarse,2)
        V_current = weight(1)*V_coarse(modulo(V_ind-1,N_coarse)+1,i1,i2)
        do i = 1, (stencil_size - 1)
          V_current = V_current + weight(i+1)*V_coarse(modulo(V_ind+i-1,N_coarse)+1,i1,i2)
        end do
        V_fine(i1,ind,i2) = V_current
      end do
    end do
  end do

end subroutine Inter_FirstDir_Permut_no_com


!> Interpolate a field along the first direction - with communication : V_fine(i,j,k) = interpolation(V_coarse(i_interpolation,j,k))
!! Variant with communication and where first direction can be different than X-axis.
!!    @param[in]        dir         = last directions (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        V_coarse    = velocity to interpolate along the last direction
!!    @param[in]        dx_c        = space step on the coarse grid (for last direction)
!!    @param[in,out]    V_fine      = interpolated velocity
!!    @param[in]        dx_f        = space step on the fine grid (for last direction)
!! @details
!!   V_fine and V_coarse must have the same resolution along second and third
!! directions.
subroutine Inter_FirstDir_com(dir, V_coarse, dx_c, V_fine, dx_f)

  

  ! Input/Output
  integer, intent(in)                       :: dir
  real(WP), intent(in)                      :: dx_f, dx_c
  real(WP), dimension(:,:,:),intent(in)     :: V_coarse
  real(WP), dimension(:,:,:),intent(inout)  :: V_fine
  ! Local variable
  real(WP), dimension(:,:,:), allocatable   :: V_beg, V_end ! received ghost values of velocity
  real(WP), dimension(:,:,:), allocatable   :: V_s1, V_s2   ! ghost values of velocity to send
  real(WP), dimension(stencil_size)         :: weight       ! interpolation weight
  integer               :: i,ind,i_bis, V_ind   ! some loop indices
  integer               :: ind_max, ind_min, ind_limit
  real(WP)              :: pos
  integer               :: N_coarse, N_fine     ! number of grid points
  integer, dimension(2) :: com_size             ! size of mpi communication for ghost points
  integer, dimension(2) :: rece_request         ! mpi communication request (handle) of nonblocking receive
  integer, dimension(MPI_STATUS_SIZE)         :: status  ! mpi status (for mpi_wait)
  integer               :: ierr                 ! mpi error code

  ! Initialisation
  com_size = size(V_coarse,2)*size(V_coarse,3)
  com_size(1) = com_size(1)*(stencil_g)
  com_size(2) = com_size(2)*(stencil_d)
  N_coarse = size(V_coarse,1)
  N_fine = size(V_fine,1)
  ind_max = ceiling((N_coarse-stencil_d)*dx_c/dx_f) - 1
  ind_min = ceiling((stencil_g)*dx_c/dx_f)+1

  ! ==== Communication ====
  if(stencil_g>0) then
    allocate(V_beg(stencil_g,size(V_coarse,2),size(V_coarse,3)))
    ! Initiate non blocking receive
    call Mpi_Irecv(V_beg(1,1,1),com_size(1),MPI_REAL_WP, &
      & neighbors(dir,-1), 1, D_comm(dir), rece_request(1), ierr)
    ! Send data
    allocate(V_s1(stencil_g,size(V_coarse,2),size(V_coarse,3)))
    V_s1 = V_coarse(N_coarse-stencil_g+1:N_coarse,:,:)
    call Mpi_Send(V_s1(1,1,1),com_size(1),MPI_REAL_WP, &
      & neighbors(dir,1), 1, D_comm(dir), ierr)
  end if

  if(stencil_d>0) then
    allocate(V_end(stencil_d,size(V_coarse,2),size(V_coarse,3)))
    ! Initiate non blocking receive
    call Mpi_Irecv(V_end(1,1,1),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,1), 2, D_comm(dir), rece_request(2), ierr)
    ! Send data
    allocate(V_s2(stencil_d,size(V_coarse,2),size(V_coarse,3)))
    V_s2 = V_coarse(1:stencil_d,:,:)
    call Mpi_Send(V_s2(1,1,1),com_size(2),MPI_REAL_WP, &
      & neighbors(dir,-1), 2, D_comm(dir), ierr)
  else
  end if

  ! ==== Interpolation ====
  ! -- For middle points --
  do ind = ind_min, ind_max
    pos = (ind-1)*dx_f/dx_c
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    V_ind = V_ind - stencil_g
    V_fine(ind,:,:) = weight(1)*V_coarse(V_ind,:,:)
    do i = 1, (stencil_size - 1)
      V_fine(ind,:,:) = V_fine(ind,:,:) + weight(i+1)*V_coarse(V_ind+i,:,:)
    end do
  end do
  ! -- For begining --
  if(stencil_g>0) call mpi_wait(rece_request(1), status, ierr)
  ! Use that interpolation formula are exact
  V_fine(1,:,:) = V_coarse(1,:,:)
  ! For other first points
  do ind = 2, ind_min-1
    pos = (ind-1)*dx_f/dx_c
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    !V_ind = V_ind - stencil_g
    V_fine(ind,:,:) = weight(1)*V_beg(V_ind,:,:) ! Array start from 1
    ind_limit = stencil_g - V_ind + 1
    do i = 2, ind_limit
      V_fine(ind,:,:) = V_fine(ind,:,:) + weight(i)*V_beg(V_ind-1+i,:,:) ! first point in V_beg stands for 1-stencil_g position
    end do
    do i_bis = ind_limit+1, stencil_size
      V_fine(ind,:,:) = V_fine(ind,:,:) + weight(i_bis)*V_coarse(i_bis-ind_limit,:,:)
    end do
  end do
  ! -- For point of at the end of a line along the current direction --
  if(stencil_d>0) call mpi_wait(rece_request(2), status, ierr)
  do ind = ind_max+1, N_fine
    pos = (ind-1)*dx_f/dx_c
    V_ind = floor(pos)+1
    call get_weight(pos-V_ind+1, weight)
    !V_ind = V_ind - stencil_g
    V_ind = V_ind - stencil_g-1
    V_fine(ind,:,:) = weight(1)*V_coarse(V_ind+1,:,:)
    ind_limit = min((stencil_size),N_coarse-V_ind)
    do i = 2, ind_limit
      V_fine(ind,:,:) = V_fine(ind,:,:) + weight(i)*V_coarse(V_ind+i,:,:)
    end do
    V_ind = V_ind - N_coarse
    do i_bis = ind_limit+1, stencil_size
      V_fine(ind,:,:) = V_fine(ind,:,:) + weight(i_bis)*V_end(i_bis+V_ind,:,:)
    end do
  end do

  ! Free memory
  if(stencil_d>0) deallocate(V_end)
  if(stencil_d>0) deallocate(V_s2)
  if(stencil_g>0) deallocate(V_beg)
  if(stencil_g>0) deallocate(V_s1)

end subroutine Inter_FirstDir_com


subroutine weight_Mprime4(pos,weight)

  real(WP), intent(in)                :: pos
  real(WP), dimension(:), intent(out) :: weight

  !weight(1)  = ((2.-(pos+1.))**2 * (1.-(pos+1.)))/2.
  weight(1) = (pos * (pos * (-pos + 2.) - 1.)) / 2.
  !weight(3) = 1.-2.5*(1.-pos)**2 + 1.5*(1.-pos)**3
  weight(3) = (pos * (pos * (-3. * pos + 4.) + 1.)) / 2.
  !weight(4) = ((2.-(2.-pos))**2 * (1.-(2.-pos)))/2.
  weight(4) = (pos * pos * (pos - 1.)) / 2.
  !weight(2) = 1.- 2.5*pos**2 + 1.5*pos**3
  weight(2) = 1. - (weight(1)+weight(3)+weight(4))


end subroutine weight_Mprime4


!> Interpolation with M4 kernel. Order 2 everywhere ?
subroutine weight_M4(pos,weight)

  real(WP), intent(in)              :: pos
  real(WP), dimension(:), intent(out) :: weight

  ! kernel =
  !(1._WP/6._WP)*((-X+2)**3)  if 1<=abs(X)<2
  !(1._WP/6._WP)*((-X+2)**3) - (4._WP/6._WP)*((-X+1)**3) if abs(X) < 1

  !weight(1) = (1._WP/6._WP)*((-(pos+1)+2)**3)
  weight(1) = (1._WP/6._WP)*((-pos+1._WP)**3)
  !weight(2) = (1._WP/6._WP)*((-pos+2)**3) - (4._WP/6._WP)*((-pos+1)**3)
  !weight(3) = (1._WP/6._WP)*((-(1-pos)+2)**3) - (4._WP/6._WP)*((-(1-pos)+1)**3)
  weight(3) = (1._WP/6._WP)*((pos+1)**3) - (4._WP/6._WP)*(pos**3)
  weight(4) = (1._WP/6._WP)*(pos**3)
  weight(2) = 1. - (weight(1)+weight(3)+weight(4))


end subroutine weight_M4


!> Interpolation with Lambda(4,4) kernel. Order 4 everywhere.
subroutine weight_Lambda4_4(pos,weight)

  real(WP), intent(in)              :: pos
  real(WP), dimension(:), intent(out) :: weight

    weight(1) = (pos*(pos*(pos*(pos*(pos*(pos*(pos*(pos*(-46. * pos + 207.) - 354.) + 273.) - 80.) + 1.) - 2.)- 1.) + 2.)) / 24.
    weight(2) = (pos*(pos*(pos*(pos*(pos*(pos*(pos*(pos*(230. * pos - 1035.) +1770.) - 1365.) + 400.) - 4.) + 4.) + 16.) - 16.)) / 24.
    weight(3) = (pos* pos*(pos*pos* (pos*(pos*(pos*(pos*(-460.* pos + 2070.) - 3540.) + 2730.) - 800.) + 6.) - 30.)+ 24.) / 24.
    weight(4) = (pos*(pos*(pos*(pos*(pos*(pos*(pos*(pos*(460. * pos - 2070.) + 3540.) - 2730.) + 800.) - 4.) - 4.) + 16.) + 16.)) / 24.
    !weight(5) = (pos*(pos*(pos*(pos*(pos*(pos*(pos*(pos * (-230. * pos + 1035.) - 1770.) + 1365.) - 400.) + 1.) + 2.) - 1.) - 2.)) / 24.
    weight(6) = (pos*pos*pos*pos*pos*(pos*(pos * (pos * (46. * pos - 207.) + 354.) - 273.) + 80.)) / 24.
    weight(5) = 1. - (weight(1)+weight(2)+weight(3)+weight(4)+weight(6))


end subroutine weight_Lambda4_4


!> Basic interpolation formula. Be careful, this kernel may create unphysical oscillation
!! in low frequency. This is rather implemented to show the requirement of "better"
!! interpolation kernel.
subroutine weight_linear(pos,weight)

  real(WP), intent(in)              :: pos
  real(WP), dimension(:), intent(out) :: weight

    weight(1) = 1-pos
    weight(2) = pos

end subroutine weight_linear

end module Interpolation_velo
!> @}
