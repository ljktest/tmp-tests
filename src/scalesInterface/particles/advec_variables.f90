!USEFORTEST advec
!USEFORTEST interpolation
!> @addtogroup part
!! @{
!------------------------------------------------------------------------------
!
! MODULE: advec_variables
!
!
! DESCRIPTION:
!> The module ``advec_variables'' gather all variables that have to been shared by diffrenrent advection
!! modules. It also provide a set of method to set the protected or private variables to the right values.
!! @details
!! It contains the variables common to the solver along each direction and other generic variables used for the
!! advection based on the particle method. It provied functions to set
!! them to right values depending on the choosen remeshing formula.
!!
!! This module is not supposed to be used by the main code but only by the other advection module.
!! More precisly, a final user must only used the generic "advec" module wich contains all the interface
!! to initialize the solver (eg choosing the remeshing formula and the dimension splitting) and to solve
!! the advection equation with the particle method.
!!
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module advec_variables

    use precision_tools

    implicit none

    ! ===== Public and protected variables =====
    ! ----- Minimal and maximal indice of the buffer used in the different communication -----
    !> minimal indice of the send buffer
    integer, public                             :: send_j_min
    !> maximal indice of the send buffer
    integer, public                             :: send_j_max
    !> To take in account diffusion inside remeshing
    real(WP), protected, dimension(:,:), allocatable :: sc_diff_dt_dx


    ! ------ Solver context -----
    ! solver choosen
    character(len=str_short), protected         :: type_solv
    integer, dimension(2), protected            :: remesh_stencil
    ! ------ Remeshing information -----
    !> number of particles in a block
    integer, protected                          :: bl_size
    !> distance between the "central" mesh point and the extream mesh point of the stencil of points used to remesh a particle
    integer, protected                          :: bl_bound_size
    !> Number of common meshes used in the remeshing of two successive particle
    !! (in case off standart (ie non corrected) remeshing formula)).
    integer, dimension(2), protected            :: bl_remesh_superposition
    !> Number of block on each processus along each direction
    integer, dimension(3), protected            :: bl_nb
    !> Maximum CFL number allowed by communications for the current parameters
    integer, protected                          :: CFL_max

    ! ------ To ensure unique mpi message tag -----
    ! Tag generate with a proc_gap
    !> To create tag used in AC_particle_velocity to send range
    integer, dimension(2), parameter            :: tag_velo_range = (/ 0,1 /)
    !> To create tag used in AC_particle_velocity to send velocity field
    integer, dimension(2), parameter            :: tag_velo_V = (/ 0,2 /)
    !> To create tag used in bufferToScalar to send range of buffer which will be send
    integer, dimension(2), parameter            :: tag_bufToScal_range = (/ 0,3 /)
    !> To create tag used in bufferToScalar to send the buffer used to remesh particles
    integer, dimension(2), parameter            :: tag_bufToScal_buffer = (/ 0,4 /)

    ! Tag generate with "compute_gap_NP"
    !> To create tag used in AC_obtain_recevers to send ghost
    integer, dimension(2), parameter            :: tag_obtrec_ghost_NP = (/ 0, 1/)
    !> To create tag used in AC_type_and_bloc to exchange ghost with neighbors
    integer, dimension(2), parameter            :: tag_part_tag_NP = (/ 0, 2/)
    !> To create tag used in AC_obtain_recevers to send message about recevers of minimal and maximal rank
    integer, dimension(2), parameter            :: tag_obtrec_NP = (/ 0, 3/)
    !> To create tag used in AC_obtain_receivers to send message about senders of minimal and maximal rank
    integer, dimension(2), parameter            :: tag_obtsend_NP = (/ 0, 4/)
    !> To create tag used in advecY_limitator_group to exchange ghost with neighbors
    integer, dimension(2), parameter            :: tag_part_slope = (/ 0, 5/)

    ! ===== Public procedures =====
    !----- Initialize solver -----
    public                                      :: AC_solver_init
    public                                      :: AC_set_part_bound_size

contains

! ====================================================================
! ====================    Initialize context      ====================
! ====================================================================

!> Initialize some variable related to the solver implementation (and which
!! depend of the resmeshing formula choosen and the dimmensionnal splitting used).
!!    @param[in]        part_solv   = remeshing formula choosen (spcae order, ...)
!!    @param[in]        verbosity   = to display info about chosen remeshing formula (optional)
subroutine AC_solver_init(part_solv, verbosity)

    use cart_topology   ! info about mesh and mpi topology

    ! Input/Output
    character(len=*), optional, intent(in)  ::  part_solv
    logical, optional, intent(in)  ::  verbosity
    ! Others
    logical :: verbose

    ! Set verbosity
    verbose = .true.
    if (present(verbosity)) verbose = verbosity

    ! Initialisation part adapted to each method
    if (present(part_solv)) type_solv = part_solv
    select case(type_solv)
        case('p_O2')
            bl_size = 2
            bl_bound_size = 1
            remesh_stencil = 1
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') ' particle method, corrected lambda 2 '
                write(*,'(6x,a)') '====================================='
            end if
        case('p_O4')
            bl_size = 4
            bl_bound_size = 2
            remesh_stencil = 2
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') ' particle method, corrected lambda 4 '
                write(*,'(6x,a)') '====================================='
            end if
        case('p_M4')
            bl_size = 1!2
            bl_bound_size = 2   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/1,2/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') ' particle method,           M prime 4'
                write(*,'(6x,a)') '====================================='
            end if
        case('p_M6')
            bl_size = 1!2
            bl_bound_size = 3   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/2,3/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') ' particle method,           M prime 6'
                write(*,'(6x,a)') '====================================='
            end if
        case('p_M8')
            bl_size = 2
            bl_bound_size = 4   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/3,4/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') ' particle method,           M prime 8'
                write(*,'(6x,a)') '====================================='
            end if
        case('d_M4')
            bl_size = 1!2
            bl_bound_size = 2   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/1,2/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '============= Advection scheme ==========='
                write(*,'(6x,a)') ' particle method, M prime 4 with diffusion'
                write(*,'(6x,a)') '=========================================='
            end if
        case('p_44')
            bl_size = 1!2
            bl_bound_size = 3   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/2,3/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') '     particle method, Lambda 4,4     '
                write(*,'(6x,a)') '====================================='
            end if
        case('p_64')
            bl_size = 1!2
            bl_bound_size = 4   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/3,4/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') '     particle method, Lambda 6,4     '
                write(*,'(6x,a)') '====================================='
            end if
        case('p_66')
            bl_size = 1!2
            bl_bound_size = 4   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/3,4/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') '     particle method, Lambda 6,6     '
                write(*,'(6x,a)') '====================================='
            end if
        case('p_84')
            bl_size = 1!2
            bl_bound_size = 5   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/4,5/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') '     particle method, Lambda 8,4     '
                write(*,'(6x,a)') '====================================='
            end if
        ! For legacy
        case('p_L4')
            bl_size = 1!2
            bl_bound_size = 3   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/2,3/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') '     particle method, Lambda 4,4     '
                write(*,'(6x,a)') '====================================='
            end if
        case('p_L6')
            bl_size = 1!2
            bl_bound_size = 4   ! Be aware : don't use it to compute superposition between
                                ! mpi processes (not as predictible as corrected scheme)
            remesh_stencil = (/3,4/)
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') '     particle method, Lambda 6,6     '
                write(*,'(6x,a)') '====================================='
            end if
        case default
            bl_size = 2
            bl_bound_size = 1
            remesh_stencil = 1
            if ((cart_rank==0).and.(verbose)) then
                write(*,'(6x,a)') '========== Advection scheme ========='
                write(*,'(6x,a)') ' particle method, corrected lambda 2 '
                write(*,'(6x,a)') '====================================='
            end if
    end select

    ! Check if the subdomain contain a number of mesh wich could be divided by bl_size
    if ((modulo(mesh_sc%N_proc(1),bl_size)/=0).OR.  &
      & (modulo(mesh_sc%N_proc(2),bl_size)/=0).OR.  &
      & (modulo(mesh_sc%N_proc(3),bl_size)/=0)) then
        if (cart_rank ==0) print*, 'Number of mesh by processus must be a muliple of ', bl_size
        stop
    end if

    ! Compute local number of block along each direction
    bl_nb = mesh_sc%N_proc/bl_size

    ! Compute maximal CFL number
    CFL_max = minval(mesh_sc%N_proc)*(size(neighbors)/2)

    ! To take in account for diffusion during the remeshing operation
    if(.not. allocated(sc_diff_dt_dx)) allocate(sc_diff_dt_dx(1,3))

end subroutine AC_solver_init

!> Manually change protected variable "bl_bound_size" - purpose test only (for
!! auto-validation tests)
!!    @param[in]        bound_size   = wanted value of "bl_bound_part"
subroutine AC_set_part_bound_size(bound_size)

    ! Input/Ouput
    integer, intent(in) :: bound_size

    bl_bound_size = bound_size

end subroutine AC_set_part_bound_size

!> Set manually the diffusion parameter for taking into account diffusion
!! directly in remeshing.
subroutine AC_set_diff_dt_dx(sc_diff)

  use cart_topology

  ! Input/Output
  real(WP), dimension(:), intent(in)  ::  sc_diff
  ! Local
  integer                             :: ind
! character(len=10)                 :: format_out

  if (size(sc_diff_dt_dx,1) /= size(sc_diff)) then
    deallocate(sc_diff_dt_dx)
    allocate(sc_diff_dt_dx(size(sc_diff),3))
  end if
  do ind =1, 3
    sc_diff_dt_dx(:,ind) = sc_diff/(mesh_sc%dx(ind)**2)
  end do

! if(cart_rank==0) then
!   write(format_out,'(a,i0,a)') '(a,', size(sc_diff_dt_dx,1), 'g15.8)'
!   write(*,format_out) 'diff along X = ', sc_diff_dt_dx(:,1)
!   write(*,format_out) 'diff along Y = ', sc_diff_dt_dx(:,2)
!   write(*,format_out) 'diff along Z = ', sc_diff_dt_dx(:,3)
! end if

end subroutine AC_set_diff_dt_dx


end module advec_variables
