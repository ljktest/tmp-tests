!USEFORTEST advec
!> @addtogroup part

!------------------------------------------------------------------------------
!
! MODULE: advecX
!
!
! DESCRIPTION:
!> The module advecX is devoted to the advection along X axis of a scalar field.
!! It used particle method and provides a parallel implementation.
!
!> @details
!! This module is a part of the advection solver based on particles method.
!! The solver uses some dimensionnal splitting and this module contains all the
!! method used to solve advection along the X-axis. This is a parallel
!! implementation using MPI and the cartesien topology it provides.
!!
!! This module can use the method and variables defined in the module
!! "advec_common" which gather information and tools shared for advection along
!! x, y and z-axis.
!!
!! The module "test_advec" can be used in order to validate the procedures
!! embedded in this module.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module advecX

    use precision_tools
    use advec_abstract_proc
    implicit none

    ! ===== Public procedures =====
    public  :: advecX_init_group! initialisation for a group of line of particles
    public  :: advecX_remesh_buffer_to_scalar

    ! -- Init remeshing context --
    public  :: advecX_remesh_init

    ! -- Remeshing algorithm --
    public  :: advecX_remesh_type_no_com
    public  :: advecX_remesh_limited_no_com
    public  :: advecX_remesh_no_type_no_com
    public  :: advecX_remesh_in_buffer_lambda
    public  :: advecX_remesh_in_buffer_limit_lambda
    public  :: advecX_remesh_in_buffer_Mprime

    ! ===== Private procedures =====
    ! -- Compute limitator --
    public :: advecX_limitator_group
    !private:: advecX_limitator_group_no_com

    procedure(advecX_remesh_type_no_com), pointer, public   ::  advecX_remesh_no_com => null()

    ! ===== Private procedures =====

    ! ===== Private variable ====
    !> Current direction = along X
    integer, private, parameter     :: direction=1

contains

! #####################################################################################
! #####                                                                           #####
! #####                         Public procedure                                  #####
! #####                                                                           #####
! #####################################################################################


! =============================================================
! ====================   Remeshing tools   ====================
! =============================================================

subroutine advecX_remesh_init()

    use advec_variables         ! solver context

    select case(trim(type_solv))
    case ('p_M8')
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    case ('p_M6')
        !advecX_remesh_com => advecX_remesh_in_buffer_Mprime6
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    case ('p_M4')
        !advecX_remesh_com => advecX_remesh_in_buffer_Mprime6
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    case ('d_M4')
        !advecX_remesh_com => advecX_remesh_in_buffer_Mprime6
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    case ('p_L2')
        !advecX_remesh_com => advecX_remesh_in_buffer_limited
        advecX_remesh_no_com => advecX_remesh_limited_no_com
    case ('p_44') ! Lambda 4_4
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    case ('p_64') ! Lambda 6_4
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    case ('p_66') ! Lambda 6_6
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    case ('p_84') ! Lambda 8_4
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    ! To ensure retro-compatibility
    case ('p_L4')
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    case ('p_L6')
        advecX_remesh_no_com => advecX_remesh_no_type_no_com
    ! Default value
    case default
        !advecX_remesh_com => advecX_remesh_in_buffer_gp
        advecX_remesh_no_com => advecX_remesh_type_no_com
    end select

end subroutine advecX_remesh_init


!> Remesh particle inside a buffer. Use corrected lambda remeshing polynoms.
!! @autor Jean-Baptiste Lagaert
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        ind_min     = indices from the original array "pos_in_buffer" does not start from 1.
!!                                    It actually start from ind_min and to avoid access out of range,
!!                                    a gap of (-ind_min) will be added to each indices from "pos_in_buffer.
!!    @param[in]        send_min    = minimal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[in]        send_max    = maximal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[in]        scalar      = the initial scalar field transported by particles
!!    @param[out]       buffer      = buffer where particles are remeshed
!!    @param[in]        remesh_line = subroutine wich remesh a line of particle with the right remeshing formula
subroutine advecX_remesh_in_buffer_lambda(gs, j, k, ind_min, p_pos_adim, bl_type, bl_tag, send_min, send_max, &
        & scalar, buffer, pos_in_buffer)

    use cart_topology           ! Description of mesh and of mpi topology
    use advec_variables         ! contains info about solver parameters and others.
    use advec_abstract_proc     ! profile of generic procedure
    use advec_remeshing_lambda  ! remeshing formula and wrapper for a line of particles

    ! Input/Output
    integer, dimension(2), intent(in)                   :: gs
    integer, intent(in)                                 :: j, k
    integer, intent(in)                                 :: ind_min
    real(WP), dimension(:,:,:), intent(in)              :: p_pos_adim   ! adimensionned particles position
    logical, dimension(:,:,:), intent(in)               :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(:,:,:), intent(in)               :: bl_tag       ! indice of tagged particles
    integer, dimension(:,:), intent(in)                 :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(in)                 :: send_max     ! distance between me and processus wich send me information
    real(WP), dimension(:,:,:), intent(inout)           :: scalar       ! the initial scalar field transported by particles
    real(WP),dimension(:), intent(out), target          :: buffer       ! buffer where particles are remeshed
    integer, dimension(:), intent(inout)                :: pos_in_buffer! describe how the one dimensionnal array "buffer" are split
                                                                        ! in part corresponding to different processes

    ! Other local variables
    integer                                 :: proc_gap     ! distance between my (mpi) coordonate and coordinate of the
    type(real_pter),dimension(:),allocatable:: remeshX_pter  ! pointer to send buffer in which scalar are sorted by line indice.
                                                            ! sorted by receivers
    integer                                 :: i1, i2       ! indice of a line into the group
    integer                                 :: ind          ! indice of the current particle inside the current line.

    ! ===== Remeshing into the buffer by using pointer array =====
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_min(i1,i2)
            send_j_max = send_max(i1,i2)

            ! -- Allocate remeshX_pter --
            allocate(remeshX_pter(send_j_min:send_j_max))
            do ind = send_j_min, send_j_max
                proc_gap = floor(real(ind-1, WP)/mesh_sc%N_proc(direction)) - (ind_min-1)
                remeshX_pter(ind)%pter => buffer(pos_in_buffer(proc_gap))
                pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + 1
            end do

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda_pter(direction, p_pos_adim(:,i1,i2), scalar(:,j+i1-1,k+i2-1), &
                & bl_type(:,i1,i2), bl_tag(:,i1,i2), send_j_min, remeshX_pter)

            deallocate(remeshX_pter)
        end do
    end do

    ! Scalar must be re-init before ending the remeshing
    scalar(:,j:j+gs(1)-1,k:k+gs(2)-1) = 0

end subroutine advecX_remesh_in_buffer_lambda


!> Remesh particle inside a buffer. Use corrected lambda remeshing polynoms.
!! @autor Jean-Baptiste Lagaert
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in]        ind_min     = indices from the original array "pos_in_buffer" does not start from 1.
!!                                    It actually start from ind_min and to avoid access out of range,
!!                                    a gap of (-ind_min) will be added to each indices from "pos_in_buffer.
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        bl_type     = table of blocks type (center of left)
!!    @param[in]        bl_tag      = inform about tagged particles (bl_tag(ind_bl)=1 if the end of the bl_ind-th block
!!                                    and the begining of the following one is tagged)
!!    @param[in]        limit       = limitator function
!!    @param[in]        send_min    = minimal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[in]        send_max    = maximal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[in]        scalar      = the initial scalar field transported by particles
!!    @param[out]       buffer      = buffer where particles are remeshed
!!    @param[in,out]    pos_in_buffer   = information about where remesing the particle inside the buffer
subroutine advecX_remesh_in_buffer_limit_lambda(gs, j, k, ind_min, p_pos_adim, bl_type, bl_tag, limit,  &
        & send_min, send_max, scalar, buffer, pos_in_buffer)

    use cart_topology           ! Description of mesh and of mpi topology
    use advec_variables         ! contains info about solver parameters and others.
    use advec_abstract_proc     ! profile of generic procedure
    use advec_remeshing_lambda  ! remeshing formula and wrapper for a line of particles

    ! Input/Output
    integer, dimension(2), intent(in)                   :: gs
    integer, intent(in)                                 :: j, k
    integer, intent(in)                                 :: ind_min
    real(WP), dimension(:,:,:), intent(in)              :: p_pos_adim   ! adimensionned particles position
    logical, dimension(:,:,:), intent(in)               :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(:,:,:), intent(in)               :: bl_tag       ! indice of tagged particles
    real(WP), dimension(:,:,:), intent(in)              :: limit        ! limitator function (divided by 8)
    integer, dimension(:,:), intent(in)                 :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(in)                 :: send_max     ! distance between me and processus wich send me information
    real(WP), dimension(:,:,:), intent(inout)           :: scalar       ! the initial scalar field transported by particles
    real(WP),dimension(:), intent(out), target          :: buffer       ! buffer where particles are remeshed
    integer, dimension(:), intent(inout)                :: pos_in_buffer! describe how the one dimensionnal array "buffer" are split
                                                                        ! in part corresponding to different processes

    ! Other local variables
    integer                                 :: proc_gap     ! distance between my (mpi) coordonate and coordinate of the
    type(real_pter),dimension(:),allocatable:: remeshX_pter  ! pointer to send buffer in which scalar are sorted by line indice.
                                                            ! sorted by receivers
    integer                                 :: i1, i2       ! indice of a line into the group
    integer                                 :: ind          ! indice of the current particle inside the current line.

    ! ===== Remeshing into the buffer by using pointer array =====
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_min(i1,i2)
            send_j_max = send_max(i1,i2)

            ! -- Allocate remeshX_pter --
            allocate(remeshX_pter(send_j_min:send_j_max))
            do ind = send_j_min, send_j_max
                proc_gap = floor(real(ind-1, WP)/mesh_sc%N_proc(direction)) - (ind_min-1)
                remeshX_pter(ind)%pter => buffer(pos_in_buffer(proc_gap))
                pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + 1
            end do

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda2limited_pter(direction, p_pos_adim(:,i1,i2), scalar(:,j+i1-1,k+i2-1), &
                & bl_type(:,i1,i2), bl_tag(:,i1,i2), send_j_min, limit(:,i1,i2), remeshX_pter)

            deallocate(remeshX_pter)
        end do
    end do

    ! Scalar must be re-init before ending the remeshing
    scalar(:,j:j+gs(1)-1,k:k+gs(2)-1) = 0

end subroutine advecX_remesh_in_buffer_limit_lambda


!> Remesh particle inside a buffer - for M'6 or M'8 remeshing formula.
!! @autor Jean-Baptiste Lagaert
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        ind_min     = indices from the original array "pos_in_buffer" does not start from 1.
!!                                    It actually start from ind_min and to avoid access out of range,
!!                                    a gap of (-ind_min) will be added to each indices from "pos_in_buffer.
!!    @param[in]        send_min    = minimal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[in]        send_max    = maximal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[in]        scalar      = the initial scalar field transported by particles
!!    @param[out]       buffer      = buffer where particles are remeshed
subroutine advecX_remesh_in_buffer_Mprime(gs, j, k, ind_min, p_pos_adim, send_min, send_max, &
        & scalar, buffer, pos_in_buffer)

    use cart_topology     ! Description of mesh and of mpi topology
    use advec_variables         ! contains info about solver parameters and others.
    use advec_abstract_proc     ! profile of generic procedure
    use advec_remeshing_Mprime  ! remeshing formula and wrapper for a line of particles

    ! Input/Output
    integer, dimension(2), intent(in)                   :: gs
    integer, intent(in)                                 :: j, k
    integer, intent(in)                                 :: ind_min
    real(WP), dimension(:,:,:), intent(in)              :: p_pos_adim   ! adimensionned particles position
    integer, dimension(:,:), intent(in)                 :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(in)                 :: send_max     ! distance between me and processus wich send me information
    real(WP), dimension(:,:,:), intent(inout)           :: scalar       ! the initial scalar field transported by particles
    real(WP),dimension(:), intent(out), target          :: buffer       ! buffer where particles are remeshed
    integer, dimension(:), intent(inout)                :: pos_in_buffer! describe how the one dimensionnal array "buffer" are split
                                                                        ! in part corresponding to different processes

    ! Other local variables
    integer                                 :: proc_gap     ! distance between my (mpi) coordonate and coordinate of the
    type(real_pter),dimension(:),allocatable:: remeshX_pter  ! pointer to send buffer in which scalar are sorted by line indice.
                                                            ! sorted by receivers
    integer                                 :: i1, i2       ! indice of a line into the group
    integer                                 :: ind          ! indice of the current particle inside the current line.
    !! real(WP), dimension(mesh_sc%N_proc(direction))  :: pos_translat ! translation of p_pos_adim as array indice
    !!                                                        ! are now starting from 1 and not ind_min


    ! ===== Remeshing into the buffer by using pointer array =====
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)

            ! -- Allocate remeshX_pter --
            allocate(remeshX_pter(send_min(i1,i2):send_max(i1,i2)))
            do ind = send_min(i1,i2), send_max(i1,i2)
                proc_gap = floor(real(ind-1, WP)/mesh_sc%N_proc(direction)) - (ind_min-1)
                remeshX_pter(ind)%pter => buffer(pos_in_buffer(proc_gap))
                pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + 1
            end do

            !! pos_translat = p_pos_adim(:,i1,i2) - send_min(i1,i2) + 1
            !! Index translation is performed in the AC_remesh_Mprime_pter subroutine on the
            !! integer adimensionned particle position instead of here on the float position

            ! -- Remesh the particles in the buffer --
            do ind = 1, mesh_sc%N_proc(direction)
                call AC_remesh_Mprime_pter(p_pos_adim(ind,i1,i2), 1-send_min(i1,i2), scalar(ind,j+i1-1,k+i2-1), remeshX_pter)
            end do

            deallocate(remeshX_pter)
        end do
    end do

    ! Scalar must be re-init before ending the remeshing
    scalar(:,j:j+gs(1)-1,k:k+gs(2)-1) = 0

end subroutine advecX_remesh_in_buffer_Mprime


!> Remesh particle inside a buffer - for corrected lambda 2 or lambda 4 - no communication variant
!! @autor Jean-Baptiste Lagaert
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        dt          = time step (needed for tag and type)
!!    @param[in]        p_V         = particle velocity to compute block type and tag
!!    @param[in,out]    scal        = scalar field to advect
subroutine advecX_remesh_type_no_com(ind_group, gs, p_pos_adim, p_V, j, k , scal, dt)

    use cart_topology           ! Description of mesh and of mpi topology
    use advec_variables         ! contains info about solver parameters and others.
    use advec_correction        ! Some procedures common to advection along all directions
    use advec_remeshing_lambda  ! remeshing formula and wrapper for a line of particles

    ! Input/Output
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP),dimension(:,:,:),intent(inout)     :: scal
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(inout)   :: p_V          ! particles velocity
    real(WP), intent(in)                        :: dt

    ! Other local variables
    integer                                             :: i1, i2       ! position insde the group of line
    ! Type and block
    logical, dimension(bl_nb(direction)+1,gs(1),gs(2))  :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction),gs(1),gs(2))    :: bl_tag       ! indice of tagged particles
    integer                                 :: N_loc

    N_loc = size(scal,1)

    ! ===== Pre-Remeshing: Determine blocks type and tag particles =====
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    ! ===== Initialize the general buffer =====

    do i1 = 1, gs(1)
        do i2 = 1, gs(2)

            ! -- [re-] init buffer --
            !remesh_buffer = 0
            p_V(:,1,1) = 0

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda_array(direction, p_pos_adim(:,i1,i2), &
                & scal(:,j+i1-1,k+i2-1), bl_type(:,i1,i2), bl_tag(:,i1,i2), p_V(:,1,1))
                !& scal(:,j+i1-1,k+i2-1), bl_type(:,i1,i2), bl_tag(:,i1,i2), remesh_buffer)

            ! -- Update scalar from buffer --
            !scal(:,i1+j-1,k+i2-1) = remesh_buffer
            scal(:,i1+j-1,k+i2-1) = p_V(1:N_loc,1,1)
        end do
    end do

end subroutine advecX_remesh_type_no_com


!> Remesh particle inside a buffer - for limited and corrected lambda 2 - no communication variant
!! @autor Jean-Baptiste Lagaert
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        dt          = time step (needed for tag and type)
!!    @param[in]        p_V         = particle velocity to compute block type and tag
!!    @param[in,out]    scal        = scalar field to advect
subroutine advecX_remesh_limited_no_com(ind_group, gs, p_pos_adim, p_V, j, k , scal, dt)

    use cart_topology           ! Description of mesh and of mpi topology
    use advec_variables         ! contains info about solver parameters and others.
    use advec_correction        ! Some procedures common to advection along all directions
    use advec_remeshing_lambda  ! remeshing formula and wrapper for a line of particles

    ! Input/Output
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP),dimension(:,:,:),intent(inout)     :: scal
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(inout)   :: p_V          ! particles velocity
    real(WP), intent(in)                        :: dt

    ! Other local variables
    integer                                             :: i1, i2       ! position insde the group of line
    ! Type and block
    logical, dimension(bl_nb(direction)+1,gs(1),gs(2))  :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction),gs(1),gs(2))    :: bl_tag       ! indice of tagged particles
    real(WP), dimension(mesh_sc%N_proc(direction)+1,gs(1),gs(2)):: limit        ! limitator function (divided by 8.)
    ! Variable used to remesh particles in a buffer
    !real(WP),dimension(mesh_sc%N(direction))                               :: remesh_buffer! buffer use to remesh the scalar
    integer                                 :: N_loc

    N_loc = size(scal,1)

    ! ===== Pre-Remeshing I: Determine blocks type and tag particles =====
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    ! ===== Pre-Remeshing II: Compute the limitor function =====
    ! Actually, this subroutine compute [limitator/8] as this is this fraction
    ! wich appear always in the remeshing polynoms.
    call advecX_limitator_group_no_com(gs, j, k, p_pos_adim, scal, limit)

    ! ===== Initialize the general buffer =====

    do i1 = 1, gs(1)
        do i2 = 1, gs(2)

            ! -- [re-] init buffer --
            !remesh_buffer = 0
            p_V(:,1,1) = 0
            !p_V(:,i1,i2) = 0

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda2limited_array(direction, p_pos_adim(:,i1,i2), &
                & scal(:,j+i1-1,k+i2-1), bl_type(:,i1,i2), bl_tag(:,i1,i2), limit(:,i1,i2), p_V(:,1,1))
                !& scal(:,j+i1-1,k+i2-1), bl_type(:,i1,i2), bl_tag(:,i1,i2), limit(:,i1,i2), remesh_buffer)

            ! -- Update scalar from buffer --
            !scal(:,i1+j-1,k+i2-1) = remesh_buffer
            scal(:,i1+j-1,k+i2-1) = p_V(1:N_loc,1,1)
        end do
    end do
    !scal(:,i1+j-1:i1+j-1+gs(1),k+i2-1:k+i2-1+gs(2)) = p_V

end subroutine advecX_remesh_limited_no_com


!> Remesh particle inside a buffer - for M'6 or M'8 - no communication variant
!! @autor Jean-Baptiste Lagaert
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        dt          = time step (needed for tag and type)
!!    @param[in]        p_V         = particle velocity to compute block type and tag
!!    @param[in]        scalar      = the initial scalar field transported by particles
!!    @param[out]       buffer      = buffer where particles are remeshed
subroutine advecX_remesh_no_type_no_com(ind_group, gs, p_pos_adim, p_V, j, k , scalar, dt)

    use cart_topology           ! Description of mesh and of mpi topology
    use advec_variables         ! contains info about solver parameters and others.
    use advec_remeshing_Mprime  ! remeshing formula and wrapper for a line of particles

    ! Input/Output
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(inout)   :: scalar       ! the initial scalar field transported by particles
    real(WP), dimension(:,:,:), intent(inout)   :: p_V          ! particles velocity
    real(WP), intent(in)                        :: dt

    ! Other local variables
    integer                                 :: i1, i2       ! indice of a line into the group
    integer                                 :: ind          ! indice of the current particle inside the current line.
    !real(WP),dimension(mesh_sc%N(direction))        :: remesh_buffer! buffer use to remesh the scalar
    integer                                 :: N_loc

    N_loc = size(scalar,1)

    ! ===== Remeshing into the buffer by using pointer array =====
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)

            ! -- [re-] init buffer --
            p_V(:,1,1) = 0

            ! -- Remesh the particles in the buffer --
            do ind = 1, mesh_sc%N_proc(direction)
                call AC_remesh_Mprime_array(direction, p_pos_adim(ind,i1,i2), scalar(ind,j+i1-1,k+i2-1), p_V(:,1,1))
                !call AC_remesh_Mprime_array(direction, p_pos_adim(ind,i1,i2), scalar(ind,j+i1-1,k+i2-1), remesh_buffer))
            end do

            ! -- Update scalar from buffer --
            !scalar(:,j+i1-1,k+i2-1) = remesh_buffer
            scalar(:,j+i1-1,k+i2-1) = p_V(1:N_loc,1,1)

        end do
    end do


end subroutine advecX_remesh_no_type_no_com


!> Update the scalar field with scalar stored into the buffer
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        j,k         = Y- and Z-coordinate of the first line along X inside the current group of lines.
!!    @param[in]        ind_proc    = algebric distance between me and the processus which send me the buffer. To read the right cartography.
!!    @param[in]        gap         = algebric distance between my local indice and the local indices from the processus which send me the buffer.
!!    @param[in]        begin_i1    = indice corresponding to the first place into the cartography
!!                                      array where indice along the the direction of the group of lines are stored.
!!    @param[in]        cartography = cartography(proc_gap) contains the set of the lines indice in the block for wich the
!!                                    current processus requiers data from proc_gap and for each of these lines the range
!!                                    of mesh points from where it requiers the velocity values.
!!    @param[in]        buffer      = buffer containing to redistribute into the scalar field.
!!    @param[out]       scalar      = scalar field (to update)
!!    @param[out]       beg_buffer  = first indice inside the current cartography where mesh indices are stored. To know where reading data into the buffer.
subroutine advecX_remesh_buffer_to_scalar(gs, j, k, ind_proc, gap, begin_i1, cartography, buffer, scalar, beg_buffer)

    ! Input/Output
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    integer, intent(in)                         :: ind_proc     ! to read the good cartography associate to the processus which send me the buffer.
    integer,intent(in)                          :: gap          ! gap between my local indices and the local indices from another processes
    integer, intent(in)                         :: begin_i1     ! indice corresponding to the first place into the cartography
                                                                ! array where indice along the the direction of the group of lines are stored.
    integer, dimension(:,:), intent(in)         :: cartography
    real(WP),dimension(:), intent(in)           :: buffer       ! buffer containing the data to redistribute into the local scalar field.
    real(WP), dimension(:,:,:), intent(inout)   :: scalar       ! the scalar field.
    integer, intent(inout)                      :: beg_buffer   ! first indice inside where the scalar values are stored into the buffer for the current sender processus.
                                                                ! To know where reading data into the buffer.

    ! Other local variables
    integer         :: i1, i2       ! indice of a line into the group
    integer         :: ind_for_i1   ! where to read the first coordinate (i1) of the current line inside the cartography?
    integer         :: ind_i1_range ! ito know where to read the first coordinate (i1) of the current line inside the cartography.
    integer         :: ind_1Dtable  ! indice of my current position inside a one-dimensionnal table
    ! To know where reading data into the buffer and where to write inside the scalar field:
    integer         :: end_buffer   ! last indice inside where the scalar values are stored into the buffer for the current sender processus.
    integer         :: beg_sca      ! first indice inside where the scalar values has to be write inside the scalar field.
    integer         :: end_sca      ! last indice inside where the scalar values has to be write inside the scalar field.

    ! Use the cartography to know which lines are concerned
    ind_1Dtable = cartography(2,ind_proc) ! carto(2) = nb of element use to store i1 and i2 indices
    ! Position in cartography(:,ind_proc) of the current i1 indice
    ind_i1_range = begin_i1
    do i2 = 1, gs(2)
        do ind_for_i1 = ind_i1_range+1, ind_i1_range + cartography(2+i2,ind_proc), 2
            do i1 = cartography(ind_for_i1,ind_proc), cartography(ind_for_i1+1,ind_proc)
                beg_sca = cartography(ind_1Dtable+1,ind_proc)+gap
                end_sca = cartography(ind_1Dtable+2,ind_proc)+gap
                end_buffer = beg_buffer + end_sca - beg_sca
                scalar(beg_sca:end_sca,j+i1-1,k+i2-1) = scalar(beg_sca:end_sca,j+i1-1,k+i2-1) &
                    & + buffer(beg_buffer:end_buffer)
                beg_buffer = end_buffer + 1
                ind_1Dtable = ind_1Dtable + 2
            end do
        end do
        ind_i1_range = ind_i1_range + cartography(2+i2,ind_proc)
    end do

end subroutine advecX_remesh_buffer_to_scalar


! ====================================================================
! ====================    Initialize particle     ====================
! ====================================================================

!> Creation and initialisation of a group of particle line
!!    @param[in]    Vx          = 3D velocity field
!!    @param[in]    j           = Y-indice of the current line
!!    @param[in]    k           = Z-indice of the current line
!!    @param[in]    Gsize       = size of groups (along X direction)
!!    @param[out]   p_V         = particle velocity
subroutine advecX_init_group(Vx, j, k, Gsize, p_V)

    use cart_topology   ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                       :: j,k
    integer, dimension(2), intent(in)         :: Gsize
    real(WP), dimension(:,:,:),intent(out)    ::  p_V
    real(WP), dimension(:,:,:), intent(in)    :: Vx
    ! Other local variables
    integer                                   :: ind          ! indice
    integer                                   :: j_gp, k_gp   ! Y and Z indice of the current line in the group

    do k_gp = 1, Gsize(2)
        do j_gp = 1, Gsize(1)
            do ind = 1, mesh_sc%N_proc(direction)
                p_V(ind, j_gp, k_gp)        = Vx(ind,j+j_gp-1,k+k_gp-1)
            end do
        end do
    end do

end subroutine advecX_init_group



! ######################################################################################
! #####                                                                            #####
! #####                         Private procedure                                  #####
! #####                                                                            #####
! ######################################################################################

! ==================================================================================================================================
! ====================     Compute scalar slope for introducing limitator (against numerical oscillations)      ====================
! ==================================================================================================================================

!> Compute scalar slopes for introducing limitator
!!    @param[in]        gp_s        = size of a group (ie number of line it gathers along the two other directions)
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        p_pos       = particles position
!!    @param[in]        scalar      = scalar advected by particles
!!    @param[out]       limit       = limitator function
!! @details
!!        This subroutine work on a groupe of line. For each line of this group, it
!!    determine the type of each block of this line and where corrected remeshing
!!    formula are required. In those points, it tagg block transition (ie the end of
!!    the current block and the beginning of the following one) in order to indicate
!!    that corrected weigth have to be used during the remeshing.
!!         Note that the subroutine actually computes limitator/8 as this is the
!!    expression which is used inside the remeshing formula and directly computes it
!!    minimize the number of operations.
subroutine advecX_limitator_group(gp_s, ind_group, j, k, p_pos, &
                & scalar, limit)

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.
    use advec_correction! contains limitator computation
    use precision_tools       ! define working precision_tools (double or simple)

    integer, dimension(2),intent(in)                            :: gp_s         ! groupe size
    integer, dimension(2), intent(in)                           :: ind_group    ! group indice
    integer , intent(in)                                        :: j,k          ! bloc coordinates
    real(WP), dimension(:,:,:), intent(in)                      :: p_pos        ! particle position
    real(WP), dimension(:,:,:), intent(in)                      :: scalar       ! scalar field to advect
    real(WP), dimension(:,:,:), intent(out)                     :: limit        ! limitator function

    ! Local variables
    real(WP),dimension(2,gp_s(1),gp_s(2))                       :: Sbuffer, Rbuffer ! buffer to exchange scalar or limitator at boundaries with neighbors.
    real(WP),dimension(gp_s(1),gp_s(2),mesh_sc%N_proc(direction)+1)     :: deltaS      ! first order scalar variation
    integer                                                     :: ind          ! loop indice on particle indice
    integer                                                     :: send_request ! mpi status of nonblocking send
    integer                                                     :: rece_request ! mpi status of nonblocking receive
    integer, dimension(MPI_STATUS_SIZE)                         :: rece_status  ! mpi status (for mpi_wait)
    integer, dimension(MPI_STATUS_SIZE)                         :: send_status  ! mpi status (for mpi_wait)
    integer, dimension(2)                                       :: tag_table    ! other tags for mpi message
    integer                                                     :: com_size     ! size of mpi message
    integer                                                     :: ierr         ! mpi error code

    ! ===== Initialisation =====
    com_size = 2*gp_s(1)*gp_s(2)

    ! ===== Exchange ghost =====
    ! Receive ghost value, ie value from neighbors boundaries.
    tag_table = compute_tag(ind_group, tag_part_slope, direction)
    call mpi_Irecv(Rbuffer(1,1,1), com_size, MPI_REAL_WP, &
            & neighbors(direction,1), tag_table(1), D_comm(direction), rece_request, ierr)
    ! Send ghost for the two first scalar values of each line
    Sbuffer = scalar(1:2,j:j+gp_s(1)-1,k:k+gp_s(2)-1)
    call mpi_ISsend(Sbuffer(1,1,1), com_size, MPI_REAL_WP, &
            & neighbors(direction,-1), tag_table(1), D_comm(direction), send_request, ierr)

    ! ===== Compute scalar variation =====
    ! -- For the "middle" block --
    do ind = 1, mesh_sc%N_proc(direction)-1
        deltaS(:,:,ind) = scalar(ind+1,j:j+gp_s(1)-1,k:k+gp_s(2)-1) - scalar(ind,j:j+gp_s(1)-1,k:k+gp_s(2)-1)
    end do
    ! -- For the last elements  of each line --
    ! Check reception
    call mpi_wait(rece_request, rece_status, ierr)
    ! Compute delta
    deltaS(:,:,mesh_sc%N_proc(direction)) = Rbuffer(1,:,:) - scalar(ind,j:j+gp_s(1)-1,k:k+gp_s(2)-1)   ! scalar(N+1) - scalar(N)
    deltaS(:,:,mesh_sc%N_proc(direction)+1) = Rbuffer(2,:,:) - Rbuffer(1,:,:)   ! scalar(N+1) - scalar(N)


    ! ===== Compute slope and limitator =====
    call AC_limitator_from_slopes(direction, gp_s, p_pos, deltaS,   &
            & limit, tag_table(2), com_size)

    ! ===== Close mpi_ISsend when done =====
    call mpi_wait(send_request, send_status, ierr)

end subroutine advecX_limitator_group


!> Compute scalar slopes for introducing limitator - no communication variant
!!(for topology without subdivision along X)
!!    @param[in]        gp_s        = size of a group (ie number of line it gathers along the two other directions)
!!    @param[in]        p_pos       = particles position
!!    @param[in]        scalar      = scalar advected by particles
!!    @param[out]       limit       = limitator function
!! @details
!!        This subroutine work on a groupe of line. For each line of this group, it
!!    determine the type of each block of this line and where corrected remeshing
!!    formula are required. In those points, it tagg block transition (ie the end of
!!    the current block and the beginning of the following one) in order to indicate
!!    that corrected weigth have to be used during the remeshing.
!!         Note that the subroutine actually computes limitator/8 as this is the
!!    expression which is used inside the remeshing formula and directly computes it
!!    minimize the number of operations.
subroutine advecX_limitator_group_no_com(gp_s, j, k, p_pos, &
                & scalar, limit)

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.
    use precision_tools       ! define working precision_tools (double or simple)

    integer, dimension(2),intent(in)                            :: gp_s         ! groupe size
    integer , intent(in)                                        :: j,k          ! bloc coordinates
    real(WP), dimension(:,:,:), intent(in)                      :: p_pos        ! particle position
    real(WP), dimension(:,:,:), intent(in)                      :: scalar       ! scalar field to advect
    real(WP), dimension(:,:,:), intent(out)                     :: limit        ! limitator function

    ! Local variables
    real(WP),dimension(gp_s(1),gp_s(2),0:mesh_sc%N_proc(direction)+1)   :: deltaS       ! first order scalar variation
    integer                                                     :: ind          ! loop indice on particle indice
    real(WP),dimension(gp_s(1),gp_s(2))                         :: afl          ! = cfl - [cfl] where [] denotes the nearest int.

    ! ===== Compute scalar variation =====
    ! -- For the "middle" block --
    do ind = 1, mesh_sc%N_proc(direction)-1
        deltaS(:,:,ind) = scalar(ind+1,j:j+gp_s(1)-1,k:k+gp_s(2)-1) - scalar(ind,j:j+gp_s(1)-1,k:k+gp_s(2)-1)
    end do
    ! -- For the first element of each line --
    deltaS(:,:,0) = scalar(1,j:j+gp_s(1)-1,k:k+gp_s(2)-1) - scalar(mesh_sc%N(direction),j:j+gp_s(1)-1,k:k+gp_s(2)-1)   ! scalar(1) - scalar(0)
    ! -- For the last element of each line --
    deltaS(:,:,mesh_sc%N(direction))    = deltaS(:,:,0)   ! scalar(N+1) - scalar(N)
    deltaS(:,:,mesh_sc%N(direction)+1)  = deltaS(:,:,1)   ! scalar(N+2) - scalar(N+1)


    ! ===== Compute slope and limitator =====
    ! Note that limit = limitator function/divided by 8
    ! Van Leer limitator
    do ind = 1, mesh_sc%N_proc(direction)
        where(deltaS(:,:,ind)/=0)
            afl = p_pos(ind,:,:)
            afl = afl - nint(afl)
            ! If (p_pos-nint(p_pos))>=0)
            where(afl>0)
                limit(ind+1,:,:) = max(0.0_WP,(deltaS(:,:,ind-1)/deltaS(:,:,ind)))
                limit(ind+1,:,:) = limit(ind+1,:,:)/(limit(ind+1,:,:)+1)
                limit(ind+1,:,:) = (4.0_WP/8._WP)*min(0.9_WP,(afl+0.5_WP)**2)*limit(ind+1,:,:)
            elsewhere
                limit(ind+1,:,:) = max(0.0_WP,(deltaS(:,:,ind+1)/deltaS(:,:,ind)))
                limit(ind+1,:,:) = limit(ind+1,:,:)/(limit(ind+1,:,:)+1)
                limit(ind+1,:,:) = (4.0_WP/8._WP)*min(0.9_WP,(afl-0.5_WP)**2)*limit(ind+1,:,:)
            end where
        elsewhere
            limit(ind+1,:,:) = 0.0_WP
        end where
    end do
    limit(1,:,:) = limit(mesh_sc%N_proc(direction)+1,:,:)
    ! Classical (corrected) lambda formula: limitator function = 1
    ! limit = 1._WP/8._WP

end subroutine advecX_limitator_group_no_com



end module advecX
!> @}
