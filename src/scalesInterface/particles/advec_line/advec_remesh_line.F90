!USEFORTEST advec
!> @addtogroup part

!------------------------------------------------------------------------------
!
! MODULE: advec_remesh_line
!
!
! DESCRIPTION:
!> The module advec_remesh_line contains different semi-optimized remeshing
!! procedure. They are here for debugging/test/comparaison purpose and will
!! be deleted in "not to far" future (after adding optimized M'6, having a lot
!! of validation and having performed benchmarks).
!
!! The module "test_advec" can be used in order to validate the procedures
!! embedded in this module.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module advec_remesh_line

    use precision_tools
    use advec_abstract_proc
    use advec_correction
    use mpi, only: MPI_INTEGER, MPI_ANY_SOURCE
    implicit none

    ! ===== Public procedures =====
    !----- (corrected) lambda 2 Remeshing method -----
    public                  :: Xremesh_O2       ! order 2
    public                  :: Yremesh_O2       ! order 2
    public                  :: Zremesh_O2       ! order 2
    !----- (corrected) lambda 4 Remeshing method -----
    public                  :: Xremesh_O4       ! order 4
    public                  :: Yremesh_O4       ! order 4
    public                  :: Zremesh_O4       ! order 4
    !----- M'6 remeshing method -----
    public                  :: Xremesh_Mprime6
    public                  :: Yremesh_Mprime6
    public                  :: Zremesh_Mprime6


    ! ===== Private variable ====

contains

! #####################################################################################
! #####                                                                           #####
! #####                          Public procedure                                 #####
! #####                                                                           #####
! #####################################################################################

! ============================================================================
! ====================   Remeshing along X subroutines    ====================
! ============================================================================

!> remeshing along Xwith an order 2 method, corrected to allow large CFL number - group version
!!    @param[in]        direction   = current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along X direction)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        p_V         = particles velocity (needed for tag and type)
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in]        dt          = time step (needed for tag and type)
!!    @param[in,out]    scal        = scalar field to advect
subroutine Xremesh_O2(direction, ind_group, gs, p_pos_adim, p_V, j, k, scal, dt)

    use advec_common            ! Some procedures common to advection along all directions
    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
    real(WP), dimension(:,:,:), intent(inout)   :: scal
    real(WP), intent(in)                        :: dt
    ! Other local variables
    ! To type and tag particles
    logical, dimension(bl_nb(direction)+1,gs(1),gs(2))  :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction),gs(1),gs(2))    :: bl_tag       ! indice of tagged particles
    ! To compute recquired communications
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    ! Variable used to remesh particles in a buffer
    real(WP),dimension(:),allocatable, target:: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
                                                            ! sorted by receivers and not by coordinate.
    integer, dimension(2,gs(1),gs(2))       :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which
                                                            ! I will receive data
    integer, dimension(gs(1),gs(2))         :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, dimension(gs(1),gs(2))         :: proc_max     ! smaller gap between me and the processes to where I send data

    integer                                 :: i1, i2       ! indice of a line into the group

    ! -- Pre-Remeshing: Determine blocks type and tag particles --
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    !  -- Compute ranges --
    where (bl_type(1,:,:))
        ! First particle is a centered one
        send_group_min = nint(p_pos_adim(1,:,:))-1
    elsewhere
        ! First particle is a left one
        send_group_min = floor(p_pos_adim(1,:,:))-1
    end where
    where (bl_type(mesh_sc%N_proc(direction)/bl_size +1,:,:))
        ! Last particle is a centered one
        send_group_max = nint(p_pos_adim(mesh_sc%N_proc(direction),:,:))+1
    elsewhere
        ! Last particle is a left one
        send_group_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:))+1
    end where

    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_group(direction, gs, ind_group, &
      & send_group_min, send_group_max, proc_min, proc_max, rece_proc)

    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_group_min(i1,i2)
            send_j_max = send_group_max(i1,i2)

            ! -- Allocate buffer for remeshing of local particles --
            allocate(send_buffer(send_j_min:send_j_max))
            send_buffer = 0.0;

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda2corrected_basic(direction, p_pos_adim(:,i1,i2), scal(:,j+i1-1,k+i2-1), &
                & bl_type(:,i1,i2), bl_tag(:,i1,i2), send_j_min, send_j_max, send_buffer)

            ! -- Send the buffer to the matching processus and update the scalar field --
            scal(:,j+i1-1,k+i2-1) = 0
            call AC_bufferToScalar_line(direction, ind_group , send_j_min, send_j_max, proc_min(i1,i2), proc_max(i1,i2), &
                & rece_proc(:,i1,i2), send_buffer, scal(:,j+i1-1,k+i2-1))

            ! Deallocate all field
            deallocate(send_buffer)

        end do
    end do

end subroutine Xremesh_O2


!> remeshing along X with an order 4 method, corrected to allow large CFL number - untagged particles
!!    @param[in]        direction   = current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along X direction)
!!    @param[in]        p_pos_adim  = adimensionned particles position
!!    @param[in]        p_V         = particles velocity (needed for tag and type)
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in,out]    scal          = scalar field to advect
!!    @param[in]        dt          = time step (needed for tag and type)
subroutine Xremesh_O4(direction, ind_group, gs, p_pos_adim, p_V, j,k, scal, dt)

    use advec_common            ! Some procedures common to advection along all directions
    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
    real(WP), dimension(:,:,:), intent(inout)   :: scal
    real(WP), intent(in)                        :: dt
    ! Other local variables
    ! To compute recquired communications
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    ! To type and tag particles
    logical, dimension(bl_nb(direction)+1,gs(1),gs(2))  :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction),gs(1),gs(2))    :: bl_tag       ! indice of tagged particles
        ! Variables used to remesh particles ...
        ! ... and to communicate between subdomains. A variable prefixed by "send_"(resp "rece")
        ! designes something I send (resp. I receive).
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
    integer, dimension(2,gs(1),gs(2))   :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which
                                                        ! I will receive data
    integer, dimension(gs(1),gs(2))     :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, dimension(gs(1),gs(2))     :: proc_max     ! smaller gap between me and the processes to where I send data
    integer                             :: i1, i2       ! indice of a line into the group

    ! -- Pre-Remeshing: Determine blocks type and tag particles --
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    !  -- Compute ranges --
    where (bl_type(1,:,:))
        ! First particle is a centered one
        send_group_min = nint(p_pos_adim(1,:,:))-2
    elsewhere
        ! First particle is a left one
        send_group_min = floor(p_pos_adim(1,:,:))-2
    end where
    where (bl_type(mesh_sc%N_proc(direction)/bl_size +1,:,:))
        ! Last particle is a centered one
        send_group_max = nint(p_pos_adim(mesh_sc%N_proc(direction),:,:))+2
    elsewhere
        ! Last particle is a left one
        send_group_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:))+2
    end where

    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_com(direction, gs, ind_group, send_group_min, &
      & send_group_max, proc_min, proc_max, rece_proc)

    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_group_min(i1,i2)
            send_j_max = send_group_max(i1,i2)

            ! -- Allocate buffer for remeshing of local particles --
            allocate(send_buffer(send_j_min:send_j_max))
            send_buffer = 0.0;

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda4corrected_basic(direction, p_pos_adim(:,i1,i2), scal(:,j+i1-1,k+i2-1), &
                & bl_type(:,i1,i2), bl_tag(:,i1,i2), send_j_min, send_j_max, send_buffer)

            ! -- Send the buffer to the matching processus and update the scalar field --
            scal(:,j+i1-1,k+i2-1) = 0
            call AC_bufferToScalar_line(direction, ind_group, send_j_min, send_j_max, proc_min(i1,i2), proc_max(i1,i2), &
                & rece_proc(:,i1,i2), send_buffer, scal(:,j+i1-1,k+i2-1))

            ! Deallocate all field
            deallocate(send_buffer)

        end do
    end do

end subroutine Xremesh_O4


!> remeshing along X with M'6 formula - No tag neither correction for large time steps.
!!    @param[in]        direction   = current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along X direction)
!!    @param[in]        p_pos_adim  = adimensionned particles position
!!    @param[in]        p_V         = particles velocity (only to have the same profile
!!                                      then other remeshing procedures)
!!    @param[in]        j,k         = indice of of the current line (y-coordinate and z-coordinate)
!!    @param[in]        dt          = time step (only to have the same profile
!!                                      then other remeshing procedures)
!!    @param[in,out]    scal        = scalar field to advect
subroutine Xremesh_Mprime6(direction, ind_group, gs, p_pos_adim, p_V, j,k,scal, dt)

    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/outpu
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
    real(WP), dimension(:,:,:), intent(inout)   :: scal
    real(WP), intent(in)                        :: dt
    ! Other local variables
    ! To compute recquired communications
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    ! Variables used to remesh particles ...
        ! ... and to communicate between subdomains. A variable prefixed by "send_"(resp "rece")
        ! designes something I send (resp. I receive).
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
    integer, dimension(2,gs(1),gs(2))   :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which
                                                        ! I will receive data
    integer, dimension(gs(1),gs(2))     :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, dimension(gs(1),gs(2))     :: proc_max     ! smaller gap between me and the processes to where I send data
    integer                             :: i1, i2       ! indice of a line into the group
    integer                             :: i            ! indice of the current particle

    !  -- Compute the remeshing domain --
    send_group_min = floor(p_pos_adim(1,:,:)-2)
    send_group_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:)+3)

    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_com(direction, gs, ind_group, send_group_min, &
      & send_group_max, proc_min, proc_max, rece_proc)

    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_group_min(i1,i2)
            send_j_max = send_group_max(i1,i2)

            ! -- Allocate buffer for remeshing of local particles --
            allocate(send_buffer(send_j_min:send_j_max))
            send_buffer = 0.0;

            ! -- Remesh the particles in the buffer --
            do i = 1, mesh_sc%N_proc(direction), 1
                call AC_remesh_Mprime6(p_pos_adim(i,i1,i2),scal(i,j+i1-1,k+i2-1), send_buffer)
            end do

            ! -- Send the buffer to the matching processus and update the scalar field --
            scal(:,j+i1-1,k+i2-1) = 0
            call AC_bufferToScalar_line(direction, ind_group, send_j_min, send_j_max, proc_min(i1,i2), proc_max(i1,i2), &
                & rece_proc(:,i1,i2), send_buffer, scal(:,j+i1-1,k+i2-1))

            ! Deallocate all field
            deallocate(send_buffer)

        end do
    end do

end subroutine Xremesh_Mprime6


! ============================================================================
! ====================   Remeshing along Y subroutines    ====================
! ============================================================================

!> remeshing along Y with an order 2 method, corrected to allow large CFL number - group version
!!    @param[in]        direction   = current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along Y direction)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        p_V         = particles velocity (needed for tag and type)
!!    @param[in]        i,k         = indice of of the current line (y-coordinate and z-coordinate)
!!    @param[in]        dt          = time step (needed for tag and type)
!!    @param[in,out]    scal        = scalar field to advect
subroutine Yremesh_O2(direction, ind_group, gs, p_pos_adim, P_V,i,k,scal, dt)

    use advec_common            ! Some procedures common to advection along all directions
    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: i, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
    real(WP), dimension(:,:,:), intent(inout)   :: scal
    real(WP), intent(in)                        :: dt
    ! Other local variables
    ! To type and tag particles
    logical, dimension(bl_nb(direction)+1,gs(1),gs(2))  :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction),gs(1),gs(2))    :: bl_tag       ! indice of tagged particles
    ! To compute recquired communications
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    ! Variable used to remesh particles in a buffer
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
    integer, dimension(2,gs(1),gs(2))   :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which
                                                        ! I will receive data
    integer, dimension(gs(1),gs(2))     :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, dimension(gs(1),gs(2))     :: proc_max     ! smaller gap between me and the processes to where I send data

    integer                             :: i1, i2       ! indice of a line into the group

    ! -- Pre-Remeshing: Determine blocks type and tag particles --
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    !  -- Compute ranges --
    where (bl_type(1,:,:))
        ! First particle is a centered one
        send_group_min = nint(p_pos_adim(1,:,:))-1
    elsewhere
        ! First particle is a left one
        send_group_min = floor(p_pos_adim(1,:,:))-1
    end where
    where (bl_type(mesh_sc%N_proc(direction)/bl_size +1,:,:))
        ! Last particle is a centered one
        send_group_max = nint(p_pos_adim(mesh_sc%N_proc(direction),:,:))+1
    elsewhere
        ! Last particle is a left one
        send_group_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:))+1
    end where

    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_group(direction, gs, ind_group, send_group_min, &
      & send_group_max, proc_min, proc_max, rece_proc)

    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_group_min(i1,i2)
            send_j_max = send_group_max(i1,i2)

            ! -- Allocate buffer for remeshing of local particles --
            allocate(send_buffer(send_j_min:send_j_max))
            send_buffer = 0.0;

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda2corrected_basic(direction, p_pos_adim(:,i1,i2), scal(i+i1-1,:,k+i2-1), &
                & bl_type(:,i1,i2), bl_tag(:,i1,i2), send_j_min, send_j_max, send_buffer)

            ! -- Send the buffer to the matching processus and update the scalar field --
            scal(i+i1-1,:,k+i2-1) = 0
            call AC_bufferToScalar_line(direction, ind_group , send_j_min, send_j_max, proc_min(i1,i2), proc_max(i1,i2), &
                & rece_proc(:,i1,i2), send_buffer, scal(i+i1-1,:,k+i2-1))

            ! Deallocate all field
            deallocate(send_buffer)

        end do
    end do

end subroutine Yremesh_O2


!> remeshing along Y with an order 4 method, corrected to allow large CFL number - untagged particles
!!    @param[in]        direction   = current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along Y direction)
!!    @param[in]        p_pos_adim  = adimensionned particles position
!!    @param[in]        p_V         = particles velocity (needed for tag and type)
!!    @param[in]        bl_tag      = contains information about block (is it tagged ?)
!!    @param[in]        i,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in]        bl_type     = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        dt          = time step (needed for tag and type)
subroutine Yremesh_O4(direction, ind_group, gs, p_pos_adim, p_V, i,k,scal, dt)

    use advec_common            ! Some procedures common to advection along all directions
    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! input/output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: i, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
    real(WP), dimension(:,:,:), intent(inout)   :: scal
    real(WP), intent(in)                        :: dt
    ! Other local variables
    ! To compute recquired communications
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    ! To type and tag particles
    logical, dimension(bl_nb(direction)+1,gs(1),gs(2))  :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction),gs(1),gs(2))    :: bl_tag       ! indice of tagged particles
    ! Variables used to remesh particles ...
        ! ... and to communicate between subdomains. A variable prefixed by "send_"(resp "rece")
        ! designes something I send (resp. I receive).
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
    integer, dimension(2,gs(1),gs(2))   :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which
                                                        ! I will receive data
    integer, dimension(gs(1),gs(2))     :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, dimension(gs(1),gs(2))     :: proc_max     ! smaller gap between me and the processes to where I send data
    integer                             :: i1, i2       ! indice of a line into the group

    ! -- Pre-Remeshing: Determine blocks type and tag particles --
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    !  -- Compute ranges --
    where (bl_type(1,:,:))
        ! First particle is a centered one
        send_group_min = nint(p_pos_adim(1,:,:))-2
    elsewhere
        ! First particle is a left one
        send_group_min = floor(p_pos_adim(1,:,:))-2
    end where
    where (bl_type(mesh_sc%N_proc(direction)/bl_size +1,:,:))
        ! Last particle is a centered one
        send_group_max = nint(p_pos_adim(mesh_sc%N_proc(direction),:,:))+2
    elsewhere
        ! Last particle is a left one
        send_group_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:))+2
    end where

    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_group(direction, gs, ind_group, send_group_min, &
      & send_group_max, proc_min, proc_max, rece_proc)

    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_group_min(i1,i2)
            send_j_max = send_group_max(i1,i2)

            ! -- Allocate buffer for remeshing of local particles --
            allocate(send_buffer(send_j_min:send_j_max))
            send_buffer = 0.0;

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda4corrected_basic(direction, p_pos_adim(:,i1,i2), scal(i+i1-1,:,k+i2-1), &
                & bl_type(:,i1,i2), bl_tag(:,i1,i2), send_j_min, send_j_max, send_buffer)

            ! -- Send the buffer to the matching processus and update the scalar field --
            scal(i+i1-1,:,k+i2-1) = 0
            call AC_bufferToScalar_line(direction, ind_group, send_j_min, send_j_max, proc_min(i1,i2), proc_max(i1,i2), &
                & rece_proc(:,i1,i2), send_buffer, scal(i+i1-1,:,k+i2-1))

            ! Deallocate all field
            deallocate(send_buffer)

        end do
    end do

end subroutine Yremesh_O4


!> remeshing along Y with M'6 formula - No tag neither correction for large time steps.
!!    @param[in]        direction   = current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along Y direction)
!!    @param[in]        p_pos_adim  = adimensionned particles position
!!    @param[in]        p_V         = particles velocity (only to have the same profile
!!                                      then other remeshing procedures)
!!    @param[in]        i,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        dt          = time step (only to have the same profile
!!                                      then other remeshing procedures)
subroutine Yremesh_Mprime6(direction, ind_group, gs, p_pos_adim, p_V, i,k,scal, dt)

    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! input/output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: i, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
    real(WP), dimension(:,:,:), intent(inout)   :: scal
    real(WP), intent(in)                        :: dt
    ! Other local variables
    ! To compute recquired communications
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    ! Variables used to remesh particles ...
        ! ... and to communicate between subdomains. A variable prefixed by "send_"(resp "rece")
        ! designes something I send (resp. I receive).
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
    integer, dimension(2,gs(1),gs(2))   :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which
                                                        ! I will receive data
    integer, dimension(gs(1),gs(2))     :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, dimension(gs(1),gs(2))     :: proc_max     ! smaller gap between me and the processes to where I send data
    integer                             :: i1, i2       ! indice of a line into the group
    integer                             :: ind_p        ! indice of the current particle

    !  -- Compute the remeshing domain --
    send_group_min = floor(p_pos_adim(1,:,:)-2)
    send_group_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:)+3)

    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_com(direction, gs, ind_group, send_group_min, &
      & send_group_max, proc_min, proc_max, rece_proc)

    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_group_min(i1,i2)
            send_j_max = send_group_max(i1,i2)

            !  -- Allocate and initialize the buffer --
            allocate(send_buffer(send_j_min:send_j_max))
            send_buffer = 0.0;

            ! -- Remesh the particles in the buffer --
            do ind_p = 1, mesh_sc%N_proc(direction), 1
                call AC_remesh_Mprime6(p_pos_adim(ind_p,i1,i2),scal(i+i1-1,ind_p,k+i2-1), send_buffer)
            end do

            ! -- Send the buffer to the matching processus and update the scalar field --
            scal(i+i1-1,:,k+i2-1) = 0
            call AC_bufferToScalar_line(direction, ind_group, send_j_min, send_j_max, proc_min(i1,i2), proc_max(i1,i2), &
                & rece_proc(:,i1,i2), send_buffer, scal(i+i1-1,:,k+i2-1))

            ! Deallocate all field
            deallocate(send_buffer)

        end do
    end do

end subroutine Yremesh_Mprime6


! ============================================================================
! ====================   Remeshing along Z subroutines    ====================
! ============================================================================

!> remeshing along Z with an order 2 method, corrected to allow large CFL number - group version
!!    @param[in]        direction   = current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along Z direction)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        p_V         = particles velocity (needed for tag and type)
!!    @param[in]        i,j         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        dt          = time step (needed for tag and type)
subroutine Zremesh_O2(direction, ind_group, gs, p_pos_adim, p_V,i,j,scal, dt)

    use advec_common            ! Some procedures common to advection along all directions
    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: i, j
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
    real(WP), dimension(:,:,:), intent(inout)   :: scal
    real(WP), intent(in)                        :: dt
    ! Other local variables
    ! To compute recquired communications
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    ! To type and tag particles
    logical, dimension(bl_nb(direction)+1,gs(1),gs(2))  :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction),gs(1),gs(2))    :: bl_tag       ! indice of tagged particles
    ! Variable used to remesh particles in a buffer
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
    integer, dimension(2,gs(1),gs(2))   :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which
                                                        ! I will receive data
    integer, dimension(gs(1),gs(2))     :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, dimension(gs(1),gs(2))     :: proc_max     ! smaller gap between me and the processes to where I send data

    integer                             :: i1, i2       ! indice of a line into the group

    ! -- Pre-Remeshing: Determine blocks type and tag particles --
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    !  -- Compute ranges --
    where (bl_type(1,:,:))
        ! First particle is a centered one
        send_group_min = nint(p_pos_adim(1,:,:))-1
    elsewhere
        ! First particle is a left one
        send_group_min = floor(p_pos_adim(1,:,:))-1
    end where
    where (bl_type(mesh_sc%N_proc(direction)/bl_size +1,:,:))
        ! Last particle is a centered one
        send_group_max = nint(p_pos_adim(mesh_sc%N_proc(direction),:,:))+1
    elsewhere
        ! Last particle is a left one
        send_group_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:))+1
    end where

    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_group(direction, gs, ind_group, send_group_min, &
      & send_group_max, proc_min, proc_max, rece_proc)

    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_group_min(i1,i2)
            send_j_max = send_group_max(i1,i2)

            ! -- Allocate buffer for remeshing of local particles --
            allocate(send_buffer(send_j_min:send_j_max))
            send_buffer = 0.0;

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda2corrected_basic(direction, p_pos_adim(:,i1,i2), scal(i+i1-1,j+i2-1,:), &
                & bl_type(:,i1,i2), bl_tag(:,i1,i2), send_j_min, send_j_max, send_buffer)

            ! -- Send the buffer to the matching processus and update the scalar field --
            scal(i+i1-1,j+i2-1,:) = 0
            call AC_bufferToScalar_line(direction, ind_group , send_j_min, send_j_max, proc_min(i1,i2), proc_max(i1,i2), &
                & rece_proc(:,i1,i2), send_buffer, scal(i+i1-1,j+i2-1,:))

            ! Deallocate all field
            deallocate(send_buffer)

        end do
    end do

end subroutine Zremesh_O2


!> remeshing along Z with an order 4 method, corrected to allow large CFL number - untagged particles
!!    @param[in]        direction   = current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along Z direction)
!!    @param[in]        p_pos_adim  = adimensionned particles position
!!    @param[in]        p_V         = particles velocity (needed for tag and type)
!!    @param[in]        i,j         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in,out]    scal          = scalar field to advect
!!    @param[in]        dt          = time step (needed for tag and type)
subroutine Zremesh_O4(direction, ind_group, gs, p_pos_adim, p_V,i,j,scal, dt)

    use advec_common            ! Some procedures common to advection along all directions
    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: i, j
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
    real(WP), dimension(:,:,:), intent(inout)   :: scal
    real(WP), intent(in)                        :: dt
    ! Other local variables
    ! To type and tag particles
    logical, dimension(bl_nb(direction)+1,gs(1),gs(2))  :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction),gs(1),gs(2))    :: bl_tag       ! indice of tagged particles
    ! To compute recquired communications
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    ! Variables used to remesh particles ...
        ! ... and to communicate between subdomains. A variable prefixed by "send_"(resp "rece")
        ! designes something I send (resp. I receive).
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
    integer, dimension(2,gs(1),gs(2))   :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which
                                                        ! I will receive data
    integer, dimension(gs(1),gs(2))     :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, dimension(gs(1),gs(2))     :: proc_max     ! smaller gap between me and the processes to where I send data
    integer                             :: i1, i2       ! indice of a line into the group

    ! -- Pre-Remeshing: Determine blocks type and tag particles --
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    !  -- Compute ranges --
    where (bl_type(1,:,:))
        ! First particle is a centered one
        send_group_min = nint(p_pos_adim(1,:,:))-2
    elsewhere
        ! First particle is a left one
        send_group_min = floor(p_pos_adim(1,:,:))-2
    end where
    where (bl_type(mesh_sc%N_proc(direction)/bl_size +1,:,:))
        ! Last particle is a centered one
        send_group_max = nint(p_pos_adim(mesh_sc%N_proc(direction),:,:))+2
    elsewhere
        ! Last particle is a left one
        send_group_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:))+2
    end where

    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_group(direction, gs, ind_group, send_group_min, &
      & send_group_max, proc_min, proc_max, rece_proc)

    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_group_min(i1,i2)
            send_j_max = send_group_max(i1,i2)

            ! -- Allocate buffer for remeshing of local particles --
            allocate(send_buffer(send_j_min:send_j_max))
            send_buffer = 0.0;

            ! -- Remesh the particles in the buffer --
            call AC_remesh_lambda4corrected_basic(direction, p_pos_adim(:,i1,i2), scal(i+i1-1,j+i2-1,:), &
                & bl_type(:,i1,i2), bl_tag(:,i1,i2), send_j_min, send_j_max, send_buffer)

            ! -- Send the buffer to the matching processus and update the scalar field --
            scal(i+i1-1,j+i2-1,:) = 0
            call AC_bufferToScalar_line(direction, ind_group , send_j_min, send_j_max, proc_min(i1,i2), proc_max(i1,i2), &
                & rece_proc(:,i1,i2), send_buffer, scal(i+i1-1,j+i2-1,:))

            ! Deallocate all field
            deallocate(send_buffer)

        end do
    end do

end subroutine Zremesh_O4


!> remeshing along Z with M'6 formula - No tag neither correction for large time steps.
!!    @param[in]        direction   = current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along Z direction)
!!    @param[in]        p_pos_adim  = adimensionned particles position
!!    @param[in]        p_V         = particles velocity (only to have the same profile
!!                                      then other remeshing procedures)
!!    @param[in]        i,j         = indice of of the current line (x-coordinate and y-coordinate)
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        dt          = time step (only to have the same profile
!!                                      then other remeshing procedures)
subroutine Zremesh_Mprime6(direction, ind_group, gs, p_pos_adim, p_V, i,j,scal, dt)

    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! input/output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: i, j
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
    real(WP), dimension(:,:,:), intent(inout)   :: scal
    real(WP), intent(in)                        :: dt
    ! Other local variables
    ! To compute recquired communications
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    ! Variables used to remesh particles ...
        ! ... and to communicate between subdomains. A variable prefixed by "send_"(resp "rece")
        ! designes something I send (resp. I receive).
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
    integer, dimension(2,gs(1),gs(2))   :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which
                                                        ! I will receive data
    integer, dimension(gs(1),gs(2))     :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, dimension(gs(1),gs(2))     :: proc_max     ! smaller gap between me and the processes to where I send data
    integer                             :: i1, i2       ! indice of a line into the group
    integer                             :: ind_p        ! indice of the current particle

    !  -- Compute the remeshing domain --
    send_group_min = floor(p_pos_adim(1,:,:)-2)
    send_group_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:)+3)

    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_com(direction, gs, ind_group, send_group_min, &
      & send_group_max, proc_min, proc_max, rece_proc)

    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            send_j_min = send_group_min(i1,i2)
            send_j_max = send_group_max(i1,i2)

            !  -- Allocate and initialize the buffer --
            allocate(send_buffer(send_j_min:send_j_max))
            send_buffer = 0.0;

            ! -- Remesh the particles in the buffer --
            do ind_p = 1, mesh_sc%N_proc(direction), 1
                call AC_remesh_Mprime6(p_pos_adim(ind_p,i1,i2),scal(i+i1-1,j+i2-1, ind_p), send_buffer)
            end do

            ! -- Send the buffer to the matching processus and update the scalar field --
            scal(i+i1-1,j+i2-1,:) = 0
            call AC_bufferToScalar_line(direction, ind_group, send_j_min, send_j_max, proc_min(i1,i2), proc_max(i1,i2), &
                & rece_proc(:,i1,i2), send_buffer, scal(i+i1-1,j+i2-1,:))

            ! Deallocate all field
            deallocate(send_buffer)

        end do
    end do

end subroutine Zremesh_Mprime6


! #####################################################################################
! #####                                                                           #####
! #####                          Private procedure                                 #####
! #####                                                                           #####
! #####################################################################################

! =====================================================================================
! ====================   Remeshing tool to determine comunications ====================
! =====================================================================================

!> Determine the set of processes wich will send me information during the remeshing
!! and compute for each of these processes the range of wanted data. Use implicit
!! computation rather than communication (only possible if particle are gather by
!! block whith contrainst on velocity variation - as corrected lambda formula.) -
!! work directly on a group of particles lines.
!     @param[in]    send_group_min  = minimal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!     @param[in]    send_group_max  = maximal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[in]    direction       = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]    ind_group       = coordinate of the current group of lines
!!    @param[out]   send_min        = minimal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[out]   send_max        = maximal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[out]   proc_min        = gap between my coordinate and the processes of minimal coordinate which will receive information from me
!!    @param[out]   proc_max        = gap between my coordinate and the processes of maximal coordinate which will receive information from me
!!    @param[out]   rece_proc       = coordinate range of processes which will send me information during the remeshing.
!!    @param[in]    gp_s            = size of group of line along the current direction
!! @details
!!    Work on a group of line of size gs(1) x gs(2))
!!    Obtain the list of processts which are associated to sub-domain where my partticles
!!    will be remeshed and the list of processes wich contains particles which
!!    have to be remeshed in my sub-domain. This way, this procedure determine
!!    which processus need to communicate together in order to proceed to the
!!    remeshing (as in a parrallel context the real space is subdivised and each
!!    processus contains a part of it)
!!        In the same time, it computes for each processus with which I will
!!    communicate, the range of mesh point involved for each line of particles
!!    inside the group and it stores it by using some sparse matrix technics
!!    (see cartography defined in the algorithm documentation)
!!        This routine does not involve any computation to determine if
!!    a processus is the first or the last processes (considering its coordinate along
!!    the current directory) to send remeshing information to a given processes.
!!    It directly compute it using contraints on velocity (as in corrected lambda
!!    scheme) When possible use it rather than AC_obtain_senders_com
subroutine AC_obtain_senders_group(direction, gp_s, ind_group, send_min, send_max, proc_min, proc_max, rece_proc)
! XXX Work only for periodic condition. For dirichlet conditions : it is
! possible to not receive either rece_proc(1), either rece_proc(2) or none of
! these two => detect it (track the first and the last particles) and deal with it.

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/output
    integer, intent(in)                             :: direction
    integer, dimension(2), intent(in)               :: ind_group
    integer(kind=4), dimension(:,:), intent(out)    :: proc_min, proc_max
    integer, dimension(:,:,:), intent(out)          :: rece_proc
    integer, dimension(2), intent(in)               :: gp_s
    integer, dimension(:,:), intent(in)             :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(in)             :: send_max     ! distance between me and processus wich send me information
    ! Other local variable
    integer(kind=4)                                 :: proc_gap         ! gap between a processus coordinate (along the current
                                                                        ! direction) into the mpi-topology and my coordinate
    integer                                         :: rankP, rankN     ! processus rank for shift (P= previous, N = next)
    integer, dimension(2)                           :: tag_table        ! mpi message tag (for communicate rece_proc(1) and rece_proc(2))
    integer                                         :: proc_max_abs     ! maximum of proc_max array
    integer                                         :: proc_min_abs     ! minimum of proc_min array
    integer, dimension(:,:), allocatable            :: first, last      ! Storage processus to which I will be the first (or the last) to send
                                                                        ! remeshed particles
    integer, dimension(2)                           :: first_condition  ! allowed range of value of proc_min and proc_max for being the first
    integer, dimension(2)                           :: last_condition   ! allowed range of value of proc_min and proc_max for being the last
    integer, dimension(:,:),allocatable             :: send_request     ! mpi status of nonblocking send
    integer                                         :: ierr             ! mpi error code
    integer, dimension(MPI_STATUS_SIZE)             :: statut           ! mpi status
    integer                                         :: ind1, ind2       ! indice of the current line inside the group
    integer                                         :: min_size         ! begin indice in first and last to stock indice along first dimension of the group line
    integer                                         :: max_size         ! maximum size of first/last along the first direction
    integer                                         :: indice           ! internal indice
    integer, dimension(1 + gp_s(2)*(2+gp_s(1)))     :: rece_buffer      ! buffer for reception of rece_max

    rece_proc = 3*mesh_sc%N(direction)
    max_size = size(rece_buffer) + 1

    proc_min = floor(real(send_min-1, WP)/mesh_sc%N_proc(direction))
    proc_max = floor(real(send_max-1, WP)/mesh_sc%N_proc(direction))
    proc_min_abs = minval(proc_min)
    proc_max_abs = maxval(proc_max)

    allocate(send_request(proc_min_abs:proc_max_abs,3))
    send_request(:,3) = 0

    ! -- Determine if I am the first or the last to send information to a given
    ! processus and sort line by target processes for which I am the first and
    ! for which I am the last. --
    tag_table = compute_tag(ind_group, tag_obtsend_NP, direction)
    min_size = 2 + gp_s(2)
    allocate(first(max_size,proc_min_abs:proc_max_abs))
    first = 0
    first(1,:) = min_size
    allocate(last(max_size,proc_min_abs:proc_max_abs))
    last = 0
    last(1,:) = min_size
    do proc_gap = proc_min_abs, proc_max_abs
        first(2,proc_gap) = -proc_gap
        last(2,proc_gap) = -proc_gap
        first_condition(2) = proc_gap*mesh_sc%N_proc(direction)+1
        first_condition(1) = 1-2*bl_bound_size + first_condition(2)
        last_condition(2)  = (proc_gap+1)*mesh_sc%N_proc(direction)
        last_condition(1)  = -1+2*bl_bound_size + last_condition(2)
        do ind2 = 1, gp_s(2)
            first(2+ind2,proc_gap) = 0
            last(2+ind2,proc_gap) =  0
            do ind1 = 1, gp_s(1)
                ! Compute if I am the first.
                if ((send_min(ind1,ind2)< first_condition(1)).AND. &
                        & (send_max(ind1,ind2)>= first_condition(2))) then
                    first(2+ind2,proc_gap) =  first(2+ind2,proc_gap)+1
                    first(1,proc_gap) = first(1,proc_gap) + 1
                    first(first(1,proc_gap),proc_gap) = ind1
                end if
                ! Compute if I am the last.
                if ((send_max(ind1,ind2) > last_condition(1)) &
                            & .AND.(send_min(ind1,ind2)<= last_condition(2))) then
                    last(2+ind2,proc_gap) =  last(2+ind2,proc_gap)+1
                    last(1,proc_gap) = last(1,proc_gap) + 1
                    last(last(1,proc_gap),proc_gap) = ind1
                end if
            end do
        end do
    end do

#ifdef PART_DEBUG
    do proc_gap = proc_min_abs, proc_max_abs
        if (first(1,proc_gap)>max_size) then
            print*, 'too big array on proc = ', cart_rank, ' - proc_gap = ', proc_gap
            print*, 'it occurs on AC_obtain_senders_group - array concerned : "first"'
            print*, 'first = ', first(1,proc_gap)
        end if
        if (last(1,proc_gap)>max_size) then
            print*, 'too big array on proc = ', cart_rank, ' - proc_gap = ', proc_gap
            print*, 'it occurs on AC_obtain_senders_group - array concerned : "last"'
            print*, 'last = ', last(1,proc_gap)
        end if
    end do
#endif

    ! -- Send information if I am the first or the last --
    do proc_gap = proc_min_abs, proc_max_abs
        ! I am the first ?
        if (first(1,proc_gap)>min_size) then
            ! Compute the rank of the target processus
            call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, rankN, ierr)
            if(rankN /= D_rank(direction)) then
                call mpi_Isend(first(2,proc_gap), first(1,proc_gap)-1, MPI_INTEGER, rankN, tag_table(1), D_comm(direction), &
                        & send_request(proc_gap,1), ierr)
                send_request(proc_gap,3) = 1
            else
                indice = min_size
                do ind2 = 1, gp_s(2)
                    do ind1 = 1, first(2+ind2,proc_gap)
                        indice = indice+1
                        rece_proc(1,first(indice,proc_gap),ind2) = -proc_gap
                    end do
                end do
            end if
        end if
        ! I am the last ?
        if (last(1,proc_gap)>min_size) then
            ! Compute the rank of the target processus
            call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, rankN, ierr)
            if(rankN /= D_rank(direction)) then
                call mpi_Isend(last(2,proc_gap), last(1,proc_gap)-1, MPI_INTEGER, rankN, tag_table(2), D_comm(direction), &
                        & send_request(proc_gap,2), ierr)
                send_request(proc_gap,3) = send_request(proc_gap, 3) + 2
            else
                indice = min_size
                do ind2 = 1, gp_s(2)
                    do ind1 = 1, last(2+ind2,proc_gap)
                        indice = indice+1
                        rece_proc(2,last(indice,proc_gap),ind2) = -proc_gap
                    end do
                end do
            end if
        end if
    end do

    ! -- Receive it --
    ! size_max = size(rece_buffer) ! 2 + 2*gp_s(1)*gp_s(2)
    max_size = max_size-1
    do while(any(rece_proc(1,:,:) == 3*mesh_sc%N(direction)))
        call mpi_recv(rece_buffer(1), max_size, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(1), D_comm(direction), statut, ierr)
        indice = min_size-1
        do ind2 = 1, gp_s(2)
            do ind1 = 1, rece_buffer(1+ind2)
                indice = indice+1
                rece_proc(1,rece_buffer(indice),ind2) = rece_buffer(1)
            end do
        end do
    end do
    do while(any(rece_proc(2,:,:) == 3*mesh_sc%N(direction)))
        call mpi_recv(rece_buffer(1), max_size, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(2), D_comm(direction), statut, ierr)
        indice = min_size-1
        do ind2 = 1, gp_s(2)
            do ind1 = 1, rece_buffer(1+ind2)
                indice = indice+1
                rece_proc(2,rece_buffer(indice),ind2) = rece_buffer(1)
            end do
        end do
    end do

    ! -- Free Isend buffer --
    do proc_gap = proc_min_abs, proc_max_abs
        select case (send_request(proc_gap,3))
            case (3)
                call mpi_wait(send_request(proc_gap,1), statut, ierr)
                call mpi_wait(send_request(proc_gap,2), statut, ierr)
            case (2)
                call mpi_wait(send_request(proc_gap,2), statut, ierr)
            case (1)
                call mpi_wait(send_request(proc_gap,1), statut, ierr)
        end select
    end do

    deallocate(first)
    deallocate(last)
    deallocate(send_request)

end subroutine AC_obtain_senders_group


!> Determine the set of processes wich will send me information during the
!!  scalar remeshing by explicit (and exensive) way : communications !
!!    @param[in]    direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]    ind_group   = coordinate of the current group of lines
!!    @param[out]   send_min        = minimal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[out]   send_max        = maximal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[out]   proc_min    = gap between my coordinate and the processes of minimal coordinate which will receive information from me
!!    @param[out]   proc_max    = gap between my coordinate and the processes of maximal coordinate which will receive information from me
!!    @param[out]   rece_proc   = coordinate range of processes which will send me information during the remeshing.
!!    @param[in]    gp_s        = size of group of line along the current direction
!!    @param[in]    com         = integer used to distinguish this function from AC_obtain_senders_group.
!! @details
!!    Obtain the list of processus which contains some particles which belong to
!!    my subdomains after their advection (and thus which will be remeshing into
!!    my subdomain). This result is return as an interval [send_min; send_max].
!!    All the processus whose coordinate (into the current direction) belong to
!!    this segment are involved into scalar remeshing into the current
!!    subdomains. Use this method when the sender are not predictable without
!!    communication, as in M'6 schemes for instance. More precisly, it
!!    correspond do scheme without bloc of particle involving velocity variation
!!    contrainsts to avoid that the distance between to particle grows (or dimishes)
!!    too much.
subroutine AC_obtain_senders_com(direction, gp_s, ind_group, send_min, send_max, proc_min, proc_max, rece_proc)
! XXX Work only for periodic condition. See AC_obtain_senders. Adapt it for
! other condition must be more easy.

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/output
    integer, intent(in)                             :: direction
    integer, dimension(2), intent(in)               :: ind_group
    integer(kind=4), dimension(:,:), intent(out)    :: proc_min, proc_max
    integer, dimension(:,:,:), intent(out)          :: rece_proc
    integer, dimension(2), intent(in)               :: gp_s
    integer, dimension(:,:), intent(in)             :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(in)             :: send_max     ! distance between me and processus wich send me information
    ! Other local variable
    integer(kind=4)                                 :: proc_gap         ! gap between a processus coordinate (along the current
                                                                        ! direction) into the mpi-topology and my coordinate
    integer                                         :: rankP, rankN     ! processus rank for shift (P= previous, N = next)
    integer, dimension(2)                           :: tag_table        ! mpi message tag (for communicate rece_proc(1) and rece_proc(2))
    integer, dimension(gp_s(1), gp_s(2))            :: proc_max_prev    ! maximum gap between previous processus and the receivers of its remeshing buffer
    integer, dimension(gp_s(1), gp_s(2))            :: proc_min_next    ! minimum gap between next processus and the receivers of its remeshing buffer
    integer                                         :: proc_max_abs     ! maximum of proc_max array
    integer                                         :: proc_min_abs     ! minimum of proc_min array
    integer, dimension(:,:), allocatable            :: first, last      ! Storage processus to which I will be the first (or the last) to send
                                                                        ! remeshed particles
    integer, dimension(:,:),allocatable             :: send_request     ! mpi status of nonblocking send
    integer                                         :: ierr             ! mpi error code
    integer, dimension(MPI_STATUS_SIZE)             :: statut           ! mpi status
    integer                                         :: ind1, ind2       ! indice of the current line inside the group
    integer                                         :: min_size         ! begin indice in first and last to stock indice along first dimension of the group line
    integer                                         :: max_size         ! maximum size of first/last along the first direction
    integer                                         :: indice           ! internal indice
    integer, dimension(1 + gp_s(2)*(2+gp_s(1)))     :: rece_buffer      ! buffer for reception of rece_max


    rece_proc = 3*mesh_sc%N(direction)
    max_size = size(rece_buffer) + 1

    proc_min = floor(real(send_min-1, WP)/mesh_sc%N_proc(direction))
    proc_max = floor(real(send_max-1, WP)/mesh_sc%N_proc(direction))
    proc_min_abs = minval(proc_min)
    proc_max_abs = maxval(proc_max)

    allocate(send_request(proc_min_abs:proc_max_abs,3))
    send_request(:,3) = 0

    ! -- Exchange send_block_min and send_block_max to determine if I am the first
    ! or the last to send information to a given target processus. --
    min_size = gp_s(1)*gp_s(2)
    ! Compute message tag - we re-use tag_part_tag_NP id as using this procedure
    ! suppose not using "AC_type_and_block"
    tag_table = compute_tag(ind_group, tag_part_tag_NP, direction)
    ! Exchange "ghost"
    call mpi_Sendrecv(proc_min(1,1), min_size, MPI_INTEGER, neighbors(direction,1), tag_table(1), &
            & proc_min_next(1,1), min_size, MPI_INTEGER, neighbors(direction,2), tag_table(1),    &
            & D_comm(direction), statut, ierr)
    call mpi_Sendrecv(proc_max(1,1), min_size, MPI_INTEGER, neighbors(direction,2), tag_table(2), &
            & proc_max_prev(1,1), min_size, MPI_INTEGER, neighbors(direction,1), tag_table(2),    &
            & D_comm(direction), statut, ierr)

    ! -- Determine if I am the first or the last to send information to a given
    ! processus and sort line by target processes for which I am the first and
    ! for which I am the last. --
    tag_table = compute_tag(ind_group, tag_obtsend_NP, direction)
    min_size = 2 + gp_s(2)
    allocate(first(max_size,proc_min_abs:proc_max_abs))
    first = 0
    first(1,:) = min_size
    allocate(last(max_size,proc_min_abs:proc_max_abs))
    last = 0
    last(1,:) = min_size
    do proc_gap = proc_min_abs, proc_max_abs
        first(2,proc_gap) = -proc_gap
        last(2,proc_gap) = -proc_gap
    end do
    do ind2 = 1, gp_s(2)
        first(2+ind2,:) = 0
        last(2+ind2,:) =  0
        do ind1 = 1, gp_s(1)
            ! Compute if I am the first, ie if:
            ! a - proc_min <= proc_gap <= proc_max,
            ! b - proc_gap > proc_max_prev -1.
            do proc_gap = max(proc_min(ind1,ind2), proc_max_prev(ind1,ind2)), proc_max(ind1,ind2)
                first(2+ind2,proc_gap) =  first(2+ind2,proc_gap)+1
                first(1,proc_gap) = first(1,proc_gap) + 1
                first(first(1,proc_gap),proc_gap) = ind1
            end do
            ! Compute if I am the last, ie if:
            ! a - proc_min <= proc_gap <= proc_max,
            ! b - proc_gap < proc_min_next+1.
            do proc_gap = proc_min(ind1,ind2), min(proc_min_next(ind1,ind2), proc_max(ind1,ind2))
                last(2+ind2,proc_gap) =  last(2+ind2,proc_gap)+1
                last(1,proc_gap) = last(1,proc_gap) + 1
                last(last(1,proc_gap),proc_gap) = ind1
            end do
        end do
    end do

#ifdef PART_DEBUG
    do proc_gap = proc_min_abs, proc_max_abs
        if (first(1,proc_gap)>max_size) then
            print*, 'too big array on proc = ', cart_rank, ' - proc_gap = ', proc_gap
            print*, 'it occurs on AC_obtain_senders_group - array concerned : "first"'
            print*, 'first = ', first(1,proc_gap)
        end if
        if (last(1,proc_gap)>max_size) then
            print*, 'too big array on proc = ', cart_rank, ' - proc_gap = ', proc_gap
            print*, 'it occurs on AC_obtain_senders_group - array concerned : "last"'
            print*, 'last = ', last(1,proc_gap)
        end if
    end do
#endif

    ! -- Send information if I am the first or the last --
    do proc_gap = proc_min_abs, proc_max_abs
        ! I am the first ?
        if (first(1,proc_gap)>min_size) then
            ! Compute the rank of the target processus
            call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, rankN, ierr)
            if(rankN /= D_rank(direction)) then
                call mpi_Isend(first(2,proc_gap), first(1,proc_gap)-1, MPI_INTEGER, rankN, tag_table(1), D_comm(direction), &
                        & send_request(proc_gap,1), ierr)
                send_request(proc_gap,3) = 1
            else
                indice = min_size
                do ind2 = 1, gp_s(2)
                    do ind1 = 1, first(2+ind2,proc_gap)
                        indice = indice+1
                        rece_proc(1,first(indice,proc_gap),ind2) = -proc_gap
                    end do
                end do
            end if
        end if
        ! I am the last ?
        if (last(1,proc_gap)>min_size) then
            ! Compute the rank of the target processus
            call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, rankN, ierr)
            if(rankN /= D_rank(direction)) then
                call mpi_Isend(last(2,proc_gap), last(1,proc_gap)-1, MPI_INTEGER, rankN, tag_table(2), D_comm(direction), &
                        & send_request(proc_gap,2), ierr)
                send_request(proc_gap,3) = send_request(proc_gap, 3) + 2
            else
                indice = min_size
                do ind2 = 1, gp_s(2)
                    do ind1 = 1, last(2+ind2,proc_gap)
                        indice = indice+1
                        rece_proc(2,last(indice,proc_gap),ind2) = -proc_gap
                    end do
                end do
            end if
        end if
    end do


    ! -- Receive it --
    ! size_max = size(rece_buffer) ! 2 + 2*gp_s(1)*gp_s(2)
    max_size = max_size-1
    do while(any(rece_proc(1,:,:) == 3*mesh_sc%N(direction)))
        call mpi_recv(rece_buffer(1), max_size, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(1), D_comm(direction), statut, ierr)
        indice = min_size-1
        do ind2 = 1, gp_s(2)
            do ind1 = 1, rece_buffer(1+ind2)
                indice = indice+1
                rece_proc(1,rece_buffer(indice),ind2) = rece_buffer(1)
            end do
        end do
    end do
    do while(any(rece_proc(2,:,:) == 3*mesh_sc%N(direction)))
        call mpi_recv(rece_buffer(1), max_size, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(2), D_comm(direction), statut, ierr)
        indice = min_size-1
        do ind2 = 1, gp_s(2)
            do ind1 = 1, rece_buffer(1+ind2)
                indice = indice+1
                rece_proc(2,rece_buffer(indice),ind2) = rece_buffer(1)
            end do
        end do
    end do

    ! -- Free Isend buffer --
    do proc_gap = proc_min_abs, proc_max_abs
        select case (send_request(proc_gap,3))
            case (3)
                call mpi_wait(send_request(proc_gap,1), statut, ierr)
                call mpi_wait(send_request(proc_gap,2), statut, ierr)
            case (2)
                call mpi_wait(send_request(proc_gap,2), statut, ierr)
            case (1)
                call mpi_wait(send_request(proc_gap,1), statut, ierr)
        end select
    end do

    deallocate(first)
    deallocate(last)
    deallocate(send_request)

end subroutine AC_obtain_senders_com


end module advec_remesh_line
!> @}
