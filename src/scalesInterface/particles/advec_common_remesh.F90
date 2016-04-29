!USEFORTEST advec
!> @addtogroup part
!! @{
!------------------------------------------------------------------------------
!
! MODULE: advec_common
!
!
! DESCRIPTION:
!> The module ``advec_common'' gather function and subroutines used to advec scalar
!! which are not specific to a direction
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
!! remeshing formula, the dimensionnal splitting and everything else.
!!
!! The module "test_advec" can be used in order to validate the procedures
!! embedded in this module.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module advec_common_remesh

    use precision_tools
    use advec_abstract_proc
    use mpi, only: MPI_REQUEST_NULL, MPI_STATUS_SIZE, MPI_INTEGER, MPI_ANY_SOURCE
    implicit none


    ! Information about the particles and their bloc
    public


    ! ===== Public procedures =====
    !----- Init remeshing context -----
    public  :: AC_setup_init
    public  :: AC_remesh_setup_alongX
    public  :: AC_remesh_setup_alongY
    public  :: AC_remesh_setup_alongZ
    !----- To remesh particles -----
    public                        :: AC_remesh_lambda_group
    public                        :: AC_remesh_Mprime_group
    !----- Tools to  remesh particles -----
    public                        :: AC_remesh_range
    public                        :: AC_remesh_determine_communication
    public                        :: AC_remesh_cartography

    ! ===== Private procedures =====
    !----- Prepare and perform communication required during remeshing -----
    private :: AC_remesh_init
    private :: AC_remesh_finalize

    ! ===== Public variables =====

    ! ===== Private variables =====
    !> Pointer to subroutine wich remesh particle to a buffer - for formula of lambda family (with tag/type).
    procedure(remesh_in_buffer_type), pointer, private      :: remesh_in_buffer_lambda_pt => null()
    !> Pointer to subroutine wich remesh particle to a buffer - for formula of lambda family (with tag/type).
    procedure(remesh_in_buffer_limit), pointer, private     :: remesh_in_buffer_limit_lambda_pt => null()
    !> Pointer to subroutine wich remesh particle to a buffer - for formula of M' family (without tag/type).
    procedure(remesh_in_buffer_notype), pointer, private    :: remesh_in_buffer_Mprime_pt => null()
    !> Pointer to subroutine wich redistribute a buffer (containing remeshed
    !! particle) inside the original scalar field.
    procedure(remesh_buffer_to_scalar), pointer, private    :: remesh_buffer_to_scalar_pt => null()
    !> Pointer to subroutine which compute scalar slope along the current
    !! direction and then computes the limitator function (divided by 8)
    procedure(advec_limitator_group), pointer, private      :: advec_limitator            => null()


contains

! ===== Public procedure =====

! ================================================================================ !
! =============     To deal with remeshing setup and generecity      ============= !
! ================================================================================ !

!> Init remesh_line_pt for the right remeshing formula
subroutine AC_setup_init()

    use advec_remeshing_lambda
    use advec_remeshing_Mprime

    call AC_remesh_init_lambda()
    call AC_remesh_init_Mprime()

end subroutine AC_setup_init

!> Setup remesh_in_buffer and remesh_in_buffer_to_scalar for remeshing along X
subroutine AC_remesh_setup_alongX()
    use advecX

    remesh_in_buffer_lambda_pt      => advecX_remesh_in_buffer_lambda
    remesh_in_buffer_limit_lambda_pt=> advecX_remesh_in_buffer_limit_lambda
    remesh_in_buffer_Mprime_pt      => advecX_remesh_in_buffer_Mprime

    remesh_buffer_to_scalar_pt      => advecX_remesh_buffer_to_scalar

    advec_limitator                 => advecX_limitator_group

end subroutine AC_remesh_setup_alongX

!> Setup remesh_in_buffer and remesh_in_buffer_to_scalar for remeshing along X
subroutine AC_remesh_setup_alongY()
    use advecY

    remesh_in_buffer_lambda_pt      => advecY_remesh_in_buffer_lambda
    remesh_in_buffer_limit_lambda_pt=> advecY_remesh_in_buffer_limit_lambda
    remesh_in_buffer_Mprime_pt      => advecY_remesh_in_buffer_Mprime
    remesh_buffer_to_scalar_pt      => advecY_remesh_buffer_to_scalar

    advec_limitator                 => advecY_limitator_group

end subroutine AC_remesh_setup_alongY

!> Setup remesh_in_buffer and remesh_in_buffer_to_scalar for remeshing along Z
subroutine AC_remesh_setup_alongZ()
    use advecZ

    remesh_in_buffer_lambda_pt      => advecZ_remesh_in_buffer_lambda
    remesh_in_buffer_limit_lambda_pt=> advecZ_remesh_in_buffer_limit_lambda
    remesh_in_buffer_Mprime_pt      => advecZ_remesh_in_buffer_Mprime
    remesh_buffer_to_scalar_pt      => advecZ_remesh_buffer_to_scalar

    advec_limitator                 => advecZ_limitator_group

end subroutine AC_remesh_setup_alongZ


! ==============================================================================================
! ====================     Remesh all the particles of a group of lines     ====================
! ==============================================================================================


!> remeshing with an order 2 or 4 lambda method, corrected to allow large CFL number - group version
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along X direction)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        p_V         = particles velocity (needed for tag and type)
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        dt          = time step (needed for tag and type)
subroutine AC_remesh_lambda_group(direction, ind_group, gs, p_pos_adim, p_V, j, k, scal, dt)

    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology
    use advec_correction        ! To compute type and tag

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
    ! Others
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    integer, dimension(gs(1),gs(2),2)       :: send_gap     ! distance between me and processus wich send me information
    integer, dimension(2)                   :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer, dimension(2 , 2)               :: rece_gap     ! distance between me and processus to wich I send information
    integer, dimension(:,:), allocatable    :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block to wich the
    ! Variable use to manage mpi communications
    integer                                 :: max_size     ! maximal size of cartography(:,proc_gap)

    ! ===== Pre-Remeshing: Determine blocks type and tag particles =====
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    ! ===== Compute range of remeshing data =====
    call AC_remesh_range(bl_type, p_pos_adim, direction, send_group_min, send_group_max, send_gap, send_gap_abs)

    ! ===== Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) =====
    ! -- Allocation --
    max_size = 2 + gs(2)*(2+3*gs(1))
    allocate(cartography(max_size,send_gap_abs(1):send_gap_abs(2)))
    ! -- Determine which processes communicate together --
    call AC_remesh_determine_communication(direction, gs, ind_group, send_group_min, send_group_max, &
        & rece_gap, send_gap, send_gap_abs, cartography)

    ! ===== Proceed to remeshing via a local buffer =====
    call AC_remesh_via_buffer_lambda(direction, ind_group, gs, p_pos_adim, j, k,&
        & scal, send_group_min, send_group_max, send_gap_abs, rece_gap,         &
        & cartography, bl_type, bl_tag)

    ! -- Free all communication buffer and data --
    deallocate(cartography)

end subroutine AC_remesh_lambda_group


!> remeshing with an order 2 limited lambda method, corrected to allow large CFL number - group version
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along X direction)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        p_V         = particles velocity (needed for tag and type)
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        dt          = time step (needed for tag and type)
subroutine AC_remesh_limit_lambda_group(direction, ind_group, gs, p_pos_adim, p_V, j, k, scal, dt)

    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology
    use advec_correction        ! To compute type and tag

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
    real(WP), dimension(mesh_sc%N_proc(direction)+1,gs(1),gs(2)):: limit        ! limitator function (divided by 8.)
    ! Others
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    integer, dimension(gs(1),gs(2),2)       :: send_gap     ! distance between me and processus wich send me information
    integer, dimension(2)                   :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer, dimension(2 , 2)               :: rece_gap     ! distance between me and processus to wich I send information
    integer, dimension(:,:), allocatable    :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block to wich the
                                                            ! current processus will send data during remeshing and for each of these lines the range
                                                            ! of mesh points from where it requiers the velocity values.
    ! Variable use to manage mpi communications
    integer                                 :: max_size     ! maximal size of cartography(:,proc_gap)

    ! ===== Pre-Remeshing I: Determine blocks type and tag particles =====
    call AC_type_and_block_group(dt, direction, gs, ind_group, p_V, bl_type, bl_tag)

    ! ===== Compute range of remeshing data =====
    call AC_remesh_range(bl_type, p_pos_adim, direction, send_group_min, send_group_max, send_gap, send_gap_abs)

    ! ===== Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) =====
    ! -- Allocation --
    max_size = 2 + gs(2)*(2+3*gs(1))
    allocate(cartography(max_size,send_gap_abs(1):send_gap_abs(2)))
    ! -- Determine which processes communicate together --
    call AC_remesh_determine_communication(direction, gs, ind_group, send_group_min, send_group_max, &
        & rece_gap, send_gap, send_gap_abs, cartography)

    ! ===== Pre-Remeshing II: Compute the limitor function =====
    ! Actually, this subroutine compute [limitator/8] as this is this fraction
    ! wich appear always in the remeshing polynoms.
    call advec_limitator(gs, ind_group, j, k, p_pos_adim, scal, limit)

    ! ===== Proceed to remeshing via a local buffer =====
    call AC_remesh_via_buffer_limit_lambda(direction, ind_group, gs, p_pos_adim,&
        & j, k, scal, send_group_min, send_group_max, send_gap_abs, rece_gap,   &
        & cartography, bl_type, bl_tag, limit)

    ! -- Free all communication buffer and data --
    deallocate(cartography)

end subroutine AC_remesh_limit_lambda_group


!> remeshing with a M'6 or M'8 remeshing formula - group version
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along X direction)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        p_V         = particles velocity (needed for tag and type)
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        dt          = time step (needed for tag and type)
subroutine AC_remesh_Mprime_group(direction, ind_group, gs, p_pos_adim, p_V, j, k, scal, dt)

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
    integer, dimension(gs(1), gs(2))        :: send_group_min     ! distance between me and processus wich send me information
    integer, dimension(gs(1), gs(2))        :: send_group_max     ! distance between me and processus wich send me information
    integer, dimension(gs(1),gs(2),2)       :: send_gap     ! distance between me and processus wich send me information
    integer, dimension(2)                   :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer, dimension(2 , 2)               :: rece_gap     ! distance between me and processus to wich I send information
    integer, dimension(:,:), allocatable    :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block to wich the
                                                            ! current processus will send data during remeshing and for each of these lines the range
                                                            ! of mesh points from where it requiers the velocity values.
    ! Variable use to manage mpi communications
    integer                                 :: max_size     ! maximal size of cartography(:,proc_gap)

    ! ===== Compute range of remeshing data =====
    call AC_remesh_range_notype(p_pos_adim, direction, send_group_min, send_group_max, send_gap, send_gap_abs)

    ! ===== Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) =====
    ! -- Allocation --
    max_size = 2 + gs(2)*(2+3*gs(1))
    allocate(cartography(max_size,send_gap_abs(1):send_gap_abs(2)))
    ! -- Determine which processes communicate together --
    call AC_remesh_determine_communication_com(direction, gs, ind_group, &
        & rece_gap, send_gap, send_gap_abs, cartography)

    ! ===== Proceed to remeshing via a local buffer =====
    call AC_remesh_via_buffer_Mprime(direction, ind_group, gs, p_pos_adim,  &
        &  j, k, scal, send_group_min, send_group_max, send_gap_abs,        &
        &  rece_gap, cartography)

    ! -- Free all communication buffer and data --
    deallocate(cartography)

end subroutine AC_remesh_Mprime_group


! ===================================================================================================
! ===== Tools to remesh particles: variant of remeshing via buffer for each family of remeshing =====
! ===================================================================================================


!> Using input information to update the scalar field by creating particle
!! weight (from scalar values), set scalar to 0, redistribute particle inside
!! - variant for corrected lambda remeshing formula.
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along X direction)
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        bl_type     = table of blocks type (center of left)
!!    @param[in]        bl_tag      = inform about tagged particles (bl_tag(ind_bl)=1 if the end of the bl_ind-th block
!!                                    and the begining of the following one is tagged)
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        send_min    = minimal indice of mesh involved in remeshing particles
!!    @param[in]        send_max    = maximal indice of mesh involved in remeshing particles
!!    @param[in]        send_gap_abs= send_gap_abs(i) is the min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
!!    @param[in]        rece_gap    = coordinate range of processes which will send me information during the remeshing.
!!    @param[in,out]    cartography = cartography(proc_gap) contains the set of the lines indice in the block for wich the
!!                                    current processus requiers data from proc_gap and for each of these lines the range
!!                                    of mesh points from where it requiers the velocity values.
!! @details
!!    This procedure manage all communication needed. To minimize communications,
!! particles are remeshing inside a local buffer wich is after send to the
!! processus wich contain the right sub-domain depending off the particle
!! position. There is no need of communication in order to remesh inside the
!! buffer. To avoid recopy in creating particle weight (which will be weight
!! = scalar), scalar is directly redistribute inside the local buffer.
!! This provides:
!!    a - Remesh particle: redistribute scalar field inside a local buffer and
!!        set scalar = 0.
!!    b - Send local buffer to its target processus and update the scalar field,
!!        ie scalar = scalar + received buffer.
!! "remesh_in_buffer_pt" do the part "a" and "remesh_buffer_to_scalar" the part
!! B except the communication. The current subroutine manage all the
!! communications (and other stuff needed to allow correctness).
subroutine AC_remesh_via_buffer_lambda(direction, ind_group, gs, p_pos_adim,   &
        & j, k, scal, send_min, send_max, send_gap_abs, rece_gap, cartography, &
        & bl_type, bl_tag)

    use advec_variables         ! contains info about solver parameters and others.
    use advec_abstract_proc     ! contain some useful procedure pointers.
    use advecX                  ! procedure specific to advection alongX
    use advecY                  ! procedure specific to advection alongY
    use advecZ                  ! procedure specific to advection alongZ
    use cart_topology           ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    logical,dimension(:,:,:),intent(in)         :: bl_type      ! is the particle block a center block or a left one ?
    logical,dimension(:,:,:),intent(in)         :: bl_tag       ! indice of tagged particles
    real(WP),dimension(:,:,:),intent(inout)     :: scal
    integer, dimension(:,:), intent(in)         :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(in)         :: send_max     ! distance between me and processus wich send me information
    integer, dimension(2), intent(in)           :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer, dimension(2, 2), intent(in)        :: rece_gap     ! distance between me and processus to wich I send information
    integer, dimension(:,:), intent(inout)      :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block to wich the
                                                            ! current processus will send data during remeshing and for each of these lines the range
                                                            ! of mesh points from where it requiers the velocity values.
    ! Other local variables
    ! Others
    integer, dimension(:,:), allocatable    :: rece_carto   ! same as abobve but for what I receive
    integer                                 :: min_size     ! minimal size of cartography(:,proc_gap)
    integer                                 :: max_size     ! maximal size of cartography(:,proc_gap)
    ! Variable used to remesh particles in a buffer
    real(WP),dimension(:),allocatable,target:: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
                                                            ! sorted by receivers and not by coordinate.
    integer, dimension(:), allocatable      :: pos_in_buffer! buffer size
    ! Variable use to manage mpi communications
    integer, dimension(:), allocatable      :: s_request_ran! mpi communication request (handle) of nonblocking send
    integer, dimension(:), allocatable      :: r_request_ran! mpi communication request (handle) of nonblocking receive
    integer, dimension(:,:), allocatable    :: r_status     ! mpi communication status of nonblocking receive
    integer, dimension(:,:), allocatable    :: s_status     ! mpi communication status of nonblocking send
    integer                                 :: ierr         ! mpi error code
    integer                                 :: nb_r, nb_s   ! number of reception/send


    ! ===== Allocation =====
    ! -- allocate request about cartography (non-blocking) reception --
    nb_r = rece_gap(1,2) - rece_gap(1,1) + 1
    allocate(r_request_ran(1:nb_r))
    r_request_ran = MPI_REQUEST_NULL
    ! -- allocate cartography about what I receive --
    max_size = 2 + gs(2)*(2+3*gs(1))
    allocate(rece_carto(max_size,rece_gap(1,1):rece_gap(1,2)))
    ! -- allocate request about cartography (non-blocking) send --
    nb_s = send_gap_abs(2) - send_gap_abs(1) + 1
    allocate(s_request_ran(1:nb_s))
    ! -- To manage buffer --
    ! Position of sub-buffer associated different mpi-processes
    allocate(pos_in_buffer(0:nb_s))

    ! ===== Init the remeshing process: pre-process  =====
    ! Perform a cartography of mesh points where particle will be remesh,
    ! create a 1D to buffer where remeshing will be performed and create
    ! tools to manage it.
    call AC_remesh_init(direction, ind_group, gs, send_min, send_max, &
        & send_gap_abs, rece_gap, nb_s, cartography, rece_carto,      &
        & pos_in_buffer, min_size, max_size, s_request_ran, r_request_ran)


    ! ===== Initialize the general buffer =====
    allocate(send_buffer(pos_in_buffer(nb_s) &
                & + cartography(1,nb_s)-1))
    send_buffer = 0.0

    ! ===== Remeshing into the buffer by using pointer array =====
    call remesh_in_buffer_lambda_pt(gs, j, k, send_gap_abs(1)-1, p_pos_adim, bl_type, bl_tag, send_min, &
            & send_max, scal, send_buffer, pos_in_buffer)
    ! Observe that now:
    ! => pos_in_buffer(i-1) = first (1D-)indice of the sub-array of send_buffer
    ! associated to he i-rd mpi-processus to wich I will send remeshed particles.

    ! ===== Wait for reception of all cartography =====
    allocate(r_status(MPI_STATUS_SIZE,1:nb_r))
    call mpi_waitall(nb_r,r_request_ran, r_status, ierr)
    deallocate(r_request_ran)
    deallocate(r_status)
    !allocate(s_status(MPI_STATUS_SIZE,1:nb_s))
    !allocate(ind_array(send_gap_abs(1):send_gap_abs(2)))
    !call mpi_testsome(size(s_request_ran),s_request_ran, ind_1Dtable, ind_array, s_status, ierr)
    !deallocate(ind_array)

    ! ===== Finish the remeshing process =====
    ! Send buffer, receive some other buffers and update scalar field.
    call AC_remesh_finalize(direction, ind_group, gs, j, k, scal, send_gap_abs, rece_gap, &
      & nb_r, nb_s, cartography, rece_carto, send_buffer, pos_in_buffer, min_size)


    ! ===== Free memory and communication buffer ====
    ! -- Deallocate all field --
    deallocate(rece_carto)
    ! -- Check if Isend are done --
    allocate(s_status(MPI_STATUS_SIZE,1:nb_s))
    call mpi_waitall(nb_s, s_request_ran, s_status, ierr)
    deallocate(s_status)
    ! -- Free all communication buffer and data --
    deallocate(send_buffer)
    deallocate(pos_in_buffer)
    deallocate(s_request_ran)

end subroutine AC_remesh_via_buffer_lambda


!> Using input information to update the scalar field by creating particle
!! weight (from scalar values), set scalar to 0, redistribute particle inside
!! - variant for corrected and limited lambda remeshing formula.
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along X direction)
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        bl_type     = table of blocks type (center of left)
!!    @param[in]        bl_tag      = inform about tagged particles (bl_tag(ind_bl)=1 if the end of the bl_ind-th block
!!                                    and the begining of the following one is tagged)
!!    @param[in]        limit       = limitator function
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        send_min    = minimal indice of mesh involved in remeshing particles
!!    @param[in]        send_max    = maximal indice of mesh involved in remeshing particles
!!    @param[in]        send_gap_abs= send_gap_abs(i) is the min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
!!    @param[in]        rece_gap    = coordinate range of processes which will send me information during the remeshing.
!!    @param[in,out]    cartography = cartography(proc_gap) contains the set of the lines indice in the block for wich the
!!                                    current processus requiers data from proc_gap and for each of these lines the range
!!                                    of mesh points from where it requiers the velocity values.
!! @details
!!    This procedure manage all communication needed. To minimize communications,
!! particles are remeshing inside a local buffer wich is after send to the
!! processus wich contain the right sub-domain depending off the particle
!! position. There is no need of communication in order to remesh inside the
!! buffer. To avoid recopy in creating particle weight (which will be weight
!! = scalar), scalar is directly redistribute inside the local buffer.
!! This provides:
!!    a - Remesh particle: redistribute scalar field inside a local buffer and
!!        set scalar = 0.
!!    b - Send local buffer to its target processus and update the scalar field,
!!        ie scalar = scalar + received buffer.
!! "remesh_in_buffer_pt" do the part "a" and "remesh_buffer_to_scalar" the part
!! B except the communication. The current subroutine manage all the
!! communications (and other stuff needed to allow correctness).
subroutine AC_remesh_via_buffer_limit_lambda(direction, ind_group, gs, p_pos_adim,  &
        & j, k, scal, send_min, send_max, send_gap_abs, rece_gap, cartography,      &
        & bl_type, bl_tag, limit)

    use advec_variables         ! contains info about solver parameters and others.
    use advec_abstract_proc     ! contain some useful procedure pointers.
    use advecX                  ! procedure specific to advection alongX
    use advecY                  ! procedure specific to advection alongY
    use advecZ                  ! procedure specific to advection alongZ
    use cart_topology           ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    logical, dimension(:,:,:), intent(in)       :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(:,:,:), intent(in)       :: bl_tag       ! indice of tagged particles
    real(WP), dimension(:,:,:), intent(in)      :: limit        ! limitator function (divided by 8.)
    real(WP),dimension(:,:,:),intent(inout)     :: scal
    integer, dimension(:,:), intent(in)         :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(in)         :: send_max     ! distance between me and processus wich send me information
    integer, dimension(2), intent(in)           :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer, dimension(2, 2), intent(in)        :: rece_gap     ! distance between me and processus to wich I send information
    integer, dimension(:,:), intent(inout)      :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block to wich the
                                                            ! current processus will send data during remeshing and for each of these lines the range
                                                            ! of mesh points from where it requiers the velocity values.
    ! Other local variables
    ! Others
    integer, dimension(:,:), allocatable    :: rece_carto   ! same as abobve but for what I receive
    integer                                 :: min_size     ! minimal size of cartography(:,proc_gap)
    integer                                 :: max_size     ! maximal size of cartography(:,proc_gap)
    ! Variable used to remesh particles in a buffer
    real(WP),dimension(:),allocatable,target:: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
                                                            ! sorted by receivers and not by coordinate.
    integer, dimension(:), allocatable      :: pos_in_buffer! buffer size
    ! Variable use to manage mpi communications
    integer, dimension(:), allocatable      :: s_request_ran! mpi communication request (handle) of nonblocking send
    integer, dimension(:), allocatable      :: r_request_ran! mpi communication request (handle) of nonblocking receive
    integer, dimension(:,:), allocatable    :: r_status     ! mpi communication status of nonblocking receive
    integer, dimension(:,:), allocatable    :: s_status     ! mpi communication status of nonblocking send
    integer                                 :: ierr         ! mpi error code
    integer                                 :: nb_r, nb_s   ! number of reception/send


    ! ===== Allocation =====
    ! -- allocate request about cartography (non-blocking) reception --
    nb_r = rece_gap(1,2) - rece_gap(1,1) + 1
    allocate(r_request_ran(1:nb_r))
    r_request_ran = MPI_REQUEST_NULL
    ! -- allocate cartography about what I receive --
    max_size = 2 + gs(2)*(2+3*gs(1))
    allocate(rece_carto(max_size,rece_gap(1,1):rece_gap(1,2)))
    ! -- allocate request about cartography (non-blocking) send --
    nb_s = send_gap_abs(2) - send_gap_abs(1) + 1
    allocate(s_request_ran(1:nb_s))
    ! -- To manage buffer --
    ! Position of sub-buffer associated different mpi-processes
    allocate(pos_in_buffer(0:nb_s))

    ! ===== Init the remeshing process: pre-process  =====
    ! Perform a cartography of mesh points where particle will be remesh,
    ! create a 1D to buffer where remeshing will be performed and create
    ! tools to manage it.
    call AC_remesh_init(direction, ind_group, gs, send_min, send_max, &
        & send_gap_abs, rece_gap, nb_s, cartography, rece_carto,      &
        & pos_in_buffer, min_size, max_size, s_request_ran, r_request_ran)


    ! ===== Initialize the general buffer =====
    allocate(send_buffer(pos_in_buffer(nb_s) &
                & + cartography(1,nb_s)-1))
    send_buffer = 0.0

    ! ===== Remeshing into the buffer by using pointer array =====
    call remesh_in_buffer_limit_lambda_pt(gs, j, k, send_gap_abs(1)-1, p_pos_adim, bl_type, bl_tag, limit,  &
            & send_min, send_max, scal, send_buffer, pos_in_buffer)
    ! Observe that now:
    ! => pos_in_buffer(i-1) = first (1D-)indice of the sub-array of send_buffer
    ! associated to he i-rd mpi-processus to wich I will send remeshed particles.

    ! ===== Wait for reception of all cartography =====
    allocate(r_status(MPI_STATUS_SIZE,1:nb_r))
    call mpi_waitall(nb_r,r_request_ran, r_status, ierr)
    deallocate(r_request_ran)
    deallocate(r_status)
    !allocate(s_status(MPI_STATUS_SIZE,1:nb_s))
    !allocate(ind_array(send_gap_abs(1):send_gap_abs(2)))
    !call mpi_testsome(size(s_request_ran),s_request_ran, ind_1Dtable, ind_array, s_status, ierr)
    !deallocate(ind_array)

    ! ===== Finish the remeshing process =====
    ! Send buffer, receive some other buffers and update scalar field.
    call AC_remesh_finalize(direction, ind_group, gs, j, k, scal, send_gap_abs, rece_gap, &
      & nb_r, nb_s, cartography, rece_carto, send_buffer, pos_in_buffer, min_size)


    ! ===== Free memory and communication buffer ====
    ! -- Deallocate all field --
    deallocate(rece_carto)
    ! -- Check if Isend are done --
    allocate(s_status(MPI_STATUS_SIZE,1:nb_s))
    call mpi_waitall(nb_s, s_request_ran, s_status, ierr)
    deallocate(s_status)
    ! -- Free all communication buffer and data --
    deallocate(send_buffer)
    deallocate(pos_in_buffer)
    deallocate(s_request_ran)

end subroutine AC_remesh_via_buffer_limit_lambda


!> Using input information to update the scalar field by creating particle
!! weight (from scalar values), set scalar to 0, redistribute particle inside
!! - variant for M' remeshing formula.
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        gs          = size of groups (along X direction)
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in,out]    scal        = scalar field to advect
!!    @param[in]        send_min    = minimal indice of mesh involved in remeshing particles
!!    @param[in]        send_max    = maximal indice of mesh involved in remeshing particles
!!    @param[in]        send_gap_abs= send_gap_abs(i) is the min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
!!    @param[in]        rece_gap    = coordinate range of processes which will send me information during the remeshing.
!!    @param[in,out]    cartography = cartography(proc_gap) contains the set of the lines indice in the block for wich the
!!                                    current processus requiers data from proc_gap and for each of these lines the range
!!                                    of mesh points from where it requiers the velocity values.
!! @details
!!    This procedure manage all communication needed. To minimize communications,
!! particles are remeshing inside a local buffer wich is after send to the
!! processus wich contain the right sub-domain depending off the particle
!! position. There is no need of communication in order to remesh inside the
!! buffer. To avoid recopy in creating particle weight (which will be weight
!! = scalar), scalar is directly redistribute inside the local buffer.
!! This provides:
!!    a - Remesh particle: redistribute scalar field inside a local buffer and
!!        set scalar = 0.
!!    b - Send local buffer to its target processus and update the scalar field,
!!        ie scalar = scalar + received buffer.
!! "remesh_in_buffer_pt" do the part "a" and "remesh_buffer_to_scalar" the part
!! B except the communication. The current subroutine manage all the
!! communications (and other stuff needed to allow correctness).
subroutine AC_remesh_via_buffer_Mprime(direction, ind_group, gs, p_pos_adim, &
        & j, k, scal, send_min, send_max, send_gap_abs, rece_gap, cartography)

    use advec_variables         ! contains info about solver parameters and others.
    use advec_abstract_proc     ! contain some useful procedure pointers.
    use advecX                  ! procedure specific to advection alongX
    use advecY                  ! procedure specific to advection alongY
    use advecZ                  ! procedure specific to advection alongZ
    use cart_topology           ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
    real(WP),dimension(:,:,:),intent(inout)     :: scal
    integer, dimension(:,:), intent(in)         :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(in)         :: send_max     ! distance between me and processus wich send me information
    integer, dimension(2), intent(in)           :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer, dimension(2, 2), intent(in)        :: rece_gap     ! distance between me and processus to wich I send information
    integer, dimension(:,:), intent(inout)      :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block to wich the
                                                            ! current processus will send data during remeshing and for each of these lines the range
                                                            ! of mesh points from where it requiers the velocity values.
    ! Other local variables
    ! Others
    integer, dimension(:,:), allocatable    :: rece_carto   ! same as abobve but for what I receive
    integer                                 :: min_size     ! minimal size of cartography(:,proc_gap)
    integer                                 :: max_size     ! maximal size of cartography(:,proc_gap)
    ! Variable used to remesh particles in a buffer
    real(WP),dimension(:),allocatable,target:: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
                                                            ! sorted by receivers and not by coordinate.
    integer, dimension(:), allocatable      :: pos_in_buffer! buffer size
    ! Variable use to manage mpi communications
    integer, dimension(:), allocatable      :: s_request_ran! mpi communication request (handle) of nonblocking send
    integer, dimension(:), allocatable      :: r_request_ran! mpi communication request (handle) of nonblocking receive
    integer, dimension(:,:), allocatable    :: r_status     ! mpi communication status of nonblocking receive
    integer, dimension(:,:), allocatable    :: s_status     ! mpi communication status of nonblocking send
    integer                                 :: ierr         ! mpi error code
    integer                                 :: nb_r, nb_s   ! number of reception/send


    ! ===== Allocation =====
    ! -- allocate request about cartography (non-blocking) reception --
    nb_r = rece_gap(1,2) - rece_gap(1,1) + 1
    allocate(r_request_ran(1:nb_r))
    r_request_ran = MPI_REQUEST_NULL
    ! -- allocate cartography about what I receive --
    max_size = 2 + gs(2)*(2+3*gs(1))
    allocate(rece_carto(max_size,rece_gap(1,1):rece_gap(1,2)))
    ! -- allocate request about cartography (non-blocking) send --
    nb_s = send_gap_abs(2) - send_gap_abs(1) + 1
    allocate(s_request_ran(1:nb_s))
    ! -- To manage buffer --
    ! Position of sub-buffer associated different mpi-processes
    allocate(pos_in_buffer(0:nb_s))

    ! ===== Init the remeshing process: pre-process  =====
    ! Perform a cartography of mesh points where particle will be remesh,
    ! create a 1D to buffer where remeshing will be performed and create
    ! tools to manage it.
    call AC_remesh_init(direction, ind_group, gs, send_min, send_max, &
        & send_gap_abs, rece_gap, nb_s, cartography, rece_carto,      &
        & pos_in_buffer, min_size, max_size, s_request_ran, r_request_ran)


    ! ===== Initialize the general buffer =====
    allocate(send_buffer(pos_in_buffer(nb_s) &
                & + cartography(1,nb_s)-1))
    send_buffer = 0.0

    ! ===== Remeshing into the buffer by using pointer array =====
    call remesh_in_buffer_Mprime_pt(gs, j, k, send_gap_abs(1)-1, p_pos_adim, send_min, &
            & send_max, scal, send_buffer, pos_in_buffer)
    ! Observe that now:
    ! => pos_in_buffer(i-1) = first (1D-)indice of the sub-array of send_buffer
    ! associated to he i-rd mpi-processus to wich I will send remeshed particles.

    ! ===== Wait for reception of all cartography =====
    allocate(r_status(MPI_STATUS_SIZE,1:nb_r))
    call mpi_waitall(nb_r,r_request_ran, r_status, ierr)
    deallocate(r_request_ran)
    deallocate(r_status)
    !allocate(s_status(MPI_STATUS_SIZE,1:nb_s))
    !allocate(ind_array(send_gap_abs(1):send_gap_abs(2)))
    !call mpi_testsome(size(s_request_ran),s_request_ran, ind_1Dtable, ind_array, s_status, ierr)
    !deallocate(ind_array)

    ! ===== Finish the remeshing process =====
    ! Send buffer, receive some other buffers and update scalar field.
    call AC_remesh_finalize(direction, ind_group, gs, j, k, scal, send_gap_abs, rece_gap, &
      & nb_r, nb_s, cartography, rece_carto, send_buffer, pos_in_buffer, min_size)


    ! ===== Free memory and communication buffer ====
    ! -- Deallocate all field --
    deallocate(rece_carto)
    ! -- Check if Isend are done --
    allocate(s_status(MPI_STATUS_SIZE,1:nb_s))
    call mpi_waitall(nb_s, s_request_ran, s_status, ierr)
    deallocate(s_status)
    ! -- Free all communication buffer and data --
    deallocate(send_buffer)
    deallocate(pos_in_buffer)
    deallocate(s_request_ran)

end subroutine AC_remesh_via_buffer_Mprime


! ==================================================================================
! ====================     Other tools to remesh particles      ====================
! ==================================================================================

!> Determine where the particles of each lines will be remeshed
!!    @param[in]    bl_type         = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]    p_pos_adim      = adimensionned  particles position
!!    @param[in]    direction       = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[out]   send_min        = minimal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[out]   send_max        = maximal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[out]   send_gap        = distance between me and processus wich send me information (for each line of the group)
!!    @param[out]   send_gap_abs    = send_gap_abs(i) is the min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
subroutine AC_remesh_range(bl_type, p_pos_adim, direction, send_min, send_max, send_gap, send_gap_abs)

    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/Output
    logical, dimension(:,:,:), intent(in)   :: bl_type      ! is the particle block a center block or a left one ?
    real(WP), dimension(:,:,:), intent(in)  :: p_pos_adim   ! adimensionned particles position
    integer, intent(in)                     :: direction
    integer, dimension(:,:), intent(out)    :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(out)    :: send_max     ! distance between me and processus wich send me information
    integer, dimension(:,:,:), intent(out)  :: send_gap     ! distance between me and processus wich send me information
    integer, dimension(2), intent(out)      :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)

    !  -- Compute ranges --
    where (bl_type(1,:,:))
        ! First particle is a centered one
        send_min = nint(p_pos_adim(1,:,:))-remesh_stencil(1)
    elsewhere
        ! First particle is a left one
        send_min = floor(p_pos_adim(1,:,:))-remesh_stencil(1)
    end where
    where (bl_type(bl_nb(direction)+1,:,:))
        ! Last particle is a centered one
        send_max = nint(p_pos_adim(mesh_sc%N_proc(direction),:,:))+remesh_stencil(2)
    elsewhere
        ! Last particle is a left one
        send_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:))+remesh_stencil(2)
    end where

    ! -- What have I to communicate ? --
    send_gap(:,:,1) = floor(real(send_min-1, WP)/mesh_sc%N_proc(direction))
    send_gap(:,:,2) = floor(real(send_max-1, WP)/mesh_sc%N_proc(direction))
    send_gap_abs(1) = minval(send_gap(:,:,1))
    send_gap_abs(2) = maxval(send_gap(:,:,2))

end subroutine AC_remesh_range


!> Determine where the particles of each lines will be remeshed - Variant for
!! remeshing without type/tag
!!    @param[in]    p_pos_adim      = adimensionned  particles position
!!    @param[in]    direction       = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[out]   send_min        = minimal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[out]   send_max        = maximal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[out]   send_gap        = distance between me and processus wich send me information (for each line of the group)
!!    @param[out]   send_gap_abs    = send_gap_abs(i) is the min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
subroutine AC_remesh_range_notype(p_pos_adim, direction, send_min, send_max, send_gap, send_gap_abs)

    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/Output
    real(WP), dimension(:,:,:), intent(in)  :: p_pos_adim   ! adimensionned particles position
    integer, intent(in)                     :: direction
    integer, dimension(:,:), intent(out)    :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(out)    :: send_max     ! distance between me and processus wich send me information
    integer, dimension(:,:,:), intent(out)  :: send_gap     ! distance between me and processus wich send me information
    integer, dimension(2), intent(out)      :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)

    !  -- Compute ranges --
    send_min = floor(p_pos_adim(1,:,:))-remesh_stencil(1)
    send_max = floor(p_pos_adim(mesh_sc%N_proc(direction),:,:))+remesh_stencil(2)

    ! -- What have I to communicate ? --
    send_gap(:,:,1) = floor(real(send_min-1, WP)/mesh_sc%N_proc(direction))
    send_gap(:,:,2) = floor(real(send_max-1, WP)/mesh_sc%N_proc(direction))
    send_gap_abs(1) = minval(send_gap(:,:,1))
    send_gap_abs(2) = maxval(send_gap(:,:,2))

end subroutine AC_remesh_range_notype


!> Determine the set of processes wich will send me information during the remeshing
!! and compute for each of these processes the range of wanted data. Use implicit
!! computation rather than communication (only possible if particle are gather by
!! block whith contrainst on velocity variation - as corrected lambda formula.) -
!! work directly on a group of particles lines.
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        remesh_min  =  minimal indice of meshes where I will remesh my particles.
!!    @param[in]        remesh_max  =  maximal indice of meshes where I will remesh my particles.
!!    @param[out]       rece_gap    = coordinate range of processes which will send me information during the remeshing.
!!    @param[in]        send_gap    = distance between me and processus wich send me information (for each line of the group)
!!    @param[in]        send_gap_abs= send_gap_abs(i) is the min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
!!    @param[in,out]    cartography = cartography(proc_gap) contains the set of the lines indice in the block for wich the
!!                                    current processus requiers data from proc_gap and for each of these lines the range
!!                                    of mesh points from where it requiers the velocity values.
!! @details
!!    Work on a group of line of size gs(1) x gs(2))
!!    Obtain the list of processus which are associated to sub-domain where my particles
!!    will be remeshed and the list of processes wich contains particles which
!!    have to be remeshed in my sub-domain. This way, this procedure determine
!!    which processus need to communicate together in order to proceed to the
!!    remeshing (as in a parrallel context the real space is subdivised and each
!!    processus contains a part of it)
!!        In the same time, it computes for each processus with which I will
!!    communicate, the range of mesh point involved for each line of particles
!!    inside the group and it stores it by using some sparse matrix technics
!!    (see cartography defined in the algorithm documentation)
!!        This routine does not involve any communication to determine if
!!    a processus is the first or the last processes (considering its coordinate along
!!    the current directory) to send remeshing information to a given processes.
!!    It directly compute it using contraints on velocity (as in corrected lambda
!!    scheme) When possible use it rather than AC_obtain_senders_com
subroutine AC_remesh_determine_communication(direction, gs, ind_group, remesh_min, remesh_max, &
    & rece_gap, send_gap, send_gap_abs, cartography)
! XXX Work only for periodic condition. For dirichlet conditions : it is
! possible to not receive either rece_gap(1), either rece_gap(2) or none of
! these two => detect it (track the first and the last particles) and deal with it.

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/output
    integer, intent(in)                                 :: direction
    integer, dimension(2), intent(in)                   :: gs           ! a group size
    integer, dimension(2), intent(in)                   :: ind_group
    integer, dimension(:,:), intent(in)                 :: remesh_min   ! minimal indice of meshes where I will remesh my particles.
    integer, dimension(:,:), intent(in)                 :: remesh_max   ! maximal indice of meshes where I will remesh my particles.
    integer, dimension(2, 2), intent(out)               :: rece_gap
    integer(kind=4), dimension(gs(1),gs(2),2),intent(in):: send_gap     ! minimal and maximal processus which contains the sub-domains where my
                                                                        ! particles will be remeshed for each line of the line group
    integer, dimension(2), intent(in)                   :: send_gap_abs ! min and maximal processus which contains the sub-domains where my particles will be remeshed.
    integer, dimension(2+gs(2)*(2+3*gs(1)), &
        & send_gap_abs(1):send_gap_abs(2)), intent(out) :: cartography

    ! To manage communications and to localize sub-domain
    integer(kind=4)                         :: proc_gap         ! gap between a processus coordinate (along the current
                                                                ! direction) into the mpi-topology and my coordinate
    integer, dimension(2)                   :: tag_table        ! mpi message tag (for communicate rece_gap(1) and rece_gap(2))
    integer, dimension(:,:),allocatable     :: send_request     ! mpi status of nonblocking send
    integer                                 :: ierr             ! mpi error code
    integer, dimension(MPI_STATUS_SIZE)     :: statut           ! mpi status
    ! To determine which processus is the first/last to send data to another
    integer, dimension(:,:), allocatable    :: first, last      ! Storage processus to which I will be the first (or the last) to send
                                                                ! remeshed particles
    integer                                 :: first_condition  ! allowed range of value of proc_min and proc_max for being the first
    integer                                 :: last_condition   ! allowed range of value of proc_min and proc_max for being the last
    ! Other local variable
    integer                                 :: ind1, ind2       ! indice of the current line inside the group
    integer                                 :: min_size         ! begin indice in first and last to stock indice along first dimension of the group line
    integer                                 :: gp_size          ! group size
    integer,dimension(2)                    :: rece_buffer      ! buffer for reception of rece_max
    logical                                 :: begin_interval   ! ware we in the start of an interval ?

    rece_gap(1,1) = 3*mesh_sc%N(direction)
    rece_gap(1,2) = -3*mesh_sc%N(direction)
    rece_gap(2,:) = 0
    gp_size = gs(1)*gs(2)

    allocate(send_request(send_gap_abs(1):send_gap_abs(2),3))
    send_request(:,3) = 0

    ! ===== Compute if I am first or last and determine the cartography =====
    min_size = 2 + gs(2)
    ! Initialize first and last to determine if I am the the first or the last processes (considering the current direction)
        ! to require information from this processus
    allocate(first(2,send_gap_abs(1):send_gap_abs(2)))
    first(2,:) = 0  ! number of lines for which I am the first
    allocate(last(2,send_gap_abs(1):send_gap_abs(2)))
    last(2,:) = 0   ! number of lines for which I am the last
    ! Initialize cartography
    cartography(1,:) = 0            ! number of velocity values to receive
    cartography(2,:) = min_size     ! number of element to send when sending cartography
    ! And compute cartography, first and last !
    do proc_gap = send_gap_abs(1), send_gap_abs(2)
        first(1,proc_gap) = -proc_gap
        last(1,proc_gap) = -proc_gap
        first_condition =  1-2*bl_bound_size + proc_gap*mesh_sc%N_proc(direction)+1
        last_condition  = -1+2*bl_bound_size + (proc_gap+1)*mesh_sc%N_proc(direction)
        do ind2 = 1, gs(2)
            cartography(2+ind2,proc_gap) = 0    ! 2 x number of interval of concern line into the column i2
            begin_interval = .true.
            do ind1 = 1, gs(1)
                ! Does proc_gap belongs to [send_gap(i1,i2,1);send_gap(i1,i2,2)]?
                if((proc_gap>=send_gap(ind1,ind2,1)).and.(proc_gap<=send_gap(ind1,ind2,2))) then
                    ! Compute if I am the first.
                    if (remesh_min(ind1,ind2)< first_condition) first(2,proc_gap) =  first(2,proc_gap)+1
                    ! Compute if I am the last.
                    if (remesh_max(ind1,ind2) > last_condition) last(2,proc_gap) =  last(2,proc_gap)+1
                    ! Update cartography // Needed even if target processus is myself as we us buffer
                    ! in all the case (scalar field cannot be used directly during the remeshing)
                    if (begin_interval) then
                        cartography(2+ind2,proc_gap) =  cartography(2+ind2,proc_gap)+2
                        cartography(cartography(2,proc_gap)+1,proc_gap) = ind1
                        cartography(2,proc_gap) = cartography(2,proc_gap) + 2
                        cartography(cartography(2,proc_gap),proc_gap) = ind1
                        begin_interval = .false.
                    else
                        cartography(cartography(2,proc_gap),proc_gap) = ind1
                    end if
                else
                    begin_interval = .true.
                end if
            end do
        end do
    end do

    ! ===== Send information about first and last  =====
    tag_table = compute_tag(ind_group, tag_obtsend_NP, direction)
    do proc_gap = send_gap_abs(1), send_gap_abs(2)
        ! I am the first ?
        if (first(2,proc_gap)>0) then
            if(neighbors(direction,proc_gap)/= D_rank(direction)) then
                call mpi_ISsend(first(1,proc_gap), 2, MPI_INTEGER, neighbors(direction,proc_gap), &
                        & tag_table(1), D_comm(direction), send_request(proc_gap,1), ierr)
                send_request(proc_gap,3) = 1
            else
                rece_gap(1,1) = min(rece_gap(1,1), -proc_gap)
                rece_gap(2,1) = rece_gap(2,1) + first(2,proc_gap)
            end if
        end if
        ! I am the last ?
        if (last(2,proc_gap)>0) then
            if(neighbors(direction,proc_gap)/= D_rank(direction)) then
                call mpi_ISsend(last(1,proc_gap), 2, MPI_INTEGER, neighbors(direction,proc_gap), &
                        & tag_table(2), D_comm(direction), send_request(proc_gap,2), ierr)
                send_request(proc_gap,3) = send_request(proc_gap, 3) + 2
            else
                rece_gap(1,2) = max(rece_gap(1,2), -proc_gap)
                rece_gap(2,2) = rece_gap(2,2) + last(2,proc_gap)
            end if
        end if
    end do

    ! ===== Receive information form the first and the last processus which need a part of my local velocity field =====
    do while(rece_gap(2,1) < gp_size)
        call mpi_recv(rece_buffer(1), 2, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(1), D_comm(direction), statut, ierr)
        rece_gap(1,1) = min(rece_gap(1,1), rece_buffer(1))
        rece_gap(2,1) = rece_gap(2,1) + rece_buffer(2)
    end do
    do while(rece_gap(2,2) < gp_size)
        call mpi_recv(rece_buffer(1), 2, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(2), D_comm(direction), statut, ierr)
        rece_gap(1,2) = max(rece_gap(1,2), rece_buffer(1))
        rece_gap(2,2) = rece_gap(2,2) + rece_buffer(2)
    end do

    ! ===== Free Isend buffer =====
    do proc_gap = send_gap_abs(1), send_gap_abs(2)
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

    ! ===== Deallocate fields =====
    deallocate(send_request)
    deallocate(first)
    deallocate(last)

end subroutine AC_remesh_determine_communication


!> Determine the set of processes wich will send me information during the remeshing
!! and compute for each of these processes the range of wanted data. Version for M'6
!! scheme (some implicitation can not be done anymore)
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[out]       rece_gap    = coordinate range of processes which will send me information during the remeshing.
!!    @param[in]        send_gap    = distance between me and processus to wich I will send information (for each line of the group)
!!    @param[in]        send_gap_abs= send_gap_abs(i) is the min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
!!    @param[in,out]    cartography = cartography(proc_gap) contains the set of the lines indice in the block for wich the
!!                                    current processus requiers data from proc_gap and for each of these lines the range
!!                                    of mesh points from where it requiers the velocity values.
!! @details
!!    Work on a group of line of size gs(1) x gs(2))
!!    Obtain the list of processus which are associated to sub-domain where my particles
!!    will be remeshed and the list of processes wich contains particles which
!!    have to be remeshed in my sub-domain. This way, this procedure determine
!!    which processus need to communicate together in order to proceed to the
!!    remeshing (as in a parrallel context the real space is subdivised and each
!!    processus contains a part of it)
!!        In the same time, it computes for each processus with which I will
!!    communicate, the range of mesh point involved for each line of particles
!!    inside the group and it stores it by using some sparse matrix technics
!!    (see cartography defined in the algorithm documentation)
!!        This routine involves communication to determine if a processus is
!!    the first or the last processes (considering its coordinate along
!!    the current directory) to send remeshing information to a given processes.
subroutine AC_remesh_determine_communication_com(direction, gs, ind_group, &
    & rece_gap, send_gap, send_gap_abs, cartography)
! XXX Work only for periodic condition.

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/output
    integer, intent(in)                                 :: direction
    integer, dimension(2), intent(in)                   :: gs           ! a group size
    integer, dimension(2), intent(in)                   :: ind_group
    integer, dimension(2, 2), intent(out)               :: rece_gap     ! minimal and maximal processus which will remesh inside me
    integer(kind=4), dimension(gs(1),gs(2),2),intent(in):: send_gap     ! minimal and maximal processus which contains the sub-domains where my
                                                                        ! particles will be remeshed for each line of the line group
    integer, dimension(2), intent(in)                   :: send_gap_abs ! min and maximal processus which contains the sub-domains where my particles will be remeshed.
    integer, dimension(2+gs(2)*(2+3*gs(1)), &
        & send_gap_abs(1):send_gap_abs(2)), intent(out) :: cartography

    ! To manage communications and to localize sub-domain
    integer(kind=4)                         :: proc_gap         ! gap between a processus coordinate (along the current
                                                                ! direction) into the mpi-topology and my coordinate
    integer, dimension(2)                   :: tag_table        ! mpi message tag (for communicate rece_gap(1) and rece_gap(2))
    integer, dimension(:,:),allocatable     :: send_request     ! mpi status of nonblocking send
    integer                                 :: ierr             ! mpi error code
    integer, dimension(MPI_STATUS_SIZE)     :: statut           ! mpi status
    ! To determine which processus is the first/last to send data to another
    integer, dimension(gs(1), gs(2))        :: send_max_prev    ! maximum gap between previous processus and the receivers of its remeshing buffer
    integer, dimension(gs(1), gs(2))        :: send_min_next    ! minimum gap between next processus and the receivers of its remeshing buffer
    integer, dimension(:,:), allocatable    :: first, last      ! Storage processus to which I will be the first (or the last) to send
                                                                ! remeshed particles
    ! Other local variable
    integer                                 :: ind1, ind2       ! indice of the current line inside the group
    integer                                 :: min_size         ! begin indice in first and last to stock indice along first dimension of the group line
    integer                                 :: gp_size          ! group size
    integer,dimension(2)                    :: rece_buffer      ! buffer for reception of rece_max
    logical                                 :: begin_interval   ! are we in the start of an interval ?

    rece_gap(1,1) = 3*mesh_sc%N(direction)
    rece_gap(1,2) = -3*mesh_sc%N(direction)
    rece_gap(2,:) = 0
    gp_size = gs(1)*gs(2)

    allocate(send_request(send_gap_abs(1):send_gap_abs(2),3))
    send_request(:,3) = 0

    ! ===== Exchange ghost =====
    ! Compute message tag - we re-use tag_part_tag_NP id as using this procedure
    ! suppose not using "AC_type_and_block"
    tag_table = compute_tag(ind_group, tag_part_tag_NP, direction)
    ! Exchange "ghost"
    call mpi_Sendrecv(send_gap(1,1,1), gp_size, MPI_INTEGER, neighbors(direction,-1), tag_table(1), &
            & send_min_next(1,1), gp_size, MPI_INTEGER, neighbors(direction,1), tag_table(1),    &
            & D_comm(direction), statut, ierr)
    call mpi_Sendrecv(send_gap(1,1,2), gp_size, MPI_INTEGER, neighbors(direction,1), tag_table(2), &
            & send_max_prev(1,1), gp_size, MPI_INTEGER, neighbors(direction,-1), tag_table(2),    &
            & D_comm(direction), statut, ierr)
    ! Translat to adapt gap to my position
    send_max_prev = send_max_prev - 1
    send_min_next = send_min_next + 1

    ! ===== Compute if I am first or last and determine the cartography =====
    min_size = 2 + gs(2)
    ! Initialize first and last to determine if I am the the first or the last processes (considering the current direction)
        ! to require information from this processus
    allocate(first(2,send_gap_abs(1):send_gap_abs(2)))
    first(2,:) = 0  ! number of lines for which I am the first
    allocate(last(2,send_gap_abs(1):send_gap_abs(2)))
    last(2,:) = 0   ! number of lines for which I am the last
    ! Initialize cartography
    cartography(1,:) = 0            ! number of velocity values to receive
    cartography(2,:) = min_size     ! number of element to send when sending cartography
    ! And compute cartography, first and last !
    do proc_gap = send_gap_abs(1), send_gap_abs(2)
        first(1,proc_gap) = -proc_gap
        last(1,proc_gap) = -proc_gap
        do ind2 = 1, gs(2)
            cartography(2+ind2,proc_gap) = 0    ! 2 x number of interval of concern line into the column i2
            begin_interval = .true.
            do ind1 = 1, gs(1)
                ! Does proc_gap belongs to [send_gap(i1,i2,1);send_gap(i1,i2,2)]?
                if((proc_gap>=send_gap(ind1,ind2,1)).and.(proc_gap<=send_gap(ind1,ind2,2))) then
                    ! Compute if I am the first.
                    if(proc_gap > send_max_prev(ind1,ind2)) first(2,proc_gap) =  first(2,proc_gap)+1
                    ! Compute if I am the last.
                    if(proc_gap < send_min_next(ind1,ind2)) last(2,proc_gap) =  last(2,proc_gap)+1
                    ! Update cartography // Needed even if target processus is myself as we us buffer
                    ! in all the case (scalar field cannot be used directly during the remeshing)
                    if (begin_interval) then
                        cartography(2+ind2,proc_gap) =  cartography(2+ind2,proc_gap)+2
                        cartography(cartography(2,proc_gap)+1,proc_gap) = ind1
                        cartography(2,proc_gap) = cartography(2,proc_gap) + 2
                        cartography(cartography(2,proc_gap),proc_gap) = ind1
                        begin_interval = .false.
                    else
                        cartography(cartography(2,proc_gap),proc_gap) = ind1
                    end if
                else
                    begin_interval = .true.
                end if
            end do
        end do
    end do

    ! ===== Send information about first and last  =====
    tag_table = compute_tag(ind_group, tag_obtsend_NP, direction)
    do proc_gap = send_gap_abs(1), send_gap_abs(2)
        ! I am the first ?
        if (first(2,proc_gap)>0) then
            if(neighbors(direction,proc_gap)/= D_rank(direction)) then
                call mpi_ISsend(first(1,proc_gap), 2, MPI_INTEGER, neighbors(direction,proc_gap), &
                        & tag_table(1), D_comm(direction), send_request(proc_gap,1), ierr)
                send_request(proc_gap,3) = 1
            else
                rece_gap(1,1) = min(rece_gap(1,1), -proc_gap)
                rece_gap(2,1) = rece_gap(2,1) + first(2,proc_gap)
            end if
        end if
        ! I am the last ?
        if (last(2,proc_gap)>0) then
            if(neighbors(direction,proc_gap)/= D_rank(direction)) then
                call mpi_ISsend(last(1,proc_gap), 2, MPI_INTEGER, neighbors(direction,proc_gap), &
                        & tag_table(2), D_comm(direction), send_request(proc_gap,2), ierr)
                send_request(proc_gap,3) = send_request(proc_gap, 3) + 2
            else
                rece_gap(1,2) = max(rece_gap(1,2), -proc_gap)
                rece_gap(2,2) = rece_gap(2,2) + last(2,proc_gap)
            end if
        end if
    end do

    ! ===== Receive information form the first and the last processus which need a part of my local velocity field =====
    do while(rece_gap(2,1) < gp_size)
        call mpi_recv(rece_buffer(1), 2, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(1), D_comm(direction), statut, ierr)
        rece_gap(1,1) = min(rece_gap(1,1), rece_buffer(1))
        rece_gap(2,1) = rece_gap(2,1) + rece_buffer(2)
    end do
    do while(rece_gap(2,2) < gp_size)
        call mpi_recv(rece_buffer(1), 2, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(2), D_comm(direction), statut, ierr)
        rece_gap(1,2) = max(rece_gap(1,2), rece_buffer(1))
        rece_gap(2,2) = rece_gap(2,2) + rece_buffer(2)
    end do

    ! ===== Free Isend buffer =====
    do proc_gap = send_gap_abs(1), send_gap_abs(2)
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

    ! ===== Deallocate fields =====
    deallocate(send_request)
    deallocate(first)
    deallocate(last)

end subroutine AC_remesh_determine_communication_com


!> Update the cartography of data which will be exchange from a processus to another in order to remesh particles.
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]        gs          = size of group of line along the current direction
!!    @param[in]        begin_i1    = indice corresponding to the first place into the cartography
!!                                      array where indice along the the direction of the group of lines are stored.
!!    @param[in]        proc_gap    = distance between my (mpi) coordonate and coordinate of the target processus
!!    @param[in]        ind_carto   = current column inside the cartography (different to proc_Gap as in this procedure
!!                                    therefore first indice = 1, carto range are not given into argument)
!!    @param[in]        send_min    = minimal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[in]        send_max    = maximal indice of mesh involved in remeshing particles (of the particles in my local subdomains)
!!    @param[in,out]    cartography = cartography(proc_gap) contains the set of the lines indice in the block for wich the
!!                                    current processus requiers data from proc_gap and for each of these lines the range
!!                                    of mesh points from where it requiers the velocity values.
!!    @param[out]       com_size    = number of elements (integers) stored into the cartography (which will be the size of some mpi communication)
subroutine AC_remesh_cartography(direction, gs, begin_i1, proc_gap, ind_carto, send_min, send_max, cartography, com_size)

    use cart_topology           ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                     :: direction
    integer, dimension(2), intent(in)       :: gs
    integer, intent(in)                     :: begin_i1     ! indice corresponding to the first place into the cartography
                                                            ! array where indice along the the direction of the group of
                                                            ! lines are stored.
    integer, intent(in)                     :: proc_gap     ! distance between my (mpi) coordonate and coordinate of the target
    integer, intent(in)                     :: ind_carto    ! current column inside the cartography (different to proc_Gap as in this procedure
                                                            ! therefore first indice = 1, carto range are not given into argument)
    integer, dimension(:,:), intent(in)     :: send_min     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(in)     :: send_max     ! distance between me and processus wich send me information
    integer, dimension(:,:), intent(inout)  :: cartography
    integer, intent(out)                    :: com_size     ! number of elements (integers) stored into the cartography (which will
                                                            ! be the size of some mpi communication)

    ! Other local variables
    integer                                 :: gap          ! gap between my local indices and the local indices from another processes
    integer                                 :: i1, i2       ! indice of a line into the group
    integer                                 :: ind_for_i1   ! where to read the first coordinate (i1) of the current line inside the cartography ?
    integer                                 :: ind_1Dtable  ! indice of my current position inside a one-dimensionnal table

    cartography(1,ind_carto) = 0
    ! Use the cartography to know which lines are concerned
    com_size = cartography(2,ind_carto)
    ! Range I want - store into the cartography
    gap = proc_gap*mesh_sc%N_proc(direction)
    ! Position in cartography(:,ind_carto) of the current i1 indice
    ind_for_i1 = begin_i1
    do i2 = 1, gs(2)
        do ind_1Dtable = ind_for_i1+1, ind_for_i1 + cartography(2+i2,ind_carto), 2
            do i1 = cartography(ind_1Dtable,ind_carto), cartography(ind_1Dtable+1,ind_carto)
                ! Interval start from:
                cartography(com_size+1,ind_carto) = max(send_min(i1,i2), gap+1) ! fortran => indice start from 0
                ! and ends at:
                cartography(com_size+2,ind_carto) = min(send_max(i1,i2), gap+mesh_sc%N_proc(direction))
                ! update number of element to send
                cartography(1,ind_carto) = cartography(1,ind_carto) &
                            & + cartography(com_size+2,ind_carto) &
                            & - cartography(com_size+1,ind_carto) + 1
                com_size = com_size+2
            end do
        end do
        ind_for_i1 = ind_for_i1 + cartography(2+i2,ind_carto)
    end do

end subroutine AC_remesh_cartography


!> Perform all the pre-process in order to remesh particle and to perform associated communication.
!! @ detail
!!     As geometric domain is subdivise among the different mpi-processes, the
!! particle remeshing involve mpi-communication in order to re-distribuate
!! particle weight to the rigth place.
!!     In order to gather theses communications for different particles lines,
!! the particle remeshing is performed into a buffer. The buffer is an 1D-array
!! which structure ensure that all the value that has to be send to a given
!! processus is memory continguous.
!!     This subroutine create this buffer and provide a map to manage it. This
!! map allow to associate a XYZ-coordinate (into the geometrical domain) to each
!! element of this 1D-array.
subroutine AC_remesh_init(direction, ind_group, gs, send_min, send_max, &
    & send_gap_abs, rece_gap, nb_s, cartography, rece_carto,            &
    & pos_in_buffer, min_size, max_size, s_request_ran, r_request_ran)

    use cart_topology     ! Description of mesh and of mpi topology
    use advec_variables         ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                     :: direction
    integer, dimension(2), intent(in)       :: ind_group
    integer, dimension(2), intent(in)       :: gs
    integer, dimension(:,:), intent(in)     :: send_min     ! distance between me and first processus wich send me information (for each line of particle)
    integer, dimension(:,:), intent(in)     :: send_max     ! distance between me and last processus wich send me information
    integer, dimension(2), intent(in)       :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer, dimension(2, 2), intent(in)    :: rece_gap     ! distance between me and processus to wich I send information
    integer, intent(in)                     :: nb_s         ! number of reception/send
    integer, dimension(:,:), intent(inout)  :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block to wich the
                                                            ! current processus will send data during remeshing and for each of these lines the range
    integer, dimension(:,:), intent(inout)  :: rece_carto   ! same as abobve but for what I receive
                                                            ! of mesh points from where it requiers the velocity values.
    integer,dimension(0:nb_s),intent(inout) :: pos_in_buffer! information about organization of the 1D buffer used to remesh
                                                            ! a 3D set of particles.
    integer, intent(out)                    :: min_size     ! tool to manage cartography
    integer, intent(in)                     :: max_size     ! tool to manage cartography
    integer, dimension(:), intent(inout)    :: s_request_ran! mpi communication request (handle) of nonblocking send
    integer, dimension(:), intent(inout)    :: r_request_ran! mpi communication request (handle) of nonblocking receive

    ! Others
    integer                                 :: proc_gap     ! distance between my (mpi) coordonate and coordinate of the
                                                            ! processus associated to a given position
    integer                                 :: ind_gap      ! loop indice
    integer                                 :: ind_1Dtable  ! indice of my current position inside a one-dimensionnal table
    ! Variable use to manage mpi communications
    integer                                 :: com_size     ! size of message send/receive
    integer                                 :: tag          ! mpi message tag
    integer                                 :: ierr         ! mpi error code

    ! ===== Receive cartography =====
    ! It is better to post recceive before sending.
    ind_1Dtable = 0
    do proc_gap = rece_gap(1,1), rece_gap(1,2)
        ind_1Dtable = ind_1Dtable + 1
        if (neighbors(direction,proc_gap)/= D_rank(direction)) then
            tag = compute_tag(ind_group, tag_bufToScal_range, direction, -proc_gap)
            call mpi_Irecv(rece_carto(1,ind_1Dtable), max_size, MPI_INTEGER,  &
                & neighbors(direction,proc_gap), tag, D_COMM(direction),      &
                & r_request_ran(ind_1Dtable), ierr)
        else
            rece_carto(1,ind_1Dtable) = 0
        end if
    end do

    ! ===== Complete cartography and send range about the particles I remesh =====
    s_request_ran = MPI_REQUEST_NULL
    min_size = 2 + gs(2)
    proc_gap = send_gap_abs(1) - 1
    do ind_gap = 1, nb_s !send_gap_abs(2), send_gap_abs(1) + 1
        proc_gap = proc_gap + 1
        !proc_gap = ind_gap+send_gap_abs(1)-1
        call AC_remesh_cartography(direction, gs, min_size, proc_gap, ind_gap, &
            & send_min, send_max, cartography, com_size)
#ifdef PART_DEBUG
            if(com_size>max_size) then
                print*, 'taille carto = ', com_size ,' plus grand que la taille théorique ', &
                    & max_size,' et carto = ', cartography(:,ind_gap)
            end if
#endif
        ! Tag = concatenation of (rank+1), ind_group(1), ind_group(2), direction and unique Id.
        tag = compute_tag(ind_group, tag_bufToScal_range, direction, proc_gap)
        ! Send message
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            call mpi_ISsend(cartography(1,ind_gap), com_size, MPI_INTEGER,&
                & neighbors(direction,proc_gap), tag, D_comm(direction),  &
                & s_request_ran(ind_gap),ierr)
        end if
    end do

    ! ===== Initialize the general buffer =====
    ! The same buffer is used to send data to all target processes. It size
    ! has to be computed as the part reserved to each processus.
    ! and it has to be splitted into parts for each target processes
    ! => pos_in_buffer(i) = first (1D-)indice of the sub-array of send_buffer
    ! associated to he i-rd mpi-processus to wich I will send remeshed particles.
    pos_in_buffer(0) = 1
    pos_in_buffer(1)   = 1
    do ind_gap =1, nb_s - 1 !send_gap_abs(2)-send_gap_abs(1)
        pos_in_buffer(ind_gap+1)= pos_in_buffer(ind_gap) + cartography(1,ind_gap)
    end do
    ! In writing values in the send buffer during the remeshing, pos_in_buffer will be update.
    ! As it has one supplementary element (the "0" one), after this process pos_in_buffer(i-1)
    ! will be equal to first (1D-)indice of the sub-array of send_buffer
    ! associated to he i-rd mpi-processus to wich I will send remeshed particles.

end subroutine AC_remesh_init

!> Perform all the staff to compute scalar value at t+dt from the buffer
!containing the remeshing of local particles.
!! @ detail
!!     After having remeshing the particles of the local sub-domain into a
!! buffer, it remains to send the buffer to the different processus according
!! to the domain sub-division into each processus. Then, the local scalar field
!! is update thanks to the received buffers.
subroutine AC_remesh_finalize(direction, ind_group, gs, j, k, scal, send_gap_abs, rece_gap, &
    & nb_r, nb_s, cartography, rece_carto, send_buffer, pos_in_buffer, min_size)

    use cart_topology     ! Description of mesh and of mpi topology
    use advec_variables         ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                         :: direction
    integer, dimension(2), intent(in)           :: ind_group
    integer, dimension(2), intent(in)           :: gs
    integer, intent(in)                         :: j, k
    real(WP),dimension(:,:,:),intent(inout)     :: scal
    integer, dimension(2), intent(in)           :: send_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer, dimension(2, 2), intent(in)        :: rece_gap     ! distance between me and processus to wich I send information
    integer, intent(in)                         :: nb_r, nb_s   ! number of reception/send
    integer, dimension(:,:), intent(in)         :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block to wich the
                                                                ! current processus will send data during remeshing and for each of these lines the range
                                                                ! of mesh points from where it requiers the velocity values.
    integer, dimension(:,:), intent(in)         :: rece_carto   ! same as above but for what I receive
    real(WP),dimension(:), intent(in)           :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
                                                                ! sorted by receivers and not by coordinate.
    integer, dimension(0:nb_s), intent(inout)   :: pos_in_buffer! buffer size
    integer, intent(in)                         :: min_size     ! tool to mange buffer - begin indice in first and last to stock indice along first dimension of the group line

    ! Other local variables
    integer                                 :: proc_gap, gap! distance between my (mpi) coordonate and coordinate of the
                                                            ! processus associated to a given position
    integer                                 :: ind_gap
    integer                                 :: ind_1Dtable  ! indice of my current position inside a one-dimensionnal table
    ! Variable used to update scalar field from the buffers
    real(WP),dimension(:),allocatable,target:: rece_buffer  ! buffer use to receive scalar field from other processes.
    integer, dimension(:), allocatable      :: rece_pos     ! cells of indice from rece_pos(i) to rece_proc(i+1) into rece_buffer
                                                            ! are devoted to the processus of relative position = i
    ! Variable use to manage mpi communications
    integer, dimension(:), allocatable      :: s_request_sca! mpi communication request (handle) of nonblocking send
    integer, dimension(:), allocatable      :: r_request_sca! mpi communication request (handle) of nonblocking receive
#ifndef BLOCKING_SEND
    integer, dimension(:,:), allocatable    :: s_status     ! mpi communication status of nonblocking send
#endif
    integer, dimension(mpi_status_size)     :: r_status   ! another mpi communication status
    integer                                 :: tag          ! mpi message tag
    integer                                 :: ierr         ! mpi error code
    integer                                 :: missing_msg  ! number of remeshing buffer not yet received


    ! ===== Receive buffer (init receive before send) =====
    ! -- Compute size of reception buffer and split it into part corresponding to each sender --
    allocate(rece_pos(rece_gap(1,1):rece_gap(1,2)+1))
    rece_pos(rece_gap(1,1)) = 1
    ind_gap = 0
    do proc_gap = rece_gap(1,1), rece_gap(1,2)
        ind_gap = ind_gap + 1
        rece_pos(proc_gap+1)= rece_pos(proc_gap) + rece_carto(1,ind_gap)
    end do
    allocate(rece_buffer(rece_pos(rece_gap(1,2)+1)-1))
    ! -- And initialize the reception --
    allocate(r_request_sca(1:nb_r))
    r_request_sca = MPI_REQUEST_NULL
    ind_gap = 0
    do proc_gap = rece_gap(1,1), rece_gap(1,2)
        ind_gap = ind_gap + 1 ! = proc_gap - rece_gap(1,1)+1
        if (neighbors(direction,proc_gap)/= D_rank(direction)) then
            tag = compute_tag(ind_group, tag_bufToScal_buffer, direction, -proc_gap)
            call mpi_Irecv(rece_buffer(rece_pos(proc_gap)), rece_carto(1,ind_gap),  &
                & MPI_REAL_WP, neighbors(direction,proc_gap), tag,         &
                & D_COMM(direction), r_request_sca(ind_gap), ierr)
        end if
    end do

    ! ===== Send buffer =====
    missing_msg = nb_r
    allocate(s_request_sca(1:nb_s))
    s_request_sca = MPI_REQUEST_NULL
    proc_gap = send_gap_abs(1)-1
    ! -- Send the buffer to the matching processus and update the scalar field --
    do ind_gap = 1, nb_s
        proc_gap = proc_gap +1
        !proc_gap = ind_gap-1+send_gap_abs(1)
        if (neighbors(direction,proc_gap)/=D_rank(direction)) then
            ! Send buffer
            tag = compute_tag(ind_group, tag_bufToScal_buffer, direction, ind_gap-1+send_gap_abs(1))
#ifdef BLOCKING_SEND
            call mpi_Send(send_buffer(pos_in_buffer(ind_gap-1)), cartography(1,ind_gap), MPI_REAL_WP, &
                & neighbors(direction,proc_gap), tag, D_comm(direction), r_status, ierr)
#else
            call mpi_ISsend(send_buffer(pos_in_buffer(ind_gap-1)), cartography(1,ind_gap), MPI_REAL_WP, &
                & neighbors(direction,proc_gap), tag, D_comm(direction), s_request_sca(ind_gap),ierr)
#endif
        else
            ! Range I want - store into the cartography
            !gap = -(ind_gap-1+send_gap_abs(1))*mesh_sc%N_proc(direction)
            gap = -proc_gap*mesh_sc%N_proc(direction)
            ! Update directly the scalar field
            call remesh_buffer_to_scalar_pt(gs, j, k, ind_gap, gap, min_size, &
                    & cartography, send_buffer, scal, pos_in_buffer(ind_gap-1))
            missing_msg = missing_msg - 1
        end if
    end do

    ! ===== Update scalar field =====
    do while (missing_msg >= 1)
        ! --- Choose one of the first available message ---
        ! more precisly: the last reception ended (and not free) and if not such
        ! message available, the first reception ended.
        call mpi_waitany(nb_r, r_request_sca, ind_1Dtable, r_status, ierr)
        ! -- Update the scalar field by using the cartography --
        ! Range I want - store into the cartography
        proc_gap = ind_1Dtable + rece_gap(1,1)-1
        gap = proc_gap*mesh_sc%N_proc(direction)
        call remesh_buffer_to_scalar_pt(gs, j, k, ind_1Dtable, gap, min_size, &
                & rece_carto, rece_buffer, scal, rece_pos(proc_gap))
        missing_msg = missing_msg - 1
    end do

    ! ===== Free memory and communication buffer ====
    ! -- Deallocate all field --
    deallocate(rece_pos)
    deallocate(rece_buffer)
    deallocate(r_request_sca)
#ifndef BLOCKING_SEND
    ! -- Check if Isend are done --
    allocate(s_status(MPI_STATUS_SIZE,1:nb_s))
    call mpi_waitall(nb_s, s_request_sca, s_status, ierr)
    deallocate(s_status)
    ! -- Free all communication buffer and data --
    deallocate(s_request_sca)
#endif

end subroutine AC_remesh_finalize


end module advec_common_remesh
!> @}
