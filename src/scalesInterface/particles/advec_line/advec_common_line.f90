!USEFORTEST advec
!> @addtogroup part
!! @{
!------------------------------------------------------------------------------
!
! MODULE: advec_common_line
!
!
! DESCRIPTION:
!> The module ``advec_common_line'' gather function and subroutines used to advec scalar
!! which are not specific to a direction. It contains some ``old''
!! functions from ``advec_common'' which are not optimized.
!! @details
!! This module gathers functions and routines used to advec scalar which are not
!! specific to a direction. More precisly, it provides function similar to
!! ``advec_common'' but which only work on single line rather than of
!! group line. Considering how mpi parallelism works, working on single
!! line are not opptimal. Therefore, these function are onbly here for
!! debbugging and testing purposes. They also could be used to compute
!! some spped-up. They are more simple and basic but less efficients.
!!
!!      This module is automatically load when advec_common is used.
!! Moreover, advec_common contains all interface to automatically use
!! the right function whenever you want work on single line or on group of
!! lines.
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

module advec_common_line

    use precision_tools
    use mpi, only: MPI_INTEGER, MPI_STATUS_SIZE, MPI_ANY_SOURCE
    implicit none

! ===== Public procedures =====

!----- To interpolate velocity -----
public                              :: AC_obtain_receivers_line
public                              :: AC_particle_velocity_line
!----- Determine block type and tag particles -----
public                              :: AC_type_and_block_line
!----- To remesh particles -----
public                              :: AC_obtain_senders_line
public                              :: AC_bufferToScalar_line

contains

! ===== Public procedure =====


! ==================================================================================
! ====================     Compute particle velocity (RK2)      ====================
! ==================================================================================

!> Determine the set of processes wich will send me information during the velocity interpolation.
!!    @param[in]    direction       = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]    ind_group       = coordinate of the current group of lines
!!    @param[in]    rece_ind_min    = minimal indice of mesh involved in remeshing particles (of the my local subdomains)
!!    @param[in]    rece_ind_max    = maximal indice of mesh involved in remeshing particles (of the my local subdomains)
!!    @param[out]   send_gap        = gap between my coordinate and the processes of minimal coordinate which will send information to me
!!    @param[out]   rece_gap        = gap between my coordinate and the processes of maximal coordinate which will receive information from me
!! @details
!!    Obtain the list of processus wich need a part of my local velocity field
!!    to interpolate the velocity used in the RK2 scheme to advect its particles.
subroutine AC_obtain_receivers_line(direction, ind_group, rece_ind_min, rece_ind_max, send_gap, rece_gap)
! XXX Work only for periodic condition.

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.
    


    ! Input/Ouput
    integer, intent(in)                 :: rece_ind_min, rece_ind_max
    integer, intent(in)                 :: direction
    integer, dimension(2), intent(in)   :: ind_group
    integer, dimension(2), intent(out)  :: rece_gap, send_gap
    integer, dimension(MPI_STATUS_SIZE) :: statut
    ! Others
    integer                             :: proc_gap         ! gap between a processus coordinate (along the current
                                                            ! direction) into the mpi-topology and my coordinate
    integer                             :: rece_gapP        ! gap between the coordinate of the previous processus (in the current direction)
                                                            ! and the processes of maximal coordinate which will receive information from it
    integer                             :: rece_gapN        ! same as above but for the next processus
    integer                             :: rankP, rankN     ! processus rank for shift (P= previous, N = next)
    integer                             :: tag_min, tag_max ! mpi message tag (for communicate rece_proc(1) and rece_proc(2))
    integer                             :: send_request     ! mpi status of nonblocking send
    integer                             :: send_request_bis ! mpi status of nonblocking send
    integer                             :: ierr             ! mpi error code
    integer, dimension(2)               :: tag_table        ! some mpi message tag
    logical, dimension(:,:), allocatable:: test_request
    integer, dimension(:,:), allocatable:: s_request

    tag_min = 5
    tag_max = 6

    send_gap = 3*mesh_sc%N(direction)

    rece_gap(1) = floor(real(rece_ind_min-1, WP)/mesh_sc%N_proc(direction))
    rece_gap(2) = floor(real(rece_ind_max-1, WP)/mesh_sc%N_proc(direction))

    ! ===== Communicate with my neigbors -> obtain ghost ! ====
    ! Compute their rank
    call mpi_cart_shift(D_comm(direction), 0, 1, rankP, rankN, ierr)
    ! Inform that about processus from which I need information
    tag_table = compute_tag(ind_group, tag_obtrec_ghost_NP, direction)
    call mpi_Isend(rece_gap(1), 1, MPI_INTEGER, rankP, tag_table(1), D_comm(direction), send_request, ierr)
    call mpi_Isend(rece_gap(2), 1, MPI_INTEGER, rankN, tag_table(2), D_comm(direction), send_request_bis, ierr)
    ! Receive the same message form my neighbors
    call mpi_recv(rece_gapN, 1, MPI_INTEGER, rankN, tag_table(1), D_comm(direction), statut, ierr)
    call mpi_recv(rece_gapP, 1, MPI_INTEGER, rankP, tag_table(2), D_comm(direction), statut, ierr)


    ! ===== Send information if I am first or last =====
    allocate(s_request(rece_gap(1):rece_gap(2),2))
    allocate(test_request(rece_gap(1):rece_gap(2),2))
    test_request = .false.
    tag_table = compute_tag(ind_group, tag_obtrec_NP, direction)
    do proc_gap = rece_gap(1), rece_gap(2)
        ! Compute the rank of the target processus
        call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, rankN, ierr)
        ! Determine if I am the the first or the last processes (considering the current directory)
            ! to require information from this processus
        if (proc_gap>rece_gapP-1) then
            if(rankN /= D_rank(direction)) then
                call mpi_Isend(-proc_gap, 1, MPI_INTEGER, rankN, tag_table(1), D_comm(direction), s_request(proc_gap,1), ierr)
                test_request(proc_gap,1) = .true.
            else
                send_gap(1) = -proc_gap
            end if
        end if
        if (proc_gap<rece_gapN+1) then
            if(rankN /= D_rank(direction)) then
                test_request(proc_gap,2) = .true.
                call mpi_Isend(-proc_gap, 1, MPI_INTEGER, rankN, tag_table(2), D_comm(direction), s_request(proc_gap,2), ierr)
            else
                send_gap(2) = -proc_gap
            end if
        end if
    end do


    ! ===== Receive information form the first and the last processus which need a part of my local velocity field =====
    if (send_gap(1) == 3*mesh_sc%N(direction)) then
        call mpi_recv(send_gap(1), 1, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(1), D_comm(direction), statut, ierr)
    end if
    if (send_gap(2) == 3*mesh_sc%N(direction)) then
        call mpi_recv(send_gap(2), 1, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(2), D_comm(direction), statut, ierr)
    end if


    call MPI_WAIT(send_request,statut,ierr)
    call MPI_WAIT(send_request_bis,statut,ierr)
    do proc_gap = rece_gap(1), rece_gap(2)
        if (test_request(proc_gap,1).eqv. .true.) call MPI_WAIT(s_request(proc_gap,1),statut,ierr)
        if (test_request(proc_gap,2)) call MPI_WAIT(s_request(proc_gap,2),statut,ierr)
    end do
    deallocate(s_request)
    deallocate(test_request)

end subroutine AC_obtain_receivers_line


!> Interpolate the velocity field used in a RK2 scheme for particle advection.
!!    @param[in]        dt          = time step
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        p_pos_adim  = adimensionned particle postion
!!    @param[in,out]    p_V         = particle velocity (along the current direction)
!! @details
!!    A RK2 scheme is used to advect the particles : the midlle point scheme. An
!!    intermediary position "p_pos_bis(i) = p_pos(i) + V(i)*dt/2" is computed and then
!!    the numerical velocity of each particles is computed as the interpolation of V  in
!!    this point. This field is used to advect the particles at the seconde order in time :
!!    p_pos(t+dt, i) = p_pos(i) + p_V(i).
!!    The group line indice is used to ensure using unicity of each mpi message tag.
subroutine AC_particle_velocity_line(dt, direction, ind_group, p_pos_adim, p_V)

    ! This code involve a recopy of p_V. It is possible to directly use the 3D velocity field but in a such code
    ! a memory copy is still needed to send velocity field to other processus : mpi send contiguous memory values

    
    use structure_tools
    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Ouput
    real(WP), intent(in)                            :: dt       ! time step
    integer, intent(in)                             :: direction
    integer, dimension(2), intent(in)               :: ind_group
    real(WP), dimension(:), intent(in)              :: p_pos_adim
    real(WP), dimension(:), intent(inout)           :: p_V
    ! Others, local
    real(WP), dimension(mesh_sc%N_proc(direction))          :: p_pos_bis    ! adimensionned position of the middle point
    real(WP), dimension(mesh_sc%N_proc(direction)), target  :: p_V_bis      ! velocity of the middle point
    real(WP), dimension(mesh_sc%N_proc(direction))          :: weight       ! interpolation weight
    type(real_pter), dimension(mesh_sc%N_proc(direction))   :: Vp, Vm       ! Velocity on previous and next mesh point
    real(WP), dimension(:), allocatable, target     :: V_buffer     ! Velocity buffer for postion outside of the local subdomain
    integer                                         :: size_buffer  ! buffer size
    integer                                         :: rece_ind_min ! the minimal indice used in velocity interpolation
    integer                                         :: rece_ind_max ! the maximal indice used in velocity interpolation
    integer                                         :: ind, ind_com ! indices
    integer                                         :: pos, pos_old ! indices of the mesh point wich preceed the particle position
    integer                                         :: proc_gap, gap! distance between my (mpi) coordonate and coordinate of the
                                                                    ! processus associated to a given position
    integer, dimension(:), allocatable              :: rece_rank    ! rank of processus wich send me information
    integer                                         :: send_rank    ! rank of processus to wich I send information
    integer                                         :: rankP        ! rank of processus ("source rank" returned by mpi_cart_shift)
    integer, dimension(2)                           :: rece_range   ! range of the velocity fields I want to receive
    integer, dimension(2)                           :: send_range   ! range of the velocity fields I send
    integer, dimension(2)                           :: rece_gap     ! distance between me and processus wich send me information
    integer, dimension(2)                           :: send_gap     ! distance between me and processus to wich I send information
    integer                                         :: msg_size     ! size of message send/receive
    integer                                         :: tag          ! mpi message tag
    integer                                         :: ierr         ! mpi error code
    integer, dimension(:), allocatable              :: s_request    ! mpi communication request (handle) of nonblocking send
    integer, dimension(:), allocatable              :: s_request_bis! mpi communication request (handle) of nonblocking send
    integer, dimension(:), allocatable              :: rece_request ! mpi communication request (handle) of nonblocking receive
    integer, dimension(MPI_STATUS_SIZE)             :: rece_status  ! mpi status (for mpi_wait)

    ! -- Initialisation --
    ind_com = 0
    do ind = 1, mesh_sc%N_proc(direction)
        nullify(Vp(ind)%pter)
        nullify(Vm(ind)%pter)
    end do
    ! Compute the midlle point
    p_pos_bis = p_pos_adim + (dt/2.0)*p_V/mesh_sc%dx(direction)
    p_V_bis = p_V
    ! Compute range of the set of point where I need the velocity value
    rece_ind_min = floor(p_pos_bis(1))
    rece_ind_max = floor(p_pos_bis(mesh_sc%N_proc(direction))) + 1
    ! Allocate the buffer
    ! If rece_ind_min and rece_ind_max are not in [mesh_sc%N_proc(direction);1] then it will change the number of communication
    ! size_buffer = max(temp - mesh_sc%N_proc(direction), 0) - min(0, temp)
    !size_buffer = - max(temp - mesh_sc%N_proc(direction), 0) - min(0, temp)
    ! It must work, but for first test we prefer compute size_buffer more simply
    size_buffer = 0

    ! -- Exchange non blocking message to do the computations during the
    ! communication process --
    call AC_obtain_receivers_line(direction, ind_group, rece_ind_min, rece_ind_max, send_gap, rece_gap)
    allocate(rece_rank(rece_gap(1):rece_gap(2)))
    ! Send messages about what I want
    allocate(s_request_bis(rece_gap(1):rece_gap(2)))
    do proc_gap = rece_gap(1), rece_gap(2)
        call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, rece_rank(proc_gap), ierr)
        if (rece_rank(proc_gap) /= D_rank(direction)) then
            ! Range I want
            gap = proc_gap*mesh_sc%N_proc(direction)
            rece_range(1) = max(rece_ind_min, gap+1) ! fortran => indice start from 0
            rece_range(2) = min(rece_ind_max, gap+mesh_sc%N_proc(direction))
            ! Tag = concatenation of (rank+1), ind_group(1), ind_group(2), direction et unique Id.
            tag = compute_tag(ind_group, tag_velo_range, direction, proc_gap)
            ! Send message
            size_buffer = size_buffer + (rece_range(2)-rece_range(1)) + 1
            call mpi_ISsend(rece_range(1), 2, MPI_INTEGER, rece_rank(proc_gap), &
                & tag, D_comm(direction), s_request_bis(proc_gap),ierr)
        end if
    end do
    allocate(V_buffer(max(size_buffer,1)))
    V_buffer = 0
    ! Send the velocity field to processus which need it
    allocate(s_request(send_gap(1):send_gap(2)))
    do proc_gap = send_gap(1), send_gap(2)
        call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, send_rank, ierr)
        if (send_rank /= D_rank(direction)) then
            ! I - Receive messages about what I have to send
            ! Ia - Compute reception tag = concatenation of (rank+1), ind_group(1), ind_group(2), direction et unique Id.
            tag = compute_tag(ind_group, tag_velo_range, direction, -proc_gap)
            ! Ib - Receive the message
            call mpi_recv(send_range(1), 2, MPI_INTEGER, send_rank, tag, D_comm(direction), rece_status, ierr)
            send_range = send_range + proc_gap*mesh_sc%N_proc(direction)
            ! II - Send it
            ! IIa - Compute send tag
            tag = compute_tag(ind_group, tag_velo_V, direction, proc_gap)
            ! IIb - Send message
            call mpi_Isend(p_V(send_range(1)), send_range(2)-send_range(1)+1, MPI_REAL_WP, &
                    & send_rank, tag, D_comm(direction), s_request(proc_gap), ierr)
        end if
    end do

    ! Non blocking reception of the velocity field
    ind = 1
    allocate(rece_request(rece_gap(1):rece_gap(2)))
    do proc_gap = rece_gap(1), rece_gap(2)
        if (rece_rank(proc_gap) /= D_rank(direction)) then
            ! IIa - Compute reception tag
            tag = compute_tag(ind_group, tag_velo_V, direction, -proc_gap)
            ! IIb - Receive message
            gap = proc_gap*mesh_sc%N_proc(direction)
            rece_range(1) = max(rece_ind_min, gap+1) ! fortran => indice start from 0
            rece_range(2) = min(rece_ind_max, gap+mesh_sc%N_proc(direction))
            msg_size = rece_range(2)-rece_range(1)+1
            call mpi_Irecv(V_buffer(ind), msg_size, MPI_REAL_WP, rece_rank(proc_gap), tag, D_comm(direction), &
                        & rece_request(proc_gap), ierr)
            ind = ind + msg_size
        end if
    end do

    ! -- Compute the interpolated velocity
    ! Compute the interpolation weight and update the pointers Vp and Vm
    ! Initialisation of reccurence process
    ind = 1
    pos = floor(p_pos_bis(ind))
    weight(ind) = p_pos_bis(ind)-pos
    ! Vm = V(pos)
    proc_gap = floor(real(pos-1, WP)/mesh_sc%N_proc(direction))
    call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, send_rank, ierr)
    if (send_rank == D_rank(direction)) then
        Vm(ind)%pter => p_V_bis(pos-proc_gap*mesh_sc%N_proc(direction))
    else
        ind_com = ind_com + 1
        Vm(ind)%pter => V_buffer(ind_com)
    end if
    ! Vp = V(pos+1)
    proc_gap = floor(real(pos+1-1, WP)/mesh_sc%N_proc(direction))
    call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, send_rank, ierr)
    if (send_rank == D_rank(direction)) then
        Vp(ind)%pter => p_V_bis(pos+1-proc_gap*mesh_sc%N_proc(direction))
    else
        ind_com = ind_com + 1
        Vp(ind)%pter => V_buffer(ind_com)
    end if
    pos_old = pos

    ! Following indice : we use previous work (already done)
    do ind = 2, mesh_sc%N_proc(direction)
        pos = floor(p_pos_bis(ind))
        weight(ind) = p_pos_bis(ind)-pos
        select case(pos-pos_old)
            case(0)
                ! The particle belongs to the same segment than the previous one
                Vm(ind)%pter => Vm(ind-1)%pter
                Vp(ind)%pter => Vp(ind-1)%pter
            case(1)
                ! The particle follows the previous one
                Vm(ind)%pter => Vp(ind-1)%pter
                ! Vp = V(pos+1)
                proc_gap = floor(real(pos+1-1, WP)/mesh_sc%N_proc(direction)) ! fortran -> indice starts from 1
                call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, send_rank, ierr)
                if (send_rank == D_rank(direction)) then
                    Vp(ind)%pter => p_V_bis(pos+1-proc_gap*mesh_sc%N_proc(direction))
                else
                    ind_com = ind_com + 1
                    Vp(ind)%pter => V_buffer(ind_com)
                end if
            case(2)
                ! pos = pos_old +2, wich correspond to "extention"
                ! Vm = V(pos)
                proc_gap = floor(real(pos-1, WP)/mesh_sc%N_proc(direction))
                call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, send_rank, ierr)
                if (send_rank == D_rank(direction)) then
                    Vm(ind)%pter => p_V_bis(pos-proc_gap*mesh_sc%N_proc(direction))
                else
                    ind_com = ind_com + 1
                    Vm(ind)%pter => V_buffer(ind_com)
                end if
                ! Vp = V(pos+1)
                proc_gap = floor(real(pos+1-1, WP)/mesh_sc%N_proc(direction))
                call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, send_rank, ierr)
                if (send_rank == D_rank(direction)) then
                    Vp(ind)%pter => p_V_bis(pos+1-proc_gap*mesh_sc%N_proc(direction))
                else
                    ind_com = ind_com + 1
                    Vp(ind)%pter => V_buffer(ind_com)
                end if
            case default
                print*, "unexpected case : pos = ", pos, " , pos_old = ", pos_old, " ind = ", ind
        end select
        pos_old = pos
    end do

    ! -- Compute the interpolate velocity --
    ! Check if communication are done
    do proc_gap = rece_gap(1), rece_gap(2)
        if (rece_rank(proc_gap)/=D_rank(direction)) then
            call mpi_wait(rece_request(proc_gap), rece_status, ierr)
        end if
    end do


    ! Then compute the field
    do ind = 1, mesh_sc%N_proc(direction)
        p_V(ind) = weight(ind)*Vp(ind)%pter + (1-weight(ind))*Vm(ind)%pter
    end do

    do ind = 1, mesh_sc%N_proc(direction)
        nullify(Vp(ind)%pter)
        nullify(Vm(ind)%pter)
    end do

    do proc_gap = send_gap(1), send_gap(2)
        call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, send_rank, ierr)
        if (send_rank /= D_rank(direction)) then
            call MPI_WAIT(s_request(proc_gap),rece_status,ierr)
        end if
    end do
    deallocate(s_request)
    do proc_gap = rece_gap(1), rece_gap(2)
        if (rece_rank(proc_gap) /= D_rank(direction)) then
            call MPI_WAIT(s_request_bis(proc_gap),rece_status,ierr)
        end if
    end do
    deallocate(s_request_bis)

    ! Deallocation
    deallocate(rece_rank)
    deallocate(rece_request)
    deallocate(V_buffer)

end subroutine AC_particle_velocity_line



! ===================================================================================================
! ====================     Others than velocity interpolation and remeshing      ====================
! ===================================================================================================
!> Determine type (center or left) of each block of a line and tag particle of this line to know where
!! corrected remeshing formula are recquired.
!!    @param[in]        dt          = time step
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        p_V         = particle velocity (along the current direction)
!!    @param[out]       bl_type     = table of blocks type (center of left)
!!    @param[out]       bl_tag      = inform about tagged particles (bl_tag(ind_bl)=1 if the end of the bl_ind-th block
!!                                    and the begining of the following one is tagged)
!! @details
!!        This subroutine deals with a single line. For each line of this group, it
!!    determine the type of each block of this line and where corrected remeshing
!!    formula are required. In those points, it tagg block transition (ie the end of
!!    the current block and the beginning of the following one) in order to indicate
!!    that corrected weigth have to be used during the remeshing.
subroutine AC_type_and_block_line(dt, direction, ind_group, p_V, &
                    & bl_type, bl_tag)

    
    use cart_topology
    use advec_variables
    use precision_tools

    ! In/Out variables
    real(WP), intent(in)                                    :: dt           ! time step
    integer, intent(in)                                     :: direction
    integer, dimension(2), intent(in)                       :: ind_group
    real(WP), dimension(:), intent(in)                      :: p_V
    logical, dimension(bl_nb(direction)+1), intent(out)     :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction)), intent(out)       :: bl_tag       ! indice of tagged particles
    ! Local variables
    real(WP), dimension(bl_nb(direction)+1)                 :: bl_lambdaMin ! for a particle, lamda = V*dt/dx ;  bl_lambdaMin = min of
                                                                            ! lambda on a block (take also into account first following particle)
    real(WP)                                                :: lambP, lambN ! buffer to exchange some lambda min with other processus
    integer, dimension(bl_nb(direction)+1)                  :: bl_ind       ! block index : integer as lambda in (bl_ind,bl_ind+1) for a left block
                                                                            ! and lambda in (bl_ind-1/2, bl_ind+1/2) for a right block
    integer                                                 :: ind, i_p     ! some indices
    real(WP)                                                :: cfl          ! = d_sc
    integer                                                 :: rankP, rankN ! processus rank for shift (P= previous, N = next)
    integer, dimension(2)                                   :: send_request ! mpi status of nonblocking send
    integer, dimension(2)                                   :: rece_request ! mpi status of nonblocking receive
    integer, dimension(MPI_STATUS_SIZE)                     :: rece_status  ! mpi status (for mpi_wait)
    integer, dimension(MPI_STATUS_SIZE)                     :: send_status  ! mpi status (for mpi_wait)
    integer, dimension(2)                                   :: tag_table    ! other tags for mpi message
    integer                                                 :: ierr         ! mpi error code

    ! ===== Initialisation =====
    cfl = dt/mesh_sc%dx(direction)

    ! ===== Compute bl_lambdaMin =====
    ! -- Compute rank of my neighbor --
    call mpi_cart_shift(D_comm(direction), 0, 1, rankP, rankN, ierr)

    ! -- For the first block (1/2) --
    ! The domain contains only its second half => exchange ghost with the previous processus
    bl_lambdaMin(1) = minval(p_V(1:(bl_size/2)+1))*cfl
    tag_table = compute_tag(ind_group, tag_part_tag_NP, direction)
    ! Send message
    call mpi_Isend(bl_lambdaMin(1), 1, MPI_REAL_WP, rankP, tag_table(1), D_comm(direction), send_request(1), ierr)
    ! Receive it
    call mpi_Irecv(lambN, 1, MPI_REAL_WP, rankN, tag_table(1), D_comm(direction), rece_request(1), ierr)

    ! -- For the last block (1/2) --
    ! The processus contains only its first half => exchange ghost with the next processus
    ind = bl_nb(direction) + 1
    bl_lambdaMin(ind) = minval(p_V(mesh_sc%N_proc(direction)-(bl_size/2)+1:mesh_sc%N_proc(direction)))*cfl
    ! Send message
    call mpi_Isend(bl_lambdaMin(ind), 1, MPI_REAL_WP, rankN, tag_table(2), D_comm(direction), send_request(2), ierr)
    ! Receive it
    call mpi_Irecv(lambP, 1, MPI_REAL_WP, rankP, tag_table(2), D_comm(direction), rece_request(2), ierr)

    ! -- For the "middle" block --
    do ind = 2, bl_nb(direction)
        i_p = ((ind-1)*bl_size) + 1 - bl_size/2
        bl_lambdaMin(ind) = minval(p_V(i_p:i_p+bl_size))*cfl
    end do

    ! -- For the first block (1/2) --
    ! The domain contains only its second half => use exchanged ghost
    ! Check reception
    call mpi_wait(rece_request(2), rece_status, ierr)
    bl_lambdaMin(1) = min(bl_lambdaMin(1), lambP)

    ! -- For the last block (1/2) --
    ! The processus contains only its first half => use exchanged ghost
    ! Check reception
    call mpi_wait(rece_request(1), rece_status, ierr)
    ind = bl_nb(direction) + 1
    bl_lambdaMin(ind) = min(bl_lambdaMin(ind), lambN)

    ! ===== Compute block type and index =====
    bl_ind = nint(bl_lambdaMin)
    bl_type = (bl_lambdaMin<dble(bl_ind))


    ! => center type if true, else left

    ! ===== Tag particles =====
    do ind = 1, bl_nb(direction)
        bl_tag(ind) = ((bl_ind(ind)/=bl_ind(ind+1)) .and. (bl_type(ind).neqv.bl_type(ind+1)))
    end do

    call mpi_wait(send_request(1), send_status, ierr)
    call mpi_wait(send_request(2), send_status, ierr)

end subroutine AC_type_and_block_line


! ===================================================================
! ====================     Remesh particles      ====================
! ===================================================================

!> Determine the set of processes wich will send me information during the
!!  scalar remeshing. Use implicit computation rather than communication (only
!!  possible if particle are gather by block whith contrainst on velocity variation
!!  - as corrected lambda formula.) - work on only a line of particles.
!!    @param[in]    send_i_min  = minimal indice of the send buffer
!!    @param[in]    send_i_max  = maximal indice of the send buffer
!!    @param[in]    direction   = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]    ind_group   = coordinate of the current group of lines
!!    @param[out]   proc_min    = gap between my coordinate and the processes of minimal coordinate which will receive information from me
!!    @param[out]   proc_max    = gap between my coordinate and the processes of maximal coordinate which will receive information from me
!!    @param[out]   rece_proc   = coordinate range of processes which will send me information during the remeshing.
!! @details
!!    Obtain the list of processus which contains some particles which belong to
!!    my subdomains after their advection (and thus which will be remeshing into
!!    my subdomain). This result is return as an interval [send_min; send_max].
!!    All the processus whose coordinate (into the current direction) belong to
!!    this segment are involved into scalar remeshing into the current
!!    subdomains. This routine does not involve any computation to determine if
!!    a processus is the first or the last processes (considering its coordinate along
!!    the current directory) to send remeshing information to a given processes.
!!    It directly compute it using contraints on velocity (as in corrected lambda
!!    scheme) When possible use it rather than AC_obtain_senders_com
subroutine AC_obtain_senders_line(send_i_min, send_i_max, direction, ind_group, proc_min, proc_max, rece_proc)
! XXX Work only for periodic condition. For dirichlet conditions : it is
! possible to not receive either rece_proc(1), either rece_proc(2) or none of
! these two => detect it (track the first and the last particles) and deal with it.

    use cart_topology   ! info about mesh and mpi topology
    
    use advec_variables

    ! Input/output
    integer, intent(in)                 :: send_i_min
    integer, intent(in)                 :: send_i_max
    integer, intent(in)                 :: direction
    integer, dimension(2), intent(in)   :: ind_group
    integer(kind=4), intent(out)        :: proc_min, proc_max
    integer, dimension(2), intent(out)  :: rece_proc
    integer, dimension(MPI_STATUS_SIZE) :: statut
    ! Other local variable
    integer(kind=4)                     :: proc_gap         ! gap between a processus coordinate (along the current
                                                            ! direction) into the mpi-topology and my coordinate
    integer                             :: rankP, rankN     ! processus rank for shift (P= previous, N = next)
    integer, dimension(2)               :: tag_table        ! mpi message tag (for communicate rece_proc(1) and rece_proc(2))
    integer, dimension(:,:),allocatable :: send_request     ! mpi status of nonblocking send
    integer                             :: ierr             ! mpi error code

    tag_table = compute_tag(ind_group, tag_obtsend_NP, direction)

    rece_proc = 3*mesh_sc%N(direction)

    proc_min = floor(real(send_i_min-1, WP)/mesh_sc%N_proc(direction))
    proc_max = floor(real(send_i_max-1, WP)/mesh_sc%N_proc(direction))

    allocate(send_request(proc_min:proc_max,3))
    send_request(:,3) = 0

    ! Send
    do proc_gap = proc_min, proc_max
        ! Compute the rank of the target processus
        call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, rankN, ierr)
        ! Determine if I am the the first or the last processes (considering my
                ! coordinate along the current directory) to send information to
                ! one of these processes.
                ! Note that local indice go from 1 to mesh_sc%N_proc (fortran).
        ! I am the first ?
        if ((send_i_min< +1-2*bl_bound_size + proc_gap*mesh_sc%N_proc(direction)+1).AND. &
                    & (send_i_max>= proc_gap*mesh_sc%N_proc(direction))) then
            if(rankN /= D_rank(direction)) then
                call mpi_Isend(-proc_gap, 1, MPI_INTEGER, rankN, tag_table(1), D_comm(direction), &
                        & send_request(proc_gap,1), ierr)
                send_request(proc_gap,3) = 1
            else
                rece_proc(1) = -proc_gap
            end if
        end if
        ! I am the last ?
        if ((send_i_max > -1+2*bl_bound_size + (proc_gap+1)*mesh_sc%N_proc(direction)) &
                    & .AND.(send_i_min<= (proc_gap+1)*mesh_sc%N_proc(direction))) then
            if(rankN /= D_rank(direction)) then
                call mpi_Isend(-proc_gap, 1, MPI_INTEGER, rankN, tag_table(2), D_comm(direction), &
                        & send_request(proc_gap,2), ierr)
                send_request(proc_gap,3) = send_request(proc_gap, 3) + 2
            else
                rece_proc(2) = -proc_gap
            end if
        end if
    end do


    ! Receive
    if (rece_proc(1) == 3*mesh_sc%N(direction)) then
        call mpi_recv(rece_proc(1), 1, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(1), D_comm(direction), statut, ierr)
    end if
    if (rece_proc(2) == 3**mesh_sc%N(direction)) then
        call mpi_recv(rece_proc(2), 1, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(2), D_comm(direction), statut, ierr)
    end if

    ! Free Isend buffer
    do proc_gap = proc_min, proc_max
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

    deallocate(send_request)

end subroutine AC_obtain_senders_line


!> Common procedure for remeshing wich perform all the communcation and provide
!! the update scalar field.
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        send_i_min  = minimal indice of the send buffer
!!    @param[in]        send_i_max  = maximal indice of the send buffer
!!    @param[out]       proc_min    = gap between my coordinate and the processes of minimal coordinate which will receive information from me
!!    @param[out]       proc_max    = gap between my coordinate and the processes of maximal coordinate which will receive information from me
!!    @param[out]       rece_proc   = coordinate range of processes which will send me information during the remeshing.
!!    @param[in]        send_buffer = buffer use to remesh the scalar before to send it to the right subdomain
!!    @param[in,out]    scal1D      = mono-dimensionnal scalar field to advect
!! @details
!!    Remeshing are done in a local buffer. This subroutine distribute this buffer
!!    to the right processes, receive the buffer send and update the scalar field.
subroutine AC_bufferToScalar_line(direction, ind_group, send_i_min, send_i_max, proc_min, proc_max, rece_proc,send_buffer, scal1D)

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.
    

    ! Input/Ouptut
    integer, intent(in)                                     :: direction
    integer, dimension(2), intent(in)                       :: ind_group
    integer, intent(in)                                     :: send_i_min
    integer, intent(in)                                     :: send_i_max
    integer, dimension(2), intent(in)                       :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the
                                                                            ! one from which  I will receive data
    integer, intent(in)                                     :: proc_min     ! smaller gap between me and the processes to where I send data
    integer, intent(in)                                     :: proc_max     ! smaller gap between me and the processes to where I send data
    real(WP), dimension(send_i_min:send_i_max), intent(in)  :: send_buffer
    real(WP), dimension(mesh_sc%N_proc(direction)), intent(inout)   :: scal1D

    ! Variables used to communicate between subdomains. A variable prefixed by "send_"(resp "rece")
    ! design something I send (resp. I receive).
    integer                             :: i            ! table indice
    integer                             :: proc_gap     ! gap between my Y-coordinate and the one of the processus
    real(WP), dimension(:), allocatable :: rece_buffer  ! buffer use to stock received scalar field
    integer                             :: send_gap     ! number of mesh between my and another processus
    integer,dimension(:,:), allocatable :: rece_range   ! range of (local) indice where the received scalar field has to be save
    integer,dimension(:,:), allocatable :: send_range   ! range of (local) indice where the send scalar field has to be save))
    integer, dimension(:), allocatable  :: rece_request ! mpi communication request (handle) of nonblocking receive
    integer, dimension(:), allocatable  :: rece_rank    ! rank of processus from wich I receive data
    integer                             :: send_rank    ! rank of processus to which I send data
    integer                             :: rankP        ! rank used in mpi_cart_shift
    integer, dimension(MPI_STATUS_SIZE) :: rece_status  ! mpi status (for mpi_wait)
    integer, dimension(MPI_STATUS_SIZE) :: send_status  ! mpi status (for mpi_wait)
    integer, dimension(:,:),allocatable :: send_request ! mpi status of nonblocking send
    integer                             :: rece_i_min   ! the minimal indice from where belong the scalar field I receive
    integer                             :: rece_i_max   ! the maximal indice from where belong the scalar field I receive
    integer                             :: ierr         ! mpi error code
    integer                             :: comm_size    ! number of element to send/receive
    integer                             :: tag          ! mpi message tag
                                                        ! with wich I communicate.

    ! ===== Receive information =====
    ! -- Allocate field --
    allocate(rece_rank(rece_proc(1):rece_proc(2)))
    allocate(rece_range(2,rece_proc(1):rece_proc(2)))  ! be careful that mpi use contiguous memory element
    allocate(rece_request(rece_proc(1):rece_proc(2)))
    ! -- Receive range --
    do proc_gap = rece_proc(1), rece_proc(2)
        call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, rece_rank(proc_gap), ierr)
        if (rece_rank(proc_gap)/=D_rank(direction)) then
            tag = compute_tag(ind_group, tag_bufToScal_range, direction, -proc_gap)
            call mpi_Irecv(rece_range(1,proc_gap), 2, MPI_INTEGER, rece_rank(proc_gap), tag, D_comm(direction), &
                        & rece_request(proc_gap), ierr) ! we use tag = source rank
        end if
    end do

    ! Send the information
    allocate(send_request(proc_min:proc_max,3))
    send_request(:,3)=0
    allocate(send_range(2,proc_min:proc_max))
    do proc_gap = proc_min, proc_max
            ! Compute the rank of the target processus
            call mpi_cart_shift(D_comm(direction), 0, proc_gap, rankP, send_rank, ierr)
            send_gap = proc_gap*mesh_sc%N_proc(direction)
            send_range(1, proc_gap) = max(send_i_min, send_gap+1) ! fortran => indice start from 0
            send_range(2, proc_gap) = min(send_i_max, send_gap+mesh_sc%N_proc(direction))
        if (send_rank/=D_rank(direction)) then
            ! Determine quantity of information to send
            comm_size = send_range(2, proc_gap)-send_range(1, proc_gap)+1
            ! Send the range of the scalar field send
            tag = compute_tag(ind_group, tag_bufToScal_range, direction, proc_gap)
            call mpi_ISsend(send_range(1, proc_gap), 2, MPI_INTEGER, send_rank, tag, D_comm(direction), send_request(proc_gap,1)&
                    & , ierr)
            ! And send the buffer
            tag = compute_tag(ind_group, tag_bufToScal_buffer, direction, proc_gap)
            call mpi_ISsend(send_buffer(send_range(1,proc_gap)),comm_size, MPI_REAL_WP, send_rank, &
                        & tag, D_comm(direction), send_request(proc_gap,2), ierr)
            send_request(proc_gap,3) = 1
        else
            ! I have to distribute the buffer in myself
            do i = send_range(1, proc_gap), send_range(2, proc_gap)
                scal1D(i-send_gap) = scal1D(i-send_gap) + send_buffer(i)
            end do
        end if
    end do

    ! Check reception
    do proc_gap = rece_proc(1), rece_proc(2)
        if (rece_rank(proc_gap)/=D_rank(direction)) then
            call mpi_wait(rece_request(proc_gap), rece_status, ierr)
        end if
    end do
    deallocate(rece_request)
    ! Receive buffer and remesh it
        ! XXX Possible optimisation : an optimal code will
        !   1 - have non-blocking reception of scalar buffers
        !   2 - check when a reception is done and then update the scalar
        !   3 - iterate step 2 until all message was rece and that the scalar
        !       field was update with all the scalar buffers
    do proc_gap = rece_proc(1), rece_proc(2)
        if (rece_rank(proc_gap)/=D_rank(direction)) then
            rece_i_min = rece_range(1,proc_gap)
            rece_i_max = rece_range(2,proc_gap)
            ! Receive information
            comm_size=(rece_i_max-rece_i_min+1)
            allocate(rece_buffer(rece_i_min:rece_i_max)) ! XXX possible optimisation
                ! by allocating one time to the max size, note that the range use in
                ! this allocation instruction is include in (1, mesh_sc%N_proc(2))
            tag = compute_tag(ind_group, tag_bufToScal_buffer, direction, -proc_gap)
            call mpi_recv(rece_buffer(rece_i_min), comm_size, MPI_REAL_WP, &
                    & rece_rank(proc_gap), tag, D_comm(direction), rece_status, ierr)
            ! Update the scalar field
            send_gap = proc_gap*mesh_sc%N_proc(direction)
            scal1D(rece_i_min+send_gap:rece_i_max+send_gap) = scal1D(rece_i_min+send_gap:rece_i_max+send_gap) &
                & + rece_buffer(rece_i_min:rece_i_max)
            deallocate(rece_buffer)
        end if
    end do


    ! Free Isend buffer
    do proc_gap = proc_min, proc_max
        if (send_request(proc_gap,3)/=0) then
            call mpi_wait(send_request(proc_gap,1), send_status, ierr)
            call mpi_wait(send_request(proc_gap,2), send_status, ierr)
        end if
    end do
    deallocate(send_request)
    deallocate(send_range)

    deallocate(rece_range)
    deallocate(rece_rank)

end subroutine AC_bufferToScalar_line



end module advec_common_line
