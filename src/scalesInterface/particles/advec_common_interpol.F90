!USEFORTEST advec
!> @addtogroup part
!! @{
!------------------------------------------------------------------------------
!
! MODULE: advec_common_velo
!
!
! DESCRIPTION:
!> The module ``advec_common_interpol'' gather function and subroutines used to interpolate
!! some quantities (velocity for instance) at particle position. Theses tools are specific to a direction
!! @details
!! This module gathers functions and routines used to interpolate some field
!! at scalar position. These subroutines are not specific to a direction.
!! This is a parallel implementation using MPI and the cartesien topology it
!! provides.
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

module advec_common_interpol

    use structure_tools
    use advec_abstract_proc
    use mpi, only:MPI_INTEGER, MPI_ANY_SOURCE, MPI_STATUS_SIZE
    implicit none


    ! Information about the particles and their bloc
    public


    ! ===== Public procedures =====
    !----- To interpolate velocity -----
    public                        :: AC_interpol_lin
    public                        :: AC_interpol_plus
    public                        :: AC_interpol_lin_no_com
    public                        :: AC_interpol_determine_communication

    ! ===== Public variables =====

    ! ===== Private variables =====


contains

! ===== Public procedure =====

! ==================================================================================
! ====================     Compute particle velocity (RK2)      ====================
! ==================================================================================

!> Interpolate the velocity field used in a RK2 scheme for particle advection -
!! version for a group of (more of one) line
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        gs          = size of a group (ie number of line it gathers along the two other directions)
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        p_inter     = adimensionned particle postion in input ; return inteprolated field as output
!!    @param[in,out]    V_comp      = field to interpolate at particle position
!! @details
!!    A RK2 scheme is used to advect the particles : the midlle point scheme. An
!!    intermediary position "p_pos_bis(i) = p_pos(i) + V(i)*dt/2" is computed and then
!!    the numerical velocity of each particles is computed as the interpolation of V  in
!!    this point. This field is used to advect the particles at the seconde order in time :
!!    p_pos(t+dt, i) = p_pos(i) + p_V(i).
!!    The group line indice is used to ensure using unicity of each mpi message tag.
!!    The interpolation is done for a group of lines, allowing to mutualise
!!    communications. Considering a group of Na X Nb lines, communication performed
!!    by this algorithm are around (Na x Nb) bigger than the alogorithm wich
!!    works on a single line but also around (Na x Nb) less frequent.
subroutine AC_interpol_lin(direction, gs, ind_group, V_comp, p_inter)

    ! This code involve a recopy of V_comp. It is possible to directly use the 3D velocity field but in a such code
    ! a memory copy is still needed to send velocity field to other processus : mpi send contiguous memory values

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Ouput
    integer, intent(in)                             :: direction    ! current direction
    integer, dimension(2),intent(in)                :: gs           ! groupe size
    integer, dimension(2), intent(in)               :: ind_group
    real(WP), dimension(:,:,:), intent(inout)       :: p_inter
    real(WP), dimension(:,:,:),intent(in),target    :: V_comp
#ifdef BLOCKING_SEND_PLUS
    real(WP)                                                    :: weight       ! interpolation weight storage
#else
    type(real_pter),dimension(mesh_sc%N_proc(direction),gs(1),gs(2))    :: Vp, Vm       ! Velocity on previous and next mesh point
#endif
    real(WP), dimension(:), allocatable, target                 :: V_buffer     ! Velocity buffer for postion outside of the local subdomain
    integer, dimension(:), allocatable                          :: pos_in_buffer! buffer size
    integer , dimension(gs(1), gs(2))           :: rece_ind_min ! minimal indice of mesh involved in remeshing particles (of my local subdomains)
    integer , dimension(gs(1), gs(2))           :: rece_ind_max ! maximal indice of mesh involved in remeshing particles (of my local subdomains)
    integer                                     :: ind, ind_com ! indices
    integer                                     :: i1, i2       ! indices in the lines group
    integer                                     :: pos, pos_old ! indices of the mesh point wich preceed the particle position
    integer                                     :: proc_gap, gap! distance between my (mpi) coordonate and coordinate of the
                                                                ! processus associated to a given position
    integer                                     :: proc_end     ! final indice of processus associate to current pos
    logical, dimension(2)                       :: myself
    integer, dimension(:), allocatable          :: send_carto   ! cartogrpahy of what I have to send
    integer                                     :: ind_1Dtable  ! indice of my current position inside a one-dimensionnal table
    integer                                     :: ind_for_i1   ! where to read the first coordinate (i1) of the current line inside the cartography ?
    real(WP), dimension(:), allocatable         :: send_buffer  ! to store what I have to send (on a contiguous way)
    integer, dimension(gs(1),gs(2),2)           :: rece_gap     ! distance between me and processus wich send me information
    integer, dimension(2 , 2)                   :: send_gap     ! distance between me and processus to wich I send information
    integer, dimension(2)                       :: rece_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer                                     :: com_size     ! size of message send/receive
    integer, dimension(:), allocatable          :: size_com     ! size of message send/receive
    integer                                     :: min_size     ! minimal size of cartography(:,proc_gap)
    integer                                     :: max_size     ! maximal size of cartography(:,proc_gap)
    integer                                     :: tag          ! mpi message tag
    integer, dimension(:), allocatable          :: tag_proc     ! mpi message tag
    integer                                     :: ierr         ! mpi error code
#ifndef BLOCKING_SEND
   integer, dimension(:), allocatable          :: s_request    ! mpi communication request (handle) of nonblocking send
#endif
    integer, dimension(:), allocatable          :: s_request_bis! mpi communication request (handle) of nonblocking send
    integer, dimension(:), allocatable          :: rece_request ! mpi communication request (handle) of nonblocking receive
    integer, dimension(MPI_STATUS_SIZE)         :: rece_status  ! mpi status (for mpi_wait)
    integer, dimension(:,:), allocatable        :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block for wich the
                                                                ! current processus requiers data from proc_gap and for each of these lines the range
                                                                ! of mesh points from where it requiers the velocity values.

    ! -- Initialisation --
#ifndef BLOCKING_SEND_PLUS
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            do ind = 1, mesh_sc%N_proc(direction)
                nullify(Vp(ind,i1,i2)%pter)
                nullify(Vm(ind,i1,i2)%pter)
            end do
        end do
    end do
#endif

    ! Compute range of the set of point where I need the velocity value
    rece_ind_min = floor(p_inter(1,:,:))
    rece_ind_max = floor(p_inter(mesh_sc%N_proc(direction),:,:)) + 1

    ! ===== Exchange velocity field if needed =====
    ! It uses non blocking message to do the computations during the communication process
    ! -- What have I to communicate ? --
    rece_gap(:,:,1) = floor(real(rece_ind_min-1, WP)/mesh_sc%N_proc(direction))
    rece_gap(:,:,2) = floor(real(rece_ind_max-1, WP)/mesh_sc%N_proc(direction))
    rece_gap_abs(1) = minval(rece_gap(:,:,1))
    rece_gap_abs(2) = maxval(rece_gap(:,:,2))
    max_size = 2 + gs(2)*(2+3*gs(1))
    allocate(cartography(max_size,rece_gap_abs(1):rece_gap_abs(2)))
    call AC_interpol_determine_communication(direction, ind_group, gs, send_gap,  &
    & rece_gap, rece_gap_abs, cartography)

    ! -- Send messages about what I want --
    allocate(s_request_bis(rece_gap_abs(1):rece_gap_abs(2)))
    allocate(size_com(rece_gap_abs(1):rece_gap_abs(2)))
    allocate(tag_proc(rece_gap_abs(1):rece_gap_abs(2)))
    min_size = 2 + gs(2)
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            cartography(1,proc_gap) = 0
            ! Use the cartography to know which lines are concerned
            size_com(proc_gap) = cartography(2,proc_gap)
            ! Range I want - store into the cartography
            gap = proc_gap*mesh_sc%N_proc(direction)
            ! Position in cartography(:,proc_gap) of the current i1 indice
            ind_for_i1 = min_size
            do i2 = 1, gs(2)
                do ind = ind_for_i1+1, ind_for_i1 + cartography(2+i2,proc_gap), 2
                    do i1 = cartography(ind,proc_gap), cartography(ind+1,proc_gap)
                        ! Interval start from:
                        cartography(size_com(proc_gap)+1,proc_gap) = max(rece_ind_min(i1,i2), gap+1) ! fortran => indice start from 0
                        ! and ends at:
                        cartography(size_com(proc_gap)+2,proc_gap) = min(rece_ind_max(i1,i2), gap+mesh_sc%N_proc(direction))
                        ! update number of element to receive
                        cartography(1,proc_gap) = cartography(1,proc_gap) &
                                    & + cartography(size_com(proc_gap)+2,proc_gap) &
                                    & - cartography(size_com(proc_gap)+1,proc_gap) + 1
                        size_com(proc_gap) = size_com(proc_gap)+2
                    end do
                end do
                ind_for_i1 = ind_for_i1 + cartography(2+i2,proc_gap)
            end do
            ! Tag = concatenation of (rank+1), ind_group(1), ind_group(2), direction et unique Id.
            tag_proc(proc_gap) = compute_tag(ind_group, tag_velo_range, direction, proc_gap)
            ! Send message
#ifdef PART_DEBUG
            if(size_com(proc_gap)>max_size) then
                print*, 'rank = ', cart_rank, ' -- bug sur taille cartography a envoyer'
                print*, 'taille carto = ', com_size, ' plus grand que la taille théorique ', &
                    & max_size, ' et carto = ', cartography(:,proc_gap)
            end if
#endif
            call mpi_ISsend(cartography(1,proc_gap), size_com(proc_gap), MPI_INTEGER,   &
                & neighbors(direction,proc_gap), tag_proc(proc_gap), D_comm(direction), &
                & s_request_bis(proc_gap),ierr)
        end if
    end do


    ! -- Non blocking reception of the velocity field --
    ! Allocate the pos_in_buffer to compute V_buffer size and to be able to
    ! allocate it.
    allocate(pos_in_buffer(rece_gap_abs(1):rece_gap_abs(2)))
    pos_in_buffer(rece_gap_abs(1)) = 1
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)-1
        pos_in_buffer(proc_gap+1)= pos_in_buffer(proc_gap) + cartography(1,proc_gap)
    end do
    allocate(V_buffer(pos_in_buffer(rece_gap_abs(2)) &
                & + cartography(1,rece_gap_abs(2))))
    V_buffer = 0
    allocate(rece_request(rece_gap_abs(1):rece_gap_abs(2)))
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            ! IIa - Compute reception tag
            tag = compute_tag(ind_group, tag_velo_V, direction, -proc_gap)
            ! IIb - Receive message
            call mpi_Irecv(V_buffer(pos_in_buffer(proc_gap)), cartography(1,proc_gap), MPI_REAL_WP, &
                    & neighbors(direction,proc_gap), tag, D_comm(direction), rece_request(proc_gap), ierr)
        end if
    end do

    ! -- Send the velocity field to processus which need it --
#ifndef BLOCKING_SEND
   allocate(s_request(send_gap(1,1):send_gap(1,2)))
#endif
    allocate(send_carto(max_size))
! XXX Todo : compter le nombre de messages à recevoir puis les traiter dans
! l'ordre où ils arrivent via un MPI_ANY_PROC ? Mais alors il faut lier rang et
! coordonnées ... ce qui signifie ajouter un appel à un mpi_cart_cood ... ou
! envoyer le rand dans la cartographie !!
! A voir ce qui est le mieux.
    do proc_gap = send_gap(1,1), send_gap(1,2)
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            ! I - Receive messages about what I have to send
            ! Ia - Compute reception tag = concatenation of (rank+1), ind_group(1), ind_group(2), direction et unique Id.
            tag = compute_tag(ind_group, tag_velo_range, direction, -proc_gap)
            ! Ib - Receive the message
            call mpi_recv(send_carto(1), max_size, MPI_INTEGER, neighbors(direction,proc_gap), &
              & tag, D_comm(direction), rece_status, ierr)
            ! II - Send it
            ! IIa - Create send buffer
            allocate(send_buffer(send_carto(1)))
            gap = proc_gap*mesh_sc%N_proc(direction)
            com_size = 0
            ind_1Dtable = send_carto(2)
            ! Position in cartography(:,proc_gap) of the current i1 indice
            ind_for_i1 = min_size
            do i2 = 1, gs(2)
                do ind = ind_for_i1+1, ind_for_i1 + send_carto(2+i2), 2
                    do i1 = send_carto(ind), send_carto(ind+1)
                        do ind_com = send_carto(ind_1Dtable+1)+gap, send_carto(ind_1Dtable+2)+gap ! indice inside the current line
                            com_size = com_size + 1
                            send_buffer(com_size) = V_comp(ind_com, i1,i2)
                        end do
                        ind_1Dtable = ind_1Dtable + 2
                    end do
                end do
                ind_for_i1 = ind_for_i1 + send_carto(2+i2)
            end do
            ! IIa_bis - check correctness
#ifdef PART_DEBUG
            if(com_size/=send_carto(1)) then
                print*, 'rank = ', cart_rank, ' -- bug sur taille champ de vitesse a envoyer'
                print*, 'taille carto = ', com_size, ' plus grand recu ', &
                    & send_carto(1), ' et carto = ', send_carto(:)
            end if
#endif
            ! IIb - Compute send tag
            tag = compute_tag(ind_group, tag_velo_V, direction, proc_gap)
            ! IIc - Send message
#ifdef BLOCKING_SEND
            call mpi_Send(send_buffer(1), com_size, MPI_REAL_WP,  &
                    & neighbors(direction,proc_gap), tag, D_comm(direction),&
                    & ierr)
#else
           call mpi_ISend(send_buffer(1), com_size, MPI_REAL_WP,  &
                   & neighbors(direction,proc_gap), tag, D_comm(direction),&
                   & s_request(proc_gap), ierr)
#endif
            deallocate(send_buffer)
        end if
    end do
    deallocate(send_carto)

    !-- Free som ISsend buffer and some array --
! XXX Todo : préférer un call MPI_WAIT_ALL couplé avec une init de s_request_bis
! sur MPI_REQUEST_NULL et enlever la boucle ET le if.
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            call MPI_WAIT(s_request_bis(proc_gap),rece_status,ierr)
        end if
    end do
    deallocate(s_request_bis)
    deallocate(cartography) ! We do not need it anymore
    deallocate(tag_proc)
    deallocate(size_com)

#ifdef BLOCKING_SEND_PLUS
    ! -- Compute the interpolate velocity --
    ! Check if communication are done
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (neighbors(direction,proc_gap)/=D_rank(direction)) then
            call mpi_wait(rece_request(proc_gap), rece_status, ierr)
        end if
    end do
    deallocate(rece_request)
#endif

    ! ===== Compute the interpolated velocity =====
    ! -- Compute the interpolation weight and update the pointers Vp and Vm --
    pos_in_buffer = pos_in_buffer - 1
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            ! Initialisation of reccurence process
            ind = 1
            pos = floor(p_inter(ind,i1,i2))
#ifndef BLOCKING_SEND_PLUS
            p_inter(ind,i1,i2) = p_inter(ind,i1,i2)-pos
#else
            weight = p_inter(ind,i1,i2)-pos
#endif
            ! Vm = V(pos)
            proc_gap = floor(real(pos-1, WP)/mesh_sc%N_proc(direction))
            if (neighbors(direction,proc_gap) == D_rank(direction)) then
#ifndef BLOCKING_SEND_PLUS
              Vm(ind,i1,i2)%pter => V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#else
              p_inter(ind,i1,i2) = (1._WP-weight)*V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#endif
              myself(1) = .true.
            else
              pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + 1  ! XXX New version only
#ifndef BLOCKING_SEND_PLUS
              Vm(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap))
#else
              p_inter(ind,i1,i2) = (1._WP-weight)*V_buffer(pos_in_buffer(proc_gap))
#endif
              myself(1) = .false.
            end if
            ! Vp = V(pos+1)
            gap = floor(real(pos+1-1, WP)/mesh_sc%N_proc(direction))
            if (neighbors(direction,gap) == D_rank(direction)) then
#ifndef BLOCKING_SEND_PLUS
              Vp(ind,i1,i2)%pter => V_comp(pos+1-gap*mesh_sc%N_proc(direction), i1,i2)
#else
              p_inter(ind,i1,i2) = p_inter(ind,i1,i2) + weight*V_comp(pos+1-gap*mesh_sc%N_proc(direction), i1,i2)
#endif
            else
              pos_in_buffer(gap) = pos_in_buffer(gap) + 1  ! XXX New version only
#ifndef BLOCKING_SEND_PLUS
              Vp(ind,i1,i2)%pter => V_buffer(pos_in_buffer(gap))
#else
              p_inter(ind,i1,i2) = p_inter(ind,i1,i2) + weight*V_buffer(pos_in_buffer(gap))
#endif
            end if
            pos_old = pos
            proc_end = (proc_gap+1)*mesh_sc%N_proc(direction)
            myself(2) = (neighbors(direction,proc_gap+1) == D_rank(direction))


            ! XXX New version XXX
            ! Following indice: new version
            ind = 2
            if (ind<=mesh_sc%N_proc(direction)) pos = floor(p_inter(ind,i1,i2))
            do while (ind<=mesh_sc%N_proc(direction))
              !pos = floor(p_inter(ind,i1,i2))
              if(myself(1)) then
                ! -- Inside the current block, it is always the same --
                do while ((pos<proc_end).and.(ind<mesh_sc%N_proc(direction)))
                  ! Computation for current step
#ifndef BLOCKING_SEND_PLUS
                  p_inter(ind,i1,i2) = p_inter(ind,i1,i2)-pos
                  Vm(ind,i1,i2)%pter => V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
                  Vp(ind,i1,i2)%pter => V_comp(pos+1-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#else
                  !weight = p_inter(ind,i1,i2)-pos
                  !p_inter = weight*Vp + (1-weight)*Vm = weight*(Vp-Vm) + Vm
                  p_inter(ind,i1,i2) = (p_inter(ind,i1,i2)-pos)*(V_comp(pos+1-proc_gap*mesh_sc%N_proc(direction), i1,i2) &
                    & - V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)) + V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#endif
                  ! Prepare next step
                  pos_old = pos
                  ind = ind + 1
                  pos = floor(p_inter(ind,i1,i2))
                end do ! ((pos<proc_end).and.(ind<mesh_sc%N_proc(direction)))
                ! -- When we are exactly on the subdomain transition --
                do while ((pos==proc_end).and.(ind<mesh_sc%N_proc(direction)))
#ifndef BLOCKING_SEND_PLUS
                  p_inter(ind,i1,i2) = p_inter(ind,i1,i2)-pos
                  ! Vm is in the same sub-domain
                  Vm(ind,i1,i2)%pter => V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#endif
                  ! Vp is in the next one (proc_gap+1)
                  if(myself(2)) then
#ifndef BLOCKING_SEND_PLUS
                    Vp(ind,i1,i2)%pter => V_comp(pos+1-(proc_gap+1)*mesh_sc%N_proc(direction), i1,i2)
#else
                    p_inter(ind,i1,i2) = ((p_inter(ind,i1,i2)-pos)*         &
                      & (V_comp(pos+1-(proc_gap+1)*mesh_sc%N_proc(direction), i1,i2) &
                      & - V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2))    )&
                      &  + V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#endif
                  else
                    ! If pos = pos_old, we must have pos_in_buffer(proc_gap+1) += 0 (no changes)
                    ! Else pos>pos_old, we must have pos_in_buffer(proc_gap+1) += 1
                    ! We use that min(1,pos-pos_old)   = 0 if pos=pos_old, 1 else
                    pos_in_buffer(proc_gap+1) = pos_in_buffer(proc_gap+1) + min(1,pos-pos_old)
#ifndef BLOCKING_SEND_PLUS
                    Vp(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap+1))
#else
                    p_inter(ind,i1,i2) = ((p_inter(ind,i1,i2)-pos)*(V_buffer(pos_in_buffer(proc_gap+1)) &
                    & - V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2))) + V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#endif
                  end if
                  ! Prepare next step
                  pos_old = pos
                  ind = ind + 1
                  pos = floor(p_inter(ind,i1,i2))
                end do
                ! -- When we reach the end of the sub-domain OR the end of the particle line --
                if (pos>proc_end) then  ! Changement of subdomain
                  ! We have reach the next subdomain => update values
                  proc_gap = floor(real(pos-1, WP)/mesh_sc%N_proc(direction)) ! "proc_gap = proc_gap + 1" does not work if N_proc = 1 and pos-pos_old = 2.
                  myself(1) = (neighbors(direction,proc_gap) == D_rank(direction)) ! For the same reason that line jsute above, we do not use "myself(1) = myself(2)"
                  proc_end = (proc_gap+1)*mesh_sc%N_proc(direction)
                  myself(2) = (neighbors(direction,proc_gap+1) == D_rank(direction))
                  ! ... and go on the next loop !
                else ! ind == N_proc and no changement of subdomain
#ifndef BLOCKING_SEND_PLUS
                  ! Computation for current step
                  p_inter(ind,i1,i2) = p_inter(ind,i1,i2)-pos
                  ! Vm
                  Vm(ind,i1,i2)%pter => V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#endif
                  ! Vp
                  if(pos<proc_end) then
#ifndef BLOCKING_SEND_PLUS
                    Vp(ind,i1,i2)%pter => V_comp(pos+1-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#else
                    p_inter(ind,i1,i2) = (p_inter(ind,i1,i2)-pos)*(V_comp(pos+1-proc_gap*mesh_sc%N_proc(direction), i1,i2) &
                      & - V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)) + V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#endif
                  else ! pos+1 is in next subdomain: use the same algorithm than line 377-390
                    if(myself(2)) then
#ifndef BLOCKING_SEND_PLUS
                      Vp(ind,i1,i2)%pter => V_comp(pos+1-(proc_gap+1)*mesh_sc%N_proc(direction), i1,i2)
#else
                      p_inter(ind,i1,i2) = (p_inter(ind,i1,i2)-pos)*(V_comp(pos+1-(proc_gap+1)*mesh_sc%N_proc(direction), i1,i2) &
                        & - V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)) + V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#endif
                    else
                      ! If pos = pos_old, we must have pos_in_buffer(proc_gap+1) += 0 (no changes)
                      ! Else pos>pos_old, we must have pos_in_buffer(proc_gap+1) += 1
                      ! We use that min(1,pos-pos_old)   = 0 if pos=pos_old, 1 else
                      pos_in_buffer(proc_gap+1) = pos_in_buffer(proc_gap+1) + min(1,pos-pos_old)
#ifndef BLOCKING_SEND_PLUS
                      Vp(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap+1))
#else
                      p_inter(ind,i1,i2) = (p_inter(ind,i1,i2)-pos)*(V_buffer(pos_in_buffer(proc_gap+1)) &
                        & - V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)) + V_comp(pos-proc_gap*mesh_sc%N_proc(direction), i1,i2)
#endif
                    end if
                  end if
                  ! Go to the next (i1,i2) value: ind must be greater than N_proc
                  ind = ind +1
                end if
              else ! => not myself(1)
                ! -- Inside the current block, it is always the same --
                do while ((pos<proc_end).and.(ind<mesh_sc%N_proc(direction)))
                  ! Computation for current step
                  pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + pos-pos_old
#ifndef BLOCKING_SEND_PLUS
                  p_inter(ind,i1,i2) = p_inter(ind,i1,i2)-pos
                  Vm(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap)-1)
                  Vp(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap))
#else
                  p_inter(ind,i1,i2) = ((p_inter(ind,i1,i2)-pos)*(V_buffer(pos_in_buffer(proc_gap)) &
                      & - V_buffer(pos_in_buffer(proc_gap)-1))) + V_buffer(pos_in_buffer(proc_gap)-1)
#endif
                  ! Prepare next step
                  pos_old = pos
                  ind = ind + 1
                  pos = floor(p_inter(ind,i1,i2))
                end do
                ! -- When we are exactly on the subdomain transition --
                do while ((pos==proc_end).and.(ind<mesh_sc%N_proc(direction)))
                  ! If pos = pos_old, we must have  pos_in_buffer(proc_gap) += 0
                  !                             and pos_in_buffer(proc_gap+1) += 0 (no changes)
                  ! Else pos>pos_old, we must have pos_in_buffer(proc_gap) += (pos-pos_old -1)
                  !                             and pos_in_buffer(proc_gap+1) += 1
                  ! We use max(0,pos-pos_old-1) = 0 if pos=pos_old, (pos-pos_old-1) else.
                  !    and min(1,pos-pos_old)   = 0 if pos=pos_old, 1 else
                  ! Vm is in the same sub-domain
                  pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + max(0,pos-pos_old-1)
#ifndef BLOCKING_SEND_PLUS
                 p_inter(ind,i1,i2) = p_inter(ind,i1,i2)-pos
                 Vm(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap))
#endif
                  ! Vp is in the next one
                  if(myself(2)) then
#ifndef BLOCKING_SEND_PLUS
                    Vp(ind,i1,i2)%pter => V_comp(pos+1-(proc_gap+1)*mesh_sc%N_proc(direction), i1,i2)
#else
                    p_inter(ind,i1,i2) = ((p_inter(ind,i1,i2)-pos)* &
                      & (V_comp(pos+1-(proc_gap+1)*mesh_sc%N_proc(direction), i1,i2)   &
                      & - V_buffer(pos_in_buffer(proc_gap))              ) )&
                      & + V_buffer(pos_in_buffer(proc_gap))
#endif
                  else
                    pos_in_buffer(proc_gap+1) = pos_in_buffer(proc_gap+1) + min(1,pos-pos_old)
#ifndef BLOCKING_SEND_PLUS
                    Vp(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap+1))
#else
                    p_inter(ind,i1,i2) = ((p_inter(ind,i1,i2)-pos)* &
                      & (V_buffer(pos_in_buffer(proc_gap+1))      &
                      & - V_buffer(pos_in_buffer(proc_gap)) )    )&
                      & + V_buffer(pos_in_buffer(proc_gap))
#endif
                  end if
                  ! Prepare next step
                  pos_old = pos
                  ind = ind + 1
                  pos = floor(p_inter(ind,i1,i2))
                end do
                ! -- When we reach the end of the sub-domain OR the end of the particle line --
                if (pos>proc_end) then  ! Changement of subdomain
                  ! We have reach the next subdomain => update values
                  proc_gap = floor(real(pos-1, WP)/mesh_sc%N_proc(direction)) ! "proc_gap = proc_gap + 1" does not work if N_proc = 1 and pos-pos_old = 2.
                  myself(1) = (neighbors(direction,proc_gap) == D_rank(direction)) ! For the same reason that line jsute above, we do not use "myself(1) = myself(2)"
                  proc_end = (proc_gap+1)*mesh_sc%N_proc(direction)
                  myself(2) = (neighbors(direction,proc_gap+1) == D_rank(direction))
                  ! ... and go on the next loop !
                else ! ind == N_proc and no changement of subdomain
                  ! Computation for current step
#ifndef BLOCKING_SEND_PLUS
                 p_inter(ind,i1,i2) = p_inter(ind,i1,i2)-pos
#endif
                  if (pos<proc_end) then
                    pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + pos-pos_old
#ifndef BLOCKING_SEND_PLUS
                    Vm(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap)-1)
                    Vp(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap))
#else
                    p_inter(ind,i1,i2) = ((p_inter(ind,i1,i2)-pos)*(V_buffer(pos_in_buffer(proc_gap)) &
                        & - V_buffer(pos_in_buffer(proc_gap)-1))) + V_buffer(pos_in_buffer(proc_gap)-1)
#endif
                  else ! pos=proc_end : same as in line 440 to 462
                    pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + max(0,pos-pos_old-1)
#ifndef BLOCKING_SEND_PLUS
                    Vm(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap))
#endif
                    ! Vp is in the next one
                    if(myself(2)) then
#ifndef BLOCKING_SEND_PLUS
                      Vp(ind,i1,i2)%pter => V_comp(pos+1-(proc_gap+1)*mesh_sc%N_proc(direction), i1,i2)
#else
                      p_inter(ind,i1,i2) = ((p_inter(ind,i1,i2)-pos)*(V_comp(pos+1-(proc_gap+1)*mesh_sc%N_proc(direction), i1,i2) &
                        & - V_buffer(pos_in_buffer(proc_gap)))) + V_buffer(pos_in_buffer(proc_gap))
#endif
                    else
                      pos_in_buffer(proc_gap+1) = pos_in_buffer(proc_gap+1) + min(1,pos-pos_old)
#ifndef BLOCKING_SEND_PLUS
                      Vp(ind,i1,i2)%pter => V_buffer(pos_in_buffer(proc_gap+1))
#else
                      p_inter(ind,i1,i2) = ((p_inter(ind,i1,i2)-pos)*(V_buffer(pos_in_buffer(proc_gap+1)) &
                        & - V_buffer(pos_in_buffer(proc_gap)))) + V_buffer(pos_in_buffer(proc_gap))
#endif
                    end if
                  end if
                  ! Go to the next (i1,i2) value: ind must be greater than N_proc
                  ind = ind +1
                end if  ! pos>proc_end
              end if ! myself(1)
            end do ! (ind<mesh_sc%N_proc(direction)

        end do ! loop on first coordinate (i1) of a line inside the block of line
    end do ! loop on second coordinate (i2) of a line inside the block of line

    deallocate(pos_in_buffer)   ! We do not need it anymore

#ifndef BLOCKING_SEND_PLUS
    ! -- Compute the interpolate velocity --
    ! Check if communication are done
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (neighbors(direction,proc_gap)/=D_rank(direction)) then
            call mpi_wait(rece_request(proc_gap), rece_status, ierr)
        end if
    end do
    deallocate(rece_request)
#endif

    ! Then compute the field
#ifndef BLOCKING_SEND_PLUS
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            do ind = 1, mesh_sc%N_proc(direction)
                p_inter(ind,i1,i2) = p_inter(ind,i1,i2)*Vp(ind,i1,i2)%pter + (1.-p_inter(ind,i1,i2))*Vm(ind,i1,i2)%pter
            end do
        end do
    end do
#endif


    ! ===== Free memory =====
    ! -- Pointeur --
#ifndef BLOCKING_SEND_PLUS
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            do ind = 1, mesh_sc%N_proc(direction)
                nullify(Vp(ind,i1,i2)%pter)
                nullify(Vm(ind,i1,i2)%pter)
            end do
        end do
    end do
#endif
#ifndef BLOCKING_SEND
    ! -- Mpi internal buffer for non blocking communication --
    do proc_gap = send_gap(1,1), send_gap(1,2)
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            call MPI_WAIT(s_request(proc_gap),rece_status,ierr)
        end if
    end do
    deallocate(s_request)
#endif
    ! -- Deallocate dynamic array --
    deallocate(V_buffer)

end subroutine AC_interpol_lin


!> Determine the set of processes wich will send me information during the velocity interpolation and compute
!! for each of these processes the range of wanted data.
!!    @param[in]    direction       = current direction (1 = along X, 2 = along Y, 3 = along Z)
!!    @param[in]    gp_s            = size of a group (ie number of line it gathers along the two other directions)
!!    @param[in]    ind_group       = coordinate of the current group of lines
!!    @param[in]    ind_group       = coordinate of the current group of lines
!!    @param[out]   send_gap        = gap between my coordinate and the processes of minimal coordinate which will send information to me
!!    @param[in]    rece_gap        = gap between my coordinate and the processes of maximal coordinate which will receive information from me
!!    @param[in]    rece_gap_abs    = min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
!!    @param[out]   cartography     = cartography(proc_gap) contains the set of the lines indice in the block for wich the
!!                                    current processus requiers data from proc_gap and for each of these lines the range
!!                                    of mesh points from where it requiers the velocity values.
!! @details
!!    Work on a group of line of size gs(1) x gs(2))
!!    Obtain the list of processus wich need a part of my local velocity field
!!    to interpolate the velocity used in the RK2 scheme to advect its particles.
!!    In the same time, it computes for each processus from which I need a part
!!    of the velocity field, the range of mesh point where I want data and store it
!!    by using some sparse matrix technics (see cartography defined in the
!!    algorithm documentation)
subroutine AC_interpol_determine_communication(direction, ind_group, gs, send_gap,  &
    & rece_gap, rece_gap_abs, cartography)
! XXX Work only for periodic condition.

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Ouput
    integer, intent(in)                                 :: direction
    integer, dimension(2), intent(in)                   :: ind_group
    integer, dimension(2), intent(in)                   :: gs
    integer, dimension(gs(1), gs(2), 2), intent(in)     :: rece_gap
    integer, dimension(2), intent(in)                   :: rece_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer, dimension(2, 2), intent(out)               :: send_gap
    integer, dimension(2+gs(2)*(2+3*gs(1)), &
        & rece_gap_abs(1):rece_gap_abs(2)), intent(out) :: cartography
    ! Others
    integer                             :: proc_gap         ! gap between a processus coordinate (along the current
                                                            ! direction) into the mpi-topology and my coordinate
    integer, dimension(gs(1), gs(2))    :: rece_gapP        ! gap between the coordinate of the previous processus (in the current direction)
                                                            ! and the processes of maximal coordinate which will receive information from it
    integer, dimension(gs(1), gs(2))    :: rece_gapN        ! same as above but for the next processus
    integer                             :: send_request_gh  ! mpi status of noindicelocking send
    integer                             :: send_request_gh2 ! mpi status of noindicelocking send
    integer                             :: ierr             ! mpi error code
    integer, dimension(2)               :: tag_table        ! some mpi message tag
    logical, dimension(:,:), allocatable:: test_request     ! for mpi non blocking communication
    integer, dimension(:,:), allocatable:: send_request     ! for mpi non blocking send
    integer                             :: ind1, ind2       ! indice of the current line inside the group
    integer,dimension(2)                :: rece_buffer      ! buffer for reception of rece_max
    integer, dimension(:,:), allocatable:: first, last      ! Storage processus to which I will be the first (or the last) to receive
    integer                             :: min_size         ! begin indice in first and last to stock indice along first dimension of the group line
    integer                             :: gp_size          ! group size
    logical                             :: begin_interval   ! ware we in the start of an interval ?
    logical                             :: not_myself       ! Is the target processus myself ?
    integer, dimension(MPI_STATUS_SIZE) :: statut

    send_gap(1,1) = 3*mesh_sc%N(direction)
    send_gap(1,2) = -3*mesh_sc%N(direction)
    send_gap(2,:) = 0
    gp_size = gs(1)*gs(2)

    ! ===== Communicate with my neigbors -> obtain ghost ! ====
    ! Inform that about processus from which I need information
    tag_table = compute_tag(ind_group, tag_obtrec_ghost_NP, direction)
    call mpi_ISsend(rece_gap(1,1,1), gp_size, MPI_INTEGER, neighbors(direction,-1), tag_table(1), &
        & D_comm(direction), send_request_gh, ierr)
    call mpi_ISsend(rece_gap(1,1,2), gp_size, MPI_INTEGER, neighbors(direction,1), tag_table(2), &
        & D_comm(direction), send_request_gh2, ierr)
    ! Receive the same message form my neighbors
    call mpi_recv(rece_gapN(1,1), gp_size, MPI_INTEGER, neighbors(direction,1), tag_table(1), D_comm(direction), statut, ierr)
    call mpi_recv(rece_gapP(1,1), gp_size, MPI_INTEGER, neighbors(direction,-1), tag_table(2), D_comm(direction), statut, ierr)

    ! ===== Compute if I am first or last and determine the carography =====
    min_size = 2 + gs(2)
    ! Initialize first and last to determine if I am the the first or the last processes (considering the current direction)
        ! to require information from this processus
    allocate(first(2,rece_gap_abs(1):rece_gap_abs(2)))
    first(2,:) = 0  ! number of lines for which I am the first
    allocate(last(2,rece_gap_abs(1):rece_gap_abs(2)))
    last(2,:) = 0   ! number of lines for which I am the last
    ! Initialize cartography
    cartography(1,:) = 0            ! number of velocity values to receive
    cartography(2,:) = min_size     ! number of element to send when sending cartography
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        first(1,proc_gap) = -proc_gap
        last(1,proc_gap) = -proc_gap
        not_myself = (neighbors(direction,proc_gap) /= D_rank(direction)) ! Is the target processus myself ?
        do ind2 = 1, gs(2)
            cartography(2+ind2,proc_gap) = 0    ! 2 x number of interval of concern line into the column i2
            begin_interval = .true.
            do ind1 = 1, gs(1)
                ! Does proc_gap belongs to [rece_gap(i1,i2,1);rece_gap(i1,i2,2)]?
                if((proc_gap>=rece_gap(ind1,ind2,1)).and.(proc_gap<=rece_gap(ind1,ind2,2))) then
                    ! Compute if I am the first.
                    if (proc_gap>rece_gapP(ind1,ind2)-1) then
                        first(2,proc_gap) =  first(2,proc_gap)+1
                    end if
                    ! Compute if I am the last.
                    if (proc_gap<rece_gapN(ind1,ind2)+1) then
                        last(2,proc_gap) =  last(2,proc_gap)+1
                    end if
                    ! Update cartography // Not need I target processus is myself
                    if (not_myself) then
                        if (begin_interval) then
                            cartography(2+ind2,proc_gap) =  cartography(2+ind2,proc_gap)+2
                            cartography(cartography(2,proc_gap)+1,proc_gap) = ind1
                            cartography(2,proc_gap) = cartography(2,proc_gap) + 2
                            cartography(cartography(2,proc_gap),proc_gap) = ind1
                            begin_interval = .false.
                        else
                            cartography(cartography(2,proc_gap),proc_gap) = ind1
                        end if
                    end if
                else
                    begin_interval = .true.
                end if
            end do
        end do
    end do

    ! ===== Free Isend buffer from first communication =====
    call MPI_WAIT(send_request_gh,statut,ierr)
    call MPI_WAIT(send_request_gh2,statut,ierr)

    ! ===== Send information about first and last  =====
    tag_table = compute_tag(ind_group, tag_obtrec_NP, direction)
    allocate(send_request(rece_gap_abs(1):rece_gap_abs(2),2))
    allocate(test_request(rece_gap_abs(1):rece_gap_abs(2),2))
    test_request = .false.
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        ! I am the first ?
        if (first(2,proc_gap)>0) then
            if(neighbors(direction,proc_gap)/= D_rank(direction)) then
                call mpi_ISsend(first(1,proc_gap), 2, MPI_INTEGER, neighbors(direction,proc_gap),&
                        & tag_table(1), D_comm(direction), send_request(proc_gap,1), ierr)
                test_request(proc_gap,1) = .true.
            else
                send_gap(1,1) = min(send_gap(1,1), -proc_gap)
                send_gap(2,1) = send_gap(2,1) + first(2,proc_gap)
            end if
        end if
        ! I am the last ?
        if (last(2,proc_gap)>0) then
            if(neighbors(direction,proc_gap)/= D_rank(direction)) then
                call mpi_ISsend(last(1,proc_gap), 2, MPI_INTEGER, neighbors(direction,proc_gap),&
                        &  tag_table(2), D_comm(direction), send_request(proc_gap,2), ierr)
                test_request(proc_gap,2) = .true.
            else
                send_gap(1,2) = max(send_gap(1,2), -proc_gap)
                send_gap(2,2) = send_gap(2,2) + last(2,proc_gap)
            end if
        end if
    end do



    ! ===== Receive information form the first and the last processus which need a part of my local velocity field =====
    do while(send_gap(2,1) < gp_size)
        call mpi_recv(rece_buffer(1), 2, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(1), D_comm(direction), statut, ierr)
        send_gap(1,1) = min(send_gap(1,1), rece_buffer(1))
        send_gap(2,1) = send_gap(2,1) + rece_buffer(2)
    end do
    do while(send_gap(2,2) < gp_size)
        call mpi_recv(rece_buffer(1), 2, MPI_INTEGER, MPI_ANY_SOURCE, tag_table(2), D_comm(direction), statut, ierr)
        send_gap(1,2) = max(send_gap(1,2), rece_buffer(1))
        send_gap(2,2) = send_gap(2,2) + rece_buffer(2)
    end do

    ! ===== Free Isend buffer =====
    !call MPI_WAIT(send_request_gh,statut,ierr)
    !call MPI_WAIT(send_request_gh2,statut,ierr)
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (test_request(proc_gap,1).eqv. .true.) call MPI_WAIT(send_request(proc_gap,1),statut,ierr)
        if (test_request(proc_gap,2)) call MPI_WAIT(send_request(proc_gap,2),statut,ierr)
    end do
    deallocate(send_request)
    deallocate(test_request)

    ! ===== Deallocate array =====
    deallocate(first)
    deallocate(last)

end subroutine AC_interpol_determine_communication


!> Interpolate the velocity field used in a RK2 scheme for particle advection -
!! version for direction with no domain subdivision ands thus no required
!! communications
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        gs          = size of a group (ie number of line it gathers along the two other directions)
!!    @param[in]        V_comp      = velocity to interpolate
!!    @param[in,out]    p_V         = particle position in input and particle velocity (along the current direction) as output
!! @details
!!    A RK2 scheme is used to advect the particles : the midlle point scheme. An
!!    intermediary position "p_pos_bis(i) = p_pos(i) + V(i)*dt/2" is computed and then
!!    the numerical velocity of each particles is computed as the interpolation of V  in
!!    this point. This field is used to advect the particles at the seconde order in time :
!!    p_pos(t+dt, i) = p_pos(i) + p_V(i).
!!    Variant for cases with no required communication.
subroutine AC_interpol_lin_no_com(direction, gs, V_comp, p_V)

    ! This code involve a recopy of p_V. It is possible to directly use the 3D velocity field but it will also limit the meroy access.

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Ouput
    integer, intent(in)                             :: direction    ! current direction
    integer, dimension(2),intent(in)                :: gs           ! groupe size
    real(WP), dimension(:,:,:), intent(in)          :: V_comp
    real(WP), dimension(:,:,:), intent(inout)       :: p_V
    ! Others, local
    integer                                             :: ind          ! indices
    integer                                             :: i1, i2       ! indices in the lines group
    integer                                             :: pos          ! indices of the mesh point wich preceed the particle position


    ! ===== Compute the interpolated velocity =====
    ! -- Compute the interpolation weight and update the velocity directly in p_V --
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            do ind = 1, mesh_sc%N(direction)

            pos = floor(p_V(ind,i1,i2))
            p_V(ind,i1,i2) = V_comp(modulo(pos-1,mesh_sc%N(direction))+1,i1,i2) + (p_V(ind,i1,i2)-pos)* &
                & (V_comp(modulo(pos,mesh_sc%N(direction))+1,i1,i2)-V_comp(modulo(pos-1,mesh_sc%N(direction))+1,i1,i2))

            end do ! loop on particle indice (ind)
        end do ! loop on first coordinate (i1) of a line inside the block of line
    end do ! loop on second coordinate (i2) of a line inside the block of line

end subroutine AC_interpol_lin_no_com


!> Interpolate the velocity field from coarse grid at particles positions
!! version for a group of (more of one) line
!!    @param[in]        dt          = time step
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        gs          = size of a group (ie number of line it gathers along the two other directions)
!!    @param[in]        ind_group   = indices of the current work item
!!    @param[in]        id1         = first coordinate of the current work item related to the total local mesh
!!    @param[in]        id2         = first coordinate of the current work item related to the total local mesh
!!    @param[in]        V_coarse    = velocity to interpolate
!!    @param[in,out]    p_V         = particle position in input and particle velocity (along the current direction) as output
!! @details
!!    A RK2 scheme is used to advect the particles : the midlle point scheme. An
!!    intermediary position "p_pos_bis(i) = p_pos(i) + V(i)*dt/2" is computed and then
!!    the numerical velocity of each particles is computed as the interpolation of V  in
!!    this point. This field is used to advect the particles at the seconde order in time :
!!    p_pos(t+dt, i) = p_pos(i) + p_V(i).
!!    The group line indice is used to ensure using unicity of each mpi message tag.
!!    The interpolation is done for a group of lines, allowing to mutualise
!!    communications. Considering a group of Na X Nb lines, communication performed
!!    by this algorithm are around (Na x Nb) bigger than the alogorithm wich
!!    works on a single line but also around (Na x Nb) less frequent.
subroutine AC_interpol_plus(direction, gs, ind_group, id1, id2, V_coarse, p_V)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.
    !use Interpolation_velo
    use interpolation_velo, only : get_weight, stencil_g, stencil_d, stencil_size

    implicit none

    ! Input/Ouput
    integer                   , intent(in)          :: direction    ! current direction
    integer, dimension(2)     , intent(in)          :: gs           ! groupe size
    integer, dimension(2)     , intent(in)          :: ind_group
    integer                   , intent(in)          :: id1, id2
    real(WP), dimension(:,:,:), intent(inout)       :: p_V
    real(WP), dimension(:,:,:), intent(in)          :: V_coarse     ! velocity on coarse grid
    ! Local
    integer                                     :: idir1, idir2 ! = (id1, id2) -1 as array indice starts from 1.
    real(WP), dimension(stencil_size)           :: weight       ! interpolation weight storage
    real(WP), dimension(:), allocatable         :: V_buffer     ! Velocity buffer for postion outside of the local subdomain
    integer, dimension(:), allocatable          :: pos_in_buffer! buffer size
    integer , dimension(gs(1), gs(2))           :: rece_ind_min ! minimal indice of mesh involved in remeshing particles (of my local subdomains)
    integer , dimension(gs(1), gs(2))           :: rece_ind_max ! maximal indice of mesh involved in remeshing particles (of my local subdomains)
    integer                                     :: ind, ind_com, V_ind ! indices
    integer                                     :: i_limit, i, ind_gap
    integer                                     :: i1, i2       ! indices in the lines group
    integer                                     :: pos, pos_old ! indices of the mesh point wich preceed the particle position
    integer                                     :: proc_gap, gap! distance between my (mpi) coordonate and coordinate of the
                                                                ! processus associated to a given position
    integer                                     :: proc_end     ! final indice of processus associate to current pos
    logical, dimension(3)                       :: myself
    integer, dimension(:), allocatable          :: send_carto   ! cartogrpahy of what I have to send
    integer                                     :: ind_1Dtable  ! indice of my current position inside a one-dimensionnal table
    integer                                     :: ind_for_i1   ! where to read the first coordinate (i1) of the current line inside the cartography ?
    real(WP), dimension(:), allocatable         :: send_buffer  ! to store what I have to send (on a contiguous way)
    integer, dimension(gs(1),gs(2),2)           :: rece_gap     ! distance between me and processus wich send me information
    integer, dimension(2 , 2)                   :: send_gap     ! distance between me and processus to wich I send information
    integer, dimension(2)                       :: rece_gap_abs ! min (resp max) value of rece_gap(:,:,i) with i=1 (resp 2)
    integer                                     :: com_size     ! size of message send/receive
    integer, dimension(:), allocatable          :: size_com     ! size of message send/receive
    integer                                     :: min_size     ! minimal size of cartography(:,proc_gap)
    integer                                     :: max_size     ! maximal size of cartography(:,proc_gap)
    integer                                     :: tag          ! mpi message tag
    integer, dimension(:), allocatable          :: tag_proc     ! mpi message tag
    integer                                     :: ierr         ! mpi error code
    integer, dimension(:), allocatable          :: s_request_bis! mpi communication request (handle) of nonblocking send
    integer, dimension(:), allocatable          :: rece_request ! mpi communication request (handle) of nonblocking receive
    integer, dimension(MPI_STATUS_SIZE)         :: rece_status  ! mpi status (for mpi_wait)
    integer, dimension(:,:), allocatable        :: cartography  ! cartography(proc_gap) contains the set of the lines indice in the block for wich the
                                                                ! current processus requiers data from proc_gap and for each of these lines the range
                                                                ! of mesh points from where it requiers the velocity values.

    ! -- Initialisation --
    idir1 = id1 - 1
    idir2 = id2 - 1
    ! Compute range of the set of point where I need the velocity value
    rece_ind_min = floor(p_V(1,:,:)) - stencil_g
    rece_ind_max = floor(p_V(mesh_sc%N_proc(direction),:,:)) + stencil_d

    ! ===== Exchange velocity field if needed =====
    ! It uses non blocking message to do the computations during the communication process
    ! -- What have I to communicate ? --
    rece_gap(:,:,1) = floor(real(rece_ind_min-1, WP)/mesh_V%N_proc(direction))
    rece_gap(:,:,2) = floor(real(rece_ind_max-1, WP)/mesh_V%N_proc(direction))
    rece_gap_abs(1) = minval(rece_gap(:,:,1))
    rece_gap_abs(2) = maxval(rece_gap(:,:,2))
    max_size = 2 + gs(2)*(2+3*gs(1))
    allocate(cartography(max_size,rece_gap_abs(1):rece_gap_abs(2)))
    call AC_interpol_determine_communication(direction, ind_group, gs, send_gap,  &
    & rece_gap, rece_gap_abs, cartography)

    ! -- Send messages about what I want --
    allocate(s_request_bis(rece_gap_abs(1):rece_gap_abs(2)))
    allocate(size_com(rece_gap_abs(1):rece_gap_abs(2)))
    allocate(tag_proc(rece_gap_abs(1):rece_gap_abs(2)))
    min_size = 2 + gs(2)
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            cartography(1,proc_gap) = 0
            ! Use the cartography to know which lines are concerned
            size_com(proc_gap) = cartography(2,proc_gap)
            ! Range I want - store into the cartography
            gap = proc_gap*mesh_V%N_proc(direction)
            ! Position in cartography(:,proc_gap) of the current i1 indice
            ind_for_i1 = min_size
            do i2 = 1, gs(2)
                do ind = ind_for_i1+1, ind_for_i1 + cartography(2+i2,proc_gap), 2
                    do i1 = cartography(ind,proc_gap), cartography(ind+1,proc_gap)
                        ! Interval start from:
                        cartography(size_com(proc_gap)+1,proc_gap) = max(rece_ind_min(i1,i2), gap+1) ! fortran => indice start from 0
                        ! and ends at:
                        cartography(size_com(proc_gap)+2,proc_gap) = min(rece_ind_max(i1,i2), gap+mesh_V%N_proc(direction))
                        ! update number of element to receive
                        cartography(1,proc_gap) = cartography(1,proc_gap) &
                                    & + cartography(size_com(proc_gap)+2,proc_gap) &
                                    & - cartography(size_com(proc_gap)+1,proc_gap) + 1
                        size_com(proc_gap) = size_com(proc_gap)+2
                    end do
                end do
                ind_for_i1 = ind_for_i1 + cartography(2+i2,proc_gap)
            end do
            ! Tag = concatenation of (rank+1), ind_group(1), ind_group(2), direction et unique Id.
            tag_proc(proc_gap) = compute_tag(ind_group, tag_velo_range, direction, proc_gap)
            ! Send message
#ifdef PART_DEBUG
            if(size_com(proc_gap)>max_size) then
                print*, 'rank = ', cart_rank, ' -- bug sur taille cartography a envoyer'
                print*, 'taille carto = ', com_size, ' plus grand que la taille théorique ', &
                    & max_size, ' et carto = ', cartography(:,proc_gap)
            end if
#endif
            call mpi_ISsend(cartography(1,proc_gap), size_com(proc_gap), MPI_INTEGER,   &
                & neighbors(direction,proc_gap), tag_proc(proc_gap), D_comm(direction), &
                & s_request_bis(proc_gap),ierr)
        end if
    end do


    ! -- Non blocking reception of the velocity field --
    ! Allocate the pos_in_buffer to compute V_buffer size and to be able to
    ! allocate it.
    allocate(pos_in_buffer(rece_gap_abs(1):rece_gap_abs(2)))
    pos_in_buffer(rece_gap_abs(1)) = 1
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)-1
        pos_in_buffer(proc_gap+1)= pos_in_buffer(proc_gap) + cartography(1,proc_gap)
    end do
    allocate(V_buffer(pos_in_buffer(rece_gap_abs(2)) &
                & + cartography(1,rece_gap_abs(2))))
    V_buffer = 0
    allocate(rece_request(rece_gap_abs(1):rece_gap_abs(2)))
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            ! IIa - Compute reception tag
            tag = compute_tag(ind_group, tag_velo_V, direction, -proc_gap)
            ! IIb - Receive message
            call mpi_Irecv(V_buffer(pos_in_buffer(proc_gap)), cartography(1,proc_gap), MPI_REAL_WP, &
                    & neighbors(direction,proc_gap), tag, D_comm(direction), rece_request(proc_gap), ierr)
        end if
    end do

    ! -- Send the velocity field to processus which need it --
    allocate(send_carto(max_size))
    do proc_gap = send_gap(1,1), send_gap(1,2)
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            ! I - Receive messages about what I have to send
            ! Ia - Compute reception tag = concatenation of (rank+1), ind_group(1), ind_group(2), direction et unique Id.
            tag = compute_tag(ind_group, tag_velo_range, direction, -proc_gap)
            ! Ib - Receive the message
            call mpi_recv(send_carto(1), max_size, MPI_INTEGER, neighbors(direction,proc_gap), &
              & tag, D_comm(direction), rece_status, ierr)
            ! II - Send it
            ! IIa - Create send buffer
            allocate(send_buffer(send_carto(1)))
            gap = proc_gap*mesh_V%N_proc(direction)
            com_size = 0
            ind_1Dtable = send_carto(2)
            ! Position in cartography(:,proc_gap) of the current i1 indice
            ind_for_i1 = min_size
            do i2 = 1, gs(2)
                do ind = ind_for_i1+1, ind_for_i1 + send_carto(2+i2), 2
                    do i1 = send_carto(ind), send_carto(ind+1)
                        do ind_com = send_carto(ind_1Dtable+1)+gap, send_carto(ind_1Dtable+2)+gap ! indice inside the current line
                            com_size = com_size + 1
                            send_buffer(com_size) = V_coarse(ind_com, i1+idir1,i2+idir2)
                        end do
                        ind_1Dtable = ind_1Dtable + 2
                    end do
                end do
                ind_for_i1 = ind_for_i1 + send_carto(2+i2)
            end do
            ! IIa_bis - check correctness
#ifdef PART_DEBUG
            if(com_size/=send_carto(1)) then
                print*, 'rank = ', cart_rank, ' -- bug sur taille champ de vitesse a envoyer'
                print*, 'taille carto = ', com_size, ' plus grand recu ', &
                    & send_carto(1), ' et carto = ', send_carto(:)
            end if
#endif
            ! IIb - Compute send tag
            tag = compute_tag(ind_group, tag_velo_V, direction, proc_gap)
            ! IIc - Send message
            call mpi_Send(send_buffer(1), com_size, MPI_REAL_WP,  &
                    & neighbors(direction,proc_gap), tag, D_comm(direction),&
                    & ierr)
                    !& ierr)
            deallocate(send_buffer)
        end if
    end do
    deallocate(send_carto)

    !-- Free som ISsend buffer and some array --
! XXX Todo : préférer un call MPI_WAIT_ALL couplé avec une init de s_request_bis
! sur MPI_REQUEST_NULL et enlever la boucle ET le if.
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (neighbors(direction,proc_gap) /= D_rank(direction)) then
            call MPI_WAIT(s_request_bis(proc_gap),rece_status,ierr)
        end if
    end do
    deallocate(s_request_bis)
    deallocate(cartography) ! We do not need it anymore
    deallocate(tag_proc)
    deallocate(size_com)

    ! Check if communication are done before starting the interpolation
    do proc_gap = rece_gap_abs(1), rece_gap_abs(2)
        if (neighbors(direction,proc_gap)/=D_rank(direction)) then
            call mpi_wait(rece_request(proc_gap), rece_status, ierr)
        end if
    end do
    deallocate(rece_request)

!print*, '#### rank = ', cart_rank, ' / V_buff = ', V_buffer


  ! ===== Compute the interpolated velocity =====
  pos_in_buffer = pos_in_buffer - 1
  do i2 = 1, gs(2)
    do i1 = 1, gs(1)
      ind = 1
      pos = floor(p_V(ind,i1,i2))-stencil_g
      pos_old = pos-1
      proc_gap = floor(dble(pos-1)/mesh_V%N_proc(direction))
      myself(1) =(D_rank(direction)== neighbors(direction,proc_gap))
      myself(2) = (D_rank(direction) == neighbors(direction,proc_gap+1))
      ind_gap = proc_gap*mesh_V%N_proc(direction)
      proc_end=(proc_gap+1)*mesh_V%N_proc(direction)
      do while (ind <= mesh_sc%N_proc(direction))
        if (myself(1)) then
          ! Case 1: If all stencil points belong to the local subdomain associate to current MPI process:
          do while((pos+stencil_size-1<=proc_end).and.(ind<=mesh_sc%N_proc(direction)))
            call get_weight(p_V(ind,i1,i2)-(pos+stencil_g), weight)
            V_ind = pos - ind_gap
            p_V(ind,i1,i2) = sum(weight*V_coarse(V_ind:V_ind+stencil_size-1,i1+idir1,i2+idir2))
            ! Update for next particle:
            ind = ind + 1
            pos_old = pos
            pos = floor(p_V(ind,i1,i2))-stencil_g
          end do ! case 1: while((pos+stencil_size<=proc_end).and.(V_ind<=mesh_sc%N_proc(direction)))
          ! Case 2: Else if the stencil intersect two local subdomain
          do while((pos<=proc_end).and.(ind <= mesh_sc%N_proc(direction)))
            call get_weight(p_V(ind,i1,i2)-(pos+stencil_g), weight)
            V_ind = pos - ind_gap
            i_limit = mesh_V%N_proc(direction) - V_ind + 1
            p_V(ind,i1, i2) = weight(1)*V_coarse(V_ind,i1+idir1,i2+idir2)
            do i = 2, i_limit
              p_V(ind,i1,i2) = p_V(ind,i1,i2) + weight(i)*V_coarse(i+V_ind-1,i1+idir1,i2+idir2)
            end do
            if(myself(2)) then
              do i = i_limit+1, stencil_size
                p_V(ind,i1,i2) = p_V(ind,i1,i2) + weight(i)*V_coarse(i-i_limit,i1+idir1,i2+idir2)
              end do
            else ! not(myself(2))
              ! Start to read in buffer at (pos_in_buffer(proc_gap+1)+1) and do
              ! not update pos_in_buffer until pos does not change of subdomain.
              do i = i_limit+1, stencil_size
                p_V(ind,i1,i2) = p_V(ind,i1,i2) + weight(i)*V_buffer(pos_in_buffer(proc_gap+1)+i-i_limit)
              end do
            end if
            ! Si non(stencil_size < N_proc(direction) +1):
            !   calculer i_limit2 = min(stencil_size, proc_end + N_proc - pos)
            !   arrêter la boucle précédente à "i_limit2"
            !   ajouter une boucle de i_limit2+1 à stencil_size
            !   Dans cette boucle utiliser proc_gap+2 et myself(3)
            ! Et ainsi de suite ...
            ! Update for next particle:
            ind = ind + 1
            pos_old = pos
            pos = floor(p_V(ind,i1,i2))-stencil_g
          end do ! case 2: ((pos<=proc_end).and.(ind <= mesh_sc%N_proc(direction)))
        else ! not(myself(1))
          ! Case 1: If all stencil points belong to the local subdomain associate to current MPI process:
          do while((pos+stencil_size-1<=proc_end).and.(ind<=mesh_sc%N_proc(direction)))
            call get_weight(p_V(ind,i1,i2)-(pos+stencil_g), weight)
            pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + pos - pos_old
            p_V(ind,i1,i2) = sum(weight*V_buffer(pos_in_buffer(proc_gap):pos_in_buffer(proc_gap)+stencil_size-1))
            ! Update for next particle:
            ind = ind + 1
            pos_old = pos
            pos = floor(p_V(ind,i1,i2))-stencil_g
          end do ! case 1: while((pos+stencil_size<=proc_end).and.(V_ind<=mesh_sc%N_proc(direction)))
          ! Case 2: Else if the stencil intersect two local subdomain
          do while((pos<=proc_end).and.(ind <= mesh_sc%N_proc(direction)) &
                                 &.and.(ind<=mesh_sc%N_proc(direction)))
            call get_weight(p_V(ind,i1,i2)-(pos+stencil_g), weight)
            i_limit = mesh_V%N_proc(direction) - (pos-ind_gap) + 1
            pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + pos - pos_old
            p_V(ind,i1,i2) = weight(1)*V_buffer(pos_in_buffer(proc_gap))
            do i = 2, i_limit
              p_V(ind,i1,i2) = p_V(ind,i1,i2) + weight(i)*V_buffer(pos_in_buffer(proc_gap)+i-1)
            end do
            if(myself(2)) then
              do i = i_limit+1, stencil_size
                p_V(ind,i1,i2) = p_V(ind,i1,i2) + weight(i)*V_coarse(i-i_limit,i1+idir1,i2+idir2)
              end do
            else ! not(myself(2))
              do i = i_limit+1, stencil_size
                p_V(ind,i1,i2) = p_V(ind,i1,i2) + weight(i)*V_buffer(pos_in_buffer(proc_gap+1)+i-i_limit)
              end do
            end if
            ! Update for next particle:
            ind = ind + 1
            pos_old = pos
            pos = floor(p_V(ind,i1,i2))-stencil_g
          end do ! case 2: ((pos<=proc_end).and.(ind <= mesh_sc%N_proc(direction)))
        end if
        ! Case 3 and 4 can be gathered, either myself is true or not.
        ! Case 3: Pos belong to the next subdomain
        if(ind<=mesh_sc%N_proc(direction)) then !Changement de proc
          proc_gap = proc_gap+1
          myself(1) = myself(2)
          ind_gap = proc_end
          proc_end = proc_end + mesh_V%N_proc(direction)
          myself(2) = (D_rank(direction) ==neighbors(direction, proc_gap+1))
        else
        ! Case 4: End of the line. Update pos_in buffer for next line.
          ! pos_in_buffer must be update to the maximal indices already used.
          if (pos_old+stencil_size<=proc_end) then
            if (.not.(myself(1))) pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + stencil_size - 1
          else
            i_limit = mesh_V%N_proc(direction) - (pos_old-ind_gap) + 1
            !i_limit = mesh_V%N_proc(direction) - (pos-ind_gap)
            if (.not.(myself(1))) pos_in_buffer(proc_gap) = pos_in_buffer(proc_gap) + i_limit - 1
            if (.not.(myself(2))) pos_in_buffer(proc_gap+1) = pos_in_buffer(proc_gap+1) + (stencil_size - i_limit)
          end if
        end if ! if case 3
      end do  ! while (ind<mesh_sc%N_proc)
    end do    ! i1 = 1, gs(1)
  end do      ! i2 = 1, gs(2)

  deallocate(pos_in_buffer)   ! We do not need it anymore

  ! ===== Free memory =====
  ! -- Deallocate dynamic array --
  deallocate(V_buffer)

end subroutine AC_interpol_plus


!> Interpolate the velocity field used in a RK2 scheme for particle advection -
!! version for direction with no domain subdivision ands thus no required
!! communications. Work with any interpolation formula.
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        gs          = size of a group (ie number of line it gathers along the two other directions)
!!    @param[in]        V_comp      = velocity to interpolate
!!    @param[in,out]    p_V         = particle position in input and particle velocity (along the current direction) as output
!! @details
!!    A RK2 scheme is used to advect the particles : the midlle point scheme. An
!!    intermediary position "p_pos_bis(i) = p_pos(i) + V(i)*dt/2" is computed and then
!!    the numerical velocity of each particles is computed as the interpolation of V  in
!!    this point. This field is used to advect the particles at the seconde order in time :
!!    p_pos(t+dt, i) = p_pos(i) + p_V(i).
!!    Variant for cases with no required communication.
subroutine AC_interpol_plus_no_com(direction, gs, id1, id2, V_coarse, p_V)

    ! This code involve a recopy of p_V. It is possible to directly use the 3D velocity field but it will also limit the meroy access.

    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.
    use interpolation_velo, only : get_weight, stencil_g, stencil_size

    ! Input/Ouput
    integer, intent(in)                         :: direction    ! current direction
    integer, dimension(2),intent(in)            :: gs           ! groupe size
    integer                   , intent(in)      :: id1, id2
    real(WP), dimension(:,:,:), intent(in)      :: V_coarse
    real(WP), dimension(:,:,:), intent(inout)   :: p_V
    ! Others, local
    integer                                     :: idir1, idir2 ! = (id1, id2) -1 as array indice starts from 1.
    real(WP), dimension(stencil_size)           :: weight       ! interpolation weight storage
    integer                                     :: ind, i_st    ! indices
    integer                                     :: i1, i2       ! indices in the lines group
    integer                                     :: pos          ! indices of the mesh point wich preceed the particle position

    idir1 = id1 - 1
    idir2 = id2 - 1

    ! ===== Compute the interpolated velocity =====
    ! -- Compute the interpolation weight and update the velocity directly in p_V --
    do i2 = 1, gs(2)
        do i1 = 1, gs(1)
            do ind = 1, mesh_sc%N(direction)
              pos = floor(p_V(ind,i1,i2))
              call get_weight(p_V(ind,i1,i2)-pos, weight)
              pos = pos - stencil_g - 1
              p_V(ind,i1,i2) = weight(1)*V_coarse(modulo(pos,mesh_V%N(direction))+1,i1+idir1,i2+idir2)
              do i_st = 2, stencil_size
                p_V(ind,i1,i2) = p_V(ind,i1,i2) + &
                    & weight(i_st)*V_coarse(modulo(pos+i_st-1,mesh_V%N(direction))+1,i1+idir1,i2+idir2)
              end do ! loop on stencil points.
            end do ! loop on particle indice (ind)
        end do ! loop on first coordinate (i1) of a line inside the block of line
    end do ! loop on second coordinate (i2) of a line inside the block of line

end subroutine AC_interpol_plus_no_com


!> Interpolate the velocity field from coarse grid at particles positions
end module advec_common_interpol
!> @}
