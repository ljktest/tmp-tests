!USEFORTEST advec
!> @addtogroup part
!! @{
!------------------------------------------------------------------------------
!
! MODULE: advec_correction
!
!
! DESCRIPTION:
!> The module ``advec_correction'' gather function and subroutines used to computed
!! eventual correction or limitator if wanted. These tools are
!! independant from the direction.
!! @details
!! This module gathers functions and routines used to determine when correction
!! are required depending on the remeshing formula. It includes particle
!! type and tag (for corrected lambda schemes) and variation computation
!! for limitator.
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

module advec_correction

  use mpi, only: MPI_STATUS_SIZE

    implicit none
    

    !----- Determine block type and tag particles -----
    public  :: AC_type_and_block_group
    public  :: AC_limitator_from_slopes

contains

! ===========================================================================================================
! ====================     Bloc type and particles tag for corrected lambda schemes      ====================
! ===========================================================================================================

!> Determine type (center or left) of each block and tag for a complete group of
!! lines.
!! corrected remeshing formula are recquired.
!!    @param[in]        dt          = time step
!!    @param[in]        dir         = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        gp_s        = size of a group (ie number of line it gathers along the two other directions)
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        p_V         = particle velocity (along the current direction)
!!    @param[out]       bl_type     = table of blocks type (center of left)
!!    @param[out]       bl_tag      = inform about tagged particles (bl_tag(ind_bl)=1 if the end of the bl_ind-th block
!!                                    and the begining of the following one is tagged)
!! @details
!!        This subroutine work on a groupe of line. For each line of this group, it
!!    determine the type of each block of this line and where corrected remeshing
!!    formula are required. In those points, it tagg block transition (ie the end of
!!    the current block and the beginning of the following one) in order to indicate
!!    that corrected weigth have to be used during the remeshing.
subroutine AC_type_and_block_group(dt, dir, gp_s, ind_group, p_V, &
                & bl_type, bl_tag)

    
    use precision_tools ! define working precision_tools (double or simple)
    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.

    real(WP), intent(in)                      :: dt           ! time step
    integer, intent(in)                       :: dir
    integer, dimension(2),intent(in)          :: gp_s         ! groupe size
    integer, dimension(2), intent(in)         :: ind_group    ! group indice
    real(WP), dimension(:,:,:), intent(in)    :: p_V
    logical,dimension(:,:,:),intent(out)      :: bl_type      ! is the particle block a center block or a left one ?
    logical,dimension(:,:,:),intent(out)      :: bl_tag       ! indice of tagged particles

    real(WP),dimension(bl_nb(dir)+1,gp_s(1),gp_s(2))            :: bl_lambdaMin ! for a particle, lamda = V*dt/dx ;  bl_lambdaMin = min of
                                                                                ! lambda on a block (take also into account first following particle)
    real(WP),dimension(gp_s(1),gp_s(2))                         :: lambP, lambN ! buffer to exchange some lambda min with other processus
    real(WP),dimension(gp_s(1),gp_s(2))                         :: lambB, lambE ! min value of lambda of the begin of the line and at the end of the line
    integer, dimension(bl_nb(dir)+1,gp_s(1),gp_s(2))            :: bl_ind       ! block index : integer as lambda in (bl_ind,bl_ind+1) for a left block
                                                                                ! and lambda in (bl_ind-1/2, bl_ind+1/2) for a right block
    integer                                                     :: ind,i_p      ! some indices
    real(WP)                                                    :: cfl          ! = d_sc
    integer, dimension(2)                                       :: send_request ! mpi status of nonblocking send
    integer, dimension(2)                                       :: rece_request ! mpi status of nonblocking receive
    integer, dimension(MPI_STATUS_SIZE)                         :: rece_status  ! mpi status (for mpi_wait)
    integer, dimension(MPI_STATUS_SIZE)                         :: send_status  ! mpi status (for mpi_wait)
    integer, dimension(2)                                       :: tag_table    ! other tags for mpi message
    integer                                                     :: com_size     ! size of mpi message
    integer                                                     :: ierr         ! mpi error code

    ! ===== Initialisation =====
    cfl = dt/mesh_sc%dx(dir)
    com_size = gp_s(1)*gp_s(2)

    ! ===== Compute bl_lambdaMin =====

    ! Receive ghost value, ie value from neighbors boundaries.
    tag_table = compute_tag(ind_group, tag_part_tag_NP, dir)
    call mpi_Irecv(lambN(1,1), com_size, MPI_REAL_WP, &
            & neighbors(dir,1), tag_table(1), D_comm(dir), rece_request(1), ierr)
    call mpi_Irecv(lambP(1,1), com_size, MPI_REAL_WP, &
            &  neighbors(dir,-1), tag_table(2), D_comm(dir), rece_request(2), ierr)

    ! -- For the first block (1/2) --
    ! The domain contains only its second half => exchange ghost with the previous processus
    lambB = minval(p_V(1:(bl_size/2)+1,:,:),1)*cfl
    !tag_table = compute_tag(ind_group, tag_part_tag_NP, dir)   ! Tag table is already equals to this.
    ! Send message
    call mpi_ISsend(lambB(1,1), com_size, MPI_REAL_WP, &
            & neighbors(dir,-1), tag_table(1), D_comm(dir), send_request(1), ierr)

    ! -- For the last block (1/2) --
    ! The processus contains only its first half => exchange ghost with the next processus
    ind = bl_nb(dir) + 1
    lambE = minval(p_V(mesh_sc%N_proc(dir) - (bl_size/2)+1 :mesh_sc%N_proc(dir),:,:),1)*cfl
    ! Send message
    call mpi_ISsend(lambE(1,1), com_size, MPI_REAL_WP, &
            & neighbors(dir,1), tag_table(2), D_comm(dir), send_request(2), ierr)

    ! -- For the "middle" block --
    do ind = 2, bl_nb(dir)
        i_p = ((ind-1)*bl_size) + 1 - bl_size/2
        bl_lambdaMin(ind,:,:) = minval(p_V(i_p:i_p+bl_size,:,:),1)*cfl
    end do

    ! -- For the first block (1/2) --
    ! The domain contains only its second half => use exchanged ghost
    ! Check reception
    call mpi_wait(rece_request(2), rece_status, ierr)
    bl_lambdaMin(1,:,:) = min(lambB(:,:), lambP(:,:))

    ! -- For the last block (1/2) --
    ! The processus contains only its first half => use exchanged ghost
    ! Check reception
    call mpi_wait(rece_request(1), rece_status, ierr)
    ind = bl_nb(dir) + 1
    bl_lambdaMin(ind,:,:) = min(lambE(:,:), lambN(:,:))

    ! ===== Compute block type and index =====
    bl_ind = nint(bl_lambdaMin)
    bl_type = (bl_lambdaMin<dble(bl_ind))

    ! ===== Tag particles =====
    do ind = 1, bl_nb(dir)
        bl_tag(ind,:,:) = ((bl_ind(ind,:,:)/=bl_ind(ind+1,:,:)) .and. &
                & (bl_type(ind,:,:).neqv.bl_type(ind+1,:,:)))
    end do

    call mpi_wait(send_request(1), send_status, ierr)
    call mpi_wait(send_request(2), send_status, ierr)

end subroutine AC_type_and_block_group


!> Compute a limitator function from scalar slope - only for corrected lambda 2 formula.
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
subroutine AC_limitator_from_slopes(direction, gp_s, p_pos, &
                & deltaS, limit, tag_mpi, com_size)

    
    use cart_topology   ! info about mesh and mpi topology
    use advec_variables ! contains info about solver parameters and others.
    use precision_tools       ! define working precision_tools (double or simple)

    integer                                     :: direction    ! current direction
    integer, dimension(2),intent(in)            :: gp_s         ! groupe size
    real(WP), dimension(:,:,:), intent(in)      :: p_pos        ! particle position
    real(WP), dimension(:,:,:), intent(in)      :: deltaS       ! scalar slope: scalar(i+1)-scalar(i) - for i=1 N_proc+1
    real(WP), dimension(:,:,:), intent(out)     :: limit        ! limitator function
    integer, intent(in)                         :: tag_mpi      ! tag for mpi message
    integer, intent(in)                         :: com_size     ! size of mpi message

    ! Local variables
    real(WP),dimension(2,gp_s(1),gp_s(2))       :: Sbuffer, Rbuffer ! buffer to exchange scalar or limitator at boundaries with neighbors.
    integer                                     :: ind          ! loop indice on particle indice
    real(WP),dimension(gp_s(1),gp_s(2))         :: afl          ! = cfl - [cfl] where [] denotes the nearest int.
!   integer,dimension(gp_s(1),gp_s(2))          :: afl_sign     ! = sign of afl, ie 1 if afl>=0, -1 if afl<0
    integer                                     :: send_request ! mpi status of nonblocking send
    integer, dimension(MPI_STATUS_SIZE)         :: rece_status  ! mpi status (for mpi_wait)
    integer, dimension(MPI_STATUS_SIZE)         :: send_status  ! mpi status (for mpi_wait)
    integer                                     :: ierr         ! mpi error code

    ! ===== Compute slope and limitator =====
    ! Van Leer limitator function (limit = limitator/8)
    ! -- For the "middle" and the "last" block --
    do ind = 2, mesh_sc%N_proc(direction)
        where(deltaS(:,:,ind)/=0)
            afl = p_pos(ind,:,:)
            afl = afl - nint(afl)
!           afl_sign = int(sign(1._WP,afl))
!           limit(ind+1,:,:) = (4.0_WP/8._WP)*min(0.9_WP,(afl_sign*afl+0.5_WP)**2)*(deltaS(:,:,ind-afl_sign)/deltaS(:,:,ind))/(1+(deltaS(:,:,ind-afl_sign)/deltaS(:,:,ind)))
            ! If (p_pos-nint(p_pos))>=0)
            where(afl>=0)
                limit(ind+1,:,:) = max(0._WP,(deltaS(:,:,ind-1)/deltaS(:,:,ind)))
                limit(ind+1,:,:) = limit(ind+1,:,:)/(limit(ind+1,:,:)+1)
                limit(ind+1,:,:) = (4.0_WP/8._WP)*min(0.9_WP,(afl+0.5_WP)**2)*limit(ind+1,:,:)
            elsewhere
                limit(ind+1,:,:) = max(0._WP,(deltaS(:,:,ind+1)/deltaS(:,:,ind)))
                limit(ind+1,:,:) = limit(ind+1,:,:)/(limit(ind+1,:,:)+1)
                limit(ind+1,:,:) = (4.0_WP/8._WP)*min(0.9_WP,(afl-0.5_WP)**2)*limit(ind+1,:,:)
            end where
        elsewhere
            limit(ind+1,:,:) = 0.0_WP
        end where
    end do
    ! -- For the "first" block --
    ! 1 - limit(1) - limitator at 1/2 is already compute on the previous mpi-rank (limit(N_proc+1) !)
    ! 2 - limit(2) - limitator at 1+1/2 requires deltaS(0) = scalar slope between scalar(0) and scalar(-1) which is already compute on previous rank
    ! Send these values
    Sbuffer(1,:,:) = limit(mesh_sc%N_proc(direction)+1,:,:)
    Sbuffer(2,:,:) = deltaS(:,:,mesh_sc%N_proc(direction))
    call mpi_ISsend(Sbuffer(1,1,1), com_size, MPI_REAL_WP, &
            & neighbors(direction,1), tag_mpi, D_comm(direction), send_request, ierr)
    ! Receive it !
    call mpi_recv(Rbuffer(1,1,1), com_size, MPI_REAL_WP, &
            &  neighbors(direction,-1), tag_mpi, D_comm(direction),rece_status, ierr)
    ! Get limit(1) = limitator at 1/2
    limit(1,:,:) = Rbuffer(1,:,:)
    ! Get limit(2) = limitator at 1+1/2
    where(deltaS(:,:,1)/=0)
        afl = p_pos(1,:,:)
        afl = afl - nint(afl)
        ! If (p_pos-nint(p_pos))>=0)
        where(afl>=0)
            limit(2,:,:) = max(0._WP,(Rbuffer(2,:,:)/deltaS(:,:,1)))
            !            = ( deltaS(:,:,0)/deltaS(:,:,1))
            limit(2,:,:) = limit(2,:,:)/(1+limit(2,:,:))
            limit(2,:,:) = (4.0_WP/8._WP)*min(0.9_WP,(afl+0.5_WP)**2)*limit(2,:,:)
        elsewhere
            limit(2,:,:) = max(0._WP,(deltaS(:,:,2)/deltaS(:,:,1)))
            limit(2,:,:) = limit(2,:,:)/(1+limit(2,:,:))
            limit(2,:,:) = (4.0_WP/8._WP)*min(0.9_WP,(afl-0.5_WP)**2)*limit(2,:,:)
        end where
    elsewhere
        limit(2,:,:) = 0.0_WP
    end where

    ! Classical (corrected) lambda formula: limitator function = 1
    ! limit = 1._WP/8._WP


    ! ===== Close mpi_ISsend when done =====
    call mpi_wait(send_request, send_status, ierr)

end subroutine AC_limitator_from_slopes

end module advec_correction
