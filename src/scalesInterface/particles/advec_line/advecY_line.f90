!USEFORTEST advec
!> @addtogroup part
!! @{

!------------------------------------------------------------------------------
!
! MODULE: advecY_line
!
!
! DESCRIPTION:
!> The module advecY_line is devoted to the simplest implementation of
!! advection along Y axis of a scalar field.
!
!> @details
!> The module advecY_line is devoted to the simplest implementation of
!! advection along Y axis of a scalar field. It is an unoptimized
!! version, useful to understand the basis and to benchmark the
!! optimisation done.
!! It used particle method and provides a parallel implementation.
!!
!! This module can use the method and variables defined in the module
!! "advec_common_line" which gather information and tools shared for advection along
!! x, y and z-axis.
!!
!! The module "test_advec" can be used in order to validate the procedures
!! embedded in this module.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module advecY_line

    use precision_tools
    use advec_abstract_proc

    implicit none

    ! ===== Public procedures =====
    !> particles solver with remeshing method at order 2
    private                 :: advecY_calc_line     ! remeshing method at order 2
    !----- (corrected) Remeshing method (these methods are set to public in validation purposes) -----
    public                  :: Yremesh_O2_line       ! order 2

    ! ===== Private porcedures =====
    ! Particles initialisation
    private                 :: advecY_init_line ! initialisation for only one line of particles

    ! ===== Private variables =====
    !> current direction = alongY (to avoid redefinition and make more easy cut/paste)
    integer, parameter, private      :: direction = 2
    !> Group size along current direction
!    integer, private, dimension(2)  :: gs

contains


! #####################################################################################
! #####                                                                           #####
! #####                         Public procedure                                  #####
! #####                                                                           #####
! #####################################################################################


!> Advection along Y during a time step dt - order 2 - "line one by one" version
!!    @param[in]        dt      = time step
!!    @param[in]        Vy      = velocity along y (could be discretised on a bigger mesh then the scalar)
!!    @param[in,out]    scal3D   = scalar field to advect
subroutine advecY_calc_line(dt,Vy,scal3D)

    use advec_common_line          ! some procedures common to advection along all directions
    use advec_variables       ! contains info about solver parameters and others.
    use cart_topology   ! description of mesh and of mpi topology

    ! input/output
    real(WP), intent(in)                                                :: dt
    real(WP), dimension(mesh_sc%N_proc(1), mesh_sc%N_proc(2), mesh_sc%N_proc(3)), intent(in)    :: Vy
    real(WP), dimension(mesh_sc%N_proc(1), mesh_sc%N_proc(2), mesh_sc%N_proc(3)), intent(inout) :: scal3D
    ! other local variables
    integer                                 :: i,k          ! indice of the currend mesh point
    integer, dimension(direction)           :: ind_group    ! indice of the currend group of line ((i,k) by default)
    real(WP), dimension(mesh_sc%N_proc(direction))  :: p_pos_adim   ! adimensionned particles position 
    real(WP), dimension(mesh_sc%N_proc(direction))  :: p_V          ! particles velocity 
    logical, dimension(bl_nb(direction)+1)  :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(direction))    :: bl_tag       ! indice of tagged particles

    ind_group = 0
    do k = 1, mesh_sc%N_proc(3)
        ind_group(2) = ind_group(2) + 1
        ind_group(1) = 0
        do i = 1, mesh_sc%N_proc(1)
            ind_group(1) = ind_group(1) + 1

            ! ===== Init particles =====
            call advecY_init_line(Vy, i, k, p_pos_adim, p_V)

            ! ===== Advection =====
            ! -- Compute velocity (with a RK2 scheme) --
            call AC_particle_velocity_line(dt, direction, ind_group, p_pos_adim, p_V)
            ! -- Advec particles --
            p_pos_adim = p_pos_adim + dt*p_V/mesh_sc%dx(direction)

            ! ===== Remeshing =====
            ! -- Pre-Remeshing: Determine blocks type and tag particles --
            call AC_type_and_block_line(dt, direction, ind_group, p_V, bl_type, bl_tag)
            ! -- Remeshing --
            call Yremesh_O2_line(ind_group, p_pos_adim, bl_type, bl_tag,i,k,scal3D)
        end do
    end do


end subroutine advecY_calc_line


! #####################################################################################
! #####                                                                           #####
! #####                         Private procedure                                 #####
! #####                                                                           #####
! #####################################################################################

! ====================================================================
! ====================   Remeshing subroutines    ====================
! ====================================================================

!> remeshing with an order 2 method, corrected to allow large CFL number - untagged particles
!!    @param[in]        ind_group   = coordinate of the current group of lines
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        bl_type     = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]        bl_tag      = contains information about bloc (is it tagged ?)
!!    @param[in]        i,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in,out]    scal        = scalar field to advect
subroutine Yremesh_O2_line(ind_group, p_pos_adim, bl_type, bl_tag,i,k,scal)

    use advec_common_line            ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/Output
    integer, dimension(2), intent(in)                                   :: ind_group
    integer, intent(in)                                                 :: i, k
    logical, dimension(:), intent(in)                                   :: bl_type
    logical, dimension(:), intent(in)                                   :: bl_tag
    real(WP), dimension(:), intent(in)                                  :: p_pos_adim
    real(WP), dimension(mesh_sc%N_proc(1), mesh_sc%N_proc(2), mesh_sc%N_proc(3)), intent(inout) :: scal
    ! Other local variables 
    ! Variable used to remesh particles in a buffer
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it 
                                                        ! to the right subdomain
    integer, dimension(2)               :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which 
                                                        ! I will receive data
    integer                             :: proc_min     ! smaller gap between me and the processes to where I send data
    integer                             :: proc_max     ! smaller gap between me and the processes to where I send data

    !  -- Compute ranges for remeshing of local particles --
    if (bl_type(1)) then
        ! First particle is a centered one
        send_j_min = nint(p_pos_adim(1))-1
    else
        ! First particle is a left one
        send_j_min = floor(p_pos_adim(1))-1
    end if
    if (bl_type(mesh_sc%N_proc(direction)/bl_size +1)) then
        ! Last particle is a centered one
        send_j_max = nint(p_pos_adim(mesh_sc%N_proc(direction)))+1
    else
        ! Last particle is a left one
        send_j_max = floor(p_pos_adim(mesh_sc%N_proc(direction)))+1
    end if
        
    ! -- Determine the communication needed : who will communicate whit who ? (ie compute sender and recer) --
    call AC_obtain_senders_line(send_j_min, send_j_max, direction, ind_group, proc_min, proc_max, rece_proc)

    ! -- Allocate buffer for remeshing of local particles --
    allocate(send_buffer(send_j_min:send_j_max))
    send_buffer = 0.0;

    ! -- Remesh the particles in the buffer --
    call AC_remesh_lambda2corrected_basic(direction, p_pos_adim, scal(i,:,k), bl_type, bl_tag, send_j_min, send_j_max, send_buffer)
    
    ! -- Send the buffer to the matching processus and update the scalar field --
    scal(i,:,k) = 0
    call AC_bufferToScalar_line(direction, ind_group , send_j_min, send_j_max, proc_min, proc_max, &
        & rece_proc, send_buffer, scal(i,:,k))

    ! -- Deallocate all field --
    deallocate(send_buffer)

end subroutine Yremesh_O2_line


! ====================================================================
! ====================    Initialize particle     ====================
! ====================================================================

!> Creation and initialisation of a particle line (ie X and Z coordinate are fixed)
!!    @param[in]    Vy          = 3D velocity field
!!    @param[in]    i           = X-indice of the current line
!!    @param[in]    k           = Z-indice of the current line
!!    @param[out]   p_pos_adim  = adimensioned particles postion
!!    @param[out]   p_V         = particle velocity 
subroutine advecY_init_line(Vy, i, k, p_pos_adim, p_V)

    use cart_topology   ! description of mesh and of mpi topology

    ! input/output
    integer, intent(in)                                 :: i,k
    real(WP), dimension(mesh_sc%N_proc(direction)), intent(out) :: p_pos_adim, p_V
    real(WP), dimension(:,:,:), intent(in)              :: Vy
    ! Other local variables
    integer                                     :: ind          ! indice

    do ind = 1, mesh_sc%N_proc(direction)
        p_pos_adim(ind) = ind
        p_V(ind)        = Vy(i,ind,k)
    end do

end subroutine advecY_init_line

end module advecY_line
!> @}
