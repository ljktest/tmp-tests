!USEFORTEST advec
!> @addtogroup part

!------------------------------------------------------------------------------
!
! MODULE: advecX_line
!
!
! DESCRIPTION:
!> The module advecX_line is devoted to the simplest implementation of
!! advection along X axis of a scalar field.
!
!> @details
!> The module advecX_line is devoted to the simplest implementation of
!! advection along X axis of a scalar field. It is an unoptimized
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

module advecX_line

    use precision_tools
    use advec_abstract_proc

    implicit none

    ! ===== Public procedures =====
    !> particles solver with remeshing method at order 2
    public                  :: advecX_calc_line ! remeshing method at order 2
    !----- (corrected) Remeshing method (these methods are set to public in validation purposes) -----
    public                  :: Xremesh_O2_line  ! order 2

    ! ===== Private porcedures =====
    ! Particles initialisation
    private                 :: advecX_init_line ! initialisation for only one line of particles

    ! ===== Private variable ====
    ! particles solver with different remeshing formula
    integer, dimension(2), private  :: gpX_size
    !> Current direction = along X
    integer, private, parameter     :: direction=1
    !> Group size along current direction
    !integer, private, dimension(2)  :: gs

contains

! #####################################################################################
! #####                                                                           #####
! #####                         Public procedure                                  #####
! #####                                                                           #####
! #####################################################################################

! =================================================================
! ====================         Solveur         ====================
! =================================================================

!> Advection during a time step dt - order 2 - "line one by one" version
!!    @param[in]        dt      = time step
!!    @param[in]        Vx      = velocity along y (could be discretised on a bigger mesh then the scalar)
!!    @param[in,out]    scal3D   = scalar field to advect
subroutine advecX_calc_line(dt,Vx,scal3D)

    use advec_common_line   ! Some procedures common to advection along all directions
    use advec_variables     ! contains info about solver parameters and others.
    use cart_topology       ! Description of mesh and of mpi topology

    ! Input/Output
    real(WP), intent(in)                                                :: dt
    real(WP), dimension(mesh_sc%N_proc(1), mesh_sc%N_proc(2), mesh_sc%N_proc(3)), intent(in)    :: Vx
    real(WP), dimension(mesh_sc%N_proc(1), mesh_sc%N_proc(2), mesh_sc%N_proc(3)), intent(inout) :: scal3D
    ! Other local variables
    integer                             :: j,k          ! indice of the currend mesh point
    integer, dimension(2)               :: ind_group    ! indice of the currend group of line (=(i,k) by default)
    integer                             :: direction=1  ! current direction = along Y
    real(WP), dimension(mesh_sc%N_proc(1))      :: p_pos_adim   ! adimensionned particles position 
    real(WP), dimension(mesh_sc%N_proc(1))      :: p_V          ! particles velocity 
    logical, dimension(bl_nb(1)+1)      :: bl_type      ! is the particle block a center block or a left one ?
    logical, dimension(bl_nb(1))        :: bl_tag       ! indice of tagged particles

    ind_group = 0
    do k = 1, mesh_sc%N_proc(3)
        ind_group(2) = ind_group(2) + 1
        ind_group(1) = 0
        do j = 1, mesh_sc%N_proc(2)
        ind_group(1) = ind_group(1) + 1

            ! ===== Init particles =====
            call advecX_init_line(Vx, j, k, p_pos_adim, p_V)

            ! ===== Advection =====
            ! -- Compute velocity (with a RK2 scheme) --
            call AC_particle_velocity_line(dt, direction, ind_group, p_pos_adim, p_V)
            ! -- Advec particles --
            p_pos_adim = p_pos_adim + dt*p_V/mesh_sc%dx(direction)

            ! ===== Remeshing =====
            ! -- Pre-Remeshing: Determine blocks type and tag particles --
            call AC_type_and_block_line(dt, direction, ind_group, p_V, &
                    & bl_type, bl_tag)
            ! -- Remeshing --
            call Xremesh_O2_line(ind_group, p_pos_adim, bl_type, bl_tag,j,k,scal3D)

        end do
    end do

end subroutine advecX_calc_line


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
!!    @param[in]        j,k         = indice of of the current line (x-coordinate and z-coordinate)
!!    @param[in,out]    scal        = scalar field to advect
subroutine Xremesh_O2_line(ind_group, p_pos_adim, bl_type, bl_tag,j,k,scal)

    use advec_common_line       ! Some procedures common to advection along all directions
    use advec_remeshing_line    ! Remeshing formula
    use advec_variables         ! contains info about solver parameters and others.
    use cart_topology     ! Description of mesh and of mpi topology

    ! Input/Output
    integer, dimension(2), intent(in)                                   :: ind_group
    integer, intent(in)                                                 :: j, k
    logical, dimension(:), intent(in)                                   :: bl_type
    logical, dimension(:), intent(in)                                   :: bl_tag
    real(WP), dimension(:), intent(in)                                  :: p_pos_adim
    real(WP), dimension(mesh_sc%N_proc(1), mesh_sc%N_proc(2), mesh_sc%N_proc(3)), intent(inout) :: scal
    ! Other local variables
    ! Variable used to remesh particles in a buffer
    real(WP),dimension(:),allocatable   :: send_buffer  ! buffer use to remesh the scalar before to send it to the right subdomain
    integer, dimension(2)               :: rece_proc    ! minimal and maximal gap between my Y-coordinate and the one from which 
                                                        ! I will receive data
    integer                             :: proc_min     ! smaller gap between me and the processes to where I send data
    integer                             :: proc_max     ! smaller gap between me and the processes to where I send data


    !  -- Compute ranges --
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
    call AC_remesh_lambda2corrected_basic(direction, p_pos_adim, scal(:,j,k), bl_type, bl_tag, send_j_min, send_j_max, send_buffer)

    ! -- Send the buffer to the matching processus and update the scalar field --
    scal(:,j,k) = 0
    call AC_bufferToScalar_line(direction, ind_group , send_j_min, send_j_max, proc_min, proc_max, &
        & rece_proc, send_buffer, scal(:,j,k))

    ! Deallocate all field
    deallocate(send_buffer)

end subroutine Xremesh_O2_line


! ====================================================================
! ====================    Initialize particle     ====================
! ====================================================================

!> Creation and initialisation of a particle line (ie Y and Z coordinate are fixed)
!!    @param[in]    Vx          = 3D velocity field
!!    @param[in]    j           = Y-indice of the current line
!!    @param[in]    k           = Z-indice of the current line
!!    @param[out]   p_pos_adim  = adimensioned particles postion
!!    @param[out]   p_V         = particle velocity
subroutine advecX_init_line(Vx, j, k, p_pos_adim, p_V)

    use cart_topology   ! Description of mesh and of mpi topology

    ! Input/Output
    integer, intent(in)                                 :: j,k
    real(WP), dimension(mesh_sc%N_proc(direction)), intent(out) :: p_pos_adim, p_V
    real(WP), dimension(:,:,:), intent(in)              :: Vx
    ! Other local variables
    integer                                     :: ind          ! indice

    do ind = 1, mesh_sc%N_proc(direction)
        p_pos_adim(ind) = ind
        p_V(ind)        = Vx(ind,j,k)
    end do

end subroutine advecX_init_line

end module advecX_line
!> @}
