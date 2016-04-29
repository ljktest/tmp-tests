!USEFORTEST advec
!> @addtogroup part

!------------------------------------------------------------------------------
!
!
!       ===================================================================
!       ====================     Remesh particles      ====================
!       ===================================================================
!
! MODULE: advec_remeshing_formula
!
!
! DESCRIPTION:
!> This module gathers all the remeshing formula of type "corrected lambda".
!! These interpolation polynoms allow to re-distribute particles on mesh grid at each
!! iterations.
!! @details
!! It provides lambda 2 corrected, lambda 4 corrected and limited lambda 2
!! corrected.
!!     The remeshing of type "lambda corrected" are design for large time
!! step. They are based on lambda formula. The stability condition does not
!! involve the CFL number but only the velocity gradient:
!! dt < constant*gradient(velocity)
!!     Note that such a remeshing formula involve different cases depending
!! of variation of the local CFL number. Thus particle are gather by group
!! and "if structure" (actually it is rather a "select case") is applied to
!! avoid such an "if".
!! each bloc to match to the right case. Only M' formula (see advec_remesh_Mprime)
!!     This module also provide some wraper to remesh a complete line
!! of particles (with the different formula) and to do it either on a
!! array or into a array of pointer to reals. In order to gather
!! communications between different lines of particles, it is better to
!! use continguous memory space for mesh point with belong to the same
!! processes and thus to use and array of pointer to easily deal with it.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module advec_remeshing_lambda

    use structure_tools
    use advec_common_line

    implicit none

    ! #############################
    ! ########## Hearder ##########
    ! #############################

    ! ===== Public procedures =====
    ! "Line remeshing" wrapper (they remesh a complete line of particle using
    ! the adapted interpolation polynom)
    procedure(AC_remesh_lambda2corrected_array), pointer, public ::  AC_remesh_lambda_array => null()    !> Generic wrapper to remesh a line of particle into an array of real
    procedure(AC_remesh_lambda2corrected_pter) , pointer, public ::  AC_remesh_lambda_pter  => null()    !> Generic wrapper to remesh a line of particle into an array of pointer
    ! Line remeshing for each corrected lambda scheme.
    public                              :: AC_remesh_lambda2corrected_pter
    public                              :: AC_remesh_lambda4corrected_pter
    public                              :: AC_remesh_lambda2corrected_array
    public                              :: AC_remesh_lambda4corrected_array
    ! To get the right "line remeshing" wrapper
!   !public                              :: AC_remesh_get_pointer

    ! ===== Private procedures =====
    !----- Order 2 remeshing formula -----
    ! Interface
    private                             :: AC_remesh_O2        ! lambda 2 remeshing formula (for left or center block - no correction)
    private                             :: AC_remesh_tag_CL    ! corrected formula for tagged particles : transition from C to L block.
    private                             :: AC_remesh_tag_LC    ! corrected formula for tagged particles : transition from L to C block
    ! Function used by the interfaces
    private                             :: AC_remesh_O2_array     ! lambda 2 remeshing formula (for left or center block - no correction)
    private                             :: AC_remesh_O2_pter      ! lambda 2 remeshing formula (for left or center block - no correction)
    private                             :: AC_remesh_tag_CL_array   ! corrected formula for tagged particles : transition from C to L block.
    private                             :: AC_remesh_tag_CL_pter    ! corrected formula for tagged particles : transition from C to L block.
    !----- Order 4 remeshing formula -----
    ! Interface
    private                             :: AC_remesh_O4_left   ! left remeshing formula
    private                             :: AC_remesh_O4_center ! centered remeshing formula
    private                             :: AC_remesh_O4_tag_CL ! corrected formula for tagged particles : transition from C to L block.
    private                             :: AC_remesh_O4_tag_LC ! corrected formula for tagged particles : transition from L to C block
    ! Function used by the interfaces
    private                             :: AC_remesh_O4_left_array  ! left remeshing formula - array of real
    private                             :: AC_remesh_O4_left_pter   ! left remeshing formula - array of pointer
    private                             :: AC_remesh_O4_center_array! centered remeshing formula
    private                             :: AC_remesh_O4_center_pter ! centered remeshing formula
    private                             :: AC_remesh_O4_tag_CL_array! corrected formula for tagged particles : transition from C to L block.
    private                             :: AC_remesh_O4_tag_CL_pter ! corrected formula for tagged particles : transition from C to L block.
    private                             :: AC_remesh_O4_tag_LC_array! corrected formula for tagged particles : transition from L to C block
    private                             :: AC_remesh_O4_tag_LC_pter ! corrected formula for tagged particles : transition from L to C block


    !===== Interface =====
    ! -- Order 2: array of real or of pointer --
    interface AC_remesh_lambda2corrected
        module procedure AC_remesh_lambda2corrected_pter, AC_remesh_lambda2corrected_array
    end interface AC_remesh_lambda2corrected

    interface AC_remesh_O2
        module procedure AC_remesh_O2_pter, AC_remesh_O2_array
    end interface AC_remesh_O2

    interface AC_remesh_tag_CL
        module procedure AC_remesh_tag_CL_pter, AC_remesh_tag_CL_array
    end interface AC_remesh_tag_CL

    interface AC_remesh_tag_LC
        module procedure AC_remesh_tag_LC_pter, AC_remesh_tag_LC_array
    end interface AC_remesh_tag_LC

    ! -- Order 4: array of real or of pointer --
    interface AC_remesh_lambda4corrected
        module procedure AC_remesh_lambda4corrected_pter, AC_remesh_lambda4corrected_array
    end interface AC_remesh_lambda4corrected

    interface AC_remesh_O4_left
        module procedure AC_remesh_O4_left_pter, AC_remesh_O4_left_array
    end interface AC_remesh_O4_left

    interface AC_remesh_O4_center
        module procedure AC_remesh_O4_center_pter, AC_remesh_O4_center_array
    end interface AC_remesh_O4_center

    interface AC_remesh_O4_tag_CL
        module procedure AC_remesh_O4_tag_CL_pter, AC_remesh_O4_tag_CL_array
    end interface AC_remesh_O4_tag_CL

    interface AC_remesh_O4_tag_LC
        module procedure AC_remesh_O4_tag_LC_pter, AC_remesh_O4_tag_LC_array
    end interface AC_remesh_O4_tag_LC

    ! -- Order 2 with limitator: array of real or of pointer --
    !interface AC_remesh_lambda2corrected
    !    module procedure AC_remesh_lambda2corrected_pter, AC_remesh_lambda2corrected_array
    !end interface AC_remesh_lambda2corrected

    interface AC_remesh_limitO2
        module procedure AC_remesh_limitO2_pter, AC_remesh_limitO2_array
    end interface AC_remesh_limitO2

    interface AC_remesh_limitO2_tag_CL
        module procedure AC_remesh_limitO2_tag_CL_pter, AC_remesh_limitO2_tag_CL_array
    end interface AC_remesh_limitO2_tag_CL

    interface AC_remesh_limitO2_tag_LC
        module procedure AC_remesh_limitO2_tag_LC_pter, AC_remesh_limitO2_tag_LC_array
    end interface AC_remesh_limitO2_tag_LC
    ! ===== Abstract procedure =====

    ! --- Abstract profile of subroutine used to remesh a line of particles ---
    ! Variant: the buffer is an array of pointer (and not a pointer to an array)
    abstract interface
        subroutine AC_remesh_line_pter(direction, p_pos_adim, scal1D, bl_type, bl_tag, ind_min, buffer)
            use structure_tools
            use advec_variables

            implicit none

            ! Input/Output
            integer, intent(in)                             :: direction
            real(WP), dimension(:), intent(in)              :: p_pos_adim
            real(WP), dimension(:), intent(in)              :: scal1D
            logical, dimension(:), intent(in)               :: bl_type
            logical, dimension(:), intent(in)               :: bl_tag
            integer, intent(in)                             :: ind_min
            type(real_pter), dimension(:), intent(inout)    :: buffer
        end subroutine AC_remesh_line_pter
    end interface

contains


! ###################################################################
! ############                                            ###########
! ############     Pointer to the right remesh formula    ###########
! ############                                            ###########
! ###################################################################

subroutine AC_remesh_get_lambda(pointer)

    use advec_variables         ! solver context

    procedure(AC_remesh_line_pter), pointer, intent(out)    :: pointer ! subroutine wich remesh a line of particle with the right remeshing formula

    select case(trim(type_solv))
    case ('p_O4')
        pointer  => AC_remesh_lambda4corrected_pter
    case default
        pointer  => AC_remesh_lambda2corrected_pter
    end select

end subroutine AC_remesh_get_lambda


subroutine AC_remesh_init_lambda()

    use advec_variables         ! solver cntext

    select case(trim(type_solv))
    case ('p_O4')
        AC_remesh_lambda_array => AC_remesh_lambda4corrected_array
        AC_remesh_lambda_pter  => AC_remesh_lambda4corrected_pter
    case default
        AC_remesh_lambda_array => AC_remesh_lambda2corrected_array
        AC_remesh_lambda_pter  => AC_remesh_lambda2corrected_pter
    end select

end subroutine AC_remesh_init_lambda


! ###################################################################
! ############                                            ###########
! ############     Wrapper to remesh a complete line      ###########
! ############                                            ###########
! ###################################################################

!> Remesh particle line with corrected lambda 2 formula - remeshing is done into
!! an array of pointer to real
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        scal1D      = scalar field to advect
!!    @param[in]        bl_type     = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]        bl_tag      = contains information about bloc (is it tagged ?)
!!    @param[in]        ind_min     = minimal indice of the send buffer
!!    @param[in, out]   send_buffer = array of pointers to the buffer use to remesh the scalar before to send it to the right subdomain
!! @details
!!     Use corrected lambda 2 remeshing formula.
!! This remeshing formula depends on the particle type :
!!     1 - Is the particle tagged ?
!!     2 - Does it belong to a centered or a left block ?
!! Observe that tagged particles go by group of two : if the particles of a
!! block end are tagged, the one first one of the following block are
!! tagged too.
!! The following algorithm is write for block of minimal size.
!! @author = Jean-Baptiste Lagaert, LEGI/Ljk
subroutine AC_remesh_lambda2corrected_pter(direction, p_pos_adim, scal1D, bl_type, bl_tag, ind_min, send_buffer)

    use cart_topology   ! Description of mesh and of mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                                         :: direction
    real(WP), dimension(:), intent(in)                          :: p_pos_adim
    real(WP), dimension(:), intent(in)                          :: scal1D
    logical, dimension(:), intent(in)                           :: bl_type
    logical, dimension(:), intent(in)                           :: bl_tag
    integer, intent(in)                                         :: ind_min
    type(real_pter), dimension(:), intent(inout)                :: send_buffer
    ! Other local variables
    integer                                     :: bl_ind       ! indice of the current "block end".
    integer                                     :: p_ind        ! indice of the current particle
    real(WP), dimension(mesh_sc%N_proc(direction))      :: pos_translat ! translation of p_pos_adim as array indice are now starting from 1 and not ind_min

    pos_translat = p_pos_adim - ind_min + 1

    do p_ind = 1, mesh_sc%N_proc(direction), bl_size
        bl_ind = p_ind/bl_size + 1
        if (bl_tag(bl_ind)) then
            ! Tag case
                ! XXX Debug : to activate only in purpose debug
                !if (bl_type(ind).neqv. (.not. bl_type(ind+1))) then
                !    write(*,'(a,x,3(L1,x),a,3(i0,a))'), 'error on remeshing particles: (tag,type(i), type(i+1)) =', &
                !    & bl_tag(ind), bl_type(ind), bl_type(ind+1), ' and type must be different. Mesh point = (',i, ', ', j,', ',k,')'
                !    write(*,'(a,x,i0)'),  'paramètres du blocs : ind =', bl_ind
                !    stop
                !end if
                ! XXX Debug - end
            if (bl_type(bl_ind)) then
                ! tagged, the first particle belong to a centered block and the last to left block.
                call AC_remesh_tag_CL(pos_translat(p_ind), scal1D(p_ind), pos_translat(p_ind+1), scal1D(p_ind+1), send_buffer)
            else
                ! tagged, the first particle belong to a left block and the last to centered block.
                call AC_remesh_tag_LC(pos_translat(p_ind), scal1D(p_ind), pos_translat(p_ind+1), scal1D(p_ind+1), send_buffer)
            end if
        else
            ! First particle
            call AC_remesh_O2(pos_translat(p_ind),scal1D(p_ind), bl_type(bl_ind), send_buffer)
            ! Second particle is remeshed with left formula
            call AC_remesh_O2(pos_translat(p_ind+1),scal1D(p_ind+1), bl_type(bl_ind+1), send_buffer)
        end if
    end do

end subroutine AC_remesh_lambda2corrected_pter


!> Remesh particle line with corrected lambda 2 formula - remeshing is done into
!! an real array - no communication variant (buffer does not have the same size)
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        scal1D      = scalar field to advect
!!    @param[in]        bl_type     = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]        bl_tag      = contains information about bloc (is it tagged ?)
!!    @param[in, out]   remesh_buffer= buffer use to remesh the scalar
!! @details
!!     Use corrected lambda 2 remeshing formula.
!! This remeshing formula depends on the particle type :
!!     1 - Is the particle tagged ?
!!     2 - Does it belong to a centered or a left block ?
!! Observe that tagged particles go by group of two : if the particles of a
!! block end are tagged, the one first one of the following block are
!! tagged too.
!! The following algorithm is write for block of minimal size.
!! @author = Jean-Baptiste Lagaert, LEGI/Ljk
subroutine ac_remesh_lambda2corrected_array(direction, p_pos_adim, scal1d, bl_type, bl_tag, remesh_buffer)

    use cart_topology   ! description of mesh and of mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! input/output
    integer, intent(in)                                 :: direction
    real(wp), dimension(:), intent(in)                  :: p_pos_adim
    real(wp), dimension(mesh_sc%N_proc(direction)), intent(in)  :: scal1d
    logical, dimension(:), intent(in)                   :: bl_type
    logical, dimension(:), intent(in)                   :: bl_tag
    real(wp), dimension(:), intent(inout)               :: remesh_buffer
    ! Other local variables
    integer     :: bl_ind                               ! indice of the current "block end".
    integer     :: p_ind                                ! indice of the current particle

    do p_ind = 1, mesh_sc%N_proc(direction), bl_size
        bl_ind = p_ind/bl_size + 1
        if (bl_tag(bl_ind)) then
            ! Tag case
                ! XXX Debug : to activate only in purpose debug
                !if (bl_type(ind).neqv. (.not. bl_type(ind+1))) then
                !    write(*,'(a,x,3(L1,x),a,3(i0,a))'), 'error on remeshing particles: (tag,type(i), type(i+1)) =', &
                !    & bl_tag(ind), bl_type(ind), bl_type(ind+1), ' and type must be different. Mesh point = (',i, ', ', j,', ',k,')'
                !    write(*,'(a,x,i0)'),  'paramètres du blocs : ind =', bl_ind
                !    stop
                !end if
                ! XXX Debug - end
            if (bl_type(bl_ind)) then
                ! tagged, the first particle belong to a centered block and the last to left block.
                call AC_remesh_tag_CL(direction, p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), scal1D(p_ind+1), remesh_buffer)
            else
                ! tagged, the first particle belong to a left block and the last to centered block.
                call AC_remesh_tag_LC(direction, p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), scal1D(p_ind+1), remesh_buffer)
            end if
        else
            ! First particle
            call AC_remesh_O2(direction, p_pos_adim(p_ind),scal1D(p_ind), bl_type(bl_ind), remesh_buffer)
            ! Second particle is remeshed with left formula
            call AC_remesh_O2(direction, p_pos_adim(p_ind+1),scal1D(p_ind+1), bl_type(bl_ind+1), remesh_buffer)
        end if
    end do

end subroutine AC_remesh_lambda2corrected_array


!> Remesh particle line with corrected lambda 4 formula - array version
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        scal1D      = scalar field to advect
!!    @param[in]        bl_type     = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]        bl_tag      = contains information about bloc (is it tagged ?)
!!    @param[in, out]   remesh_buffer = buffer use to remesh the scalar before to send it to the right subdomain
!! @details
!!     Use corrected lambda 2 remeshing formula.
!! This remeshing formula depends on the particle type :
!!     1 - Is the particle tagged ?
!!     2 - Does it belong to a centered or a left block ?
!! Observe that tagged particles go by group of two : if the particles of a
!! block end are tagged, the one first one of the following block are
!! tagged too.
!! The following algorithm is write for block of minimal size.
!! @author = Jean-Baptiste Lagaert, LEGI/Ljk
subroutine AC_remesh_lambda4corrected_array(direction, p_pos_adim, scal1D, bl_type, bl_tag, remesh_buffer)

    use cart_topology   ! Description of mesh and of mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                                 :: direction
    real(WP), dimension(:), intent(in)                  :: p_pos_adim
    real(WP), dimension(mesh_sc%N_proc(direction)), intent(in)  :: scal1D
    logical, dimension(:), intent(in)                   :: bl_type
    logical, dimension(:), intent(in)                   :: bl_tag
    real(WP), dimension(:), intent(inout)               :: remesh_buffer
    ! Other local variables
    integer     :: bl_ind                               ! indice of the current "block end".
    integer     :: p_ind                                ! indice of the current particle

    do p_ind = 1, mesh_sc%N_proc(direction), bl_size
        bl_ind = p_ind/bl_size + 1
        if (bl_tag(bl_ind)) then
            ! Tagged case
            if (bl_type(bl_ind)) then
                ! tagged, the first particle belong to a centered block and the last to left block.
                call AC_remesh_O4_tag_CL(direction, p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), scal1D(p_ind+1), &
                        & p_pos_adim(p_ind+2), scal1D(p_ind+2), p_pos_adim(p_ind+3), scal1D(p_ind+3), remesh_buffer)
            else
                ! tagged, the first particle belong to a left block and the last to centered block.
                call AC_remesh_O4_tag_LC(direction, p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), scal1D(p_ind+1), &
                        & p_pos_adim(p_ind+2), scal1D(p_ind+2), p_pos_adim(p_ind+3), scal1D(p_ind+3), remesh_buffer)
            end if
        else
            ! No tag
            if (bl_type(bl_ind)) then
                call AC_remesh_O4_center(direction, p_pos_adim(p_ind),scal1D(p_ind), remesh_buffer)
                call AC_remesh_O4_center(direction, p_pos_adim(p_ind+1),scal1D(p_ind+1), remesh_buffer)
            else
                call AC_remesh_O4_left(direction, p_pos_adim(p_ind),scal1D(p_ind), remesh_buffer)
                call AC_remesh_O4_left(direction, p_pos_adim(p_ind+1),scal1D(p_ind+1), remesh_buffer)
            end if
            if (bl_type(bl_ind+1)) then
                call AC_remesh_O4_center(direction, p_pos_adim(p_ind+2),scal1D(p_ind+2), remesh_buffer)
                call AC_remesh_O4_center(direction, p_pos_adim(p_ind+3),scal1D(p_ind+3), remesh_buffer)
            else
                call AC_remesh_O4_left(direction, p_pos_adim(p_ind+2),scal1D(p_ind+2), remesh_buffer)
                call AC_remesh_O4_left(direction, p_pos_adim(p_ind+3),scal1D(p_ind+3), remesh_buffer)
            end if
        end if
    end do

end subroutine AC_remesh_lambda4corrected_array


!> Remesh particle line with corrected lambda 4 formula - array pter version
!!    @param[in]        direction       = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        p_pos_adim      = adimensionned  particles position
!!    @param[in]        scal1D          = scalar field to advect
!!    @param[in]        bl_type         = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]        bl_tag          = contains information about bloc (is it tagged ?)
!!    @param[in]        ind_min         = minimal indice of the send buffer
!!    @param[in, out]   remesh_buffer   = array of pointer to the buffer use to locally remesh the scalar
!! @details
!!     Use corrected lambda 4 remeshing formula.
!! This remeshing formula depends on the particle type :
!!     1 - Is the particle tagged ?
!!     2 - Does it belong to a centered or a left block ?
!! Observe that tagged particles go by group of two : if the particles of a
!! block end are tagged, the one first one of the following block are
!! tagged too.
!! The following algorithm is write for block of minimal size.
!! @author = Jean-Baptiste Lagaert, LEGI/Ljk
subroutine AC_remesh_lambda4corrected_pter(direction, p_pos_adim, scal1D, bl_type, bl_tag, ind_min, remesh_buffer)

    use cart_topology   ! Description of mesh and of mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                                 :: direction
    real(WP), dimension(:), intent(in)                  :: p_pos_adim
    real(WP), dimension(:), intent(in)                  :: scal1D
    logical, dimension(:), intent(in)                   :: bl_type
    logical, dimension(:), intent(in)                   :: bl_tag
    integer, intent(in)                                 :: ind_min
    type(real_pter), dimension(:), intent(inout)        :: remesh_buffer
    ! Other local variables
    integer     :: bl_ind                               ! indice of the current "block end".
    integer     :: p_ind                                ! indice of the current particle
    real(WP), dimension(mesh_sc%N_proc(direction))      :: pos_translat ! translation of p_pos_adim as array indice are now starting from 1 and not ind_min

    pos_translat = p_pos_adim - ind_min + 1

    do p_ind = 1, mesh_sc%N_proc(direction), bl_size
        bl_ind = p_ind/bl_size + 1
        if (bl_tag(bl_ind)) then
            ! Tagged case
            if (bl_type(bl_ind)) then
                ! tagged, the first particle belong to a centered block and the last to left block.
                call AC_remesh_O4_tag_CL(pos_translat(p_ind), scal1D(p_ind), pos_translat(p_ind+1), scal1D(p_ind+1), &
                        & pos_translat(p_ind+2), scal1D(p_ind+2), pos_translat(p_ind+3), scal1D(p_ind+3), remesh_buffer)
            else
                ! tagged, the first particle belong to a left block and the last to centered block.
                call AC_remesh_O4_tag_LC(pos_translat(p_ind), scal1D(p_ind), pos_translat(p_ind+1), scal1D(p_ind+1), &
                        & pos_translat(p_ind+2), scal1D(p_ind+2), pos_translat(p_ind+3), scal1D(p_ind+3), remesh_buffer)
            end if
        else
            ! No tag
            if (bl_type(bl_ind)) then
                call AC_remesh_O4_center(pos_translat(p_ind),scal1D(p_ind), remesh_buffer)
                call AC_remesh_O4_center(pos_translat(p_ind+1),scal1D(p_ind+1), remesh_buffer)
            else
                call AC_remesh_O4_left(pos_translat(p_ind),scal1D(p_ind), remesh_buffer)
                call AC_remesh_O4_left(pos_translat(p_ind+1),scal1D(p_ind+1), remesh_buffer)
            end if
            if (bl_type(bl_ind+1)) then
                call AC_remesh_O4_center(pos_translat(p_ind+2),scal1D(p_ind+2), remesh_buffer)
                call AC_remesh_O4_center(pos_translat(p_ind+3),scal1D(p_ind+3), remesh_buffer)
            else
                call AC_remesh_O4_left(pos_translat(p_ind+2),scal1D(p_ind+2), remesh_buffer)
                call AC_remesh_O4_left(pos_translat(p_ind+3),scal1D(p_ind+3), remesh_buffer)
            end if
        end if
    end do

end subroutine AC_remesh_lambda4corrected_pter


!> Remesh particle line with corrected and limited lambda 2 formula - remeshing is done into
!! an array of pointer to real
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        scal1D      = scalar field to advect
!!    @param[in]        bl_type     = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]        bl_tag      = contains information about bloc (is it tagged ?)
!!    @param[in]        limit       = limitator function value associated to the right and the left scalar variations
!!    @param[in]        ind_min     = minimal indice of the send buffer
!!    @param[in, out]   send_buffer = array of pointers to the buffer use to remesh the scalar before to send it to the right subdomain
!! @details
!!     Use corrected lambda 2 remeshing formula.
!! This remeshing formula depends on the particle type :
!!     1 - Is the particle tagged ?
!!     2 - Does it belong to a centered or a left block ?
!! Observe that tagged particles go by group of two : if the particles of a
!! block end are tagged, the one first one of the following block are
!! tagged too.
!! The following algorithm is write for block of minimal size.
!!    Note that instead of the value of the limitator function, it is actually
!! these values divided by 8 wich are given as arguments. As the limitator function
!! always appear divided by 8 in the remeshing polynom, perform this division
!! during the computation of the limitator function enhances the performances.
!! @author = Jean-Baptiste Lagaert, LEGI/Ljk
subroutine AC_remesh_lambda2limited_pter(direction, p_pos_adim, scal1D, bl_type, bl_tag, ind_min, limit, send_buffer)

    use cart_topology   ! Description of mesh and of mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                                         :: direction
    real(WP), dimension(:), intent(in)                          :: p_pos_adim
    real(WP), dimension(:), intent(in)                          :: scal1D
    real(WP), dimension(:), intent(in)                          :: limit
    logical, dimension(:), intent(in)                           :: bl_type
    logical, dimension(:), intent(in)                           :: bl_tag
    integer, intent(in)                                         :: ind_min
    type(real_pter), dimension(:), intent(inout)                :: send_buffer
    ! Other local variables
    integer                                     :: bl_ind       ! indice of the current "block end".
    integer                                     :: p_ind        ! indice of the current particle
    real(WP), dimension(mesh_sc%N_proc(direction))      :: pos_translat ! translation of p_pos_adim as array indice are now starting from 1 and not ind_min

    pos_translat = p_pos_adim - ind_min + 1

    do p_ind = 1, mesh_sc%N_proc(direction), bl_size
        bl_ind = p_ind/bl_size + 1
        if (bl_tag(bl_ind)) then
            ! Tag case
                ! XXX Debug : to activate only in purpose debug
                !if (bl_type(ind).neqv. (.not. bl_type(ind+1))) then
                !    write(*,'(a,x,3(L1,x),a,3(i0,a))'), 'error on remeshing particles: (tag,type(i), type(i+1)) =', &
                !    & bl_tag(ind), bl_type(ind), bl_type(ind+1), ' and type must be different. Mesh point = (',i, ', ', j,', ',k,')'
                !    write(*,'(a,x,i0)'),  'paramètres du blocs : ind =', bl_ind
                !    stop
                !end if
                ! XXX Debug - end
            if (bl_type(bl_ind)) then
                ! tagged, the first particle belong to a centered block and the last to left block.
                call AC_remesh_limitO2_tag_CL(pos_translat(p_ind), scal1D(p_ind), pos_translat(p_ind+1), &
                        & scal1D(p_ind+1), limit(p_ind:p_ind+2), send_buffer)
            else
                ! tagged, the first particle belong to a left block and the last to centered block.
                call AC_remesh_limitO2_tag_LC(pos_translat(p_ind), scal1D(p_ind), pos_translat(p_ind+1), &
                        & scal1D(p_ind+1), limit(p_ind:p_ind+2), send_buffer)
            end if
        else
            ! First particle
            call AC_remesh_limitO2(pos_translat(p_ind),scal1D(p_ind), bl_type(bl_ind), limit(p_ind:p_ind+1), send_buffer)
            ! Second particle is remeshed with left formula
            call AC_remesh_limitO2(pos_translat(p_ind+1),scal1D(p_ind+1), bl_type(bl_ind+1), limit(p_ind+1:p_ind+2), send_buffer)
        end if
    end do

end subroutine AC_remesh_lambda2limited_pter


!> Remesh particle line with corrected lambda 2 formula - remeshing is done into
!! an real array - no communication variant (buffer does not have the same size)
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        scal1D      = scalar field to advect
!!    @param[in]        bl_type     = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]        bl_tag      = contains information about bloc (is it tagged ?)
!!    @param[in]        limit       = limitator function value associated to the right and the left scalar variations
!!    @param[in, out]   remesh_buffer= buffer use to remesh the scalar
!! @details
!!     Use corrected lambda 2 remeshing formula.
!! This remeshing formula depends on the particle type :
!!     1 - Is the particle tagged ?
!!     2 - Does it belong to a centered or a left block ?
!! Observe that tagged particles go by group of two : if the particles of a
!! block end are tagged, the one first one of the following block are
!! tagged too.
!! The following algorithm is write for block of minimal size.
!!    Note that instead of the value of the limitator function, it is actually
!! these values divided by 8 wich are given as arguments. As the limitator function
!! always appear divided by 8 in the remeshing polynom, perform this division
!! during the computation of the limitator function enhances the performances.
!! @author = Jean-Baptiste Lagaert, LEGI/Ljk
subroutine AC_remesh_lambda2limited_array(direction, p_pos_adim, scal1d, bl_type, bl_tag, limit, remesh_buffer)

    use cart_topology   ! description of mesh and of mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! input/output
    integer, intent(in)                                 :: direction
    real(wp), dimension(:), intent(in)                  :: p_pos_adim
    real(wp), dimension(:), intent(in)                  :: scal1d
    real(WP), dimension(:), intent(in)                  :: limit
    logical, dimension(:), intent(in)                   :: bl_type
    logical, dimension(:), intent(in)                   :: bl_tag
    real(wp), dimension(:), intent(inout)               :: remesh_buffer
    ! Other local variables
    integer     :: bl_ind                               ! indice of the current "block end".
    integer     :: p_ind                                ! indice of the current particle

    do p_ind = 1, mesh_sc%N_proc(direction), bl_size
        bl_ind = p_ind/bl_size + 1
        if (bl_tag(bl_ind)) then
            ! Tag case
                ! XXX Debug : to activate only in purpose debug
                !if (bl_type(ind).neqv. (.not. bl_type(ind+1))) then
                !    write(*,'(a,x,3(L1,x),a,3(i0,a))'), 'error on remeshing particles: (tag,type(i), type(i+1)) =', &
                !    & bl_tag(ind), bl_type(ind), bl_type(ind+1), ' and type must be different. Mesh point = (',i, ', ', j,', ',k,')'
                !    write(*,'(a,x,i0)'),  'paramètres du blocs : ind =', bl_ind
                !    stop
                !end if
                ! XXX Debug - end
            if (bl_type(bl_ind)) then
                ! tagged, the first particle belong to a centered block and the last to left block.
                call AC_remesh_limitO2_tag_CL(direction, p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), &
                        & scal1D(p_ind+1), limit(p_ind:p_ind+2), remesh_buffer)
            else
                ! tagged, the first particle belong to a left block and the last to centered block.
                call AC_remesh_limitO2_tag_LC(direction, p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), &
                        & scal1D(p_ind+1), limit(p_ind:p_ind+2), remesh_buffer)
            end if
        else
            ! First particle
            call AC_remesh_limitO2(direction, p_pos_adim(p_ind),scal1D(p_ind), bl_type(bl_ind), limit(p_ind:p_ind+1), remesh_buffer)
            ! Second particle is remeshed with left formula
            call AC_remesh_limitO2(direction, p_pos_adim(p_ind+1),scal1D(p_ind+1), bl_type(bl_ind+1), limit(p_ind+1:p_ind+2), remesh_buffer)
        end if
    end do

end subroutine AC_remesh_lambda2limited_array


! #########################################################################
! ############                                                  ###########
! ############     Interpolation polynom used for remeshing     ###########
! ############                                                  ###########
! #########################################################################

! ============================================================
! ============     Lambda 2 corrected formula     ============
! ============================================================

!> (center or left) lambda remeshing formula of order 2 - version for classical array
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in]       bl_type = equal 0 (resp 1) if the block is left (resp centered)
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_O2_array(dir, pos_adim, sca, bl_type, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                    :: pos_adim, sca
    integer, intent(in)                     :: dir
    logical, intent(in)                     :: bl_type
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Ohter local variables
    integer     :: j0, j1                   ! indice of the the nearest mesh points
    real(WP)    :: bM, b0, bP               ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    if (bl_type) then
        ! Center remeshing
        j0 = nint(pos_adim)
    else
        ! Left remeshing
        j0 = floor(pos_adim)
    end if

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    bM=0.5*y0*(y0-1.)
    b0=1.-y0**2
    !bP=0.5*y0*(y0+1.)
    bP=1. - (b0+bM)

    ! remeshing
    j1 = modulo(j0-2,mesh_sc%N(dir))+1 ! j0-1
    buffer(j1) = buffer(j1) + bM*sca
    j1 = modulo(j0-1,mesh_sc%N(dir))+1 ! j0
    buffer(j1) = buffer(j1) + b0*sca
    j1 = modulo(j0,mesh_sc%N(dir))+1   ! j0+1
    buffer(j1) = buffer(j1) + bP*sca

end subroutine AC_remesh_O2_array


!> (center or left) lambda remeshing formula of order 2 - version for array of pointer
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in]       bl_type = equal 0 (resp 1) if the block is left (resp centered)
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_O2_pter(pos_adim, sca, bl_type, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                :: pos_adim, sca
    logical, intent(in)                                 :: bl_type
    type(real_pter), dimension(:), intent(inout)        :: buffer
    ! Ohter local variables
    integer     :: j0                       ! indice of the the nearest mesh points
    real(WP)    :: bM, b0, bP               ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    if (bl_type) then
        ! Center remeshing
        j0 = nint(pos_adim)
    else
        ! Left remeshing
        j0 = floor(pos_adim)
    end if

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    bM=0.5*y0*(y0-1.)
    b0=1.-y0**2
    !bP=0.5*y0*(y0+1.)
    bP=1. - (b0+bM)

    ! remeshing
    buffer(j0-1)%pter = buffer(j0-1)%pter + bM*sca
    buffer(j0)%pter   = buffer(j0)%pter   + b0*sca
    buffer(j0+1)%pter = buffer(j0+1)%pter + bP*sca

end subroutine AC_remesh_O2_pter


!> Corrected remeshing formula for transition from Centered block to a Left block with a different indice (tagged particles)
!!    @param[in]       dir     = current direction
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @detail
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a centered block
!!    and the last to a left one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_tag_CL_array(dir, pos_adim, sca, posP_ad, scaP, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca, posP_ad, scaP
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: jM, j0, jP               ! indice of the the nearest mesh points
                                            ! (they depend on the block type)
    integer     :: j0_bis                   ! indice of the the nearest mesh point for the indP=ind+1 particle
    real(WP)    :: aM, a0, bP, b0           ! interpolation weight for the particles
    real(WP)    :: y0, y0_bis               ! adimensionned distance to mesh points

    j0 = nint(pos_adim)
    !j0 = nint(pos/d_sc(2))
    j0_bis = floor(posP_ad)
    !j0_bis = floor(posP/d_sc(2))

    y0 = (pos_adim - real(j0, WP))
    !y0 = (pos - real(j0, WP)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - real(j0_bis, WP)*d_sc(2))/d_sc(2)

    aM=0.5*y0*(y0-1)
    a0=1.-aM
    bP=0.5*y0_bis*(y0_bis+1.)
    b0=1.-bP

    ! Remeshing
    jM = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    jP = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    j0 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(jM)=buffer(jM)+aM*sca
    buffer(j0)=buffer(j0)+a0*sca+b0*scaP
    buffer(jP)=buffer(jP)+bP*scaP

end subroutine AC_remesh_tag_CL_array


!> Corrected remeshing formula for transition from Centered block to a Left block with a different indice (tagged particles)
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a centered block
!!    and the last to a left one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_tag_CL_pter(pos_adim, sca, posP_ad, scaP, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                            :: pos_adim, sca, posP_ad, scaP
    type(real_pter), dimension(:), intent(inout)    :: buffer
    ! Other local variables
    integer     :: jM, j0, jP               ! indice of the the nearest mesh points
                                            ! (they depend on the block type)
    integer     :: j0_bis                   ! indice of the the nearest mesh point for the indP=ind+1 particle
    real(WP)    :: aM, a0, bP, b0           ! interpolation weight for the particles
    real(WP)    :: y0, y0_bis               ! adimensionned distance to mesh points

    j0 = nint(pos_adim)
    !j0 = nint(pos/d_sc(2))
    j0_bis = floor(posP_ad)
    !j0_bis = floor(posP/d_sc(2))
    jM=j0-1
    jP=j0+1

    y0 = (pos_adim - real(j0, WP))
    !y0 = (pos - real(j0, WP)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - real(j0_bis, WP)*d_sc(2))/d_sc(2)

    aM=0.5*y0*(y0-1)
    a0=1.-aM
    bP=0.5*y0_bis*(y0_bis+1.)
    b0=1.-bP

    ! Remeshing
    buffer(jM)%pter=buffer(jM)%pter+aM*sca
    buffer(j0)%pter=buffer(j0)%pter+a0*sca+b0*scaP
    buffer(jP)%pter=buffer(jP)%pter+bP*scaP

end subroutine AC_remesh_tag_CL_pter


!> Corrected remeshing formula for transition from Left block to a Centered  block with a different indice (tagged particles)
!!    @param[in]       dir     = current direction
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a left block
!!    and the last to a centered one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_tag_LC_array(dir, pos_adim, sca, posP_ad, scaP, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca, posP_ad, scaP
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: jM, j0, jP, jP2, jP3             ! indice of the the nearest mesh points
                                                    ! (they depend on the block type)
    integer     :: j0_bis                           ! indice of the the nearest mesh point for the indP=ind+1 particle
    real(WP)    :: aM, a0, aP,aP2, b0, bP, bP2, bP3 ! interpolation weight for the particles
    real(WP)    :: y0, y0_bis                       ! adimensionned distance to mesh points


    ! Indice of mesh point used in order to remesh
    j0 = floor(pos_adim)
    !j0 = floor(pos/d_sc(2))
    j0_bis = nint(posP_ad)
    !j0_bis = nint(posP/d_sc(2))
    jM=j0-1
    jP=j0+1
    jP2=j0+2
    jP3=j0+3

    ! Distance to mesh point
    y0 = (pos_adim - real(j0, WP))
    !y0 = (pos - real(j0, WP)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - real(j0_bis, WP)*d_sc(2))/d_sc(2)

    ! Interpolation weight
    a0=1-y0**2
    aP=y0
    !aM=y0*yM/2.
    aM = 0.5-(a0+aP)/2.
    aP2=aM
    bP=-y0_bis
    bP2=1-y0_bis**2
    !b0=y0_bis*yP_bis/2.
    b0 = 0.5-(bP+bP2)/2.
    bP3=b0

    ! Remeshing
    jM = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    jP = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    jP2= modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    jP3= modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    j0 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(jM)= buffer(jM)+aM*sca
    buffer(j0)= buffer(j0)+a0*sca+b0*scaP
    buffer(jP)= buffer(jP)+aP*sca+bP*scaP
    buffer(jP2)=buffer(jP2)+aP2*sca+bP2*scaP
    buffer(jP3)=buffer(jP3)+bP3*scaP

end subroutine AC_remesh_tag_LC_array


!> Corrected remeshing formula for transition from Left block to a Centered  block with a different indice (tagged particles)
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a left block
!!    and the last to a centered one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_tag_LC_pter(pos_adim, sca, posP_ad, scaP, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                            :: pos_adim, sca, posP_ad, scaP
    type(real_pter), dimension(:), intent(inout)    :: buffer
    ! Other local variables
    integer     :: jM, j0, jP, jP2, jP3             ! indice of the the nearest mesh points
                                                    ! (they depend on the block type)
    integer     :: j0_bis                           ! indice of the the nearest mesh point for the indP=ind+1 particle
    real(WP)    :: aM, a0, aP,aP2, b0, bP, bP2, bP3 ! interpolation weight for the particles
    real(WP)    :: y0, y0_bis                       ! adimensionned distance to mesh points


    ! Indice of mesh point used in order to remesh
    j0 = floor(pos_adim)
    !j0 = floor(pos/d_sc(2))
    j0_bis = nint(posP_ad)
    !j0_bis = nint(posP/d_sc(2))
    jM=j0-1
    jP=j0+1
    jP2=j0+2
    jP3=j0+3

    ! Distance to mesh point
    y0 = (pos_adim - real(j0, WP))
    !y0 = (pos - real(j0, WP)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - real(j0_bis, WP)*d_sc(2))/d_sc(2)

    ! Interpolation weight
    a0=1-y0**2
    aP=y0
    !aM=y0*yM/2.
    aM = 0.5-(a0+aP)/2.
    aP2=aM
    bP=-y0_bis
    bP2=1-y0_bis**2
    !b0=y0_bis*yP_bis/2.
    b0 = 0.5-(bP+bP2)/2.
    bP3=b0

    ! Remeshing
    buffer(jM)%pter= buffer(jM)%pter+aM*sca
    buffer(j0)%pter= buffer(j0)%pter+a0*sca+b0*scaP
    buffer(jP)%pter= buffer(jP)%pter+aP*sca+bP*scaP
    buffer(jP2)%pter=buffer(jP2)%pter+aP2*sca+bP2*scaP
    buffer(jP3)%pter=buffer(jP3)%pter+bP3*scaP

end subroutine AC_remesh_tag_LC_pter


! ========================================================================
! ============     Lambda 2 corrected and limited formula     ============
! ========================================================================

!> (center or left) remeshing formula for lambda 2 corrected and limited - classical array
!! version
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in]       bl_type = equal 0 (resp 1) if the block is left (resp centered)
!!      @param[in]       limit   = limitator function value associated to the right and the left scalar variations
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Note that instead of the value of the limitator funciton, it is actually
!! these values divided by 8 wich are given as arguments. As the limitator function
!! always appear divided by 8 in the remeshing polynom, perform this division
!! during the computation of the limitator function enhances the performances.
subroutine AC_remesh_limitO2_array(dir, pos_adim, sca, bl_type, limit, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                    :: pos_adim, sca
    logical, intent(in)                     :: bl_type
    real(WP), dimension(2), intent(in)      :: limit
    integer, intent(in)                     :: dir
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Ohter local variables
    integer     :: j0, j1                   ! indice of the the nearest mesh points
    real(WP)    :: bM, b0, bP               ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    if (bl_type) then
        ! Center remeshing
        j0 = nint(pos_adim)
    else
        ! Left remeshing
        j0 = floor(pos_adim)
    end if

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    bM=0.5*((y0-0.5)**2) - limit(1)
    b0=0.75_WP - y0**2 + limit(1) + limit(2)
    !bP=0.5*((y0+0.5)**2) - limit(2)
    bP=1. - (b0+bM)

    ! remeshing
    j1 = modulo(j0-2,mesh_sc%N(dir))+1 ! j0-1
    buffer(j1) = buffer(j1) + bM*sca
    j1 = modulo(j0-1,mesh_sc%N(dir))+1 ! j0
    buffer(j1) = buffer(j1) + b0*sca
    j1 = modulo(j0,mesh_sc%N(dir))+1   ! j0+1
    buffer(j1) = buffer(j1) + bP*sca

end subroutine AC_remesh_limitO2_array


!> Left remeshing formula for lambda 2 corrected and limited - array of pointer
!! version
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in]       bl_type = equal 0 (resp 1) if the block is left (resp centered)
!!      @param[in]       limit   = limitator function value associated to the right and the left scalar variations
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Note that instead of the value of the limitator funciton, it is actually
!! these values divided by 8 wich are given as arguments. As the limitator function
!! always appear divided by 8 in the remeshing polynom, perform this division
!! during the computation of the limitator function enhances the performances.
subroutine AC_remesh_limitO2_pter(pos_adim, sca, bl_type, limit, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                            :: pos_adim, sca
    logical, intent(in)                             :: bl_type
    real(WP), dimension(2), intent(in)              :: limit
    type(real_pter), dimension(:), intent(inout)    :: buffer

    ! Ohter local variables
    integer     :: j0                       ! indice of the the nearest mesh points
    real(WP)    :: bM, b0, bP               ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    if (bl_type) then
        ! Center remeshing
        j0 = nint(pos_adim)
    else
        ! Left remeshing
        j0 = floor(pos_adim)
    end if

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    bM=0.5*((y0-0.5)**2) - limit(1)
    b0=0.75_WP - y0**2 + limit(1) + limit(2)
    !bP=0.5*((y0+0.5)**2) - limit(2)
    bP=1. - (b0+bM)

    ! remeshing
    buffer(j0-1)%pter = buffer(j0-1)%pter + bM*sca
    buffer(j0)%pter   = buffer(j0)%pter   + b0*sca
    buffer(j0+1)%pter = buffer(j0+1)%pter + bP*sca

end subroutine AC_remesh_limitO2_pter


!> Corrected remeshing formula for transition from Centered block to a Left block with a different indice (tagged particles)
!!    @param[in]       dir     = current direction
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in]       limit   = limitator function value associated to the right and the left scalar variations
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @detail
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a centered block
!!    and the last to a left one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_limitO2_tag_CL_array(dir, pos_adim, sca, posP_ad, scaP, limit, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca, posP_ad, scaP
    real(WP), dimension(3), intent(in)      :: limit    ! to remesh particles of indices i, i+1, limitator must be known at i-1/2, i+1/2=(i+1)-1/2 and (i+1)+1/2
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: jM, j0, jP               ! indice of the the nearest mesh points
                                            ! (they depend on the block type)
    integer     :: j0_bis                   ! indice of the the nearest mesh point for the indP=ind+1 particle
    real(WP)    :: aM, a0, bP, b0           ! interpolation weight for the particles
    real(WP)    :: y0, y0_bis               ! adimensionned distance to mesh points

    j0 = nint(pos_adim)
    !j0 = nint(pos/d_sc(2))
    j0_bis = floor(posP_ad)
    !j0_bis = floor(posP/d_sc(2))

    y0 = (pos_adim - real(j0, WP))
    !y0 = (pos - real(j0, WP)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - real(j0_bis, WP)*d_sc(2))/d_sc(2)

    aM=0.5*((y0-0.5)**2) - limit(1)  ! = (lambda 2 limited) alpha(y0_bis)
    a0=1.-aM
    bP=0.5*((y0_bis+0.5)**2) - limit(3)  ! = (lambda 2 limited) gamma(y0_bis)
    ! note that limit(3) is the limitator function at (i+1)+1/2), with (i+1) the
    ! indice of particle of wieght scaP
    b0=1.-bP

    ! Remeshing
    jM = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    jP = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    j0 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(jM)=buffer(jM)+aM*sca
    buffer(j0)=buffer(j0)+a0*sca+b0*scaP
    buffer(jP)=buffer(jP)+bP*scaP

end subroutine AC_remesh_limitO2_tag_CL_array


!> Corrected remeshing formula for transition from Centered block to a Left block with a different indice (tagged particles)
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in]       limit   = limitator function value associated to the right and the left scalar variations
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a centered block
!!    and the last to a left one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_limitO2_tag_CL_pter(pos_adim, sca, posP_ad, scaP, limit, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                            :: pos_adim, sca, posP_ad, scaP
    real(WP), dimension(3), intent(in)              :: limit    ! to remesh particles of indices i, i+1, limitator must be known at i-1/2, i+1/2=(i+1)-1/2 and (i+1)+1/2
    type(real_pter), dimension(:), intent(inout)    :: buffer
    ! Other local variables
    integer     :: jM, j0, jP               ! indice of the the nearest mesh points
                                            ! (they depend on the block type)
    integer     :: j0_bis                   ! indice of the the nearest mesh point for the indP=ind+1 particle
    real(WP)    :: aM, a0, bP, b0           ! interpolation weight for the particles
    real(WP)    :: y0, y0_bis               ! adimensionned distance to mesh points

    j0 = nint(pos_adim)
    !j0 = nint(pos/d_sc(2))
    j0_bis = floor(posP_ad)
    !j0_bis = floor(posP/d_sc(2))
    jM=j0-1
    jP=j0+1

    y0 = (pos_adim - real(j0, WP))
    !y0 = (pos - real(j0, WP)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - real(j0_bis, WP)*d_sc(2))/d_sc(2)

    aM=0.5*((y0-1.)**2) - limit(1)  ! = (lambda 2 limited) alpha(y0_bis)
    a0=1.-aM
    bP=0.5*((y0_bis+1.)**2) - limit(3)  ! = (lambda 2 limited) gamma(y0_bis)
    ! note that limit(3) is the limitator function at (i+1)+1/2), with (i+1) the
    ! indice of particle of wieght scaP
    b0=1.-bP

    ! Remeshing
    buffer(jM)%pter=buffer(jM)%pter+aM*sca
    buffer(j0)%pter=buffer(j0)%pter+a0*sca+b0*scaP
    buffer(jP)%pter=buffer(jP)%pter+bP*scaP

end subroutine AC_remesh_limitO2_tag_CL_pter


!> Corrected remeshing formula for transition from Left block to a Centered  block with a different indice (tagged particles)
!!    @param[in]       dir     = current direction
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in]       limit   = limitator function value associated to the right and the left scalar variations
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a left block
!!    and the last to a centered one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_limitO2_tag_LC_array(dir, pos_adim, sca, posP_ad, scaP, limit, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca, posP_ad, scaP
    real(WP), dimension(3), intent(in)      :: limit    ! to remesh particles of indices i, i+1, limitator must be known at i-1/2, i+1/2=(i+1)-1/2 and (i+1)+1/2
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: jM, j0, jP, jP2, jP3             ! indice of the the nearest mesh points
                                                    ! (they depend on the block type)
    integer     :: j0_bis                           ! indice of the the nearest mesh point for the indP=ind+1 particle
    real(WP)    :: aM, a0, aP,aP2, b0, bP, bP2, bP3 ! interpolation weight for the particles
    real(WP)    :: y0, y0_bis                       ! adimensionned distance to mesh points


    ! Indice of mesh point used in order to remesh
    j0 = floor(pos_adim)
    !j0 = floor(pos/d_sc(2))
    j0_bis = nint(posP_ad)
    !j0_bis = nint(posP/d_sc(2))
    jM=j0-1
    jP=j0+1
    jP2=j0+2
    jP3=j0+3

    ! Distance to mesh point
    y0 = (pos_adim - real(j0, WP))
    !y0 = (pos - real(j0, WP)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - real(j0_bis, WP)*d_sc(2))/d_sc(2)

    ! Interpolation weight
    ! Use limit(1) and limit(2) to remesh particle i (they are limitator at i-1/2, i+1/2)
    aM = 0.5*((y0-0.5)**2) - limit(1)
    a0=0.75_WP - y0**2 + limit(1) + limit(2)
    aP=y0
    aP2=1._WP - aM - a0 - aP

    ! Use limit(2) and limit(3) to remesh particle i+1 (they are limitator at i+1-1/2, i+1+1/2)
    bP  = -y0_bis
    bP2 = 0.75_WP - y0_bis**2 + limit(2) + limit(3)
    bP3 = 0.5*((y0_bis+0.5)**2) - limit(3)
    b0 = 1._WP - bP - bP2 - bP3

    ! Remeshing
    jM = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    jP = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    jP2= modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    jP3= modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    j0 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(jM)= buffer(jM)  +aM *sca
    buffer(j0)= buffer(j0)  +a0 *sca+b0 *scaP
    buffer(jP)= buffer(jP)  +aP *sca+bP *scaP
    buffer(jP2)=buffer(jP2) +aP2*sca+bP2*scaP
    buffer(jP3)=buffer(jP3)         +bP3*scaP

end subroutine AC_remesh_limitO2_tag_LC_array


!> Corrected remeshing formula for transition from Left block to a Centered  block with a different indice (tagged particles)
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in]       limit   = limitator function value associated to the right and the left scalar variations
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a left block
!!    and the last to a centered one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_limitO2_tag_LC_pter(pos_adim, sca, posP_ad, scaP, limit, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                            :: pos_adim, sca, posP_ad, scaP
    real(WP), dimension(3), intent(in)              :: limit    ! to remesh particles of indices i, i+1, limitator must be known at i-1/2, i+1/2=(i+1)-1/2 and (i+1)+1/2
    type(real_pter), dimension(:), intent(inout)    :: buffer
    ! Other local variables
    integer     :: jM, j0, jP, jP2, jP3             ! indice of the the nearest mesh points
                                                    ! (they depend on the block type)
    integer     :: j0_bis                           ! indice of the the nearest mesh point for the indP=ind+1 particle
    real(WP)    :: aM, a0, aP,aP2, b0, bP, bP2, bP3 ! interpolation weight for the particles
    real(WP)    :: y0, y0_bis                       ! adimensionned distance to mesh points


    ! Indice of mesh point used in order to remesh
    j0 = floor(pos_adim)
    !j0 = floor(pos/d_sc(2))
    j0_bis = nint(posP_ad)
    !j0_bis = nint(posP/d_sc(2))
    jM=j0-1
    jP=j0+1
    jP2=j0+2
    jP3=j0+3

    ! Distance to mesh point
    y0 = (pos_adim - real(j0, WP))
    !y0 = (pos - real(j0, WP)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - real(j0_bis, WP)*d_sc(2))/d_sc(2)

    ! Interpolation weight
    ! Use limit(1) and limit(2) to remesh particle i (they are limitator at i-1/2, i+1/2)
    aM = 0.5*((y0-0.5)**2) - limit(1)
    a0=0.75_WP - y0**2 + limit(1) + limit(2)
    aP=y0
    aP2=1._WP - aM - a0 - aP

    ! Use limit(2) and limit(3) to remesh particle i+1 (they are limitator at i+1-1/2, i+1+1/2)
    bP  = -y0_bis
    bP2 = 0.75_WP - y0_bis**2 + limit(2) + limit(3)
    bP3 = 0.5*((y0_bis+0.5)**2) - limit(3)
    b0 = 1._WP - bP - bP2 - bP3

    ! Remeshing
    buffer(jM)%pter= buffer(jM)%pter+aM*sca
    buffer(j0)%pter= buffer(j0)%pter+a0*sca+b0*scaP
    buffer(jP)%pter= buffer(jP)%pter+aP*sca+bP*scaP
    buffer(jP2)%pter=buffer(jP2)%pter+aP2*sca+bP2*scaP
    buffer(jP3)%pter=buffer(jP3)%pter+bP3*scaP

end subroutine AC_remesh_limitO2_tag_LC_pter


! ============================================================
! ============     Lambda 4 corrected formula     ============
! ============================================================

!> Left remeshing formula of order 4
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_O4_left_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Ohter local variables
    integer     :: j0, j1                   ! indice of the the nearest mesh points
    real(WP)    :: bM2, bM, b0, bP, bP2     ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points
    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)
    !j0 = floor(pos/d_sc(2))

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    !bM2=(y0-2.)*(y0-1.)*y0*(y0+1.)/24.0
    bM2=y0*(2.+y0*(-1.+y0*(-2.+y0)))/24.0
    !bM =(2.-y0)*(y0-1.)*y0*(y0+2.)/6.0
    bM =y0*(-4.+y0*(4.+y0*(1.-y0)))/6.0
    !bP =(2.-y0)*y0*(y0+1.)*(y0+2.)/6.0
    bP =y0*(4+y0*(4-y0*(1.+y0)))/6.0
    !bP2=(y0-1.)*y0*(y0+1.)*(y0+2.)/24.0
    bP2=y0*(-2.+y0*(-1.+y0*(2.+y0)))/24.0
    !b0 =(y0-2.)*(y0-1.)*(y0+1.)*(y0+2.)/4.0
    b0 = 1. -(bM2+bM+bP+bP2)

    ! remeshing
    j1 = modulo(j0-3,mesh_sc%N(dir))+1  ! j0-2
    buffer(j1) = buffer(j1) + bM2*sca
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + bM*sca
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + b0*sca
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + bP*sca
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + bP2*sca

end subroutine AC_remesh_O4_left_array

!> Left remeshing formula of order 4 - surcharge for pointer
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_O4_left_pter(pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)        :: buffer
    ! Ohter local variables
    integer     :: j0                       ! indice of the the nearest mesh points
    real(WP)    :: bM2, bM, b0, bP, bP2     ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points
    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)
    !j0 = floor(pos/d_sc(2))

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    !bM2=(y0-2.)*(y0-1.)*y0*(y0+1.)/24.0
    bM2=y0*(2.0_WP+y0*(-1.0_WP+y0*(-2.0_WP+y0)))/24.0_WP
    !bM =(2.-y0)*(y0-1.)*y0*(y0+2.)/6.0
    bM =y0*(-4.0_WP+y0*(4.+y0*(1.-y0)))/6.0_WP
    !bP =(2.-y0)*y0*(y0+1.)*(y0+2.)/6.0
    bP =y0*(4._WP+y0*(4._WP-y0*(1._WP+y0)))/6._WP
    !bP2=(y0-1.)*y0*(y0+1.)*(y0+2.)/24.0
    bP2=y0*(-2._WP+y0*(-1._WP+y0*(2._WP+y0)))/24._WP
    !b0 =(y0-2.)*(y0-1.)*(y0+1.)*(y0+2.)/4.0
    b0 = 1._WP -(bM2+bM+bP+bP2)

    ! remeshing
    buffer(j0-2)%pter = buffer(j0-2)%pter  + bM2*sca
    buffer(j0-1)%pter = buffer(j0-1)%pter  + bM*sca
    buffer(j0  )%pter = buffer(j0  )%pter  + b0*sca
    buffer(j0+1)%pter = buffer(j0+1)%pter  + bP*sca
    buffer(j0+2)%pter = buffer(j0+2)%pter  + bP2*sca

end subroutine AC_remesh_O4_left_pter

!> Centered remeshing formula of order 4 - array version
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_O4_center_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/output
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: j0,j1                    ! indice of the the nearest mesh points
    real(WP)    :: bM2, bM, b0, bP, bP2     ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points
    ! Mesh point used in remeshing formula
    j0 = nint(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    !bM2=(y0-2.)*(y0-1.)*y0*(y0+1.)/24.0
    bM2=y0*(2._WP+y0*(-1._WP+y0*(-2._WP+y0)))/24._WP
    !bM =(2.-y0)*(y0-1.)*y0*(y0+2.)/6.0
    bM =y0*(-4._WP+y0*(4._WP+y0*(1._WP-y0)))/6._WP
    !bP =(2.-y0)*y0*(y0+1.)*(y0+2.)/6.0
    bP =y0*(4._WP+y0*(4._WP-y0*(1._WP+y0)))/6._WP
    !bP2=(y0-1.)*y0*(y0+1.)*(y0+2.)/24.0
    bP2=y0*(-2._WP+y0*(-1._WP+y0*(2._WP+y0)))/24._WP
    !b0 =(y0-2.)*(y0-1.)*(y0+1.)*(y0+2.)/4.0
    b0 = 1._WP -(bM2+bM+bP+bP2)

    ! remeshing
    j1 = modulo(j0-3,mesh_sc%N(dir))+1  ! j0-2
    buffer(j1) = buffer(j1) + bM2*sca
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + bM*sca
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + b0*sca
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + bP*sca
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + bP2*sca

end subroutine AC_remesh_O4_center_array

!> Centered remeshing formula of order 4 - array version
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_O4_center_pter(pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/output
    real(WP), intent(in)                                        :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)                :: buffer
    ! Other local variables
    integer     :: j0                       ! indice of the the nearest mesh points
    real(WP)    :: bM2, bM, b0, bP, bP2     ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points
    ! Mesh point used in remeshing formula
    j0 = nint(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    !bM2=(y0-2.)*(y0-1.)*y0*(y0+1.)/24.0
    bM2=y0*(2._WP+y0*(-1._WP+y0*(-2._WP+y0)))/24._WP
    !bM =(2.-y0)*(y0-1.)*y0*(y0+2.)/6.0
    bM =y0*(-4._WP+y0*(4._WP+y0*(1._WP-y0)))/6._WP
    !bP =(2.-y0)*y0*(y0+1.)*(y0+2.)/6.0
    bP =y0*(4._WP+y0*(4._WP-y0*(1._WP+y0)))/6._WP
    !bP2=(y0-1.)*y0*(y0+1.)*(y0+2.)/24.0
    bP2=y0*(-2._WP+y0*(-1._WP+y0*(2.+y0)))/24._WP
    !b0 =(y0-2.)*(y0-1.)*(y0+1.)*(y0+2.)/4.0
    b0 = 1._WP -(bM2+bM+bP+bP2)

    ! remeshing
    buffer(j0-2)%pter = buffer(j0-2)%pter   + bM2*sca
    buffer(j0-1)%pter = buffer(j0-1)%pter   + bM*sca
    buffer(j0  )%pter = buffer(j0  )%pter   + b0*sca
    buffer(j0+1)%pter = buffer(j0+1)%pter   + bP*sca
    buffer(j0+2)%pter = buffer(j0+2)%pter   + bP2*sca


end subroutine AC_remesh_O4_center_pter


!> Order 4 corrected remeshing formula for transition from Centered block to a Left block with a different indice (tagged particles)
!! - version for array of real.
!!    @param[in]       dir     = current direction
!!    @param[in]       posM_ad = adimensionned position of the first particle
!!    @param[in]       scaM    = scalar advected by the first particle
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in]       posP2_ad= adimensionned position of the fourth (and last) particle
!!    @param[in]       scaP2   = scalar advected by this particle
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a centered block
!!    and the last to a left one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_O4_tag_CL_array(dir, posM_ad, scaM, pos_adim, sca, posP_ad, scaP, posP2_ad, scaP2, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca, posP_ad, scaP
    real(WP), intent(in)                    :: posM_ad, scaM, posP2_ad, scaP2
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: jM, j0, jP, jP2, j1      ! indice of the the nearest mesh points
                                            ! (they depend on the block type)
    real(WP)    :: aM3, aM2, aM, a0         ! interpolation weight for the particles
    real(WP)    :: bM2, bM, b0, bP          ! interpolation weight for the particles
    real(WP)    :: cM, c0, cP, cP2          ! interpolation weight for the particles
    real(WP)    :: e0, eP, eP2, eP3         ! interpolation weight for the particles
    real(WP)    :: yM, y0, yP, yP2          ! adimensionned distance to mesh points for each particles

    ! Indice of mesh point used in order to remesh
    jM = nint(posM_ad)
    j0 = nint(pos_adim)
    jP = floor(posP_ad)
    jP2= floor(posP2_ad)

    ! Distance to mesh point
    yM = (posM_ad  - real(jM, WP))
    y0 = (pos_adim - real(j0, WP))
    yP = (posP_ad  - real(jP, WP))
    yP2= (posP2_ad - real(jP2, WP))

    ! Interpolation weights
    !aM3=(yM-2.)*(yM-1.)*yM*(yM+1.)/24.0
    aM3=yM*(2.+yM*(-1.+yM*(-2.+yM)))/24.0
    !aM2=(2.-yM)*(yM-1.)*yM*(yM+2.)/6.0
    aM2=yM*(-4.+yM*(4.+yM*(1.-yM)))/6.0
    !aM =(yM-2.)*(yM-1.)*(yM+1.)*(yM+2.)/4.0
    aM =(4.+(yM**2)*(-5.+yM**2))/4.0
    !a0 =((2.-yM)*yM*(yM+1.)*(yM+2.)/6.0) + ((yM-1.)*yM*(yM+1.)*(yM+2.)/24.0)
    a0 = 1. - (aM3+aM2+aM)

    !bM2=(y0-2.)*(y0-1.)*y0*(y0+1.)/24.0
    bM2=y0*(2.+y0*(-1.+y0*(-2.+y0)))/24.0
    !bM =(2.-y0)*(y0-1.)*y0*(y0+2.)/6.0
    bM =y0*(-4.+y0*(4.+y0*(1.-y0)))/6.0
    !bP =((y0+1)-1.)*(y0+1)*((y0+1)+1.)*((y0+1)+2.)/24.0
    bP =y0*(6.+y0*(11+y0*(6+y0)))/24.0
    !b0 =((y0-2.)*(y0-1.)*(y0+1.)*(y0+2.)/4.0) + ((2.-y0)*y0*(y0+1.)*(y0+2.)/6.0) &
    !        & + ((y0-1.)*y0*(y0+1.)*(y0+2.)/24.0) - bP
    b0 = 1. - (bM2+bM+bP)

    !cM =((yP-1.)-2.)*((yP-1.)-1.)*(yP-1.)*((yP-1.)+1.)/24.0
    cM =yP*(-6.+yP*(11.+yP*(-6.+yP)))/24.0
    !cP =(2.-yP)*yP*(yP+1.)*(yP+2.)/6.0
    cP =yP*(4.+yP*(4.-yP*(1.+yP)))/6.0
    !cP2=(yP-1.)*yP*(yP+1.)*(yP+2.)/24.0
    cP2=yP*(-2.+yP*(-1.+yP*(2.+yP)))/24.0
    !c0 =((yP-2.)*(yP-1.)*yP*(yP+1.)/24.0)+((2.-yP)*(yP-1.)*yP*(yP+2.)/6.0) &
    !        & + ((yP-2.)*(yP-1.)*(yP+1.)*(yP+2.)/4.0) - cM
    c0 = 1. - (cM+cP+cP2)

    !eP =(yP2-2.)*(yP2-1.)*(yP2+1.)*(yP2+2.)/4.0
    eP =1.+((yP2**2)*(-5+yP2**2)/4.0)
    !eP2=(2.-yP2)*yP2*(yP2+1.)*(yP2+2.)/6.0
    eP2=yP2*(4.+yP2*(4.-yP2*(1+yP2)))/6.0
    !eP3=(yP2-1.)*yP2*(yP2+1.)*(yP2+2.)/24.0
    eP3=yP2*(-2.+yP2*(-1.+yP2*(2+yP2)))/24.0
    !e0 =((yP2-2.)*(yP2-1.)*yP2*(yP2+1.)/24.0) + ((2.-yP2)*(yP2-1.)*yP2*(yP2+2.)/6.0)
    e0 = 1. - (eP+eP2+eP3)

    ! -- remeshing --
    ! j0-3
    j1 = modulo(j0-4,mesh_sc%N(dir))+1
    buffer(j1) = buffer(j1) + aM3*scaM
    ! j0-2
    j1 = modulo(j0-3,mesh_sc%N(dir))+1
    buffer(j1) = buffer(j1) + aM2*scaM + bM2*sca
    ! j0-1
    j1 = modulo(j0-2,mesh_sc%N(dir))+1
    buffer(j1) = buffer(j1) + aM*scaM  + bM*sca   + cM*scaP
    ! j0
    j1 = modulo(j0-1,mesh_sc%N(dir))+1
    buffer(j1) = buffer(j1) + a0*scaM  + b0*sca   + c0*scaP  + e0*scaP2
    ! j0+1
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1)            + bP*sca   + cP*scaP  + eP*scaP2
    ! j0+2
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1)                       + cP2*scaP + ep2*scaP2
    ! j0+3
    j1 = modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    buffer(j1) = buffer(j1)                                  + ep3*scaP2

end subroutine AC_remesh_O4_tag_CL_array


!> Order 4 corrected remeshing formula for transition from Centered block to a Left block with a different indice (tagged particles)
!! - version for array of pointer.
!!    @param[in]       posM_ad = adimensionned position of the first particle
!!    @param[in]       scaM    = scalar advected by the first particle
!!    @param[in]       pos_adim= adimensionned particle position
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the second particle
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in]       posP2_ad= adimensionned position of the fourth (and last) particle
!!    @param[in]       scaP2   = scalar advected by this particle
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a centered block
!!    and the last to a left one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_O4_tag_CL_pter(posM_ad, scaM, pos_adim, sca, posP_ad, scaP, posP2_ad, scaP2, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                            :: pos_adim, sca, posP_ad, scaP
    real(WP), intent(in)                            :: posM_ad, scaM, posP2_ad, scaP2
    type(real_pter), dimension(:), intent(inout)    :: buffer
    ! Other local variables
    integer     :: jM, j0, jP, jP2          ! indice of the the nearest mesh points
                                            ! (they depend on the block type)
    real(WP)    :: aM3, aM2, aM, a0         ! interpolation weight for the particles
    real(WP)    :: bM2, bM, b0, bP          ! interpolation weight for the particles
    real(WP)    :: cM, c0, cP, cP2          ! interpolation weight for the particles
    real(WP)    :: e0, eP, eP2, eP3         ! interpolation weight for the particles
    real(WP)    :: yM, y0, yP, yP2          ! adimensionned distance to mesh points for each particles

    ! Indice of mesh point used in order to remesh
    jM = nint(posM_ad)
    j0 = nint(pos_adim)
    jP = floor(posP_ad)
    jP2= floor(posP2_ad)

    ! Distance to mesh point
    yM = (posM_ad  - real(jM, WP))
    y0 = (pos_adim - real(j0, WP))
    yP = (posP_ad  - real(jP, WP))
    yP2= (posP2_ad - real(jP2, WP))

    ! Interpolation weights
    !aM3=(yM-2.)*(yM-1.)*yM*(yM+1.)/24.0
    aM3=yM*(2._WP+yM*(-1._WP+yM*(-2._WP+yM)))/24._WP
    !aM2=(2.-yM)*(yM-1.)*yM*(yM+2.)/6.0
    aM2=yM*(-4._WP+yM*(4._WP+yM*(1._WP-yM)))/6._WP
    !aM =(yM-2.)*(yM-1.)*(yM+1.)*(yM+2.)/4.0
    aM =(4._WP+(yM**2._WP)*(-5._WP+yM**2))/4._WP
    !a0 =((2.-yM)*yM*(yM+1.)*(yM+2.)/6.0) + ((yM-1.)*yM*(yM+1.)*(yM+2.)/24.0)
    a0 = 1._WP - (aM3+aM2+aM)

    !bM2=(y0-2.)*(y0-1.)*y0*(y0+1.)/24.0
    bM2=y0*(2._WP+y0*(-1._WP+y0*(-2._WP+y0)))/24._WP
    !bM =(2.-y0)*(y0-1.)*y0*(y0+2.)/6.0
    bM =y0*(-4._WP+y0*(4._WP+y0*(1._WP-y0)))/6._WP
    !bP =((y0+1)-1.)*(y0+1)*((y0+1)+1.)*((y0+1)+2.)/24.0
    bP =y0*(6._WP+y0*(11._WP+y0*(6._WP+y0)))/24._WP
    !b0 =((y0-2.)*(y0-1.)*(y0+1.)*(y0+2.)/4.0) + ((2.-y0)*y0*(y0+1.)*(y0+2.)/6.0) &
    !        & + ((y0-1.)*y0*(y0+1.)*(y0+2.)/24.0) - bP
    b0 = 1._WP - (bM2+bM+bP)

    !cM =((yP-1.)-2.)*((yP-1.)-1.)*(yP-1.)*((yP-1.)+1.)/24.0
    cM =yP*(-6._WP+yP*(11._WP+yP*(-6._WP+yP)))/24._WP
    !cP =(2.-yP)*yP*(yP+1.)*(yP+2.)/6.0
    cP =yP*(4._WP+yP*(4._WP-yP*(1._WP+yP)))/6._WP
    !cP2=(yP-1.)*yP*(yP+1.)*(yP+2.)/24.0
    cP2=yP*(-2._WP+yP*(-1._WP+yP*(2._WP+yP)))/24._WP
    !c0 =((yP-2.)*(yP-1.)*yP*(yP+1.)/24.0)+((2.-yP)*(yP-1.)*yP*(yP+2.)/6.0) &
    !        & + ((yP-2.)*(yP-1.)*(yP+1.)*(yP+2.)/4.0) - cM
    c0 = 1._WP - (cM+cP+cP2)

    !eP =(yP2-2.)*(yP2-1.)*(yP2+1.)*(yP2+2.)/4.0
    eP =1._WP+((yP2**2)*(-5._WP+yP2**2)/4._WP)
    !eP2=(2.-yP2)*yP2*(yP2+1.)*(yP2+2.)/6.0
    eP2=yP2*(4._WP+yP2*(4._WP-yP2*(1._WP+yP2)))/6._WP
    !eP3=(yP2-1.)*yP2*(yP2+1.)*(yP2+2.)/24.0
    eP3=yP2*(-2._WP+yP2*(-1._WP+yP2*(2._WP+yP2)))/24._WP
    !e0 =((yP2-2.)*(yP2-1.)*yP2*(yP2+1.)/24.0) + ((2.-yP2)*(yP2-1.)*yP2*(yP2+2.)/6.0)
    e0 = 1._WP - (eP+eP2+eP3)

    ! remeshing
    buffer(j0-3)%pter = buffer(j0-3)%pter +aM3*scaM
    buffer(j0-2)%pter = buffer(j0-2)%pter +aM2*scaM +bM2*sca
    buffer(j0-1)%pter = buffer(j0-1)%pter + aM*scaM + bM*sca  + cM*scaP
    buffer(j0  )%pter = buffer(j0  )%pter + a0*scaM + b0*sca  + c0*scaP + e0*scaP2
    buffer(j0+1)%pter = buffer(j0+1)%pter           + bP*sca  + cP*scaP + eP*scaP2
    buffer(j0+2)%pter = buffer(j0+2)%pter                     +cP2*scaP +eP2*scaP2
    buffer(j0+3)%pter = buffer(j0+3)%pter                               +eP3*scaP2

end subroutine AC_remesh_O4_tag_CL_pter


!> Corrected remeshing formula of order 3 for transition from Left block to a centered
!! block with a different indice (tagged particles). Use it for lambda 4 corrected scheme.
!! - version for array of real.
!!    @param[in]       dir     = current direction
!!    @param[in]       posM_ad = adimensionned position of the first particle
!!    @param[in]       scaM    = scalar advected by the first particle
!!    @param[in]       pos_adim= adimensionned position of the second particle (the last of the first block)
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the third particle (wich is the first of the second block)
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in]       posP2_ad= adimensionned position of the fourth (and last) particle
!!    @param[in]       scaP2   = scalar advected by this particle
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a left block
!!    and the last to a centered one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_O4_tag_LC_array(dir, posM_ad, scaM, pos_adim, sca, posP_ad, scaP, posP2_ad, scaP2, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca, posP_ad, scaP
    real(WP), intent(in)                    :: posM_ad, scaM, posP2_ad, scaP2
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: jM, j0, jP, jP2          ! indice of the the nearest mesh points
                                            ! (they depend on the block type)
    integer     :: j1                       ! equal to previous j but with a modulo
    real(WP)    :: aM3, aM2, aM, a0, aP,aP2 ! interpolation weight for the particles
    real(WP)    :: bM2, bM, b0, bP, bP2,bP3 ! interpolation weight for the particles
    real(WP)    :: cM, c0, cP, cP2, cP3,cP4 ! interpolation weight for the particles
    real(WP)    :: e0, eP, eP2, eP3,eP4,ep5 ! interpolation weight for the particles
    real(WP)    :: yM, y0, yP, yP2          ! adimensionned distance to mesh points for each particles


    ! Indice of mesh point used in order to remesh
    jM = floor(posM_ad)
    j0 = floor(pos_adim)
    jP = nint(posP_ad)
    jP2= nint(posP2_ad)

    ! Distance to mesh point
    yM = (posM_ad  - real(jM, WP))
    y0 = (pos_adim - real(j0, WP))
    yP = (posP_ad  - real(jP, WP))
    yP2= (posP2_ad - real(jP2, WP))

    ! Interpolation weights
    !aM3=(yM-2.)*(yM-1.)*yM*(yM+1.)/24.0
    aM3=yM*(2.+yM*(-1.+yM*(-2.+yM)))/24.0
    !aM2=(2.-yM)*(yM-1.)*yM*(yM+2.)/6.0
    aM2 =yM*(-4.+yM*(4.+yM*(1.-yM)))/6.0
    !aM =(yM-2.)*(yM-1.)*(yM+1.)*(yM+2.)/4.0
    aM =(4.+(yM**2)*(-5.+yM**2))/4.0
    !a0 =((2.-yM)*yM*(yM+1.)*(yM+2.)/6.0)
    a0 =yM*(4+yM*(4-yM*(1.+yM)))/6.0
    !aP2=(((yM-1.)-1.)*(yM-1.)*((yM-1.)+1.)*((yM-1.)+2.)/24.0)
    !aP2=yM*(yM-2.)*(yM-1.)*(yM+1.)/24.0
    aP2=aM3
    !aP =((yM-1.)*yM*(yM+1.)*(yM+2.)/24.0) - aP2
    !aP = 1.0 - (aM3+aM2+aM+a0+aP2)
    aP = 1.0 - (2.*aM3+aM2+aM+a0)

    !bM2=(y0-2.)*(y0-1.)*y0*(y0+1.)/24.0
    bM2=y0*(2.+y0*(-1.+y0*(-2.+y0)))/24.0
    !bM =(2.-y0)*(y0-1.)*y0*(y0+2.)/6.0
    bM =y0*(-4.+y0*(4.+y0*(1.-y0)))/6.0
    !b0 =(y0-2.)*(y0-1.)*(y0+1.)*(y0+2.)/4.0
    b0 =(4.+(y0**2)*(-5.+y0**2))/4.0
    !bP2=(2.-(y0-1.))*(y0-1.)*((y0-1.)+1.)*((y0-1.)+2.)/6.0
    !bP2=y0*(3.-y0)*(y0-1.)*(y0+1.)/6.0
    bP2=y0*(-3.+y0*(1.+y0*(3.-y0)))/6.0
    !bP3=((y0-1.)-1.)*(y0-1.)*((y0-1.)+1.)*((y0-1.)+2.)/24.0
    !bP3=y0*(y0-2.)*(y0-1.)*(y0+1.)/24.0
    bP3 = bM2
    !bP =(2.-y0)*y0*(y0+1.)*(y0+2.)/6.0 + ((y0-1.)*y0*(y0+1.)*(y0+2.)/24.0) &
    !       & - (bP2 + bP3)
    !bP = 1.0 - (bM2 + bM + b0 + bP2 + bP3)
    bP = 1.0 - (2*bM2 + bM + b0 + bP2)

    !cM =((yP+1)-2.)*((yP+1)-1.)*(yP+1)*((yP+1)+1.)/24.0
    cM =(yP-1.)*yP*(yP+1)*(yP+2.)/24.0
    !cM =yP*(-2.+yP*(-1.+yP*(2.+yP)))/24.0
    !c0 =(2.-(yP+1))*((yP+1)-1.)*(yP+1)*((yP+1)+2.)/6.0
    !c0 =(1.-yP)*yP*(yP+1)*(yP+3.)/6.0
    c0 =yP*(3.+yP*(1.-yP*(3.+yP)))/6.0
    !cP2=(yP-2.)*(yP-1.)*(yP+1.)*(yP+2.)/4.0
    cP2=(4.+(yP**2)*(-5.+yP**2))/4.0
    !cP3=(2.-yP)*yP*(yP+1.)*(yP+2.)/6.0
    cP3=yP*(4+yP*(4-yP*(1.+yP)))/6.0
    !cP4=(yP-1.)*yP*(yP+1.)*(yP+2.)/24.0
    cP4=cM
    !cP =(yP-2.)*(yP-1.)*yP*(yP+1.)/24.0 + ((2.-yP)*(yP-1.)*yP*(yP+2.)/6.0) &
    !        & - (cM + c0)
    cP = 1.0 - (cM+c0+cP2+cP3+cP4)

    !e0 =((yP2+1)-2.)*((yP2+1)-1.)*(yP2+1)*((yP2+1)+1.)/24.0
    !e0 =(yP2-1.)*yP2*(yP2+1)*(yP2+2.)/24.0
    e0 =yP2*(-2.+yP2*(-1.+yP2*(2.+yP2)))/24.0
    !eP2=(2.-yP2)*(yP2-1.)*yP2*(yP2+2.)/6.0
    eP2=yP2*(-4.+yP2*(4.+yP2*(1.-yP2)))/6.0
    !eP3=(yP2-2.)*(yP2-1.)*(yP2+1.)*(yP2+2.)/4.0
    eP3=(4.+(yP2**2)*(-5.+yP2**2))/4.0
    !eP4=(2.-yP2)*yP2*(yP2+1.)*(yP2+2.)/6.0
    eP4=yP2*(4+yP2*(4-yP2*(1.+yP2)))/6.0
    !eP5=(yP2-1.)*yP2*(yP2+1.)*(yP2+2.)/24.0
    eP5=e0
    !eP =((yP2-2.)*(yP2-1.)*yP2*(yP2+1.)/24.0) - e0
    eP = 1.0 - (e0+eP2+eP3+eP4+eP5)

    ! remeshing
    ! j0-3
    j1 = modulo(j0-4,mesh_sc%N(dir))+1
    buffer(j1) = buffer(j1) + aM3*scaM
    ! j0-2
    j1 = modulo(j0-3,mesh_sc%N(dir))+1
    buffer(j1) = buffer(j1) + aM2*scaM + bM2*sca
    ! j0-1
    j1 = modulo(j0-2,mesh_sc%N(dir))+1
    buffer(j1) = buffer(j1) + aM*scaM  + bM*sca   + cM*scaP
    ! j0
    j1 = modulo(j0-1,mesh_sc%N(dir))+1
    buffer(j1) = buffer(j1) + a0*scaM  + b0*sca   + c0*scaP  + e0*scaP2
    ! j0+1
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + aP*scaM  + bP*sca   + cP*scaP  + eP*scaP2
    ! j0+2
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1)  + aP2*scaM + bP2*sca + cP2*scaP + ep2*scaP2
    ! j0+3
    j1 = modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    buffer(j1) = buffer(j1)             + bP3*sca + cP3*scaP + ep3*scaP2
    ! j0+3
    j1 = modulo(j0+3,mesh_sc%N(dir))+1  ! j0+3
    buffer(j1) = buffer(j1)                       + cP4*scaP + ep4*scaP2
    ! j0+3
    j1 = modulo(j0+4,mesh_sc%N(dir))+1  ! j0+5
    buffer(j1) = buffer(j1)                                  + ep5*scaP2

end subroutine AC_remesh_O4_tag_LC_array


!> Corrected remeshing formula of order 3 for transition from Left block to a centered
!! block with a different indice (tagged particles). Use it for lambda 4 corrected scheme.
!! - version for array of pointer.
!!    @param[in]       posM_ad = adimensionned position of the first particle
!!    @param[in]       scaM    = scalar advected by the first particle
!!    @param[in]       pos_adim= adimensionned position of the second particle (the last of the first block)
!!    @param[in]       sca     = scalar advected by this particle
!!    @param[in]       posP_ad = adimensionned position of the third particle (wich is the first of the second block)
!!    @param[in]       scaP    = scalar advected by this particle
!!    @param[in]       posP2_ad= adimensionned position of the fourth (and last) particle
!!    @param[in]       scaP2   = scalar advected by this particle
!!    @param[in,out]   buffer  = temporaly remeshed scalar field
!! @details
!!    Remeshing formula devoted to tagged particles.
!!    The particle group send into argument is composed of a block end and of the
!!    begining of the next block. The first particles belong to a left block
!!    and the last to a centered one. The block have difference indice (tagged
!!    particles) and we have to use corrected formula.
subroutine AC_remesh_O4_tag_LC_pter(posM_ad, scaM, pos_adim, sca, posP_ad, scaP, posP2_ad, scaP2, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                            :: pos_adim, sca, posP_ad, scaP
    real(WP), intent(in)                            :: posM_ad, scaM, posP2_ad, scaP2
    type(real_pter), dimension(:), intent(inout)    :: buffer
    ! Other local variables
    integer     :: jM, j0, jP, jP2          ! indice of the the nearest mesh points
                                            ! (they depend on the block type)
    real(WP)    :: aM3, aM2, aM, a0, aP,aP2 ! interpolation weight for the particles
    real(WP)    :: bM2, bM, b0, bP, bP2,bP3 ! interpolation weight for the particles
    real(WP)    :: cM, c0, cP, cP2, cP3,cP4 ! interpolation weight for the particles
    real(WP)    :: e0, eP, eP2, eP3,eP4,ep5 ! interpolation weight for the particles
    real(WP)    :: yM, y0, yP, yP2          ! adimensionned distance to mesh points for each particles


    ! Indice of mesh point used in order to remesh
    jM = floor(posM_ad)
    j0 = floor(pos_adim)
    jP = nint(posP_ad)
    jP2= nint(posP2_ad)

    ! Distance to mesh point
    yM = (posM_ad  - real(jM, WP))
    y0 = (pos_adim - real(j0, WP))
    yP = (posP_ad  - real(jP, WP))
    yP2= (posP2_ad - real(jP2, WP))

    ! Interpolation weights
    !aM3=(yM-2.)*(yM-1.)*yM*(yM+1.)/24.0
    aM3=yM*(2._WP+yM*(-1._WP+yM*(-2._WP+yM)))/24._WP
    !aM2=(2.-yM)*(yM-1.)*yM*(yM+2.)/6.0
    aM2 =yM*(-4._WP+yM*(4._WP+yM*(1._WP-yM)))/6._WP
    !aM =(yM-2.)*(yM-1.)*(yM+1.)*(yM+2.)/4.0
    aM =(4._WP+(yM**2)*(-5._WP+yM**2))/4._WP
    !a0 =((2.-yM)*yM*(yM+1.)*(yM+2.)/6.0)
    a0 =yM*(4._WP+yM*(4._WP-yM*(1._WP+yM)))/6._WP
    !aP2=(((yM-1.)-1.)*(yM-1.)*((yM-1.)+1.)*((yM-1.)+2.)/24.0)
    !aP2=yM*(yM-2.)*(yM-1.)*(yM+1.)/24.0
    aP2=aM3
    !aP =((yM-1.)*yM*(yM+1.)*(yM+2.)/24.0) - aP2
    !aP = 1.0 - (aM3+aM2+aM+a0+aP2)
    aP = 1._WP - (2._WP*aM3+aM2+aM+a0)

    !bM2=(y0-2.)*(y0-1.)*y0*(y0+1.)/24.0
    bM2=y0*(2._WP+y0*(-1._WP+y0*(-2._WP+y0)))/24._WP
    !bM =(2.-y0)*(y0-1.)*y0*(y0+2.)/6.0
    bM =y0*(-4._WP+y0*(4._WP+y0*(1._WP-y0)))/6._WP
    !b0 =(y0-2.)*(y0-1.)*(y0+1.)*(y0+2.)/4.0
    b0 =(4._WP+(y0**2)*(-5._WP+y0**2))/4._WP
    !bP2=(2.-(y0-1.))*(y0-1.)*((y0-1.)+1.)*((y0-1.)+2.)/6.0
    !bP2=y0*(3.-y0)*(y0-1.)*(y0+1.)/6.0
    bP2=y0*(-3._WP+y0*(1._WP+y0*(3._WP-y0)))/6._WP
    !bP3=((y0-1.)-1.)*(y0-1.)*((y0-1.)+1.)*((y0-1.)+2.)/24.0
    !bP3=y0*(y0-2.)*(y0-1.)*(y0+1.)/24.0
    bP3 = bM2
    !bP =(2.-y0)*y0*(y0+1.)*(y0+2.)/6.0 + ((y0-1.)*y0*(y0+1.)*(y0+2.)/24.0) &
    !       & - (bP2 + bP3)
    !bP = 1.0 - (bM2 + bM + b0 + bP2 + bP3)
    bP = 1._WP - (2._WP*bM2 + bM + b0 + bP2)

    !cM =((yP+1)-2.)*((yP+1)-1.)*(yP+1)*((yP+1)+1.)/24.0
    !cM =(yP-1._WP)*yP*(yP+1._WP)*(yP+2._WP)/24._WP
    cM =yP*(-2._WP+yP*(-1._WP+yP*(2._WP+yP)))/24._WP
    !c0 =(2.-(yP+1))*((yP+1)-1.)*(yP+1)*((yP+1)+2.)/6.0
    !c0 =(1.-yP)*yP*(yP+1)*(yP+3.)/6.0
    c0 =yP*(3._WP+yP*(1._WP-yP*(3._WP+yP)))/6._WP
    !cP2=(yP-2.)*(yP-1.)*(yP+1.)*(yP+2.)/4.0
    cP2=(4._WP+(yP**2)*(-5._WP+yP**2))/4._WP
    !cP3=(2.-yP)*yP*(yP+1.)*(yP+2.)/6.0
    cP3=yP*(4._WP+yP*(4._WP-yP*(1._WP+yP)))/6._WP
    !cP4=(yP-1.)*yP*(yP+1.)*(yP+2.)/24.0
    cP4=cM
    !cP =(yP-2.)*(yP-1.)*yP*(yP+1.)/24.0 + ((2.-yP)*(yP-1.)*yP*(yP+2.)/6.0) &
    !        & - (cM + c0)
    cP = 1._WP - (cM+c0+cP2+cP3+cP4)

    !e0 =((yP2+1)-2.)*((yP2+1)-1.)*(yP2+1)*((yP2+1)+1.)/24.0
    !e0 =(yP2-1.)*yP2*(yP2+1)*(yP2+2.)/24.0
    e0 =yP2*(-2._WP+yP2*(-1._WP+yP2*(2._WP+yP2)))/24._WP
    !eP2=(2.-yP2)*(yP2-1.)*yP2*(yP2+2.)/6.0
    eP2=yP2*(-4._WP+yP2*(4._WP+yP2*(1._WP-yP2)))/6._WP
    !eP3=(yP2-2.)*(yP2-1.)*(yP2+1.)*(yP2+2.)/4.0
    eP3=(4._WP+(yP2**2)*(-5._WP+yP2**2))/4._WP
    !eP4=(2.-yP2)*yP2*(yP2+1.)*(yP2+2.)/6.0
    eP4=yP2*(4._WP+yP2*(4._WP-yP2*(1._WP+yP2)))/6._WP
    !eP5=(yP2-1.)*yP2*(yP2+1.)*(yP2+2.)/24.0
    eP5=e0
    !eP =((yP2-2.)*(yP2-1.)*yP2*(yP2+1.)/24.0) - e0
    eP = 1._WP - (e0+eP2+eP3+eP4+eP5)

    ! remeshing
    buffer(j0-3)%pter = buffer(j0-3)%pter +aM3*scaM
    buffer(j0-2)%pter = buffer(j0-2)%pter +aM2*scaM +bM2*sca
    buffer(j0-1)%pter = buffer(j0-1)%pter + aM*scaM + bM*sca  + cM*scaP
    buffer(j0  )%pter = buffer(j0  )%pter + a0*scaM + b0*sca  + c0*scaP + e0*scaP2
    buffer(j0+1)%pter = buffer(j0+1)%pter + aP*scaM + bP*sca  + cP*scaP + eP*scaP2
    buffer(j0+2)%pter = buffer(j0+2)%pter +aP2*scaM +bP2*sca  +cP2*scaP +eP2*scaP2
    buffer(j0+3)%pter = buffer(j0+3)%pter           +bP3*sca  +cP3*scaP +eP3*scaP2
    buffer(j0+4)%pter = buffer(j0+4)%pter                     +cP4*scaP +eP4*scaP2
    buffer(j0+5)%pter = buffer(j0+5)%pter                               +eP5*scaP2

end subroutine AC_remesh_O4_tag_LC_pter


end module advec_remeshing_lambda
!> @}
