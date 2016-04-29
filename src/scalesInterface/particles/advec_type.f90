!USEFORTEST advec
!> @addtogroup part
!! @{
!------------------------------------------------------------------------------
!
! MODULE: advec_abstract_proc
!
!
! DESCRIPTION:
!> The module ``advec_abstract_procedure'' gather all user abstract procedure that are used by the different advection
!! modules. It allow to share that function/procediure profile and to safetly use procedural argument or pointer.
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

module advec_abstract_proc

    implicit none


    ! --- Abstract profile of subroutine used as wrapper for remeshing ---
    ! Such a procedure will call all the needed other subroutine to
    ! remesh in a buffer (procedure itself use a AC_remesh_line_pter subroutine)
    ! and to redristibute this buffer into the scalar field (and deal with
    ! all the communication)
    abstract interface
      subroutine AC_init_p_V(V_comp, j, k, Gsize, p_V)

        use precision_tools
        implicit none

        ! Input/Output
        integer, intent(in)                       :: j,k
        integer, dimension(2), intent(in)         :: Gsize
        real(WP), dimension(:,:,:),intent(out)    :: p_V
        real(WP), dimension(:,:,:), intent(in)    :: V_comp

      end subroutine AC_init_p_V
    end interface

    ! --- Abstract profile of subroutine used as wrapper for remeshing ---
    ! Such a procedure will call all the needed other subroutine to
    ! remesh in a buffer (procedure itself use a AC_remesh_line_pter subroutine)
    ! and to redristibute this buffer into the scalar field (and deal with
    ! all the communication)
    abstract interface
        subroutine AC_remesh(direction, ind_group, gs, p_pos_adim, p_V, j, k, scal, dt)

            use precision_tools
            implicit none

            ! Input/Output
            integer, intent(in)                         :: direction
            integer, dimension(2), intent(in)           :: ind_group
            integer, dimension(2), intent(in)           :: gs
            integer, intent(in)                         :: j, k
            real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
            real(WP), dimension(:,:,:), intent(in)      :: p_V          ! particles velocity
            real(WP), dimension(:,:,:), intent(inout)   :: scal
            real(WP), intent(in)                        :: dt

        end subroutine AC_remesh
    end interface


    ! --- Abstract profile of subroutine used to compute limitator function ---
    ! Note that such a function actually computes limitator/8 as it is always
    ! this fraction which appears in the remeshing polynoms (and thsu directly
    ! divided limitator function by 8 avoids to have to do it several times later)
    abstract interface
        !!    @param[in]        gp_s        = size of a group (ie number of line it gathers along the two other directions)
        !!    @param[in]        ind_group   = coordinate of the current group of lines
        !!    @param[in]        p_pos       = particles position
        !!    @param[in]        scalar      = scalar advected by particles
        !!    @param[out]       limit       = limitator function
        subroutine advec_limitator_group(gp_s, ind_group, j, k, p_pos, &
                & scalar, limit)

            use precision_tools
            implicit none

            integer, dimension(2),intent(in)                            :: gp_s         ! groupe size
            integer, dimension(2), intent(in)                           :: ind_group    ! group indice
            integer , intent(in)                                        :: j,k          ! bloc coordinates
            real(WP), dimension(:,:,:), intent(in)                      :: p_pos        ! particle position
            real(WP), dimension(:,:,:), intent(in)                      :: scalar       ! scalar field to advect
            real(WP), dimension(:,:,:), intent(out)                     :: limit        ! limitator function

        end subroutine advec_limitator_group
    end interface

    ! --- Abstract profile of subroutine used to remesh scalar inside a buffer - lambda formula ---
    abstract interface
        subroutine remesh_in_buffer_type(gs, j, k, ind_min, p_pos_adim, bl_type, bl_tag, send_min, send_max, &
        & scalar, buffer, pos_in_buffer)

            use precision_tools
            implicit none

            ! Input/Output
            integer, dimension(2), intent(in)           :: gs
            integer, intent(in)                         :: j, k
            integer, intent(in)                         :: ind_min
            real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
            logical, dimension(:,:,:), intent(in)       :: bl_type      ! is the particle block a center block or a left one ?
            logical, dimension(:,:,:), intent(in)       :: bl_tag       ! indice of tagged particles
            integer, dimension(:,:), intent(in)         :: send_min     ! distance between me and processus wich send me information
            integer, dimension(:,:), intent(in)         :: send_max     ! distance between me and processus wich send me information
            real(WP), dimension(:,:,:), intent(inout)   :: scalar       ! the initial scalar field transported by particles
            real(WP),dimension(:), intent(out), target  :: buffer       ! buffer where particles are remeshed
            integer, dimension(:), intent(inout)        :: pos_in_buffer! describe how the one dimensionnal array "buffer" are split
                                                                        ! in part corresponding to different processes

        end subroutine remesh_in_buffer_type
    end interface

    ! --- Abstract profile of subroutine used to remesh scalar inside a buffer - limited lambda formula ---
    abstract interface
        subroutine remesh_in_buffer_limit(gs, j, k, ind_min, p_pos_adim, bl_type, bl_tag, limit,&
            & send_min, send_max, scalar, buffer, pos_in_buffer)

            use precision_tools
            implicit none

            ! Input/Output
            integer, dimension(2), intent(in)           :: gs
            integer, intent(in)                         :: j, k
            integer, intent(in)                         :: ind_min
            real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
            logical, dimension(:,:,:), intent(in)       :: bl_type      ! is the particle block a center block or a left one ?
            logical, dimension(:,:,:), intent(in)       :: bl_tag       ! indice of tagged particles
            real(WP), dimension(:,:,:), intent(in)              :: limit        ! limitator function (divided by 8)
            integer, dimension(:,:), intent(in)         :: send_min     ! distance between me and processus wich send me information
            integer, dimension(:,:), intent(in)         :: send_max     ! distance between me and processus wich send me information
            real(WP), dimension(:,:,:), intent(inout)   :: scalar       ! the initial scalar field transported by particles
            real(WP),dimension(:), intent(out), target  :: buffer       ! buffer where particles are remeshed
            integer, dimension(:), intent(inout)        :: pos_in_buffer! describe how the one dimensionnal array "buffer" are split
                                                                        ! in part corresponding to different processes

        end subroutine remesh_in_buffer_limit
    end interface

    ! --- Abstract profile of subroutine used to remesh scalar inside a buffer - variant with no type/tag ---
    abstract interface
        subroutine remesh_in_buffer_notype(gs, j, k, ind_min, p_pos_adim, send_min, send_max, &
        & scalar, buffer, pos_in_buffer)

            use precision_tools
            implicit none

            ! Input/Output
            integer, dimension(2), intent(in)           :: gs
            integer, intent(in)                         :: j, k
            integer, intent(in)                         :: ind_min
            real(WP), dimension(:,:,:), intent(in)      :: p_pos_adim   ! adimensionned particles position
            integer, dimension(:,:), intent(in)         :: send_min     ! distance between me and processus wich send me information
            integer, dimension(:,:), intent(in)         :: send_max     ! distance between me and processus wich send me information
            real(WP), dimension(:,:,:), intent(inout)   :: scalar       ! the initial scalar field transported by particles
                                                                        ! the right remeshing formula
            real(WP),dimension(:), intent(out), target  :: buffer       ! buffer where particles are remeshed
            integer, dimension(:), intent(inout)        :: pos_in_buffer! describe how the one dimensionnal array "buffer" are split
                                                                        ! in part corresponding to different processes

        end subroutine remesh_in_buffer_notype
    end interface

    ! --- Abstract profile of subroutine used to redistribute a buffer
    ! containing remeshing particle to a scalar field ---
    abstract interface
        subroutine remesh_buffer_to_scalar(gs, j, k, ind_proc, gap, begin_i1, cartography, buffer, scalar, beg_buffer)

            use precision_tools
            implicit none

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
            integer, intent(inout)                      :: beg_buffer   ! first indice inside where the scalar values are stored into the buffer
                                                                        ! for the current sender processus. To know where reading data into the buffer.
        end subroutine remesh_buffer_to_scalar
    end interface

end module advec_abstract_proc
