!USEFORTEST toolbox
!USEFORTEST postprocess
!USEFORTEST advec
!USEFORTEST interpolation
!USEFORTEST io
!USEFORTEST topo
!USEFORTEST avgcond
!> @addtogroup toolbox
!! @{

!-----------------------------------------------------------------------------
!
! MODULE: cart_mesh_tools
!
!
! DESCRIPTION:
!>  This module provide a mesh structure. It is used for output and as future
!! base to deal with different scalar field computed with eventually different
!! resolutions.
!
!> @details
!!  This module provide structure to save mesh context associated to a field.
!! This allow to easily work with different resolutions and to know how
!! mesh interact with the mpi topology.
!! It provide the different tools to initialise the type to some default
!! value or to auto-complete it.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module cart_mesh_tools

    use precision_tools

    implicit none

    public

    ! ===== Type =====
    ! > Information about mesh subdivision and on the global grid
    type cartesian_mesh
        !> number of grid points in each direction
        integer, dimension(3)   :: N
        !> number of grid point for the local subgrid in each direction
        integer, dimension(3)   :: N_proc
        !> information about min and max local indice on the current directory
        integer, dimension(3,2) :: relative_extend
        !> information about min and max global indice associated to the current processus
        integer, dimension(3,2) :: absolute_extend
        !> space step for field discretisation
        real(WP), dimension(3)  :: dx
        !> Physical size
        real(WP), dimension(3)  :: length
    end type cartesian_mesh


    ! ===== Public procedures =====
    ! Auto-complete cartesian_mesh data field.
    public      :: mesh_save


contains

!> Auto-complete some field about "cartesian_mesh" variables.
!>    @param[out]   mesh    = variable of type cartesian_mesh where the data about mesh are save
!>    @param[in]    Nb      = number of grid point along each direction
!>    @param[in]    Nb_proc = number of grid point along each direction associated to the current processus
!>    @param[in]    d_space = space step
!>    @param[in]    coord   = coordinate of the current processus in the 3D mpi-topology
subroutine mesh_save(mesh, Nb, Nb_proc, d_space, coord)

    implicit none

    ! Input/Output
    type(cartesian_mesh), intent(out)       :: mesh
    integer, dimension(3), intent(in)       :: Nb
    integer, dimension(3), intent(in)       :: Nb_proc
    integer, dimension(3), intent(in)       :: coord
    real(WP), dimension(3), intent(in)      :: d_space
    ! Other local variables
    integer                                 :: direction    ! integer matching to a direction (X, Y or Z)

    ! Number of mesh
    mesh%N = Nb
    mesh%N_proc = Nb_proc

    ! Relative extend
    mesh%relative_extend(:,1) = 1
    mesh%relative_extend(:,2) = Nb_proc
    ! Absolute one
    do direction = 1, 3
        mesh%absolute_extend(direction,:) = coord(1)*Nb_proc(direction) + mesh%relative_extend(direction,:)
    end do

    ! Space step
    mesh%dx = d_space

end subroutine mesh_save


end module cart_mesh_tools
!> @}
