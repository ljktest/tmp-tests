!USEFORTEST advec
!> @addtogroup part
!! @{
!------------------------------------------------------------------------------
!
! MODULE: advec
!
!
! DESCRIPTION:
!> The module advec provides all public interfaces to solve an advection equation
!! with a particle method.
!
!> @details
!!     This module contains the generic procedure to initialize and parametrise the
!! advection solver based on particles method. It also contains the subroutine
!! "advec_step" wich solves the equation for a given time step. It is the only one
!! module which is supposed to be included by a code using this library of
!! particle methods.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------
module advec_Vect

    use advec, only : advec_init

    use precision_tools
    use advec_abstract_proc
    implicit none

    ! ===== Private variables =====
    !> numerical method use to advect the scalar
    character(len=str_short), private   :: type_part_solv
    !> dimensionnal splitting (eg classical, Strang or particle)
    character(len=str_short), private   :: dim_splitting


    ! ===== Public procedures =====
    ! Scheme used to advec the scalar (order 2 or 4 ?)
!    public                              :: type_part_solver

    ! Advection methods
!    public                              :: advec_init           ! initialize the scalar solver
    public                              :: advec_step_Vect      ! advec the scalar field during a time step.
!    procedure(advec_step_Torder2), pointer, public    :: advec_step => null()
!    public                              :: advec_step_Torder1   ! advec the scalar field during a time step.
!    public                              :: advec_step_Torder2   ! advec the scalar field during a time step.
!
    ! Remeshing formula
    procedure(AC_remesh), pointer, private :: advec_remesh_bis => null()

contains

! ===== Public methods =====

!> Return the name of the particle method used for the advection
!!    @return type_part_solver      = numerical method used for advection
function type_part_solver()
    character(len=str_short)    :: type_part_solver

    type_part_solver = type_part_solv
end function

!> Solve advection equation - order 2 - with basic velocity interpolation
!!    @param[in]        dt          = time step
!!    @param[in]        Vx          = velocity along x (could be discretised on a bigger mesh then the scalar)
!!    @param[in]        Vy          = velocity along y
!!    @param[in]        Vz          = velocity along z
!!    @param[in,out]    VectX       = X component of vector to advect
!!    @param[in,out]    VectY       = Y component of vector to advect
!!    @param[in,out]    VectZ       = Z component of vector to advect
subroutine advec_step_Inter_basic_Vect(dt, Vx, Vy, Vz, vectX, vectY, vectZ)

    use Interpolation_velo

    ! Input/Output
    real(WP), intent(in)                        :: dt
    real(WP), dimension(:,:,:), intent(in)      :: Vx, Vy, Vz
    real(WP), dimension(:,:,:), intent(inout)   :: vectX, vectY, vectZ
    ! Local
    real(WP), dimension(:,:,:), allocatable   :: Vx_f, Vy_f, Vz_f
    integer                                   :: ierr                ! Error code.

    allocate(Vx_f(mesh_sc%N_proc(1),mesh_sc%N_proc(2),mesh_sc%N_proc(3)),stat=ierr)
    if (ierr/=0) write(6,'(a,i0,a)') '[ERROR] on cart_rank ', cart_rank, ' - not enough memory for Vx_f'
    allocate(Vy_f(mesh_sc%N_proc(1),mesh_sc%N_proc(2),mesh_sc%N_proc(3)),stat=ierr)
    if (ierr/=0) write(6,'(a,i0,a)') '[ERROR] on cart_rank ', cart_rank, ' - not enough memory for Vy_f'
    allocate(Vz_f(mesh_sc%N_proc(1),mesh_sc%N_proc(2),mesh_sc%N_proc(3)),stat=ierr)
    if (ierr/=0) write(6,'(a,i0,a)') '[ERROR] on cart_rank ', cart_rank, ' - not enough memory for Vz_f'

    call Interpol_3D(Vx, mesh_V%dx, Vx_f, mesh_sc%dx)
    call Interpol_3D(Vy, mesh_V%dx, Vy_f, mesh_sc%dx)
    call Interpol_3D(Vz, mesh_V%dx, Vz_f, mesh_sc%dx)
    if (cart_rank==0) write(6,'(a)') '        [INFO PARTICLES] Interpolation done'

    call advec_step_Vect(dt, Vx_f, Vy_f, Vz_f, vectX, vectY, vectZ)

    deallocate(Vx_f)
    deallocate(Vy_f)
    deallocate(Vz_f)

end subroutine advec_step_Inter_basic_Vect

!> Solve advection equation - order 2 in time (order 2 dimensional splitting)
!!    @param[in]        dt          = time step
!!    @param[in]        Vx          = velocity along x (could be discretised on a bigger mesh then the scalar)
!!    @param[in]        Vy          = velocity along y
!!    @param[in]        Vz          = velocity along z
!!    @param[in,out]    VectX       = X component of vector to advect
!!    @param[in,out]    VectY       = Y component of vector to advect
!!    @param[in,out]    VectZ       = Z component of vector to advect
subroutine advec_step_Vect(dt, Vx, Vy, Vz, vectX, vectY, vectZ)

    use advec, only : advec_setup_alongX, advec_setup_alongY, &
        & advec_setup_alongZ, gsX, gsY, gsZ
    use advecX          ! Method to advec along X
    use advecY          ! Method to advec along Y
    use advecZ          ! Method to advec along Z

    ! Input/Output
    real(WP), intent(in)                        :: dt
    real(WP), dimension(:,:,:), intent(in)      :: Vx, Vy, Vz
    real(WP), dimension(:,:,:), intent(inout)   :: vectX, vectY, vectZ

    call advec_setup_alongX()
    call advec_vector_X_basic_no_com(dt/2.0, gsX, Vx, vectX, vectY, vectZ)
    call advec_setup_alongY()
    call advec_vector_1D_basic(dt/2.0, gsY, Vy, vectX, vectY, vectZ)
    call advec_setup_alongZ()
    call advec_vector_1D_basic(dt/2.0, gsZ, Vz, vectX, vectY, vectZ)
    call advec_vector_1D_basic(dt/2.0, gsZ, Vz, vectX, vectY, vectZ)
    call advec_setup_alongY()
    call advec_vector_1D_basic(dt/2.0, gsY, Vy, vectX, vectY, vectZ)
    call advec_setup_alongX()
    call advec_vector_X_basic_no_com(dt/2.0, gsX, Vx, vectX, vectY, vectZ)

end subroutine advec_step_Vect


!> Scalar advection along one direction (this procedure call the right solver, depending on the simulation setup).
!! Variant for advection of a 3D-vector.
!!    @param[in]        dt          = time step
!!    @param[in]        gs          = size of the work item along transverse direction
!!    @param[in]        V_comp      = velocity component
!!    @param[in,out]    VectX       = X component of vector to advect
!!    @param[in,out]    VectY       = Y component of vector to advect
!!    @param[in,out]    VectZ       = Z component of vector to advect
subroutine advec_vector_1D_basic(dt, gs, V_comp, vectX, vectY, vectZ)

    use advec, only : advec_init_velo, advec_remesh, line_dir, gp_dir1, gp_dir2
    use advecX, only : advecX_init_group    ! procdure devoted to advection along Z
    use advecY, only : advecY_init_group    ! procdure devoted to advection along Z
    use advecZ, only : advecZ_init_group    ! procdure devoted to advection along Z
    use advec_variables ! contains info about solver parameters and others.
    use cart_topology   ! Description of mesh and of mpi topology
    use advec_common    ! some procedures common to advection along all line_dirs

    ! Input/Output
    real(WP), intent(in)                          :: dt
    integer, dimension(2), intent(in)             :: gs
    real(WP), dimension(:,:,:), intent(in)        :: V_comp
    real(WP), dimension(:,:,:), intent(inout)     :: vectX, vectY, vectZ
    ! Other local variables
    integer                                       :: i,j          ! indice of the currend mesh point
    integer, dimension(2)                         :: ind_group    ! indice of the currend group of line (=(i,k) by default)
    real(WP), dimension(mesh_sc%N_proc(line_dir),gs(1),gs(2))  :: p_pos_adim ! adimensionned particles position
    real(WP), dimension(mesh_sc%N_proc(line_dir),gs(1),gs(2))  :: p_V        ! particles velocity

    ind_group = 0

    do j = 1, mesh_sc%N_proc(gp_dir2), gs(2)
        ind_group(2) = ind_group(2) + 1
        ind_group(1) = 0
        do i = 1, mesh_sc%N_proc(gp_dir1), gs(1)
            ind_group(1) = ind_group(1) + 1

            ! ===== Init particles =====
            call advec_init_velo(V_comp, i, j, gs, p_pos_adim)
            ! p_pos is used to store velocity at grid point
            call AC_get_p_pos_adim(p_V, p_pos_adim, 0.5_WP*dt, mesh_sc%dx(line_dir), mesh_sc%N_proc(line_dir))
            ! p_V = middle point position = position at middle point for RK2 scheme

            ! ===== Advection =====
            ! -- Compute velocity (with a RK2 scheme) --
            ! Note that p_pos is used as velocity component storage
            call AC_interpol_lin(line_dir, gs, ind_group, p_pos_adim, p_V)
            ! p_v = velocity at middle point position
            ! -- Push particles --
            call AC_get_p_pos_adim(p_pos_adim, p_V, dt, mesh_sc%dx(line_dir), mesh_sc%N_proc(line_dir))
            ! Now p_pos = particle position and p_V = particle velocity

            ! ===== Remeshing =====
            call advec_remesh(line_dir, ind_group, gs, p_pos_adim, p_V, i,j,vectX, dt)
            call advec_remesh(line_dir, ind_group, gs, p_pos_adim, p_V, i,j,vectY, dt)
            call advec_remesh(line_dir, ind_group, gs, p_pos_adim, p_V, i,j,vectZ, dt)

        end do
    end do

end subroutine advec_vector_1D_basic

!> Scalar advection along one direction - variant for cases with no communication
!!    @param[in]        dt          = time step
!!    @param[in]        V_comp      = velocity along X (could be discretised on a bigger mesh then the scalar)
!!    @param[in,out]    scal3D      = scalar field to advect
!> Details
!!   Work only for direction = X. Basic (and very simple) remeshing has just to
!! be add for other direction.
subroutine advec_vector_X_basic_no_com(dt, gs, V_comp, vectX, vectY, vectZ)

    use advec, only : advec_init_velo, advec_remesh, line_dir, gp_dir1, gp_dir2
    use advecX          ! Procedure specific to advection along X
    use advec_common    ! Some procedures common to advection along all directions
    use advec_variables ! contains info about solver parameters and others.
    use cart_topology   ! Description of mesh and of mpi topology

    ! Input/Output
    real(WP), intent(in)                          :: dt
    integer, dimension(2), intent(in)             :: gs
    real(WP), dimension(:,:,:), intent(in)        :: V_comp
    real(WP), dimension(:,:,:), intent(inout)     :: vectX, vectY, vectZ
    ! Other local variables
    integer                                             :: j,k          ! indice of the currend mesh point
    integer, dimension(2)                               :: ind_group    ! indice of the currend group of line (=(i,k) by default)
    real(WP),dimension(mesh_sc%N_proc(line_dir),gs(1),gs(2)) :: p_pos_adim   ! adimensionned particles position
    real(WP),dimension(mesh_sc%N_proc(line_dir),gs(1),gs(2)) :: p_V          ! particles velocity

    ind_group = 0

    do k = 1, mesh_sc%N_proc(gp_dir2), gs(2)
        ind_group(2) = ind_group(2) + 1
        ind_group(1) = 0
        do j = 1, mesh_sc%N_proc(gp_dir1), gs(1)
            ind_group(1) = ind_group(1) + 1

            ! ===== Init particles =====
            ! p_pos is used to store velocity at grid point
            call advec_init_velo(V_comp, j, k, gs, p_pos_adim)
            ! p_V = middle point position = position at middle point for RK2 scheme
            call AC_get_p_pos_adim(p_V, p_pos_adim, 0.5_WP*dt, mesh_sc%dx(line_dir), mesh_sc%N_proc(line_dir))

            ! ===== Advection =====
            ! -- Compute velocity (with a RK2 scheme): p_V = velocity at middle point position --
            ! Note that p_pos is used as velocity component storage
            call AC_interpol_lin_no_com(line_dir, gs, p_pos_adim, p_V)
            ! p_v = velocity at middle point position
            ! -- Push particles --
            call AC_get_p_pos_adim(p_pos_adim, p_V, dt, mesh_sc%dx(line_dir), mesh_sc%N_proc(line_dir))
            ! Now p_pos = particle position and p_V = particle velocity

            ! ===== Remeshing =====
            call advecX_remesh_no_com(ind_group, gs, p_pos_adim, p_V, j, k, vectX, dt)
            call advecX_remesh_no_com(ind_group, gs, p_pos_adim, p_V, j, k, vectY, dt)
            call advecX_remesh_no_com(ind_group, gs, p_pos_adim, p_V, j, k, vectZ, dt)

        end do
    end do

end subroutine advec_vector_X_basic_no_com


end module advec_Vect
