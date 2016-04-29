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

module advec_common

    use precision_tools

    ! Velocity interpolation at particle position
    use advec_common_interpol ,only:AC_interpol_lin, AC_interpol_lin_no_com, &
      & AC_interpol_plus, AC_interpol_plus_no_com
    ! Particles remeshing
    use advec_common_remesh,only: AC_setup_init,                &
            & AC_remesh_setup_alongX, AC_remesh_setup_alongY, AC_remesh_setup_alongZ,&
            & AC_remesh_lambda_group, AC_remesh_limit_lambda_group, AC_remesh_Mprime_group

    implicit none

    ! To get particle position - if particles are created everywhere
    interface AC_get_p_pos_adim
      module procedure AC_init_pos, AC_get_pos_V, AC_get_pos_other_mesh, AC_get_pos_other_mesh_big
    end interface AC_get_p_pos_adim
    public :: AC_get_p_pos_adim
    private:: AC_init_pos
    private:: AC_get_pos_V
    private:: AC_get_pos_other_mesh


contains

!> Init particle position at mesh point
!!   @param[out] p_pos = adimensionned particle position
subroutine AC_init_pos(p_pos)

    real(WP), dimension(:,:,:), intent(out) :: p_pos

    integer :: i2,i1,i_p

    do i2 = 1, size(p_pos,3)
      do i1 = 1, size(p_pos,2)
        do i_p = 1, size(p_pos,1)
          p_pos(i_p,i1,i2) = i_p
        end do
      end do
    end do
    !do i_p = 1, size(p_pos,1)
    !  p_pos(i_p,:,:) = i_p
    !end do

end subroutine AC_init_pos


!> Init particle position (adimensionned by dx) at initial position + dt*velocity
!!   @param[in]  p_pos  = adimensionned particle position
!!   @param[in]  p_V    = particle velocity
!!   @param[in]  dt     = time step
!!   @param[in]  dx_sc  = spatial step for scalar
!!   @param[in]  Np     = number of particle for each line (=number of mesh point along current direction)
subroutine AC_get_pos_V(p_pos, p_V, dt, dx_sc, Np)

    real(WP), dimension(:,:,:), intent(out) :: p_pos
    real(WP), dimension(:,:,:), intent(in)  :: p_V
    real(WP)                  , intent(in)  :: dt, dx_sc
    integer                   , intent(in)  :: Np

    integer :: i2,i1,i_p
    real(WP):: coef

    coef = dt/dx_sc
    do i2 = 1, size(p_pos,3)
      do i1 = 1, size(p_pos,2)
        do i_p = 1, Np
          p_pos(i_p,i1,i2) = i_p + coef*p_V(i_p,i1,i2)
        end do
      end do
    end do
    !do i_p = 1, size(p_pos,1)
    !  p_pos(i_p,:,:) = i_p + coef*p_V(i_p,:,:)
    !end do

end subroutine AC_get_pos_V


!> Init particle position (adimensionned by dx_V) at initial position +
!! dt*velocity - use this variant if velocity and scalr resolution are different.
!!   @param[in]  p_pos  = adimensionned particle position
!!   @param[in]  p_V    = particle velocity
!!   @param[in]  dt     = time step
!!   @param[in]  dx_sc  = spatial step for scalar
!!   @param[in]  dx_V   = spatial step for velocity
!!   @param[in]  Np     = number of particle for each line (=number of mesh point along current direction)
subroutine AC_get_pos_other_mesh(p_pos, p_V, dt, dx_sc, dx_V, Np)

    real(WP), dimension(:,:,:), intent(out) :: p_pos
    real(WP), dimension(:,:,:), intent(in)  :: p_V
    real(WP)                  , intent(in)  :: dt, dx_sc, dx_V
    integer                   , intent(in)  :: Np

    integer :: i2,i1,i_p
    real(WP):: coef1, coef2

    coef1 = dx_sc/dx_V
    coef2 = dt/dx_V
    do i2 = 1, size(p_pos,3)
      do i1 = 1, size(p_pos,2)
        do i_p = 1, Np
          p_pos(i_p,i1,i2) = (coef1*i_p) + (coef2*p_V(i_p,i1,i2))
        end do
      end do
    end do
    !do i_p = 1, size(p_pos,1)
    !  p_pos(i_p,:,:) = (coef1*i_p) + (coef2*p_V(i_p,:,:))
    !end do

end subroutine AC_get_pos_other_mesh


!> Init particle position (adimensionned by dx_V) at initial position +
!! dt*velocity - use this variant if velocity and scalar resolution are different
!! and if V_comp contain not only velocity for the current work item.
!!   @param[in]  p_pos  = adimensionned particle position
!!   @param[in]  p_V    = particle velocity
!!   @param[in]  dt     = time step
!!   @param[in]  dx_sc  = spatial step for scalar
!!   @param[in]  dx_V   = spatial step for velocity
!!   @param[in]  id1,id2= coordinate of the current work item
!!   @param[in]  Np     = number of particle for each line (=number of mesh point along current direction)
subroutine AC_get_pos_other_mesh_big(p_pos, p_V, dt, dx_sc, dx_V, Np, id1, id2)

    real(WP), dimension(:,:,:), intent(out) :: p_pos
    real(WP), dimension(:,:,:), intent(in)  :: p_V
    real(WP)                  , intent(in)  :: dt, dx_sc, dx_V
    integer                   , intent(in)  :: id1, id2, Np

    integer :: i2,i1,i_p, idir1, idir2
    real(WP):: coef1, coef2

    idir1 = id1 - 1
    idir2 = id2 - 1

    coef1 = dx_sc/dx_V
    coef2 = dt/dx_V

    do i2 = 1, size(p_pos,3)
      do i1 = 1, size(p_pos,2)
        do i_p = 1, Np
          p_pos(i_p,i1,i2) = (coef1*i_p) + (coef2*p_V(i_p,i1+idir1,i2+idir2))
        end do
      end do
    end do
    !do i_p = 1, size(p_pos,1)
    !  p_pos(i_p,:,:) = (coef1*i_p) + (coef2*p_V(i_p,:,:))
    !end do

end subroutine AC_get_pos_other_mesh_big

end module advec_common
