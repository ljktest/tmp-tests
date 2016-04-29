!USEFORTEST advec
!> @addtogroup part

!------------------------------------------------------------------------------
!
! MODULE: advec_remeshing_formula
!
!
! DESCRIPTION:
!> This module gathers all the remeshing formula of ``Mprime'' family.
!! These interpolation polynoms allow to re-distribute particles on mesh grid at each
!! iteration.
!! @details
!! It provides M'6 and M'8 remeshing formula.
!!   These M' formula appear as only involving stability condition depending on
!! velocity gradient rather than CFL number. Thus, they allow us to use large
!! time-step. The stability constant is equal to 1 (ie the condition is
!! dt < gradiend(velocity)) where the numerical gradient is computed with
!! finite-difference scheme.
!!   In praxis, the' M'6 method appears to offer the better ratio between
!! precision and numerical cost. It is locally of order 4 and generically of order
!! 2 (the spatial order is decreased in location associated with important
!! velocity variation).
!!   The local accuracy of M'8 scheme can be better.
!!     This module also provides some wrapper to remesh a complete line
!! of particles (with the different formula) and to do it either on a
!! array or into a array of pointer to reals. In order to gather
!! communications between different lines of particles, it is better to
!! use continguous memory space for mesh point which belong to the same
!! processes and thus to use an array of pointer to easily deal with it.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module advec_remeshing_Mprime

    use structure_tools
    use advec_common_line

    implicit none

    ! #############################
    ! ########## Hearder ##########
    ! #############################

!   ! ===== Abstract profile of M' remeshing subroutines =====
!   ! --- Abstract profile of subroutine used to remesh a line of particles ---
!   ! Variant: the buffer is an array of pointer (and not a pointer to an array)
!   abstract interface
!       subroutine AC_remesh_Mprime(p_pos_adim, scal1D, bl_type, bl_tag, ind_min, buffer)
!           use precision_tools
!           use advec_variables

!           implicit none

!           ! Input/Output
!           real(WP), intent(in)                        :: p_pos_adim
!           real(WP), intent(in)                        :: scal1D
!           type(real_pter), dimension(:), intent(inout):: buffer
!       end subroutine AC_remesh_Mprime
!   end interface

    ! ===== Public variable =====
    !> To know wich diffusion coefficient to use.
    integer, public                                     :: sc_remesh_ind
    integer, public                                     :: current_dir = 1

    ! ===== Public procedures =====
    ! Wrapper to M' remeshing formula (actually pointer to the right subroutine)
    procedure(AC_remesh_Mstar6_array), pointer, public ::  AC_remesh_Mprime_array  => null()   !> wrapper to M' remeshing formula - buffer are stored in classical array
    procedure(AC_remesh_Mstar6_pter),  pointer, public ::  AC_remesh_Mprime_pter   => null()   !> wrapper to M' remeshing formula - buffer are stored via an array of pointer
    ! To get the right "line remeshing" wrapper
    public                              :: AC_remesh_init_Mprime
    !----- M'4 remeshing formula -----
    public                              :: AC_remesh_Mprime4    ! use 4 grid point, 2 for each side of the particle.
    public                              :: AC_remesh_Mprime4_array      ! use 4 grid point, 2 for each side of the particle.
    public                              :: AC_remesh_Mprime4_pter       ! use 4 grid point, 2 for each side of the particle.
    !----- M'6 remeshing formula -----
    public                              :: AC_remesh_Mstar6    ! use 6 grid point, 3 for each side of the particle.
    public                              :: AC_remesh_Mstar6_array      ! use 6 grid point, 3 for each side of the particle.
    public                              :: AC_remesh_Mstar6_pter       ! use 6 grid point, 3 for each side of the particle.
    !----- M'8 remeshing formula -----
    public                              :: AC_remesh_Mprime8    ! use 8 grid point, 4 for each side of the particle.
    public                              :: AC_remesh_Mprime8_array
    public                              :: AC_remesh_Mprime8_pter


    !===== Interface =====
    ! -- M'4: array of real or of pointer --
    interface AC_remesh_Mprime4
        module procedure AC_remesh_Mprime4_pter, AC_remesh_Mprime4_array
    end interface AC_remesh_Mprime4
    ! -- M'6: array of real or of pointer --
    interface AC_remesh_Mstar6
        module procedure AC_remesh_Mstar6_pter, AC_remesh_Mstar6_array
    end interface AC_remesh_Mstar6

    ! -- M'8: array of real or of pointer --
    interface AC_remesh_Mprime8
        module procedure AC_remesh_Mprime8_pter, AC_remesh_Mprime8_array
    end interface AC_remesh_Mprime8

contains

! ===================================================================
! ============     Pointer to the right remesh formula    ===========
! ===================================================================

subroutine AC_remesh_init_Mprime()

    use advec_variables         ! solver context

    select case(trim(type_solv))
    case ('d_M4')
        AC_remesh_Mprime_array => AC_remesh_Mprime4_diff_array
        AC_remesh_Mprime_pter  => AC_remesh_Mprime4_diff_pter
    case ('p_M4')
        AC_remesh_Mprime_array => AC_remesh_Mprime4_array
        AC_remesh_Mprime_pter  => AC_remesh_Mprime4_pter
    case ('p_M8')
        AC_remesh_Mprime_array => AC_remesh_Mprime8_array
        AC_remesh_Mprime_pter  => AC_remesh_Mprime8_pter
    case ('p_44')
        AC_remesh_Mprime_array => AC_remesh_L4_4_array
        AC_remesh_Mprime_pter  => AC_remesh_L4_4_pter
    case ('p_64')
        AC_remesh_Mprime_array => AC_remesh_L6_4_array
        AC_remesh_Mprime_pter  => AC_remesh_L6_4_pter
    case ('p_66')
        AC_remesh_Mprime_array => AC_remesh_L6_6_array
        AC_remesh_Mprime_pter  => AC_remesh_L6_6_pter
    case ('p_84')
        AC_remesh_Mprime_array => AC_remesh_L8_4_array
        AC_remesh_Mprime_pter  => AC_remesh_L8_4_pter
    ! To ensure retro-compatibility
    case ('p_L4')
        AC_remesh_Mprime_array => AC_remesh_L4_4_array
        AC_remesh_Mprime_pter  => AC_remesh_L4_4_pter
    case ('p_L6')
        AC_remesh_Mprime_array => AC_remesh_L6_6_array
        AC_remesh_Mprime_pter  => AC_remesh_L6_6_pter
    case default
        AC_remesh_Mprime_array => AC_remesh_Mstar6_array
        AC_remesh_Mprime_pter  => AC_remesh_Mstar6_pter
    end select

end subroutine AC_remesh_init_Mprime

! =========================================================================
! ============     Interpolation polynom used for remeshing    ============
! =========================================================================

!> M'4 remeshing formula - version for array of real
!! @author Chloe Mimeau, LJK
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_Mprime4_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: j0, j1                   ! indice of the nearest mesh points
    real(WP)    :: bM, b0, bP, bP2          ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    !bM  = ((2.-(y0+1.))**2 * (1.-(y0+1.)))/2.
    bM = (y0 * (y0 * (-y0 + 2.) - 1.)) / 2.
    !bP = 1.-2.5*(1.-y0)**2 + 1.5*(1.-y0)**3
    bP = (y0 * (y0 * (-3. * y0 + 4.) + 1.)) / 2.
    !bP2 = ((2.-(2.-y0))**2 * (1.-(2.-y0)))/2.
    bP2 = (y0 * y0 * (y0 - 1.)) / 2.
    !b0 = 1.- 2.5*y0**2 + 1.5*y0**3
    b0 = 1. - (bM+bP+bP2)

    ! remeshing
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + sca*bM
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + sca*b0
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + sca*bP
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + sca*bP2

end subroutine AC_remesh_Mprime4_array


!> M'4 remeshing formula. - version for array of pointer
!! @author Chloe Mimeau, LJK
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       translat= translation to convert adimensionned particle position to the proper array index
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_Mprime4_pter(pos_adim, translat, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                        :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)                :: buffer
    integer, intent(in)                                         :: translat
    ! Other local variables
    integer     :: j0                       ! indice of the nearest mesh points
    real(WP)    :: bM, b0, bP, bP2          ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! translation to obtain the array index
    j0 = j0 + translat

    ! Interpolation weights
    !bM  = ((2.-(y0+1.))**2 * (1.-(y0+1.)))/2.
    bM = (y0 * (y0 * (-y0 + 2.) - 1.)) / 2.
    !bP = 1.-2.5*(1.-y0)**2 + 1.5*(1.-y0)**3
    bP = (y0 * (y0 * (-3. * y0 + 4.) + 1.)) / 2.
    !bP2 = ((2.-(2.-y0))**2 * (1.-(2.-y0)))/2.
    bP2 = (y0 * y0 * (y0 - 1.)) / 2.
    !b0 = 1.- 2.5*y0**2 + 1.5*y0**3
    b0 = 1. - (bM+bP+bP2)

    ! remeshing
    buffer(j0-1)%pter = buffer(j0-1)%pter + sca*bM
    buffer(j0  )%pter = buffer(j0  )%pter + sca*b0
    buffer(j0+1)%pter = buffer(j0+1)%pter + sca*bP
    buffer(j0+2)%pter = buffer(j0+2)%pter + sca*bP2

end subroutine AC_remesh_Mprime4_pter


!> M'4 remeshing formula with diffusion - version for array of real
!! @author Jean-baptiste Lagaert, LEGI
!!      @param[in]       dir     = current direction
!!      @param[in]       diff_dt_dx = to take in account diffusion, diff = (diffusivity*time step)/((space step)^2)
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_Mprime4_diff_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: j0, j1                   ! indice of the nearest mesh points
    real(WP)    :: bM, b0, bP, bP2          ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points
    real(WP)    :: diff1, diff2             ! remeshing correction to take into account for diffusion term

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Compute coefficient for diffusion part
    diff1 = 1.5*(1.-0.5*4.*sc_diff_dt_dx(sc_remesh_ind,dir))
    diff2 = 0.5*(1.-1.5*4.*sc_diff_dt_dx(sc_remesh_ind,dir))

    ! Interpolation weights
    !bM = .5*((2.-(y0+1))**2)*(diff1*(2.-(y0+1))/3.-diff2*(y0+1))
    bM = (-1./6)*((y0-1.)**2)*(diff1*(y0-1.)+diff2*(3.*y0+3.))
    !b0 =.5*((2.-y0)**2)*(diff1*(2-y0)/3.-diff2*y0)-((1.-y0)**2)*(2.*diff1*(1.-y0)/3.-2.*diff2*y0)
    b0 =(y0**2)*((diff1*(0.5*y0-1.))+(diff2*(1.5*y0-2.))) + (diff1*2./3._WP)
    !bP =.5*((2.-(1-y0))**2)*(diff1*(2-(1-y0))/3.-diff2*(1-y0))-((1.-(1-y0))**2)*(2.*diff1*(1.-(1-y0))/3.-2.*diff2*(1-y0))
    bP = diff1*(y0*(y0*(0.5-0.5*y0)+0.5)+(1._WP/6._WP))+diff2*(y0*(y0*(2.5-1.5*y0)-0.5)-0.5)
    !bP2= .5*((2.-(2-y0))**2)*(diff1*(2.-(2-y0))/3.-diff2*(2-y0))
    bP2 = 0.5_WP*(y0**2)*((1._WP/3._WP)*diff1*y0 - diff2*(2.-y0))
    !bP = 1._WP - (bM + b0 + bP2)


    ! remeshing
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + sca*bM
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + sca*b0
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + sca*bP
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + sca*bP2

end subroutine AC_remesh_Mprime4_diff_array


!> M'4 remeshing formula with diffusion - version for array of pointer.
!! @author Jean-baptiste Lagaert, LEGI
!!      @param[in]       diff    = diffusivity
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       translat= translation to convert adimensionned particle position to the proper array index
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_Mprime4_diff_pter(pos_adim, translat, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                        :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)                :: buffer
    integer, intent(in)                                         :: translat
    ! Other local variables
    integer     :: j0                       ! indice of the nearest mesh points
    real(WP)    :: bM, b0, bP, bP2          ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points
    real(WP)    :: diff1, diff2             ! remeshing correction to take into account for diffusion term

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! translation to obtain the array index
    j0 = j0 + translat

    ! Compute coefficient for diffusion part
    diff1 = 1.5*(1.-0.5*4.*sc_diff_dt_dx(sc_remesh_ind,current_dir))
    diff2 = 0.5*(1.-1.5*4.*sc_diff_dt_dx(sc_remesh_ind,current_dir))

    ! Interpolation weights
    !bM = .5*((2.-(y0+1))**2)*(diff1*(2.-(y0+1))/3.-diff2*(y0+1))
    bM = (-1./6)*((y0-1.)**2)*(diff1*(y0-1.)+diff2*(3.*y0+3.))
    !b0 =.5*((2.-y0)**2)*(diff1*(2-y0)/3.-diff2*y0)-((1.-y0)**2)*(2.*diff1*(1.-y0)/3.-2.*diff2*y0)
    b0 =(y0**2)*((diff1*(0.5*y0-1.))+(diff2*(1.5*y0-2.))) + (diff1*2./3._WP)
    !bP =.5*((2.-(1-y0))**2)*(diff1*(2-(1-y0))/3.-diff2*(1-y0))-((1.-(1-y0))**2)*(2.*diff1*(1.-(1-y0))/3.-2.*diff2*(1-y0))
    bP = diff1*(y0*(y0*(0.5-0.5*y0)+0.5)+(1._WP/6._WP))+diff2*(y0*(y0*(2.5-1.5*y0)-0.5)-0.5)
    !bP2= .5*((2.-(2-y0))**2)*(diff1*(2.-(2-y0))/3.-diff2*(2-y0))
    bP2 = 0.5_WP*(y0**2)*((1._WP/3._WP)*diff1*y0 - diff2*(2.-y0))
    !bP = 1._WP - (bM + b0 + bP2)

    ! remeshing
    buffer(j0-1)%pter = buffer(j0-1)%pter + sca*bM
    buffer(j0  )%pter = buffer(j0  )%pter + sca*b0
    buffer(j0+1)%pter = buffer(j0+1)%pter + sca*bP
    buffer(j0+2)%pter = buffer(j0+2)%pter + sca*bP2

end subroutine AC_remesh_Mprime4_diff_pter


!> M'6 remeshing formula - version for array of real
!! @author Jean-Baptiste Lagaert, LEGI/LJK
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_Mstar6_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: j0, j1                   ! indice of the nearest mesh points
    real(WP)    :: bM, bM2, b0, bP, bP2, bP3! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    !bM2 =-(((y0+2.)-2)*(5.*(y0+2.)-8.)*((y0+2.)-3.)**3)/24.
    bM2 = y0*(2. + y0*(-1. + y0*(-9. + (13. - 5.*y0)*y0)))/24.
    !bM  =(y0+1.-1.)*(y0+1.-2.)*(25.*(y0+1.)**3-114.*(y0+1.)**2+153.*(y0+1.)-48.)/24.
    bM = y0*(-16. + y0*(16. + y0*(39. + y0*(-64. + 25.*y0))))/24.
    !bP  =-((1.-y0)-1.)*(25.*(1.-y0)**4-38.*(1.-y0)**3-3.*(1.-y0)**2+12.*(1.-y0)+12)/12.
    bP = ( y0*(8. + y0*(8. + y0*(33. + y0*(-62. + 25.*y0)))))/12.
    !bP2 = ((2.-y0)-1.)*((2.-y0)-2.)*(25.*(2.-y0)**3-114.*(2.-y0)**2+153.*(2.-y0)-48.)/24.
    bP2 = (y0*(-2. + y0*(-1. + y0*(-33. + (61. - 25.*y0)*y0))))/24.
    !bP3 =-(((3.-y0)-2)*(5.*(3.-y0)-8.)*((3.-y0)-3.)**3)/24.
    bP3 = (y0**3)*(7. + y0*(5.*y0 - 12.))/24.
    !b0  =-(y0-1.)*(25.*y0**4-38.*y0**3-3.*y0**2+12.*y0+12)/12.
    !b0 = (12. + y0**2*(-15. + y0*(-35. + (63. - 25.*y0)*y0)))/12.
    b0 = 1. - (bM2+bM+bP+bP2+bP3)

    ! remeshing
    j1 = modulo(j0-3,mesh_sc%N(dir))+1  ! j0-2
    buffer(j1) = buffer(j1) + sca*bM2
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + sca*bM
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + sca*b0
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + sca*bP
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + sca*bP2
    j1 = modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    buffer(j1) = buffer(j1) + sca*bP3

end subroutine AC_remesh_Mstar6_array


!> M'6 remeshing formula (order is more than 2, JM Ethancelin is working on
!! determining order). - version for array of pointer
!! @author Jean-Baptiste Lagaert, LEGI/LJK
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       translat= translation to convert adimensionned particle position to the proper array index
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_Mstar6_pter(pos_adim, translat, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                        :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)                :: buffer
    integer, intent(in)                                         :: translat
    ! Other local variables
    integer     :: j0                       ! indice of the nearest mesh points
    real(WP)    :: bM, bM2, b0, bP, bP2, bP3! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! translation to obtain the array index
    j0 = j0 + translat

    ! Interpolation weights
    !bM2 =-(((y0+2.)-2)*(5.*(y0+2.)-8.)*((y0+2.)-3.)**3)/24.
    bM2 = y0*(2. + y0*(-1. + y0*(-9. + (13. - 5.*y0)*y0)))/24.
    !bM  =(y0+1.-1.)*(y0+1.-2.)*(25.*(y0+1.)**3-114.*(y0+1.)**2+153.*(y0+1.)-48.)/24.
    bM = y0*(-16. + y0*(16. + y0*(39. + y0*(-64. + 25.*y0))))/24.
    !bP  =-((1.-y0)-1.)*(25.*(1.-y0)**4-38.*(1.-y0)**3-3.*(1.-y0)**2+12.*(1.-y0)+12)/12.
    bP = ( y0*(8. + y0*(8. + y0*(33. + y0*(-62. + 25.*y0)))))/12.
    !bP2 = ((2.-y0)-1.)*((2.-y0)-2.)*(25.*(2.-y0)**3-114.*(2.-y0)**2+153.*(2.-y0)-48.)/24.
    bP2 = (y0*(-2. + y0*(-1. + y0*(-33. + (61. - 25.*y0)*y0))))/24.
    !bP3 =-(((3.-y0)-2)*(5.*(3.-y0)-8.)*((3.-y0)-3.)**3)/24.
    bP3 = (y0**3)*(7. + y0*(5.*y0 - 12.))/24.
    !b0  =-(y0-1.)*(25.*y0**4-38.*y0**3-3.*y0**2+12.*y0+12)/12.
    !b0 = (12. + y0**2*(-15. + y0*(-35. + (63. - 25.*y0)*y0)))/12.
    b0 = 1. - (bM2+bM+bP+bP2+bP3)

    !print *, j0, pos_adim

    ! remeshing
    buffer(j0-2)%pter = buffer(j0-2)%pter + sca*bM2
    buffer(j0-1)%pter = buffer(j0-1)%pter + sca*bM
    buffer(j0  )%pter = buffer(j0  )%pter + sca*b0
    buffer(j0+1)%pter = buffer(j0+1)%pter + sca*bP
    buffer(j0+2)%pter = buffer(j0+2)%pter + sca*bP2
    buffer(j0+3)%pter = buffer(j0+3)%pter + sca*bP3

end subroutine AC_remesh_Mstar6_pter


!> Lambda(4,4) remeshing formula (without correction), order = 4 everywhere - version for array of real
!! @author Jean-Baptiste Lagaert, LEGI/LJK
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_L4_4_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: j0, j1                   ! indice of the nearest mesh points
    real(WP)    :: bM, bM2, b0, bP, bP2, bP3! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    bM2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-46. * y0 + 207.) - 354.) + 273.) - 80.) + 1.) - 2.)- 1.) + 2.)) / 24.
    bM  = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(230. * y0 - 1035.) +1770.) - 1365.) + 400.) - 4.) + 4.) + 16.) - 16.)) / 24.
    b0  = (y0* y0*(y0*y0* (y0*(y0*(y0*(y0*(-460.* y0 + 2070.) - 3540.) + 2730.) - 800.) + 6.) - 30.)+ 24.) / 24.
    bP  = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(460. * y0 - 2070.) + 3540.) - 2730.) + 800.) - 4.) - 4.) + 16.) + 16.)) / 24.
    !bP2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0 * (-230. * y0 + 1035.) - 1770.) + 1365.) - 400.) + 1.) + 2.) - 1.) - 2.)) / 24.
    bP3 = (y0*y0*y0*y0*y0*(y0*(y0 * (y0 * (46. * y0 - 207.) + 354.) - 273.) + 80.)) / 24.
    bP2 = 1. - (bM2+bM+bP+b0+bP3)

    ! remeshing
    j1 = modulo(j0-3,mesh_sc%N(dir))+1  ! j0-2
    buffer(j1) = buffer(j1) + sca*bM2
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + sca*bM
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + sca*b0
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + sca*bP
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + sca*bP2
    j1 = modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    buffer(j1) = buffer(j1) + sca*bP3

end subroutine AC_remesh_L4_4_array


!> Lambda(4,4) uncorrected remeshing formula (order 4 everywhere)  - version for array of pointer
!! @author Jean-Baptiste Lagaert, LEGI/LJK
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       translat= translation to convert adimensionned particle position to the proper array index
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_L4_4_pter(pos_adim, translat, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                        :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)                :: buffer
    integer, intent(in)                                         :: translat
    ! Other local variables
    integer     :: j0                       ! indice of the nearest mesh points
    real(WP)    :: bM, bM2, b0, bP, bP2, bP3! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! translation to obtain the array index
    j0 = j0 + translat

    ! Interpolation weights
    bM2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-46. * y0 + 207.) - 354.) + 273.) - 80.) + 1.) - 2.)- 1.) + 2.)) / 24.
    bM  = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(230. * y0 - 1035.) +1770.) - 1365.) + 400.) - 4.) + 4.) + 16.) - 16.)) / 24.
    b0  = (y0* y0*(y0*y0* (y0*(y0*(y0*(y0*(-460.* y0 + 2070.) - 3540.) + 2730.) - 800.) + 6.) - 30.)+ 24.) / 24.
    bP  = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(460. * y0 - 2070.) + 3540.) - 2730.) + 800.) - 4.) - 4.) + 16.) + 16.)) / 24.
    !bP2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0 * (-230. * y0 + 1035.) - 1770.) + 1365.) - 400.) + 1.) + 2.) - 1.) - 2.)) / 24.
    bP3 = (y0*y0*y0*y0*y0*(y0*(y0 * (y0 * (46. * y0 - 207.) + 354.) - 273.) + 80.)) / 24.
    bP2 = 1. - (bM2+bM+bP+b0+bP3)

    ! remeshing
    buffer(j0-2)%pter = buffer(j0-2)%pter + sca*bM2
    buffer(j0-1)%pter = buffer(j0-1)%pter + sca*bM
    buffer(j0  )%pter = buffer(j0  )%pter + sca*b0
    buffer(j0+1)%pter = buffer(j0+1)%pter + sca*bP
    buffer(j0+2)%pter = buffer(j0+2)%pter + sca*bP2
    buffer(j0+3)%pter = buffer(j0+3)%pter + sca*bP3

end subroutine AC_remesh_L4_4_pter


!> Lambda(6,4) remeshing formula (without correction),
!! order = 6 locally with regularity C4- version for array of real
!! @author Chloe Mimeau, LJK
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_L6_4_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: j0, j1            ! indice of the nearest mesh points
    real(WP)    :: bM3, bM2, bM, b0  ! interpolation weight for the particles
    real(WP)    :: bP, bP2, bP3, bP4 ! interpolation weight for the particles
    real(WP)    :: y0                ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    bM3 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(290. * y0 - 1305.) + 2231.) &
        & - 1718.) + 500.) - 5.) + 15.) + 4.) - 12.)) / 720.
    bM2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-2030. * y0 + 9135.) - 15617.) &
        & + 12027.) - 3509.) + 60.) - 120.) - 54.) + 108.)) / 720.
    bM = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(6090. * y0 - 27405.) + 46851.) &
        & - 36084.) + 10548.) - 195.) + 195.) + 540.) - 540.)) / 720.
    b0 = (y0*y0*(y0*y0*(y0*(y0*(y0*(y0*(-10150. * y0 + 45675.) - 78085.) &
        & + 60145.) - 17605.) + 280.) - 980.) + 720.) / 720.
    bP = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(10150. * y0 - 45675.) + 78085.) &
        & - 60150.) + 17620.) - 195.) - 195.) + 540.) + 540.)) / 720.
    bP2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-6090. * y0 + 27405.) - 46851.) &
        & + 36093.) - 10575.) + 60.) + 120.) - 54.) - 108.)) / 720.
    bP3 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(2030. * y0 - 9135.) + 15617.) &
        & - 12032.) + 3524.) - 5.) - 15.) + 4.) + 12.)) / 720.
    bP4 = (y0*y0*y0*y0*y0*(y0*(y0*(y0*(-290. * y0 + 1305.) - 2231.) + 1719.) &
        & - 503.)) / 720.
!    b0 = 1. - (bM3+bM2+bM+bP+bP2+bP3+bP4)

    ! remeshing
    j1 = modulo(j0-4,mesh_sc%N(dir))+1  ! j0-3
    buffer(j1) = buffer(j1) + sca*bM3
    j1 = modulo(j0-3,mesh_sc%N(dir))+1  ! j0-2
    buffer(j1) = buffer(j1) + sca*bM2
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + sca*bM
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + sca*b0
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + sca*bP
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + sca*bP2
    j1 = modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    buffer(j1) = buffer(j1) + sca*bP3
    j1 = modulo(j0+3,mesh_sc%N(dir))+1  ! j0+4
    buffer(j1) = buffer(j1) + sca*bP4

end subroutine AC_remesh_L6_4_array


!> Lambda(6,4) uncorrected remeshing formula (order 6 locally with regularity C4)
!! - version for array of pointer
!! @author Chloe Mimeau, LJK
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       translat= translation to convert adimensionned particle position to the proper array index
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_L6_4_pter(pos_adim, translat, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                             :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)     :: buffer
    integer, intent(in)                              :: translat
    ! Other local variables
    integer     :: j0                ! indice of the nearest mesh points
    real(WP)    :: bM3, bM2, bM, b0  ! interpolation weight for the particles
    real(WP)    :: bP, bP2, bP3, bP4 ! interpolation weight for the particles
    real(WP)    :: y0                ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! translation to obtain the array index
    j0 = j0 + translat

    ! Interpolation weights
    bM3 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(290. * y0 - 1305.) + 2231.) &
        & - 1718.) + 500.) - 5.) + 15.) + 4.) - 12.)) / 720.
    bM2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-2030. * y0 + 9135.) - 15617.) &
        & + 12027.) - 3509.) + 60.) - 120.) - 54.) + 108.)) / 720.
    bM = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(6090. * y0 - 27405.) + 46851.) &
        & - 36084.) + 10548.) - 195.) + 195.) + 540.) - 540.)) / 720.
    b0 = (y0*y0*(y0*y0*(y0*(y0*(y0*(y0*(-10150. * y0 + 45675.) - 78085.) &
        & + 60145.) - 17605.) + 280.) - 980.) + 720.) / 720.
    bP = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(10150. * y0 - 45675.) + 78085.) &
        & - 60150.) + 17620.) - 195.) - 195.) + 540.) + 540.)) / 720.
    bP2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-6090. * y0 + 27405.) - 46851.) &
        & + 36093.) - 10575.) + 60.) + 120.) - 54.) - 108.)) / 720.
    bP3 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(2030. * y0 - 9135.) + 15617.) &
        & - 12032.) + 3524.) - 5.) - 15.) + 4.) + 12.)) / 720.
    bP4 = (y0*y0*y0*y0*y0*(y0*(y0*(y0*(-290. * y0 + 1305.) - 2231.) + 1719.) &
        & - 503.)) / 720.
!    b0 = 1. - (bM3+bM2+bM+bP+bP2+bP3+bP4)


    ! remeshing
    buffer(j0-3)%pter = buffer(j0-3)%pter + sca*bM3
    buffer(j0-2)%pter = buffer(j0-2)%pter + sca*bM2
    buffer(j0-1)%pter = buffer(j0-1)%pter + sca*bM
    buffer(j0  )%pter = buffer(j0  )%pter + sca*b0
    buffer(j0+1)%pter = buffer(j0+1)%pter + sca*bP
    buffer(j0+2)%pter = buffer(j0+2)%pter + sca*bP2
    buffer(j0+3)%pter = buffer(j0+3)%pter + sca*bP3
    buffer(j0+4)%pter = buffer(j0+4)%pter + sca*bP4

end subroutine AC_remesh_L6_4_pter


!> Lambda(6,6) remeshing formula (without correction), order = 6 everywhere - version for array of real
!! @author Jean-Baptiste Lagaert, LEGI/LJK
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_L6_6_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: j0, j1                   ! indice of the nearest mesh points
    real(WP)    :: bM, bM2, bM3, b0, bP, bP2, bP3, bP4! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    bM3 = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (3604. * y0 - 23426.) + 63866.) &
             & - 93577.) + 77815.) - 34869.) + 6587.) + 1.) - 3.) - 5.) + 15.) + &
             & 4.) - 12.)) / 720.
    bM2 = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (-25228. * y0 + 163982.) - 447062.) &
             & + 655039.) - 544705.) + 244083.) - 46109.) - 6.) + 12.) + 60.) - &
             & 120.) - 54.) + 108.)) / 720.
    bM  = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (75684. * y0 - 491946.) + 1341186.) &
             & - 1965117.) + 1634115.) - 732249.) + 138327.) + 15.) - 15.) - 195.) &
             & + 195.) + 540.) - 540.)) / 720.
    b0  = (y0 * y0 * (y0 * y0 * (y0 * y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (-126140. * y0 + 819910.) - 2235310.) &
             & + 3275195.) - 2723525.) + 1220415.) - 230545.) - 20.) + 280.) - &
             & 980.) + 720.) / 720.
    !bP  = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (126140. * y0 - 819910.) + 2235310.) &
    !         & - 3275195.) + 2723525.) - 1220415.) + 230545.) + 15.) + 15.) - &
    !         & 195.) - 195.) + 540.) + 540.)) / 720.
    bP2 = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (-75684. * y0 + 491946.) - 1341186.) &
             & + 1965117.) - 1634115.) + 732249.) - 138327.) - 6.) - 12.) + 60.) + &
             & 120.) - 54.) - 108.)) / 720.
    bP3 = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (25228. * y0 - 163982.) + 447062.) &
             & - 655039.) + 544705.) - 244083.) + 46109.) + 1.) + 3.) - 5.) - 15.) &
             & + 4.) + 12.)) / 720.
    bp4 = (y0 * y0 * y0 * y0 * y0 * y0 * y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (-3604. * y0 + &
             & 23426.) - 63866.) + 93577.) - 77815.) + 34869.) - 6587.)) / 720.
    bP = 1. - (bM3+bM2+bM+b0+bP2+bP3+bP4)

    ! remeshing
    j1 = modulo(j0-4,mesh_sc%N(dir))+1  ! j0-3
    buffer(j1) = buffer(j1) + sca*bM3
    j1 = modulo(j0-3,mesh_sc%N(dir))+1  ! j0-2
    buffer(j1) = buffer(j1) + sca*bM2
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + sca*bM
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + sca*b0
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + sca*bP
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + sca*bP2
    j1 = modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    buffer(j1) = buffer(j1) + sca*bP3
    j1 = modulo(j0+3,mesh_sc%N(dir))+1  ! j0+4
    buffer(j1) = buffer(j1) + sca*bP4

end subroutine AC_remesh_L6_6_array


!> Lambda(6,6) uncorrected remeshing formula (order 6 everywhere)  - version for array of pointer
!! @author Jean-Baptiste Lagaert, LEGI/LJK
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       translat= translation to convert adimensionned particle position to the proper array index
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_L6_6_pter(pos_adim, translat, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                        :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)                :: buffer
    integer, intent(in)                                         :: translat
    ! Other local variables
    integer     :: j0                       ! indice of the nearest mesh points
    real(WP)    :: bM, bM2, bM3, b0, bP, bP2, bP3, bP4! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! translation to obtain the array index
    j0 = j0 + translat

    ! Interpolation weights
    bM3 = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (3604. * y0 - 23426.) + 63866.) &
             & - 93577.) + 77815.) - 34869.) + 6587.) + 1.) - 3.) - 5.) + 15.) + &
             & 4.) - 12.)) / 720.
    bM2 = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (-25228. * y0 + 163982.) - 447062.) &
             & + 655039.) - 544705.) + 244083.) - 46109.) - 6.) + 12.) + 60.) - &
             & 120.) - 54.) + 108.)) / 720.
    bM  = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (75684. * y0 - 491946.) + 1341186.) &
             & - 1965117.) + 1634115.) - 732249.) + 138327.) + 15.) - 15.) - 195.) &
             & + 195.) + 540.) - 540.)) / 720.
    b0  = (y0 * y0 * (y0 * y0 * (y0 * y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (-126140. * y0 + 819910.) - 2235310.) &
             & + 3275195.) - 2723525.) + 1220415.) - 230545.) - 20.) + 280.) - &
             & 980.) + 720.) / 720.
    !bP  = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (126140. * y0 - 819910.) + 2235310.) &
    !         & - 3275195.) + 2723525.) - 1220415.) + 230545.) + 15.) + 15.) - &
    !         & 195.) - 195.) + 540.) + 540.)) / 720.
    bP2 = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (-75684. * y0 + 491946.) - 1341186.) &
             & + 1965117.) - 1634115.) + 732249.) - 138327.) - 6.) - 12.) + 60.) + &
             & 120.) - 54.) - 108.)) / 720.
    bP3 = (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (25228. * y0 - 163982.) + 447062.) &
             & - 655039.) + 544705.) - 244083.) + 46109.) + 1.) + 3.) - 5.) - 15.) &
             & + 4.) + 12.)) / 720.
    bp4 = (y0 * y0 * y0 * y0 * y0 * y0 * y0 * (y0 * (y0 * (y0 * (y0 * (y0 * (-3604. * y0 + &
             & 23426.) - 63866.) + 93577.) - 77815.) + 34869.) - 6587.)) / 720.
    bP = 1. - (bM3+bM2+bM+b0+bP2+bP3+bP4)

    ! remeshing
    buffer(j0-3)%pter = buffer(j0-3)%pter + sca*bM3
    buffer(j0-2)%pter = buffer(j0-2)%pter + sca*bM2
    buffer(j0-1)%pter = buffer(j0-1)%pter + sca*bM
    buffer(j0  )%pter = buffer(j0  )%pter + sca*b0
    buffer(j0+1)%pter = buffer(j0+1)%pter + sca*bP
    buffer(j0+2)%pter = buffer(j0+2)%pter + sca*bP2
    buffer(j0+3)%pter = buffer(j0+3)%pter + sca*bP3
    buffer(j0+4)%pter = buffer(j0+4)%pter + sca*bP4

end subroutine AC_remesh_L6_6_pter


!> Lambda(8,4) remeshing formula (without correction),
!! order = 8 locally with regularity C4 - version for array of real
!! @author Chloe Mimeau, LJK
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_L8_4_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    integer, intent(in)                     :: dir
    real(WP), intent(in)                    :: pos_adim, sca
    real(WP), dimension(:), intent(inout)   :: buffer
    ! Other local variables
    integer     :: j0, j1                 ! indice of the nearest mesh points
    real(WP)    :: bM4, bM3, bM2, bM, b0  ! interpolation weight for the particles
    real(WP)    :: bP, bP2, bP3, bP4, bP5 ! interpolation weight for the particles
    real(WP)    :: y0                     ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    bM4 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-3569. * y0 + 16061.) &
        & - 27454.) + 21126.) - 6125.) + 49.) - 196.) - 36.) + 144.)) / 40320.
    bM3 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(32121. * y0 - 144548.) &
        & + 247074.) - 190092.) + 55125.) - 672.) + 2016.) + 512.) &
        & - 1536.)) / 40320.
    bM2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-128484. * y0 + 578188.) &
        & - 988256.) + 760312.) - 221060.) + 4732.) - 9464.) - 4032.) &
        & + 8064.)) / 40320.
    bM = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(299796. * y0 - 1349096.) &
        & + 2305856.) - 1774136.) + 517580.) - 13664.) + 13664.) &
        & + 32256.) - 32256.)) / 40320.
    b0 = (y0*y0*(y0*y0*(y0*(y0*(y0*(y0*(-449694. * y0 + 2023630.) &
        & - 3458700.) + 2661540.) - 778806.) + 19110.) - 57400.) &
        & + 40320.) / 40320.
    bP = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(449694. * y0 - 2023616.) &
        & + 3458644.) - 2662016.) + 780430.) - 13664.) - 13664.) &
        & + 32256.) + 32256.)) / 40320.
    bP2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-299796. * y0 + 1349068.) &
        & - 2305744.) + 1775032.) - 520660.) + 4732.) + 9464.) - 4032.) &
        & - 8064.)) / 40320.
    bP3 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(128484. * y0 - 578168.) &
        & + 988176.) - 760872.) + 223020.) - 672.) - 2016.) + 512.) &
        & + 1536.)) / 40320.
    bP4 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-32121. * y0 + 144541.) &
        & - 247046.) + 190246.) - 55685.) + 49.) + 196.) - 36.) &
        & - 144.)) / 40320.
    bP5 = (y0*y0*y0*y0*y0*(y0*(y0*(y0*(3569. * y0 - 16060.) + 27450.) &
        & - 21140.) + 6181.)) / 40320.
!    b0 = 1. - (bM3+bM2+bM+bP+bP2+bP3+bP4)

    ! remeshing
    j1 = modulo(j0-5,mesh_sc%N(dir))+1  ! j0-4
    buffer(j1) = buffer(j1) + sca*bM4
    j1 = modulo(j0-4,mesh_sc%N(dir))+1  ! j0-3
    buffer(j1) = buffer(j1) + sca*bM3
    j1 = modulo(j0-3,mesh_sc%N(dir))+1  ! j0-2
    buffer(j1) = buffer(j1) + sca*bM2
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + sca*bM
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + sca*b0
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + sca*bP
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + sca*bP2
    j1 = modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    buffer(j1) = buffer(j1) + sca*bP3
    j1 = modulo(j0+3,mesh_sc%N(dir))+1  ! j0+4
    buffer(j1) = buffer(j1) + sca*bP4
    j1 = modulo(j0+4,mesh_sc%N(dir))+1  ! j0+5
    buffer(j1) = buffer(j1) + sca*bP5


end subroutine AC_remesh_L8_4_array


!> Lambda(8,4) uncorrected remeshing formula (order 8 locally with regularity C4)
!! - version for array of pointer
!! @author Chloe Mimeau, LJK
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       translat= translation to convert adimensionned particle position to the proper array index
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_L8_4_pter(pos_adim, translat, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                             :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)     :: buffer
    integer, intent(in)                              :: translat
    ! Other local variables
    integer     :: j0                     ! indice of the nearest mesh points
    real(WP)    :: bM4, bM3, bM2, bM, b0  ! interpolation weight for the particles
    real(WP)    :: bP, bP2, bP3, bP4, bP5 ! interpolation weight for the particles
    real(WP)    :: y0                     ! adimensionned distance to mesh points

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! translation to obtain the array index
    j0 = j0 + translat

    ! Interpolation weights
    bM4 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-3569. * y0 + 16061.) &
        & - 27454.) + 21126.) - 6125.) + 49.) - 196.) - 36.) + 144.)) / 40320.
    bM3 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(32121. * y0 - 144548.) &
        & + 247074.) - 190092.) + 55125.) - 672.) + 2016.) + 512.) &
        & - 1536.)) / 40320.
    bM2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-128484. * y0 + 578188.) &
        & - 988256.) + 760312.) - 221060.) + 4732.) - 9464.) - 4032.) &
        & + 8064.)) / 40320.
    bM = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(299796. * y0 - 1349096.) &
        & + 2305856.) - 1774136.) + 517580.) - 13664.) + 13664.) &
        & + 32256.) - 32256.)) / 40320.
    b0 = (y0*y0*(y0*y0*(y0*(y0*(y0*(y0*(-449694. * y0 + 2023630.) &
        & - 3458700.) + 2661540.) - 778806.) + 19110.) - 57400.) &
        & + 40320.) / 40320.
    bP = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(449694. * y0 - 2023616.) &
        & + 3458644.) - 2662016.) + 780430.) - 13664.) - 13664.) &
        & + 32256.) + 32256.)) / 40320.
    bP2 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-299796. * y0 + 1349068.) &
        & - 2305744.) + 1775032.) - 520660.) + 4732.) + 9464.) - 4032.) &
        & - 8064.)) / 40320.
    bP3 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(128484. * y0 - 578168.) &
        & + 988176.) - 760872.) + 223020.) - 672.) - 2016.) + 512.) &
        & + 1536.)) / 40320.
    bP4 = (y0*(y0*(y0*(y0*(y0*(y0*(y0*(y0*(-32121. * y0 + 144541.) &
        & - 247046.) + 190246.) - 55685.) + 49.) + 196.) - 36.) &
        & - 144.)) / 40320.
    bP5 = (y0*y0*y0*y0*y0*(y0*(y0*(y0*(3569. * y0 - 16060.) + 27450.) &
        & - 21140.) + 6181.)) / 40320.
!    b0 = 1. - (bM3+bM2+bM+bP+bP2+bP3+bP4)

    ! remeshing
    buffer(j0-4)%pter = buffer(j0-4)%pter + sca*bM4
    buffer(j0-3)%pter = buffer(j0-3)%pter + sca*bM3
    buffer(j0-2)%pter = buffer(j0-2)%pter + sca*bM2
    buffer(j0-1)%pter = buffer(j0-1)%pter + sca*bM
    buffer(j0  )%pter = buffer(j0  )%pter + sca*b0
    buffer(j0+1)%pter = buffer(j0+1)%pter + sca*bP
    buffer(j0+2)%pter = buffer(j0+2)%pter + sca*bP2
    buffer(j0+3)%pter = buffer(j0+3)%pter + sca*bP3
    buffer(j0+4)%pter = buffer(j0+4)%pter + sca*bP4
    buffer(j0+5)%pter = buffer(j0+5)%pter + sca*bP5

end subroutine AC_remesh_L8_4_pter


!> M'8 remeshing formula - version for array of pointer.
!! @author Jean-Baptiste Lagaert, LEGI/LJK
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_Mprime8_array(dir, pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    integer, intent(in)                      :: dir
    real(WP), intent(in)                     :: pos_adim, sca
    real(WP), dimension(:), intent(inout)    :: buffer
    ! Other local variables
    integer     :: j0, j1                   ! indice of the nearest mesh points
    real(WP)    :: y0                       ! adimensionned distance to mesh points
    real(WP)    :: bM, bM2, bM3, b0, bP, bP2, bP3, bP4  ! interpolation weight for the particles

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    ! M'8 = 15/8*M8 + 9/8 * x * M'8 + 1/8 * x^2 * M''8
    ! y8 = y1 + 4
    ! bP4=(y0**7)/2688.-(4-y0)*(y0**6)/640.+((4-y0)**2)*(y0**5)/960
    bP4=(y0**5)*(y0*(y0/336. - 7./480.) + 1./60.)
    ! bM3=(1-y0)**7/2688.-(y0+3)*(1-y0)**6/640.+(y0+3)**2*(1-y0)**5/960
    bM3=y0*(y0*(y0*(y0*(y0*(y0*(-y0/336. + 1./160.) + 1./120.)    &
        & - 1./32.) + 1./48.) + 1./96.) - 1./60.) + 17./3360.
    ! bP3=(y0+1)**7/2688.-(3-y0)*(y0+1)**6/640.+(3-y0)**2*(y0+1)**5/960
    !     -y0**7/336+(3-y0)*y0**6/80.-(3-y0)**2*y0**5/120.
    bP3=y0*(y0*(y0*(y0*(y0*(y0*(-y0/48. + 3./32.) - 1./12.)       &
        & - 1./32.) - 1./48.) + 1./96.) + 1./60.) + 17./3360.
    ! bM2=(2-y0)**7/2688.-(y0+2)*(2-y0)**6/640.+(y0+2)**2*(2-y0)**5/960
    !     -xx2**7/336+(y0+2)*xx2**6/80.-(y0+2)**2*xx2**5/120.
    bM2=y0*(y0*(y0*(y0*(y0*(y0*(y0/48. - 5./96.) - 1./24.)        &
        & + 11./48.) - 1./6.) - 5./48.) + 3./20.) - 17./560.
    ! bP2=(y0+2)**7/2688.-(2-y0)*(y0+2)**6/640.+(2-y0)**2*(y0+2)**5/960
    !       -(y0+1)**7/336+(2-y0)*(y0+1)**6/80.-(2-y0)**2*(y0+1)**5/120.
    !       +y0**7/96.-7.*(2-y0)*y0**6/160.+7.*(2-y0)**2*y0**5/240.
    bP2=y0*(y0*(y0*(y0*(y0*(y0*(y0/16. - 41./160.) + 19./120.)    &
        & + 11./48.) + 1./6.) - 5./48.) - 3./20.) - 17./560.
    ! bM=(3-y0)**7/2688.-(y0+1)*(3-y0)**6/640.+(y0+1)**2*(3-y0)**5/960
    !       -(2-y0)**7/336+(y0+1)*(2-y0)**6/80.-(y0+1)**2*(2-y0)**5/120.
    !       +(1-y0)**7/96.-7.*(y0+1)*(1-y0)**6/160.+7.*(y0+1)**2*(1-y0)**5/240.
    bM=y0*(y0*(y0*(y0*(y0*(y0*(-y0/16. + 29./160.) + 1./15.)     &
        & - 61./96.) + 13./48.) + 79./96.) - 3./4.) + 17./224.
    ! bP=(y0+3)**7/2688.-(1-y0)*(y0+3)**6/640.+(1-y0)**2*(y0+3)**5/960
    !       -(y0+2)**7/336+(1-y0)*(y0+2)**6/80.-(1-y0)**2*(y0+2)**5/120.
    !       +(y0+1)**7/96.-7.*(1-y0)*(y0+1)**6/160.+7.*(1-y0)**2*(y0+1)**5/240.
    !       -y0**7/48.+7.*(1-y0)*y0**6/80.-7.*(1-y0)**2*y0**5/120.
    ! bP=y0*(y0*(y0*(y0*(y0*(y0*(-5.*y0/48. + 37./96.) - 1./8.)    &
    !    & - 61./96.) - 13./48.) + 79./96.) + 3./4.) + 17./224.
    ! See below : bP = 1 - (b0+bM+bP2+bM2+bP3+q7+bP4)
    ! b0=(4-y0)**7/2688.-y0*(4-y0)**6/640.+y0**2*(4-y0)**5/960
    !       -(3-y0)**7/336+y0*(3-y0)**6/80.-y0**2*(3-y0)**5/120.
    !       +(2-y0)**7/96.-7.*y0*(2-y0)**6/160.+7.*y0**2*(2-y0)**5/240.
    !       -(1-y0)**7/48.+7.*y0*(1-y0)**6/80.-7.*y0**2*(1-y0)**5/120.
    b0=y0**2*((y0**2)*(y0**2*(5.*y0/48. - 11./32.) + 7./8.)     &
        & - 35./24.) + 151./168.

    bP = 1. - bM3 - bM2 - bM - b0 - bP2 - bP3 - bP4

    ! remeshing
    j1 = modulo(j0-4,mesh_sc%N(dir))+1  ! j0-3
    buffer(j1) = buffer(j1) + sca*bM3
    j1 = modulo(j0-3,mesh_sc%N(dir))+1  ! j0-2
    buffer(j1) = buffer(j1) + sca*bM2
    j1 = modulo(j0-2,mesh_sc%N(dir))+1  ! j0-1
    buffer(j1) = buffer(j1) + sca*bM
    j1 = modulo(j0-1,mesh_sc%N(dir))+1  ! j0
    buffer(j1) = buffer(j1) + sca*b0
    j1 = modulo(j0,mesh_sc%N(dir))+1    ! j0+1
    buffer(j1) = buffer(j1) + sca*bP
    j1 = modulo(j0+1,mesh_sc%N(dir))+1  ! j0+2
    buffer(j1) = buffer(j1) + sca*bP2
    j1 = modulo(j0+2,mesh_sc%N(dir))+1  ! j0+3
    buffer(j1) = buffer(j1) + sca*bP3
    j1 = modulo(j0+3,mesh_sc%N(dir))+1  ! j0+4
    buffer(j1) = buffer(j1) + sca*bP4

end subroutine AC_remesh_Mprime8_array


!> M'8 remeshing formula - version for array of pointer.
!! @author Jean-Baptiste Lagaert, LEGI/LJK
!!      @param[in]       dir     = current direction
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       translat= translation to convert adimensionned particle position to the proper array index
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_Mprime8_pter(pos_adim, translat, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                            :: pos_adim, sca
    type(real_pter), dimension(:), intent(inout)    :: buffer
    integer, intent(in)                             :: translat
    ! Other local variables
    integer     :: j0                       ! indice of the nearest mesh points
    real(WP)    :: y0                       ! adimensionned distance to mesh points
    real(WP)    :: bM, bM2, bM3, b0, bP, bP2, bP3, bP4  ! interpolation weight for the particles

    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! translation to obtain the array index
    j0 = j0 + translat

    ! Interpolation weights
    ! M'8 = 15/8*M8 + 9/8 * x * M'8 + 1/8 * x^2 * M''8
    ! y8 = y1 + 4
    ! bP4=(y0**7)/2688.-(4-y0)*(y0**6)/640.+((4-y0)**2)*(y0**5)/960
    bP4=(y0**5)*(y0*(y0/336. - 7./480.) + 1./60.)
    ! bM3=(1-y0)**7/2688.-(y0+3)*(1-y0)**6/640.+(y0+3)**2*(1-y0)**5/960
    bM3=y0*(y0*(y0*(y0*(y0*(y0*(-y0/336. + 1./160.) + 1./120.)    &
        & - 1./32.) + 1./48.) + 1./96.) - 1./60.) + 17./3360.
    ! bP3=(y0+1)**7/2688.-(3-y0)*(y0+1)**6/640.+(3-y0)**2*(y0+1)**5/960
    !     -y0**7/336+(3-y0)*y0**6/80.-(3-y0)**2*y0**5/120.
    bP3=y0*(y0*(y0*(y0*(y0*(y0*(-y0/48. + 3./32.) - 1./12.)       &
        & - 1./32.) - 1./48.) + 1./96.) + 1./60.) + 17./3360.
    ! bM2=(2-y0)**7/2688.-(y0+2)*(2-y0)**6/640.+(y0+2)**2*(2-y0)**5/960
    !     -xx2**7/336+(y0+2)*xx2**6/80.-(y0+2)**2*xx2**5/120.
    bM2=y0*(y0*(y0*(y0*(y0*(y0*(y0/48. - 5./96.) - 1./24.)        &
        & + 11./48.) - 1./6.) - 5./48.) + 3./20.) - 17./560.
    ! bP2=(y0+2)**7/2688.-(2-y0)*(y0+2)**6/640.+(2-y0)**2*(y0+2)**5/960
    !       -(y0+1)**7/336+(2-y0)*(y0+1)**6/80.-(2-y0)**2*(y0+1)**5/120.
    !       +y0**7/96.-7.*(2-y0)*y0**6/160.+7.*(2-y0)**2*y0**5/240.
    bP2=y0*(y0*(y0*(y0*(y0*(y0*(y0/16. - 41./160.) + 19./120.)    &
        & + 11./48.) + 1./6.) - 5./48.) - 3./20.) - 17./560.
    ! bM=(3-y0)**7/2688.-(y0+1)*(3-y0)**6/640.+(y0+1)**2*(3-y0)**5/960
    !       -(2-y0)**7/336+(y0+1)*(2-y0)**6/80.-(y0+1)**2*(2-y0)**5/120.
    !       +(1-y0)**7/96.-7.*(y0+1)*(1-y0)**6/160.+7.*(y0+1)**2*(1-y0)**5/240.
    bM=y0*(y0*(y0*(y0*(y0*(y0*(-y0/16. + 29./160.) + 1./15.)     &
        & - 61./96.) + 13./48.) + 79./96.) - 3./4.) + 17./224.
    ! bP=(y0+3)**7/2688.-(1-y0)*(y0+3)**6/640.+(1-y0)**2*(y0+3)**5/960
    !       -(y0+2)**7/336+(1-y0)*(y0+2)**6/80.-(1-y0)**2*(y0+2)**5/120.
    !       +(y0+1)**7/96.-7.*(1-y0)*(y0+1)**6/160.+7.*(1-y0)**2*(y0+1)**5/240.
    !       -y0**7/48.+7.*(1-y0)*y0**6/80.-7.*(1-y0)**2*y0**5/120.
    ! bP=y0*(y0*(y0*(y0*(y0*(y0*(-5.*y0/48. + 37./96.) - 1./8.)    &
    !    & - 61./96.) - 13./48.) + 79./96.) + 3./4.) + 17./224.
    ! See below : bP = 1 - (b0+bM+bP2+bM2+bP3+q7+bP4)
    ! b0=(4-y0)**7/2688.-y0*(4-y0)**6/640.+y0**2*(4-y0)**5/960
    !       -(3-y0)**7/336+y0*(3-y0)**6/80.-y0**2*(3-y0)**5/120.
    !       +(2-y0)**7/96.-7.*y0*(2-y0)**6/160.+7.*y0**2*(2-y0)**5/240.
    !       -(1-y0)**7/48.+7.*y0*(1-y0)**6/80.-7.*y0**2*(1-y0)**5/120.
    b0=y0**2*((y0**2)*(y0**2*(5.*y0/48. - 11./32.) + 7./8.)     &
        & - 35./24.) + 151./168.

    bP = 1. - bM3 - bM2 - bM - b0 - bP2 - bP3 - bP4

    ! remeshing
    buffer(j0-3)%pter = buffer(j0-3)%pter + sca*bM3
    buffer(j0-2)%pter = buffer(j0-2)%pter + sca*bM2
    buffer(j0-1)%pter = buffer(j0-1)%pter + sca*bM
    buffer(j0  )%pter = buffer(j0  )%pter + sca*b0
    buffer(j0+1)%pter = buffer(j0+1)%pter + sca*bP
    buffer(j0+2)%pter = buffer(j0+2)%pter + sca*bP2
    buffer(j0+3)%pter = buffer(j0+3)%pter + sca*bP3
    buffer(j0+4)%pter = buffer(j0+4)%pter + sca*bP4

end subroutine AC_remesh_Mprime8_pter


end module advec_remeshing_Mprime
!> @}
