!USEFORTEST advec
!> @addtogroup part
!! @{
!------------------------------------------------------------------------------
!
! MODULE: advec_remeshing_line
!
!
! DESCRIPTION:
!> This module gathers all the remeshing formula. These interpolation
!!polynom allow to re-distribute particles on mesh grid at each
!! iterations. - old version for advection without group of line
!! @details
!! It provides lambda 2 corrected, lambda 4 corrected and M'6 remeshing formula.
!! The remeshing of type "lambda corrected" are design for large time
!! step. The M'6 formula appears as being stable for large time step, but
!! the numerical analysis remains todo.
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

module advec_remeshing_line

    public AC_remesh_lambda4corrected_basic
    public AC_remesh_lambda2corrected_basic



contains

!> Remesh particle line with corrected lambda 2 formula - remeshing is done into
!! an real array
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        scal1D      = scalar field to advect
!!    @param[in]        bl_type     = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]        bl_tag      = contains information about bloc (is it tagged ?)
!!    @param[in]        ind_min     = minimal indice of the send buffer
!!    @param[in]        ind_max     = maximal indice of the send buffer
!!    @param[in, out]   send_buffer = buffer use to remesh the scalar before to send it to the right subdomain
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
subroutine AC_remesh_lambda2corrected_basic(direction, p_pos_adim, scal1D, bl_type, bl_tag, ind_min, ind_max, send_buffer)

    use cart_topology   ! Description of mesh and of mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                                 :: direction
    real(WP), dimension(:), intent(in)                  :: p_pos_adim
    real(WP), dimension(mesh_sc%N_proc(direction)), intent(in)  :: scal1D
    logical, dimension(:), intent(in)                   :: bl_type
    logical, dimension(:), intent(in)                   :: bl_tag
    integer, intent(in)                                 :: ind_min, ind_max
    real(WP), dimension(ind_min:ind_max), intent(inout) :: send_buffer
    ! Other local variables
    integer     :: bl_ind                               ! indice of the current "block end".
    integer     :: p_ind                                ! indice of the current particle

    send_j_min = ind_min
    send_j_max = ind_max

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
                call AC_remesh_tag_CL(p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), scal1D(p_ind+1), send_buffer)
            else
                ! tagged, the first particle belong to a left block and the last to centered block.
                call AC_remesh_tag_LC(p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), scal1D(p_ind+1), send_buffer)
            end if
        else
            if (bl_type(bl_ind)) then
                ! First particle is remeshed with center formula
                call AC_remesh_center(p_pos_adim(p_ind),scal1D(p_ind), send_buffer)
            else
                ! First particle is remeshed with left formula
                call AC_remesh_left(p_pos_adim(p_ind),scal1D(p_ind), send_buffer)
            end if
            if (bl_type(bl_ind+1)) then
                ! Second particle is remeshed with center formula
                call AC_remesh_center(p_pos_adim(p_ind+1),scal1D(p_ind+1), send_buffer)
            else
                ! Second particle is remeshed with left formula
                call AC_remesh_left(p_pos_adim(p_ind+1),scal1D(p_ind+1), send_buffer)
            end if
        end if
    end do

end subroutine AC_remesh_lambda2corrected_basic


!> Remesh particle line with corrected lambda 4 formula - array version
!!    @param[in]        direction   = current direction (1 = along X, 2 = along Y and 3 = along Z)
!!    @param[in]        p_pos_adim  = adimensionned  particles position
!!    @param[in]        scal1D      = scalar field to advect
!!    @param[in]        bl_type     = equal 0 (resp 1) if the block is left (resp centered)
!!    @param[in]        bl_tag      = contains information about bloc (is it tagged ?)
!!    @param[in]        ind_min     = minimal indice of the send buffer
!!    @param[in]        ind_max     = maximal indice of the send buffer
!!    @param[in, out]   send_buffer = buffer use to remesh the scalar before to send it to the right subdomain
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
subroutine AC_remesh_lambda4corrected_basic(direction, p_pos_adim, scal1D, bl_type, bl_tag, ind_min, ind_max, send_buffer)

    use cart_topology   ! Description of mesh and of mpi topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    integer, intent(in)                                 :: direction
    real(WP), dimension(:), intent(in)                  :: p_pos_adim
    real(WP), dimension(mesh_sc%N_proc(direction)), intent(in)  :: scal1D
    logical, dimension(:), intent(in)                   :: bl_type
    logical, dimension(:), intent(in)                   :: bl_tag
    integer, intent(in)                                 :: ind_min, ind_max
    real(WP), dimension(ind_min:ind_max), intent(inout) :: send_buffer
    ! Other local variables
    integer     :: bl_ind                               ! indice of the current "block end".
    integer     :: p_ind                                ! indice of the current particle

    send_j_min = ind_min
    send_j_max = ind_max

    do p_ind = 1, mesh_sc%N_proc(direction), bl_size
        bl_ind = p_ind/bl_size + 1
        if (bl_tag(bl_ind)) then
            ! Tagged case
            if (bl_type(bl_ind)) then
                ! tagged, the first particle belong to a centered block and the last to left block.
                call AC_remesh_O4_tag_CL(p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), scal1D(p_ind+1), &
                        & p_pos_adim(p_ind+2), scal1D(p_ind+2), p_pos_adim(p_ind+3), scal1D(p_ind+3), send_buffer)
            else
                ! tagged, the first particle belong to a left block and the last to centered block.
                call AC_remesh_O4_tag_LC(p_pos_adim(p_ind), scal1D(p_ind), p_pos_adim(p_ind+1), scal1D(p_ind+1), &
                        & p_pos_adim(p_ind+2), scal1D(p_ind+2), p_pos_adim(p_ind+3), scal1D(p_ind+3), send_buffer)
            end if
        else
            ! No tag
            if (bl_type(bl_ind)) then
                call AC_remesh_O4_center(p_pos_adim(p_ind),scal1D(p_ind), send_buffer)
                call AC_remesh_O4_center(p_pos_adim(p_ind+1),scal1D(p_ind+1), send_buffer)
            else
                call AC_remesh_O4_left(p_pos_adim(p_ind),scal1D(p_ind), send_buffer)
                call AC_remesh_O4_left(p_pos_adim(p_ind+1),scal1D(p_ind+1), send_buffer)
            end if
            if (bl_type(bl_ind+1)) then
                call AC_remesh_O4_center(p_pos_adim(p_ind+2),scal1D(p_ind+2), send_buffer)
                call AC_remesh_O4_center(p_pos_adim(p_ind+3),scal1D(p_ind+3), send_buffer)
            else
                call AC_remesh_O4_left(p_pos_adim(p_ind+2),scal1D(p_ind+2), send_buffer)
                call AC_remesh_O4_left(p_pos_adim(p_ind+3),scal1D(p_ind+3), send_buffer)
            end if
        end if
    end do

end subroutine AC_remesh_lambda4corrected_basic


!> Left remeshing formula of order 2
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_left(pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                        :: pos_adim, sca
    real(WP), dimension(send_j_min:send_j_max), intent(inout)   :: buffer
    ! Ohter local variables
    integer     :: j0                       ! indice of the the nearest mesh points
    real(WP)    :: bM, b0, bP               ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points
    ! Mesh point used in remeshing formula
    j0 = floor(pos_adim)
    !j0 = floor(pos/d_sc(2))

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))

    ! Interpolation weights
    bM=0.5*y0*(y0-1.)
    b0=1.-y0**2
    !bP=0.5*y0*(y0+1.)
    bP=1. - (b0+bM)

    ! remeshing
    buffer(j0-1) = buffer(j0-1)   + bM*sca
    buffer(j0)   = buffer(j0)     + b0*sca
    buffer(j0+1) = buffer(j0+1)   + bP*sca

end subroutine AC_remesh_left


!> Centered remeshing formula of order 2
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_center(pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/output
    real(WP), intent(in)                                        :: pos_adim, sca
    real(WP), dimension(send_j_min:send_j_max), intent(inout)   :: buffer
    ! Other local variables
    integer     :: j0                       ! indice of the the nearest mesh points
    real(WP)    :: bM, b0, bP               ! interpolation weight for the particles
    real(WP)    :: y0                       ! adimensionned distance to mesh points

    j0 = nint(pos_adim)
    !j0 = nint(pos/d_sc(2))

    ! Distance to mesh points
    y0 = (pos_adim - real(j0, WP))
    !y0 = (pos - dble(j0)*d_sc(2))/d_sc(2)

    ! Interpolation weights
    bM=0.5*y0*(y0-1.)
    b0=1.-y0**2
    !bP=0.5*y0*(y0+1.)
    bP=1. -b0 - bM

    ! remeshing
    buffer(j0-1) = buffer(j0-1)   + bM*sca
    buffer(j0)   = buffer(j0)     + b0*sca
    buffer(j0+1) = buffer(j0+1)   + bP*sca

end subroutine AC_remesh_center


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
subroutine AC_remesh_tag_CL(pos_adim, sca, posP_ad, scaP, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                                        :: pos_adim, sca, posP_ad, scaP
    real(WP), dimension(send_j_min:send_j_max), intent(inout)   :: buffer
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
    !y0 = (pos - dble(j0)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - dble(j0_bis)*d_sc(2))/d_sc(2)

    aM=0.5*y0*(y0-1)
    a0=1.-aM
    bP=0.5*y0_bis*(y0_bis+1.)
    b0=1.-bP

    ! Remeshing
    buffer(jM)=buffer(jM)+aM*sca
    buffer(j0)=buffer(j0)+a0*sca+b0*scaP
    buffer(jP)=buffer(jP)+bP*scaP

end subroutine AC_remesh_tag_CL


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
subroutine AC_remesh_tag_LC(pos_adim, sca, posP_ad, scaP, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                                        :: pos_adim, sca, posP_ad, scaP
    real(WP), dimension(send_j_min:send_j_max), intent(inout)   :: buffer
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
    !y0 = (pos - dble(j0)*d_sc(2))/d_sc(2)
    y0_bis = (posP_ad - real(j0_bis, WP))
    !y0_bis = (posP - dble(j0_bis)*d_sc(2))/d_sc(2)

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
    buffer(jM)= buffer(jM)+aM*sca
    buffer(j0)= buffer(j0)+a0*sca+b0*scaP
    buffer(jP)= buffer(jP)+aP*sca+bP*scaP
    buffer(jP2)=buffer(jP2)+aP2*sca+bP2*scaP
    buffer(jP3)=buffer(jP3)+bP3*scaP

end subroutine AC_remesh_tag_LC


!> Left remeshing formula of order 4
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_O4_left(pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                        :: pos_adim, sca
    real(WP), dimension(send_j_min:send_j_max), intent(inout)   :: buffer
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
    buffer(j0-2) = buffer(j0-2)   + bM2*sca
    buffer(j0-1) = buffer(j0-1)   + bM*sca
    buffer(j0)   = buffer(j0)     + b0*sca
    buffer(j0+1) = buffer(j0+1)   + bP*sca
    buffer(j0+2) = buffer(j0+2)   + bP2*sca

end subroutine AC_remesh_O4_left


!> Centered remeshing formula of order 4 - array version
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_O4_center(pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/output
    real(WP), intent(in)                                        :: pos_adim, sca
    real(WP), dimension(send_j_min:send_j_max), intent(inout)   :: buffer
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
    bM2=y0*(2._WP+y0*(-1.+y0*(-2._WP+y0)))/24._WP
    !bM =(2.-y0)*(y0-1.)*y0*(y0+2.)/6.0
    bM =y0*(-4._WP+y0*(4._WP+y0*(1._WP-y0)))/6._WP
    !bP =(2.-y0)*y0*(y0+1.)*(y0+2.)/6.0
    bP =y0*(4._WP+y0*(4._WP-y0*(1._WP+y0)))/6._WP
    !bP2=(y0-1.)*y0*(y0+1.)*(y0+2.)/24.0
    bP2=y0*(-2._WP+y0*(-1._WP+y0*(2._WP+y0)))/24._WP
    !b0 =(y0-2.)*(y0-1.)*(y0+1.)*(y0+2.)/4.0
    b0 = 1._WP -(bM2+bM+bP+bP2)

    ! remeshing
    buffer(j0-2) = buffer(j0-2)   + bM2*sca
    buffer(j0-1) = buffer(j0-1)   + bM*sca
    buffer(j0)   = buffer(j0)     + b0*sca
    buffer(j0+1) = buffer(j0+1)   + bP*sca
    buffer(j0+2) = buffer(j0+2)   + bP2*sca

end subroutine AC_remesh_O4_center


!> Order 4 corrected remeshing formula for transition from Centered block to a Left block with a different indice (tagged particles)
!! - version for array of real.
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
subroutine AC_remesh_O4_tag_CL(posM_ad, scaM, pos_adim, sca, posP_ad, scaP, posP2_ad, scaP2, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                                        :: pos_adim, sca, posP_ad, scaP
    real(WP), intent(in)                                        :: posM_ad, scaM, posP2_ad, scaP2
    real(WP), dimension(send_j_min:send_j_max), intent(inout)   :: buffer
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

    ! remeshing
    buffer(j0-3) = buffer(j0-3)   +aM3*scaM
    buffer(j0-2) = buffer(j0-2)   +aM2*scaM +bM2*sca
    buffer(j0-1) = buffer(j0-1)   + aM*scaM + bM*sca  + cM*scaP
    buffer(j0)   = buffer(j0)     + a0*scaM + b0*sca  + c0*scaP + e0*scaP2
    buffer(j0+1) = buffer(j0+1)             + bP*sca  + cP*scaP + eP*scaP2
    buffer(j0+2) = buffer(j0+2)                       +cP2*scaP +eP2*scaP2
    buffer(j0+3) = buffer(j0+3)                                 +eP3*scaP2

end subroutine AC_remesh_O4_tag_CL


!> Corrected remeshing formula of order 3 for transition from Left block to a centered
!! block with a different indice (tagged particles). Use it for lambda 4 corrected scheme.
!! - version for array of real.
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
subroutine AC_remesh_O4_tag_LC(posM_ad, scaM, pos_adim, sca, posP_ad, scaP, posP2_ad, scaP2, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    ! Input/Output
    real(WP), intent(in)                                        :: pos_adim, sca, posP_ad, scaP
    real(WP), intent(in)                                        :: posM_ad, scaM, posP2_ad, scaP2
    real(WP), dimension(send_j_min:send_j_max), intent(inout)   :: buffer
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
    yP2= (posP2_ad - real(j0, WP))

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
    buffer(j0-3) = buffer(j0-3)   +aM3*scaM
    buffer(j0-2) = buffer(j0-2)   +aM2*scaM +bM2*sca
    buffer(j0-1) = buffer(j0-1)   + aM*scaM + bM*sca  + cM*scaP
    buffer(j0)   = buffer(j0)     + a0*scaM + b0*sca  + c0*scaP + e0*scaP2
    buffer(j0+1) = buffer(j0+1)   + aP*scaM + bP*sca  + cP*scaP + eP*scaP2
    buffer(j0+2) = buffer(j0+2)   +aP2*scaM +bP2*sca  +cP2*scaP +eP2*scaP2
    buffer(j0+3) = buffer(j0+3)             +bP3*sca  +cP3*scaP +eP3*scaP2
    buffer(j0+4) = buffer(j0+4)                       +cP4*scaP +eP4*scaP2
    buffer(j0+5) = buffer(j0+5)                                 +eP5*scaP2

end subroutine AC_remesh_O4_tag_LC


!> M'6 remeshing formula - version for array of pointer.
!!      @param[in]       pos_adim= adimensionned particle position
!!      @param[in]       sca     = scalar advected by the particle
!!      @param[in,out]   buffer  = temporaly remeshed scalar field
subroutine AC_remesh_Mprime6(pos_adim, sca, buffer)

    use cart_topology
    use advec_variables ! contains info about solver parameters and others.

    !Input/Ouput
    real(WP), intent(in)                                        :: pos_adim, sca
    real(WP), dimension(send_j_min:send_j_max), intent(inout)   :: buffer
    ! Ohter local variables
    integer     :: j0                       ! indice of the the nearest mesh points
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
    buffer(j0-2) = buffer(j0-2)   + sca*bM2
    buffer(j0-1) = buffer(j0-1)   + sca*bM
    buffer(j0)   = buffer(j0)     + sca*b0
    buffer(j0+1) = buffer(j0+1)   + sca*bP
    buffer(j0+2) = buffer(j0+2)   + sca*bP2
    buffer(j0+3) = buffer(j0+3)   + sca*bP3

end subroutine AC_remesh_Mprime6

end module advec_remeshing_line
