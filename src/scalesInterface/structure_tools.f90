!------------------------------------------------------------------------------
!
! MODULE: structure_tool
!
!
! DESCRIPTION:
!> This module provides some useful structure like array of pointer (basic
!fortran only defines pointer to an array) and array of pointer to array.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module structure_tools

  use precision_tools

  implicit none

  ! --- In order to create an array of pointer to real ---
  type real_pter
      real(WP), pointer                   :: pter
  end type real_pter
  ! --- In order to create an array of pointer to array ---
  type int_1D_pter
      integer, dimension(:), pointer      :: pter
  end type int_1D_pter
  type real_1D_pter
      real(WP), dimension(:), pointer     :: pter
  end type real_1D_pter
  ! ---------------------------------------------

end module structure_tools
