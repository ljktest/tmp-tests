!> Global parameters for f2py interface.
!!
!! see https://github.com/numpy/numpy/issues/2428
!! for some issues
module hysopparam

  implicit none

  ! double precision kind
  integer, parameter :: pk = 8
  ! integer precision kind
  integer, parameter :: ik = 8

end module hysopparam

module hysopparam_sp

  implicit none

  ! single precision kind
  integer, parameter :: pk = 4
  ! integer precision kind
  integer, parameter :: ik = 8

end module hysopparam_sp
