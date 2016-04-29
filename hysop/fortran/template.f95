  !! Template file - Provide an example
  !! to write f2py interface between fortran and python


module template_f2py

  implicit none

  !> a global var
  integer, parameter :: var1 = 12
 
contains

  !> do something ...
  subroutine check_f2py(input, output)
    
    !> input array
    real(kind=8), dimension(:,:), intent(in) :: input
    !> output array
    real(kind=8), dimension(:,:), intent(inout) :: output

    print *, 'template f2py for tab'
    output(:,:) = 2 * input(:,:)
    print *, 'aha hah', output(1,1)
  end subroutine check_f2py
  
end module template_f2py
