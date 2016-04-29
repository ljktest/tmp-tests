#include "fftw.hpp"
#include <iostream>


namespace fftw
{

  FFTW_solver::FFTW_solver(int d):_dimension(d)
  {
    std::cout << "init fftw c++, dim =" << _dimension <<  std::endl;
  }

  int FFTW_solver::run()
  {
    std::cout << "run fftw c++" << std::endl;
    return 1;
  }
 
  
}
