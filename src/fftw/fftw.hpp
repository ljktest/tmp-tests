#ifndef FFTW_HPP
#define FFTW_HPP

namespace fftw
{

  /** */
  class FFTW_solver
  {
  private:

    /** */
    int _dimension;
    
  public:

    FFTW_solver(int d);

    int run();

  };
  
}


#endif
