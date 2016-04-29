
#include "fft/fftw3.h"

namespace hysop {
    namespace fft {
        template struct Fftw3<float>;
        template struct Fftw3<double>;
        template struct Fftw3<long double>;
        #ifdef HAS_QUADMATHS
            template struct Fftw3<__float128>;
        #endif
    }
}

/* needed for swig to avoid undefined reference link error */
#if !defined(FFTW_HAS_FFTW3F_THREADS) || !defined(FFTW_HAS_FFTW3F_OMP)
    int  fftwf_init_threads()          { return 0; }
    void fftwf_plan_with_nthreads(int) {}
    void fftwf_cleanup_threads()       {}
#endif

#if !defined(FFTW_HAS_FFTW3D_THREADS) || !defined(FFTW_HAS_FFTW3D_OMP)
    int  fftw_init_threads()          { return 0; }
    void fftw_plan_with_nthreads(int) {}
    void fftw_cleanup_threads()       {}
#endif

#if !defined(FFTW_HAS_FFTW3L_THREADS) || !defined(FFTW_HAS_FFTW3L_OMP)
    int  fftwl_init_threads()          { return 0; }
    void fftwl_plan_with_nthreads(int) {}
    void fftwl_cleanup_threads()       {}
#endif

#ifdef HAS_QUADMATHS
    #if !defined(FFTW_HAS_FFTW3Q_THREADS) || !defined(FFTW_HAS_FFTW3Q_OMP)
        int  fftwq_init_threads()          { return 0; }
        void fftwq_plan_with_nthreads(int) {}
        void fftwq_cleanup_threads()       {}
    #endif
#endif
