
#ifndef FFTWCOMPLEX_H
#define FFTWCOMPLEX_H

#include <type_traits>
#include <complex>
#include <fft/fftw3.h>

namespace hysop {
    namespace fft {

        template <typename T> struct fftw_complex_type {
            typedef void value_type;
            typedef void fftw_type; 
            typedef void std_type; 
        };

        template <> struct fftw_complex_type<long double> { 
            typedef long double               value_type;
            typedef fftwl_complex             fftw_type; 
            typedef std::complex<long double> std_type; 
        };
        template <> struct fftw_complex_type<double> { 
            typedef double               value_type;
            typedef fftw_complex         fftw_type; 
            typedef std::complex<double> std_type; 
        };
        template <> struct fftw_complex_type<float> { 
            typedef float               value_type;
            typedef fftwf_complex       fftw_type; 
            typedef std::complex<float> std_type; 
        };
        
        template <> struct fftw_complex_type<std::complex<long double>> { 
            typedef long double               value_type;
            typedef fftwl_complex             fftw_type; 
            typedef std::complex<long double> std_type; 
        };
        template <> struct fftw_complex_type<std::complex<double>> { 
            typedef double               value_type;
            typedef fftw_complex         fftw_type; 
            typedef std::complex<double> std_type; 
        };
        template <> struct fftw_complex_type<std::complex<float>> { 
            typedef float               value_type;
            typedef fftwf_complex       fftw_type; 
            typedef std::complex<float> std_type; 
        };
        
        template <> struct fftw_complex_type<fftwl_complex> { 
            typedef long double               value_type;
            typedef fftwl_complex             fftw_type; 
            typedef std::complex<long double> std_type; 
        };
        template <> struct fftw_complex_type<fftw_complex> { 
            typedef double               value_type;
            typedef fftw_complex         fftw_type; 
            typedef std::complex<double> std_type; 
        };
        template <> struct fftw_complex_type<fftwf_complex> { 
            typedef float               value_type;
            typedef fftwf_complex       fftw_type; 
            typedef std::complex<float> std_type; 
        };

#ifdef HAS_QUADMATHS
        template <> struct fftw_complex_type<__float128> { 
            typedef __float128               value_type;
            typedef fftwq_complex            fftw_type; 
            typedef std::complex<__float128> std_type; 
        };
        template <> struct fftw_complex_type<std::complex<__float128>> { 
            typedef __float128               value_type;
            typedef fftwq_complex            fftw_type; 
            typedef std::complex<__float128> std_type; 
        };
        template <> struct fftw_complex_type<fftwq_complex> { 
            typedef __float128               value_type;
            typedef fftwq_complex            fftw_type; 
            typedef std::complex<__float128> std_type; 
        };
#endif

    }
}

#endif /* end of include guard: FFTWCOMPLEX_H */
