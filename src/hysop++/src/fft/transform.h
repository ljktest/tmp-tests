
#ifndef FFTTRANSFORM_H
#define FFTTRANSFORM_H

#include <complex>
#include <iostream> 
#include <fftw3.h>

#include "utils/constants.h"

namespace hysop {
    namespace fft {

        struct Transform;
        std::ostream& operator<<(std::ostream& os, const Transform &tr);

        enum BaseFunc {
            CEXP,
            SIN,
            COS
        };

        struct Transform {
            public:
                int kind;

                Transform(int p_kind = 0):
                    kind(p_kind) {}
    
                bool isR2C() const {
                    return kind == FFTW_FORWARD;
                }
                bool isR2R() const {
                    return !this->isR2C();
                }

                BaseFunc basefunc() const {
                    switch(kind) {
                        case(FFTW_REDFT00):
                        case(FFTW_REDFT01):
                        case(FFTW_REDFT10):
                        case(FFTW_REDFT11):
                            return COS;
                        case(FFTW_RODFT00):
                        case(FFTW_RODFT01):
                        case(FFTW_RODFT10):
                        case(FFTW_RODFT11):
                            return SIN;
                        case(FFTW_R2HC):
                        case(FFTW_HC2R):
                        case(FFTW_FORWARD):
                            return CEXP;
                        default:
                            throw std::runtime_error("Unknown transform !");
                    }
                }

                template <typename T>
                std::complex<T> omega(std::size_t k, std::size_t N, T L = T(1), bool lastDim=false) const {
                    using namespace hysop::constants;
                    switch(kind) {
                        case(FFTW_FORWARD):
                            if(lastDim) {
                                return std::complex<T>(T(0),T(2)*pi*T(k)/L);
                            }
                            else {
                                T kk;
                                if(k <= N/2 -1)
                                    kk = T(k);
                                else if(k==N/2)
                                    kk = T(0);
                                else 
                                    kk = double(k)-double(N);
                                return std::complex<T>(T(0),T(2)*pi*kk/L);
                            }
                        case(FFTW_REDFT00):
                            return std::complex<T>(pi*T(k)/L, T(0));
                        case(FFTW_RODFT00): /* offset +1 */
                            return std::complex<T>(pi*T(k+1)/L, T(0));
                        case(FFTW_REDFT01):
                            return std::complex<T>(pi*(T(k)+T(0.5))/L, T(0));
                        case(FFTW_RODFT01): /* -0.5 + 1 offset = +0.5 */
                            return std::complex<T>(pi*(T(k)+T(0.5))/L, T(0));
                        default:
                            throw std::runtime_error("Not implemented yet !");
                    }
                }
                
                template <typename T>
                    T normalisation(std::size_t n) const {
                    switch(kind) {
                        case(FFTW_FORWARD):
                            return T(n);
                        case(FFTW_RODFT00):
                            return T(2*(n+1));
                        case(FFTW_REDFT00):
                            return T(2*(n-1));
                        case(FFTW_REDFT01):
                        case(FFTW_REDFT10):
                        case(FFTW_REDFT11):
                        case(FFTW_RODFT01):
                        case(FFTW_RODFT10):
                        case(FFTW_RODFT11):
                            return T(2*n);
                        default:
                            return T(n);
                    }
                }

                int inverseTransform() const {
                    switch(kind) {
                        case(FFTW_REDFT00):
                        case(FFTW_RODFT00):
                            return kind;
                        case(FFTW_REDFT01):
                            return FFTW_REDFT10;
                        case(FFTW_REDFT10):
                            return FFTW_REDFT01;
                        case(FFTW_RODFT01):
                            return FFTW_RODFT10;
                        case(FFTW_RODFT10):
                            return FFTW_RODFT01;
                        case(FFTW_R2HC):
                            return FFTW_HC2R;
                        case(FFTW_HC2R):
                            return FFTW_R2HC;
                        case(FFTW_FORWARD):
                            return FFTW_BACKWARD;
                        default:
                            throw std::runtime_error("Unknown transform !");
                    }
                }

                int conjugateInverseTransform() const {
                    switch(kind) {
                        case(FFTW_REDFT00):
                            return FFTW_RODFT00;
                        case(FFTW_REDFT01):
                            return FFTW_RODFT10;
                        case(FFTW_REDFT10):
                            return FFTW_RODFT01;
                        case(FFTW_RODFT00):
                            return FFTW_REDFT00;
                        case(FFTW_RODFT01):
                            return FFTW_REDFT10;
                        case(FFTW_RODFT10):
                            return FFTW_REDFT01;
                        default:
                            return this->inverseTransform();
                    }
                }

                std::string toString() const {
                    switch(kind) {
                        case(FFTW_REDFT00):
                            return "FFTW_REDFT00";
                        case(FFTW_RODFT00):
                            return "FFTW_RODFT00";
                        case(FFTW_REDFT01):
                            return "FFTW_REDFT01";
                        case(FFTW_REDFT11):
                            return "FFTW_REDFT11";
                        case(FFTW_REDFT10):
                            return "FFTW_REDFT10";
                        case(FFTW_RODFT01):
                            return "FFTW_RODFT01";
                        case(FFTW_RODFT10):
                            return "FFTW_RODFT10";
                        case(FFTW_RODFT11):
                            return "FFTW_RODFT11";
                        case(FFTW_R2HC):
                            return "FFTW_R2HC";
                        //case(FFTW_HC2R):
                            //return "FFTW_HC2R";
                        case(FFTW_BACKWARD):
                            return "FFTW_BACKWARD";
                        case(FFTW_FORWARD):
                            return "FFTW_FORWARD";
                        default:
                            return "FFTW_TRANSFORM_KIND_STRING_NOT_FOUND";
                    }
                }
        };

    }
}

#endif /* end of include guard: FFTTRANSFORM_H */
