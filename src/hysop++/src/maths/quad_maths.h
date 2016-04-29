
#ifdef HAS_QUADMATHS

#ifndef HYSOP_QUAD_MATHS_H
#define HYSOP_QUAD_MATHS_H

#include <cfloat>
#include <cmath>
#include <limits>
#include <quadmath.h>

#include <iostream>
#include <iomanip>

/* missing gcc defines */
#define FLT128_RADIX FLT_RADIX
#define FLT128_HAS_DENORM      true
#define FLT128_HAS_INFINITY    true
#define FLT128_HAS_QUIET_NAN   true

namespace std {

    template<> struct numeric_limits<__float128> {
        static constexpr bool is_specialized = true;

        static constexpr __float128 min() { return FLT128_MIN; }
        static constexpr __float128 max() { return FLT128_MAX; }
        static constexpr __float128 lowest() noexcept { return -FLT128_MAX; }

        static constexpr int digits = FLT128_MANT_DIG;
        static constexpr int digits10 = FLT128_DIG;
        static constexpr int max_digits10 = (2 + (FLT128_MANT_DIG) * 643L / 2136);

        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr int radix = FLT128_RADIX;

        static constexpr __float128 epsilon()      { return FLT128_EPSILON; }
        static constexpr __float128 round_error()  { return 0.5; }

        static constexpr int min_exponent = FLT128_MIN_EXP;
        static constexpr int min_exponent10 = FLT128_MIN_10_EXP;
        static constexpr int max_exponent = FLT128_MAX_EXP;
        static constexpr int max_exponent10 = FLT128_MAX_10_EXP;

        static constexpr bool has_infinity = FLT128_HAS_INFINITY;
        static constexpr bool has_quiet_NaN = FLT128_HAS_QUIET_NAN;
        static constexpr bool has_signaling_NaN = has_quiet_NaN;
        static constexpr float_denorm_style has_denorm = bool(FLT128_HAS_DENORM) ? denorm_present : denorm_absent;
        static constexpr bool has_denorm_loss = std::numeric_limits<float>::has_denorm_loss;

        static constexpr __float128 infinity()      { return std::numeric_limits<float>::infinity(); }
        static constexpr __float128 quiet_NaN()     { return std::numeric_limits<float>::quiet_NaN(); }
        static constexpr __float128 signaling_NaN() { return std::numeric_limits<float>::signaling_NaN(); }
        static constexpr __float128 denorm_min()    { return FLT128_DENORM_MIN; }

        static constexpr bool is_iec559 = has_infinity && has_quiet_NaN && has_denorm == denorm_present;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;

        static constexpr bool traps = false;
        static constexpr bool tinyness_before = false;
        static constexpr float_round_style round_style = round_to_nearest;
    };

    inline int fpclassify(__float128 arg) { return std::fpclassify(static_cast<long double>(arg)); }

    inline __float128 abs(__float128 x) { return cabsq(__complex128{x,0.0Q}); }
    inline __float128 acos(__float128 x) { return acosq(x); }
    inline __float128 acosh(__float128 x) { return acoshq(x); }
    inline __float128 asin(__float128 x) { return asinq(x); }
    inline __float128 asinh(__float128 x) { return asinhq(x); }
    inline __float128 atan(__float128 x) { return atanq(x); }
    inline __float128 atanh(__float128 x) { return atanhq(x); }
    inline __float128 cbrt(__float128 x) { return cbrtq(x); }
    inline __float128 ceil(__float128 x) { return ceilq(x); }
    inline __float128 cosh(__float128 x) { return coshq(x); }
    inline __float128 cos(__float128 x) { return cosq(x); }
    inline __float128 erf(__float128 x) { return erfq(x); }
    inline __float128 erfc(__float128 x) { return erfcq(x); }
    inline __float128 exp(__float128 x) { return expq(x); }
    inline __float128 expm1(__float128 x) { return expm1q(x); }
    inline __float128 fabs(__float128 x) { return fabsq(x); }
    inline int        finite(__float128 x) { return finiteq(x); }
    inline __float128 floor(__float128 x) { return floorq(x); }
    inline __float128 frexp(__float128 x, int* p) { return frexpq(x,p); }
    inline int        isinf(__float128 x) { return isinfq(x); }
    inline int        ilogb(__float128 x) { return ilogbq(x); }
    inline int        isnan(__float128 x) { return isnanq(x); }
    inline __float128 j0(__float128 x) { return j0q(x); }
    inline __float128 j1(__float128 x) { return j1q(x); }
    inline __float128 jn(int i, __float128 x) { return jnq(i,x); }
    inline __float128 ldexp(__float128 x, int i) { return ldexpq(x,i); }
    inline __float128 lgamma(__float128 x) { return lgammaq(x); }
    inline long long int llrint(__float128 x) { return llrintq(x); }
    inline long long int llround(__float128 x) { return llroundq(x); }
    inline __float128 log(__float128 x) { return logq(x); }
    inline __float128 log10(__float128 x) { return log10q(x); }
    inline __float128 log2(__float128 x) { return log2q(x); }
    inline __float128 log1p(__float128 x) { return log1pq(x); }
    inline long int   lrint(__float128 x) { return lrintq(x); }
    inline long int   lround(__float128 x) { return lroundq(x); }
    inline __float128 nearbyint(__float128 x) { return nearbyintq(x); }
    inline __float128 pow(__float128 x, __float128 y) { return powq(x,y); }
    inline __float128 rint(__float128 x) { return rintq(x); }
    inline __float128 round(__float128 x) { return roundq(x); }
    inline __float128 scalbln(__float128 x, long int li) { return scalblnq(x,li); }
    inline __float128 scalbn(__float128 x, int i) { return scalbnq(x,i); }
    inline int        signbit(__float128 x) { return signbitq(x); }
    inline __float128 sinh(__float128 x) { return sinhq(x); }
    inline __float128 sin(__float128 x) { return sinq(x); }
    inline __float128 sqrt(__float128 x) { return sqrtq(x); }
    inline __float128 tan(__float128 x) { return tanq(x); }
    inline __float128 tanh(__float128 x) { return tanhq(x); }
    inline __float128 tgamma(__float128 x) { return tgammaq(x); }
    inline __float128 trunc(__float128 x) { return truncq(x); }
    inline __float128 y0(__float128 x) { return y0q(x); }
    inline __float128 y1(__float128 x) { return y1q(x); }
    inline __float128 yn(int i, __float128 x) { return ynq(i,x); }


    /* Prototypes for complex functions */
    inline __float128 abs(__complex128 x) { return cabsq(x); }
    inline __float128 arg(__complex128 x) { return cargq(x); }
    inline __float128 imag(__complex128 x) { return cimagq(x); }
    inline __float128 real(__complex128 x) { return crealq(x); }
    inline __complex128 acos(__complex128 x) { return cacosq(x); }
    inline __complex128 acosh(__complex128 x) { return cacoshq(x); }
    inline __complex128 asin(__complex128 x) { return casinq(x); }
    inline __complex128 asinh(__complex128 x) { return casinhq(x); }
    inline __complex128 atan(__complex128 x) { return catanq(x); }
    inline __complex128 atanh(__complex128 x) { return catanhq(x); }
    inline __complex128 cos(__complex128 x) { return ccosq(x); }
    inline __complex128 cosh(__complex128 x) { return ccoshq(x); }
    inline __complex128 exp(__complex128 x) { return cexpq(x); }
    inline __complex128 expi(__float128 x) { return cexpiq(x); }
    inline __complex128 log10(__complex128 x) { return clog10q(x); }
    inline __complex128 conj(__complex128 x) { return conjq(x); }
    inline __complex128 pow(__complex128 x, __complex128 y) { return cpowq(x,y); }
    inline __complex128 proj(__complex128 x) { return cprojq(x); }
    inline __complex128 sin(__complex128 x) { return csinq(x); }
    inline __complex128 sinh(__complex128 x) { return csinhq(x); }
    inline __complex128 sqrt(__complex128 x) { return csqrtq(x); }
    inline __complex128 tan(__complex128 x) { return ctanq(x); }
    inline __complex128 tanh(__complex128 x) { return ctanhq(x); }

    inline std::ostream& operator<<(std::ostream& os, __float128 x) {
        const int prec = os.precision();

        if(prec==0) {
            os << static_cast<long long int>(x);
        }
        else { 
            char buf[128];
            const std::string format = "%+-#"+std::to_string(prec)+".*Qe";
            const int n = quadmath_snprintf(buf,128,format.c_str(),prec,x);
            if(n>127) {
                char *str = new char[n+1];
                if (str)
                    quadmath_snprintf (str,n+1,format.c_str(),prec,x);
                os << str;
                delete(str);
            }
            else {
                os << buf;
            }
        }

        return os;
    }
}

#include <complex>

namespace std {

    inline __float128 abs(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        return cabsq(X);
    }
    inline __float128 arg(std::complex<__float128> x) {
        __complex128 X{x.real(),x.imag()};
        return cargq(X);
    }
    inline __float128 imag(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        return cimagq(X);
    }
    inline __float128 real(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        return crealq(X);
    }
    inline std::complex<__float128> acos(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = cacosq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> acosh(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = cacoshq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> asin(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = casinq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> asinh(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = casinhq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> atan(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = catanq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> atanh(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = catanhq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> cos(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = ccosq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> cosh(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = ccoshq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> exp(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = cexpq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    //inline std::complex<__float128> expi(__float128 x) { 
        //__complex128 X = cexpiq(x);
        //return std::complex<__float128>(__real__ X, __imag__ X);
    //}
    inline std::complex<__float128> log10(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = clog10q(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> conj(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = conjq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> pow(std::complex<__float128> x, std::complex<__float128> y) {
        __complex128 X{x.real(),x.imag()};
        __complex128 Y{y.real(),y.imag()};
        X = cpowq(X,Y);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> proj(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = cprojq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> sin(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = csinq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> sinh(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = csinhq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> sqrt(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = csqrtq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> tan(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = ctanq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }
    inline std::complex<__float128> tanh(std::complex<__float128> x) { 
        __complex128 X{x.real(),x.imag()};
        X = ctanhq(X);
        return std::complex<__float128>(__real__ X, __imag__ X);
    }

    inline std::complex< __float128 > pow(std::complex< __float128> x , __float128 y) {
        __float128 R = powq(std::abs(x), y);
        __float128 phi = atanq(x.imag()/x.real());
        return std::complex<__float128 >(R*cosq(y*phi), R*sinq(y*phi));
    }
}

#endif /* end of include guard: HYSOP_QUAD_MATHS_H */

#endif
