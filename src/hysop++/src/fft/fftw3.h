
#ifndef HYSOP_FFTW3_H
#define HYSOP_FFTW3_H

#include <complex>
#include <fftw3.h>

#ifdef HAS_QUADMATHS
#include "maths/quad_maths.h"
#endif

/*                                                         */
/* fftw3 c++ wrapper based on original macros in <fftw3.h> */
/*                                                         */

/* macros normally already defined in <fftw3.h> */
#ifndef FFTW_CONCAT
#define FFTW_CONCAT(prefix, name) prefix ## name
#endif
#ifndef FFTW_MANGLE_FLOAT
#define FFTW_MANGLE_FLOAT(name)       FFTW_CONCAT(fftwf_, name)
#endif
#ifndef FFTW_MANGLE_LONG_DOUBLE
#define FFTW_MANGLE_LONG_DOUBLE(name) FFTW_CONCAT(fftwl_, name)
#endif
#ifndef FFTW_MANGLE_QUAD
#define FFTW_MANGLE_QUAD(name)        FFTW_CONCAT(fftwq_, name)
#endif
/***********************************************/

#undef FFTW_MANGLE_DOUBLE
#define FFTW_MANGLE_DOUBLE(name)      FFTW_CONCAT(::fftw_, name)

/* prefix for function wrappers inside the class */
#define FFTW_MANGLE_CLASS(name)      FFTW_CONCAT(fftw_, name)

/* macro used to generate a full template specialisation of class Fftw3 for each type */
#define FFTW_DEFINE_CXX_API(X, Y, REAL, has_thread_support)                                                                                   \
template <>                                                                                                                                   \
    struct Fftw3<REAL> {                                                                                                                      \
        typedef REAL R;                                                                                                                       \
        typedef Y(complex) C;                                                                                                                 \
        typedef Y(plan) plan;                                                                                                                 \
        typedef Y(iodim) iodim;                                                                                                               \
        typedef Y(iodim64) iodim64;                                                                                                           \
        typedef Y(r2r_kind) r2r_kind;                                                                                                         \
        typedef Y(read_char_func) read_char_func;                                                                                             \
        typedef Y(write_char_func) write_char_func;                                                                                           \
        typedef std::complex<REAL> stdC;                                                                                                      \
                                                                                                                                              \
        void X(execute)(const plan p) const {                                                                                                 \
            Y(execute)(p);                                                                                                                    \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_dft)(int rank, const int *n,                                                                                              \
                C *in, C *out, int sign, unsigned int flags) const {                                                                          \
            return Y(plan_dft)(rank, n, in, out, sign, flags);                                                                                \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_dft_1d)(int n, C *in, C *out, int sign,                                                                                   \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_dft_1d)(n, in, out, sign, flags);                                                                                   \
        }                                                                                                                                     \
        plan X(plan_dft_2d)(int n0, int n1,                                                                                                   \
                C *in, C *out, int sign, unsigned int flags) const {                                                                          \
            return Y(plan_dft_2d)(n0, n1, in, out, sign, flags);                                                                              \
        }                                                                                                                                     \
        plan X(plan_dft_3d)(int n0, int n1, int n2,                                                                                           \
                C *in, C *out, int sign, unsigned int flags) const {                                                                          \
            return Y(plan_dft_3d)(n0, n1, n2, in, out, sign, flags);                                                                          \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_many_dft)(int rank, const int *n,                                                                                         \
                int howmany,                                                                                                                  \
                C *in, const int *inembed,                                                                                                    \
                int istride, int idist,                                                                                                       \
                C *out, const int *onembed,                                                                                                   \
                int ostride, int odist,                                                                                                       \
                int sign, unsigned int flags) const {                                                                                         \
            return Y(plan_many_dft)(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags);                \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_guru_dft)(int rank, const iodim *dims,                                                                                    \
                int howmany_rank,                                                                                                             \
                const iodim *howmany_dims,                                                                                                    \
                C *in, C *out,                                                                                                                \
                int sign, unsigned int flags) const {                                                                                         \
            return Y(plan_guru_dft)(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);                                            \
        }                                                                                                                                     \
        plan X(plan_guru_split_dft)(int rank, const iodim *dims,                                                                              \
                int howmany_rank,                                                                                                             \
                const iodim *howmany_dims,                                                                                                    \
                R *ri, R *ii, R *ro, R *io,                                                                                                   \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru_split_dft)(rank, dims, howmany_rank, howmany_dims, ri, ii, ro, io, flags);                                     \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_guru64_dft)(int rank,                                                                                                     \
                const iodim64 *dims,                                                                                                          \
                int howmany_rank,                                                                                                             \
                const iodim64 *howmany_dims,                                                                                                  \
                C *in, C *out,                                                                                                                \
                int sign, unsigned int flags) const {                                                                                         \
            return Y(plan_guru64_dft)(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);                                          \
        }                                                                                                                                     \
        plan X(plan_guru64_split_dft)(int rank,                                                                                               \
                const iodim64 *dims,                                                                                                          \
                int howmany_rank,                                                                                                             \
                const iodim64 *howmany_dims,                                                                                                  \
                R *ri, R *ii, R *ro, R *io,                                                                                                   \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru64_split_dft)(rank, dims, howmany_rank, howmany_dims, ri, ii, ro, io, flags);                                   \
        }                                                                                                                                     \
                                                                                                                                              \
        void X(execute_dft)(const plan p, C *in, C *out) const {                                                                              \
            Y(execute_dft)(p, in, out);                                                                                                       \
        }                                                                                                                                     \
        void X(execute_split_dft)(const plan p, R *ri, R *ii, R *ro, R *io) const {                                                           \
            Y(execute_split_dft)(p, ri, ii, ro, io);                                                                                          \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_many_dft_r2c)(int rank, const int *n,                                                                                     \
                int howmany,                                                                                                                  \
                R *in, const int *inembed,                                                                                                    \
                int istride, int idist,                                                                                                       \
                C *out, const int *onembed,                                                                                                   \
                int ostride, int odist,                                                                                                       \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_many_dft_r2c)(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, flags);                  \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_dft_r2c)(int rank, const int *n,                                                                                          \
                R *in, C *out, unsigned int flags) const {                                                                                    \
            return Y(plan_dft_r2c)(rank, n, in, out, flags);                                                                                  \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_dft_r2c_1d)(int n,R *in,C *out,unsigned int flags) const {                                                                \
            return Y(plan_dft_r2c_1d)(n,in,out,flags);                                                                                        \
        }                                                                                                                                     \
        plan X(plan_dft_r2c_2d)(int n0, int n1,                                                                                               \
                R *in, C *out, unsigned int flags) const {                                                                                    \
            return Y(plan_dft_r2c_2d)(n0, n1, in, out, flags);                                                                                \
        }                                                                                                                                     \
        plan X(plan_dft_r2c_3d)(int n0, int n1,                                                                                               \
                int n2,                                                                                                                       \
                R *in, C *out, unsigned int flags) const {                                                                                    \
            return Y(plan_dft_r2c_3d)(n0, n1, n2, in, out, flags);                                                                            \
        }                                                                                                                                     \
                                                                                                                                              \
                                                                                                                                              \
        plan X(plan_many_dft_c2r)(int rank, const int *n,                                                                                     \
                int howmany,                                                                                                                  \
                C *in, const int *inembed,                                                                                                    \
                int istride, int idist,                                                                                                       \
                R *out, const int *onembed,                                                                                                   \
                int ostride, int odist,                                                                                                       \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_many_dft_c2r)(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, flags);                  \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_dft_c2r)(int rank, const int *n,                                                                                          \
                C *in, R *out, unsigned int flags) const {                                                                                    \
            return Y(plan_dft_c2r)(rank, n, in, out, flags);                                                                                  \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_dft_c2r_1d)(int n,C *in,R *out,unsigned int flags) const {                                                                \
            return Y(plan_dft_c2r_1d)(n,in,out,flags);                                                                                        \
        }                                                                                                                                     \
        plan X(plan_dft_c2r_2d)(int n0, int n1,                                                                                               \
                C *in, R *out, unsigned int flags) const {                                                                                    \
            return Y(plan_dft_c2r_2d)(n0, n1, in, out, flags);                                                                                \
        }                                                                                                                                     \
        plan X(plan_dft_c2r_3d)(int n0, int n1,                                                                                               \
                int n2,                                                                                                                       \
                C *in, R *out, unsigned int flags) const {                                                                                    \
            return Y(plan_dft_c2r_3d)(n0, n1, n2, in, out, flags);                                                                            \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_guru_dft_r2c)(int rank, const iodim *dims,                                                                                \
                int howmany_rank,                                                                                                             \
                const iodim *howmany_dims,                                                                                                    \
                R *in, C *out,                                                                                                                \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru_dft_r2c)(rank, dims, howmany_rank, howmany_dims, in, out, flags);                                              \
        }                                                                                                                                     \
        plan X(plan_guru_dft_c2r)(int rank, const iodim *dims,                                                                                \
                int howmany_rank,                                                                                                             \
                const iodim *howmany_dims,                                                                                                    \
                C *in, R *out,                                                                                                                \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru_dft_c2r)(rank, dims, howmany_rank, howmany_dims, in, out, flags);                                              \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_guru_split_dft_r2c)(                                                                                                      \
                int rank, const iodim *dims,                                                                                                  \
                int howmany_rank,                                                                                                             \
                const iodim *howmany_dims,                                                                                                    \
                R *in, R *ro, R *io,                                                                                                          \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru_split_dft_r2c)( rank, dims, howmany_rank, howmany_dims, in, ro, io, flags);                                    \
        }                                                                                                                                     \
        plan X(plan_guru_split_dft_c2r)(                                                                                                      \
                int rank, const iodim *dims,                                                                                                  \
                int howmany_rank,                                                                                                             \
                const iodim *howmany_dims,                                                                                                    \
                R *ri, R *ii, R *out,                                                                                                         \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru_split_dft_c2r)( rank, dims, howmany_rank, howmany_dims, ri, ii, out, flags);                                   \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_guru64_dft_r2c)(int rank,                                                                                                 \
                const iodim64 *dims,                                                                                                          \
                int howmany_rank,                                                                                                             \
                const iodim64 *howmany_dims,                                                                                                  \
                R *in, C *out,                                                                                                                \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru64_dft_r2c)(rank, dims, howmany_rank, howmany_dims, in, out, flags);                                            \
        }                                                                                                                                     \
        plan X(plan_guru64_dft_c2r)(int rank,                                                                                                 \
                const iodim64 *dims,                                                                                                          \
                int howmany_rank,                                                                                                             \
                const iodim64 *howmany_dims,                                                                                                  \
                C *in, R *out,                                                                                                                \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru64_dft_c2r)(rank, dims, howmany_rank, howmany_dims, in, out, flags);                                            \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_guru64_split_dft_r2c)(                                                                                                    \
                int rank, const iodim64 *dims,                                                                                                \
                int howmany_rank,                                                                                                             \
                const iodim64 *howmany_dims,                                                                                                  \
                R *in, R *ro, R *io,                                                                                                          \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru64_split_dft_r2c)( rank, dims, howmany_rank, howmany_dims, in, ro, io, flags);                                  \
        }                                                                                                                                     \
        plan X(plan_guru64_split_dft_c2r)(                                                                                                    \
                int rank, const iodim64 *dims,                                                                                                \
                int howmany_rank,                                                                                                             \
                const iodim64 *howmany_dims,                                                                                                  \
                R *ri, R *ii, R *out,                                                                                                         \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_guru64_split_dft_c2r)( rank, dims, howmany_rank, howmany_dims, ri, ii, out, flags);                                 \
        }                                                                                                                                     \
                                                                                                                                              \
        void X(execute_dft_r2c)(const plan p, R *in, C *out) const {                                                                          \
            Y(execute_dft_r2c)(p, in, out);                                                                                                   \
        }                                                                                                                                     \
        void X(execute_dft_c2r)(const plan p, C *in, R *out) const {                                                                          \
            Y(execute_dft_c2r)(p, in, out);                                                                                                   \
        }                                                                                                                                     \
                                                                                                                                              \
        void X(execute_split_dft_r2c)(const plan p,                                                                                           \
                R *in, R *ro, R *io) const {                                                                                                  \
            return Y(execute_split_dft_r2c)(p, in, ro, io);                                                                                   \
        }                                                                                                                                     \
        void X(execute_split_dft_c2r)(const plan p,                                                                                           \
                R *ri, R *ii, R *out) const {                                                                                                 \
            return Y(execute_split_dft_c2r)(p, ri, ii, out);                                                                                  \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_many_r2r)(int rank, const int *n,                                                                                         \
                int howmany,                                                                                                                  \
                R *in, const int *inembed,                                                                                                    \
                int istride, int idist,                                                                                                       \
                R *out, const int *onembed,                                                                                                   \
                int ostride, int odist,                                                                                                       \
                const r2r_kind *kind, unsigned int flags) const {                                                                             \
            return Y(plan_many_r2r)(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, kind, flags);                \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_r2r)(int rank, const int *n, R *in, R *out,                                                                               \
                const r2r_kind *kind, unsigned int flags) const {                                                                             \
            return Y(plan_r2r)(rank, n, in, out, kind, flags);                                                                                \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_r2r_1d)(int n, R *in, R *out,                                                                                             \
                r2r_kind kind, unsigned int flags) const {                                                                                    \
            return Y(plan_r2r_1d)(n, in, out, kind, flags);                                                                                   \
        }                                                                                                                                     \
        plan X(plan_r2r_2d)(int n0, int n1, R *in, R *out,                                                                                    \
                r2r_kind kind0, r2r_kind kind1,                                                                                               \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_r2r_2d)(n0, n1, in, out, kind0, kind1, flags);                                                                      \
        }                                                                                                                                     \
        plan X(plan_r2r_3d)(int n0, int n1, int n2,                                                                                           \
                R *in, R *out, r2r_kind kind0,                                                                                                \
                r2r_kind kind1, r2r_kind kind2,                                                                                               \
                unsigned int flags) const {                                                                                                   \
            return Y(plan_r2r_3d)(n0, n1, n2, in, out, kind0, kind1, kind2, flags);                                                           \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_guru_r2r)(int rank, const iodim *dims,                                                                                    \
                int howmany_rank,                                                                                                             \
                const iodim *howmany_dims,                                                                                                    \
                R *in, R *out,                                                                                                                \
                const r2r_kind *kind, unsigned int flags) const {                                                                             \
            return Y(plan_guru_r2r)(rank, dims, howmany_rank, howmany_dims, in, out, kind, flags);                                            \
        }                                                                                                                                     \
                                                                                                                                              \
        plan X(plan_guru64_r2r)(int rank, const iodim64 *dims,                                                                                \
                int howmany_rank,                                                                                                             \
                const iodim64 *howmany_dims,                                                                                                  \
                R *in, R *out,                                                                                                                \
                const r2r_kind *kind, unsigned int flags) const {                                                                             \
            return Y(plan_guru64_r2r)(rank, dims, howmany_rank, howmany_dims, in, out, kind, flags);                                          \
        }                                                                                                                                     \
                                                                                                                                              \
        void X(execute_r2r)(const plan p, R *in, R *out) const {                                                                              \
            Y(execute_r2r)(p, in, out);                                                                                                       \
        }                                                                                                                                     \
                                                                                                                                              \
        void X(destroy_plan)(plan p) const {                                                                                                  \
            Y(destroy_plan)(p);                                                                                                               \
        }                                                                                                                                     \
        void X(forget_wisdom)() const {                                                                                                       \
            Y(forget_wisdom)();                                                                                                               \
        }                                                                                                                                     \
        void X(cleanup)() const {                                                                                                             \
            Y(cleanup)();                                                                                                                     \
        }                                                                                                                                     \
                                                                                                                                              \
        void X(set_timelimit)(double t) const {                                                                                               \
            Y(set_timelimit)(t);                                                                                                              \
        }                                                                                                                                     \
                                                                                                                                              \
        template <typename T = void>                                                                                                          \
        typename std::enable_if<has_thread_support, T>::type                                                                                  \
        X(plan_with_nthreads)(int nthreads) const {                                                                                           \
            Y(plan_with_nthreads)(nthreads);                                                                                                  \
        }                                                                                                                                     \
        template <typename T = int>                                                                                                           \
        typename std::enable_if<has_thread_support, T>::type                                                                                  \
        X(init_threads)() const {                                                                                                             \
            return Y(init_threads)();                                                                                                         \
        }                                                                                                                                     \
        template <typename T=void>                                                                                                            \
        typename std::enable_if<has_thread_support, T>::type                                                                                  \
        X(cleanup_threads)() const {                                                                                                          \
            Y(cleanup_threads)();                                                                                                             \
        }                                                                                                                                     \
                                                                                                                                              \
        int X(export_wisdom_to_filename)(const char *filename) const {                                                                        \
            return Y(export_wisdom_to_filename)(filename);                                                                                    \
        }                                                                                                                                     \
        void X(export_wisdom_to_file)(FILE *output_file) const {                                                                              \
            Y(export_wisdom_to_file)(output_file);                                                                                            \
        }                                                                                                                                     \
        char *X(export_wisdom_to_string)() const {                                                                                            \
            return Y(export_wisdom_to_string)();                                                                                              \
        }                                                                                                                                     \
        void X(export_wisdom)(write_char_func write_char,                                                                                     \
                void *data) const {                                                                                                           \
            return Y(export_wisdom)(write_char, data);                                                                                        \
        }                                                                                                                                     \
        int X(import_system_wisdom)() const {                                                                                                 \
            return Y(import_system_wisdom)();                                                                                                 \
        }                                                                                                                                     \
        int X(import_wisdom_from_filename)(const char *filename) const {                                                                      \
            return Y(import_wisdom_from_filename)(filename);                                                                                  \
        }                                                                                                                                     \
        int X(import_wisdom_from_file)(FILE *input_file) const {                                                                              \
            return Y(import_wisdom_from_file)(input_file);                                                                                    \
        }                                                                                                                                     \
        int X(import_wisdom_from_string)(const char *input_string) const {                                                                    \
            return Y(import_wisdom_from_string)(input_string);                                                                                \
        }                                                                                                                                     \
        int X(import_wisdom)(read_char_func read_char, void *data) const {                                                                    \
            return Y(import_wisdom)(read_char, data);                                                                                         \
        }                                                                                                                                     \
                                                                                                                                              \
        void X(fprint_plan)(const plan p, FILE *output_file) const {                                                                          \
            Y(fprint_plan)(p, output_file);                                                                                                   \
        }                                                                                                                                     \
        void X(print_plan)(const plan p) const {                                                                                              \
            Y(print_plan)(p);                                                                                                                 \
        }                                                                                                                                     \
                                                                                                                                              \
        void *X(malloc)(size_t n) const {                                                                                                     \
            return Y(malloc)(n);                                                                                                              \
        }                                                                                                                                     \
        R *X(alloc_real)(size_t n) const {                                                                                                    \
            return Y(alloc_real)(n);                                                                                                          \
        }                                                                                                                                     \
        C *X(alloc_complex)(size_t n) const {                                                                                                 \
            return Y(alloc_complex)(n);                                                                                                       \
        }                                                                                                                                     \
        void X(free)(void *p) const {                                                                                                         \
            Y(free)(p);                                                                                                                       \
        }                                                                                                                                     \
                                                                                                                                              \
        void X(flops)(const plan p,                                                                                                           \
                double *add, double *mul, double *fmas) const {                                                                               \
            return Y(flops)(p, add, mul, fmas);                                                                                               \
        }                                                                                                                                     \
        double X(estimate_cost)(const plan p) const {                                                                                         \
            return Y(estimate_cost)(p);                                                                                                       \
        }                                                                                                                                     \
        double X(cost)(const plan p) const {                                                                                                  \
            return Y(cost)(p);                                                                                                                \
        }                                                                                                                                     \
    };


/* Constants */
namespace hysop {
    namespace fft {
        /* float */
        #ifdef FFTW_HAS_FFTW3F
            static constexpr bool fftw_has_float_support = true;
            #ifdef FFTW_HAS_FFTW3F_MPI
                static constexpr bool fftw_has_float_mpi_support = true;
            #else 
                static constexpr bool fftw_has_float_mpi_support = false;
            #endif
            #if defined(FFTW_HAS_FFTW3F_THREADS) || defined(FFTW_HAS_FFTW3F_OMP)
                static constexpr bool fftw_has_float_thread_support = true;
            #else
                static constexpr bool fftw_has_float_thread_support = false;
            #endif 
        #else
            static constexpr bool fftw_has_float_support        = false;
            static constexpr bool fftw_has_float_thread_support = false;
            static constexpr bool fftw_has_float_mpi_support    = false;
        #endif

        /* double */
        #ifdef FFTW_HAS_FFTW3D
            static constexpr bool fftw_has_double_support = true;
            #ifdef FFTW_HAS_FFTW3D_MPI
                static constexpr bool fftw_has_double_mpi_support = true;
            #else 
                static constexpr bool fftw_has_double_mpi_support = false;
            #endif
            #if defined(FFTW_HAS_FFTW3D_THREADS) || defined(FFTW_HAS_FFTW3D_OMP)
                static constexpr bool fftw_has_double_thread_support = true;
            #else
                static constexpr bool fftw_has_double_thread_support = false;
            #endif 
        #else
            static constexpr bool fftw_has_double_support        = false;
            static constexpr bool fftw_has_double_thread_support = false;
            static constexpr bool fftw_has_double_mpi_support    = false;
        #endif
        
        /* long double */
        #ifdef FFTW_HAS_FFTW3L
            static constexpr bool fftw_has_long_double_support = true;
            #ifdef FFTW_HAS_FFTW3L_MPI
                static constexpr bool fftw_has_long_double_mpi_support = true;
            #else 
                static constexpr bool fftw_has_long_double_mpi_support = false;
            #endif
            #if defined(FFTW_HAS_FFTW3L_THREADS) || defined(FFTW_HAS_FFTW3L_OMP)
                static constexpr bool fftw_has_long_double_thread_support = true;
            #else
                static constexpr bool fftw_has_long_double_thread_support = false;
            #endif 
        #else
            static constexpr bool fftw_has_long_double_support        = false;
            static constexpr bool fftw_has_long_double_thread_support = false;
            static constexpr bool fftw_has_long_double_mpi_support    = false;
        #endif
        
        /* __float128  */
        #ifdef FFTW_HAS_FFTW3Q
            static constexpr bool fftw_has_quad_float_support = true;
            #ifdef FFTW_HAS_FFTW3Q_MPI
                static constexpr bool fftw_has_quad_float_mpi_support = true;
            #else 
                static constexpr bool fftw_has_quad_float_mpi_support = false;
            #endif
            #if defined(FFTW_HAS_FFTW3Q_THREADS) || defined(FFTW_HAS_FFTW3Q_OMP)
                static constexpr bool fftw_has_quad_float_thread_support = true;
            #else
                static constexpr bool fftw_has_quad_float_thread_support = false;
            #endif 
        #else
            static constexpr bool fftw_has_quad_float_support        = false;
            static constexpr bool fftw_has_quad_float_thread_support = false;
            static constexpr bool fftw_has_quad_float_mpi_support    = false;
        #endif
    }
}

/* Wrappers */
namespace hysop {
    namespace fft {

        template <typename T>
        struct Fftw3 {
                Fftw3() { 
                    throw std::runtime_error(
                            "Can only use Fftw3 wrapper with types {float, double, long double, __float128} ! "
                            "Note: __float128 type is enabled only if HAS_QUADMATHS is defined."
                            );
                }
            };
        template <typename T> struct is_fftw_supported_type          { static constexpr bool value = false; };
        template <typename T> struct is_fftw_supported_complex_type  { static constexpr bool value = false; };
        
        
        /* Generate Ffftw<> template specialisations */
        FFTW_DEFINE_CXX_API(FFTW_MANGLE_CLASS, FFTW_MANGLE_FLOAT, float, hysop::fft::fftw_has_float_thread_support)
        FFTW_DEFINE_CXX_API(FFTW_MANGLE_CLASS, FFTW_MANGLE_DOUBLE, double, hysop::fft::fftw_has_double_thread_support)
        FFTW_DEFINE_CXX_API(FFTW_MANGLE_CLASS, FFTW_MANGLE_LONG_DOUBLE, long double, hysop::fft::fftw_has_long_double_thread_support)

        template <> struct is_fftw_supported_type<float>       { static constexpr bool value = true;  };
        template <> struct is_fftw_supported_type<double>      { static constexpr bool value = true;  };
        template <> struct is_fftw_supported_type<long double> { static constexpr bool value = true;  };
        
        template <> struct is_fftw_supported_complex_type<std::complex<float>>       { static constexpr bool value = true;  };
        template <> struct is_fftw_supported_complex_type<std::complex<double>>      { static constexpr bool value = true;  };
        template <> struct is_fftw_supported_complex_type<std::complex<long double>> { static constexpr bool value = true;  };
        template <> struct is_fftw_supported_complex_type<fftwf_complex>             { static constexpr bool value = true;  };
        template <> struct is_fftw_supported_complex_type<fftw_complex>              { static constexpr bool value = true;  };
        template <> struct is_fftw_supported_complex_type<fftwl_complex>             { static constexpr bool value = true;  };

#ifdef HAS_QUADMATHS
        FFTW_DEFINE_CXX_API(FFTW_MANGLE_CLASS, FFTW_MANGLE_QUAD, __float128, hysop::fft::fftw_has_quad_float_thread_support)
        template <> struct is_fftw_supported_type<__float128>                        { static constexpr bool value = true;  };
        template <> struct is_fftw_supported_complex_type<std::complex<__float128>>  { static constexpr bool value = true;  };
        template <> struct is_fftw_supported_complex_type<fftwq_complex>             { static constexpr bool value = true;  };
#endif

        
    }
}

#undef FFTW_DEFINE_CXX_API
#undef FFTW_MANGLE_CLASS

#endif /* end of include guard: HYSOP_FFTW3_H */

