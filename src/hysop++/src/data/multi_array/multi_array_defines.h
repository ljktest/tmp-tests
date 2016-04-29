
#ifndef HYSOP_MULTI_ARRAY_H
#include "data/multi_array/multi_array.h"
#else 

#ifndef HYSOP_MULTI_ARRAY_DEFINES_H
#define HYSOP_MULTI_ARRAY_DEFINES_H

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "utils/utils.h"
#include "utils/types.h"
#include "data/index.h"
#include "fft/fftwComplex.h"


/* helper macros to generate all hysop multi_array wrappers */
#define SINGLE_ARG(...) __VA_ARGS__

/* declaration of all public class types */
#define PUBLIC_CLASS_TYPES()                                                      \
    template <typename Alloc>                                                     \
    using array = multi_array<T,Dim,Alloc>;                                       \
                                                                                  \
    using array_ref        = multi_array_ref<T,Dim>;                              \
    using array_view       = multi_array_view<T,Dim>;                             \
    using const_array_ref  = const_multi_array_ref<T,Dim>;                        \
    using const_array_view = const_multi_array_view<T,Dim>;                       \
                                                                                  \
    template <std::size_t NExtents>                                               \
    using extents_gen = boost::detail::multi_array::extent_gen<NExtents>;         \
                                                                                  \
    template <std::size_t NumRanges, std::size_t NumDim>                          \
    using index_gen   = boost::detail::multi_array::index_gen<NumRanges, NumDim>; \
                                                                                  \
    using index_range = boost::multi_array_types::index_range;


/* various enable if macros */
#define ENABLE_IF_BOOL(TYPE,DEFAULT)                                     \
            template <typename TT DEFAULT>                               \
            typename std::enable_if<                                     \
                       std::is_same<TT,bool>::value                      \
                , TYPE>::type
#define ENABLE_IF_FFTW_REAL(TYPE,DEFAULT)                                \
            template <typename TT DEFAULT>                               \
            typename std::enable_if<                                     \
                    hysop::fft::is_fftw_supported_type<TT>::value        \
                , TYPE>::type 
#define ENABLE_IF_FFTW_COMPLEX(TYPE,DEFAULT)                               \
            template <typename TT DEFAULT>                                 \
            typename std::enable_if<                                       \
                    hysop::fft::is_fftw_supported_complex_type<TT>::value  \
                , TYPE>::type

/* CLASS INTERFACES */
/* public const common interfaces for views and referencess */
#define PUBLIC_COMMON_CONST_INTERFACE(TYPENAME)                                                                   \
        /* shape related */                                                                                      \
        typename Shape<Dim>::type shape() const;                                                                                \
                                                                                                                 \
        /* print data */                                                                                         \
        const TYPENAME& print(const std::string& name, std::ostream& os = std::cout,                             \
                unsigned int precision=2u, unsigned int width=6u) const;                                         \
                                                                                                                 \
        /* boolean array_view specific functions */                                                              \
        ENABLE_IF_BOOL(bool,=T) all() const;                                                                     \
        ENABLE_IF_BOOL(bool,=T) any() const;                                                                     \
        ENABLE_IF_BOOL(bool,=T) none() const;


/* public const reference interface */
#define PUBLIC_CONST_REF_INTERFACE(TYPENAME)                                                                    \
        /* common const interface */                                                                            \
        PUBLIC_COMMON_CONST_INTERFACE(SINGLE_ARG(TYPENAME))                                                      \
        /* real and complex data accessors, usefull for FFT like transforms */                                  \
        ENABLE_IF_FFTW_REAL   (const T*,=T) rdata() const;                                                           \
        ENABLE_IF_FFTW_REAL   (const typename fft::fftw_complex_type<T>::std_type*,=T)  asStdComplexData() const;    \
        ENABLE_IF_FFTW_REAL   (const typename fft::fftw_complex_type<T>::fftw_type*,=T) asFftwComplexData() const;   \
        ENABLE_IF_FFTW_COMPLEX(const T*,=T) cdata() const;                                                           \
        ENABLE_IF_FFTW_COMPLEX(const typename fft::fftw_complex_type<T>::std_type*,=T)    std_cdata() const;         \
        ENABLE_IF_FFTW_COMPLEX(const typename fft::fftw_complex_type<T>::fftw_type*,=T)  fftw_cdata() const;         \
        ENABLE_IF_FFTW_COMPLEX(const typename fft::fftw_complex_type<T>::value_type*,=T) asRealData() const;


/* public const view interface */
#define PUBLIC_CONST_VIEW_INTERFACE(TYPENAME)                                                     \
        /* common const interface */                                                              \
        PUBLIC_COMMON_CONST_INTERFACE(SINGLE_ARG(TYPENAME))


/* non const interfaces */
#define PUBLIC_COMMON_NON_CONST_INTERFACE(TYPENAME)                                               \
        /* Apply function to all elements */                                                      \
        TYPENAME& apply(const std::function<void(T&)>& func);                                     \
        TYPENAME& apply(const std::function<void(T&, const Index<Dim>&)>& func);                  \
        template <typename Functor, typename Arg0, typename... Args>                              \
        TYPENAME& apply(const Functor& func, Arg0&& farg0, Args&&... fargs);


#define PUBLIC_NON_CONST_REF_INTERFACE(TYPENAME)                                                  \
        /* common non const interface */                                                          \
        PUBLIC_COMMON_NON_CONST_INTERFACE(SINGLE_ARG(TYPENAME))                                   \
        /* real and complex data accessors, usefull for FFT like transforms */                    \
        ENABLE_IF_FFTW_REAL(T*,=T) rdata();                                                            \
        ENABLE_IF_FFTW_REAL(typename fft::fftw_complex_type<T>::std_type*,=T) asStdComplexData();      \
        ENABLE_IF_FFTW_REAL(typename fft::fftw_complex_type<T>::fftw_type*,=T) asFftwComplexData();    \
        ENABLE_IF_FFTW_COMPLEX(T*,=T) cdata();                                                         \
        ENABLE_IF_FFTW_COMPLEX(typename fft::fftw_complex_type<T>::std_type*,=T) std_cdata();          \
        ENABLE_IF_FFTW_COMPLEX(typename fft::fftw_complex_type<T>::fftw_type*,=T) fftw_cdata();        \
        ENABLE_IF_FFTW_COMPLEX(typename fft::fftw_complex_type<T>::value_type*,=T) asRealData();


#define PUBLIC_NON_CONST_VIEW_INTERFACE(TYPENAME)                                                 \
        /* common non const interface */                                                          \
        PUBLIC_COMMON_NON_CONST_INTERFACE(SINGLE_ARG(TYPENAME))



/* CLASS IMPLEMENTATIONS */

/* Loop dependant implementation macros (references contain contiguous data but not the views) */
#define LOOP_VARNAME multi_array_index
#define NO_DEFAULT_TEMPLATES 

/* A reference has a contiguous memory layout and can be accessed by data offsets */
#define LOOP_OVER_ALL_REF_ELEMENTS(ARRAY) \
    for (std::size_t LOOP_VARNAME=0; LOOP_VARNAME<((ARRAY).num_elements()); LOOP_VARNAME++)

#define REF_DATA_ACCESS(ARRAY) (ARRAY).data()[LOOP_VARNAME]

/* A view is non a contiguous memory access and can be accessed only by index list */
#define LOOP_OVER_ALL_VIEW_ELEMENTS(ARRAY)      \
    Index<Dim> LOOP_VARNAME((ARRAY).shape());     \
    LOOP_VARNAME.setIndexToMinusOne();          \
    while((++LOOP_VARNAME)() != LOOP_VARNAME.maxId())

#define VIEW_DATA_ACCESS(ARRAY) (ARRAY).operator()(LOOP_VARNAME.ids())


/* CONST IMPLEMENTATIONS */
/* Common const implementation */
#define COMMON_CONST_IMPLEMENTATION(TYPENAME,TEMPLATES)          \
        /* shape related */                                      \
        TEMPLATES                                                \
        typename Shape<Dim>::type TYPENAME::shape() const {                     \
            typename Shape<Dim>::type shape;                                    \
            const std::size_t* extents = this->super::shape();   \
            for (std::size_t d = 0; d < Dim; d++)                \
                shape[d] = static_cast<std::size_t>(extents[d]); \
            return shape;                                        \
        }

/* Loop dependant const implementation */
#define LOOP_DEPENDENT_CONST_IMPLEMENTATION(TYPENAME,TEMPLATES,LOOP_OVER_ALL_ELEMENTS,DATA_ACCESS)                            \
        /* print data */                                                                                                      \
        TEMPLATES                                                                                                             \
        const TYPENAME& TYPENAME::print(const std::string& name, std::ostream& os,                                            \
                unsigned int precision, unsigned int width) const {                                                           \
            auto S = this->super::shape();                                                                                    \
            std::size_t id = 0;                                                                                               \
                                                                                                                              \
            os << name << " = [";                                                                                             \
            if(Dim==1) {                                                                                                      \
                for(std::size_t k=0; k<this->num_elements(); k++) {                                                           \
                    T x = this->operator()(boost::array<std::size_t,1>{k});                                                   \
                    os << std::fixed << std::showpos << std::setprecision(precision) << std::setw(width) << x << " ";         \
                }                                                                                                             \
            }                                                                                                                 \
            else if(Dim==2) {                                                                                                 \
                std::cout << std::endl;                                                                                       \
                for(std::size_t i=0; i<S[0]; i++) {                                                                           \
                    os << "\t[";                                                                                              \
                    for(std::size_t j=0; j<S[1]; j++) {                                                                       \
                        T x = this->operator()(boost::array<std::size_t,2>{i,j});                                             \
                        os << std::fixed << std::showpos << std::setprecision(precision) << std::setw(width) << x << " ";     \
                    }                                                                                                         \
                    os << "]" << std::endl;                                                                                   \
                }                                                                                                             \
            }                                                                                                                 \
            else if(Dim==3) {                                                                                                 \
                std::cout << std::endl;                                                                                       \
                for(std::size_t i=0; i<S[0]; i++) {                                                                           \
                    os << "\t[[";                                                                                             \
                    for(std::size_t j=0; j<S[1]; j++) {                                                                       \
                        if(j>0)                                                                                               \
                            os << "\t [";                                                                                     \
                        for(std::size_t k=0; k<S[2]; k++) {                                                                   \
                            T x = this->operator()(boost::array<std::size_t,3>{i,j,k});                                       \
                            os << std::fixed << std::showpos << std::setprecision(precision) << std::setw(width) << x << " "; \
                        }                                                                                                     \
                        if(j!=S[1]-1)                                                                                         \
                            os << "]" << std::endl;                                                                           \
                        else                                                                                                  \
                            os << "]]," << std::endl;                                                                         \
                    }                                                                                                         \
                }                                                                                                             \
            }                                                                                                                 \
            else {                                                                                                            \
                LOOP_OVER_ALL_ELEMENTS(*this) {                                                                               \
                    T x = DATA_ACCESS(*this);                                                                                 \
                    os << std::fixed << std::showpos << std::setprecision(precision) << std::setw(width) << x << " ";         \
                }                                                                                                             \
            }                                                                                                                 \
            os << "];" << std::endl;                                                                                          \
            return *this;                                                                                                     \
        }                                                                                                   \
                                                                                                            \
        /* boolean array_view specific functions */                                                         \
        TEMPLATES ENABLE_IF_BOOL(bool,NO_DEFAULT_TEMPLATES) TYPENAME::all() const {                         \
            LOOP_OVER_ALL_ELEMENTS(*this) {                                                                 \
                const bool val = DATA_ACCESS(*this);                                                        \
                if(!val)                                                                                    \
                    return false;                                                                           \
            }                                                                                               \
            return true;                                                                                    \
        }                                                                                                   \
        TEMPLATES ENABLE_IF_BOOL(bool,NO_DEFAULT_TEMPLATES) TYPENAME::any() const {                         \
            LOOP_OVER_ALL_ELEMENTS(*this) {                                                                 \
                const bool val = DATA_ACCESS(*this);                                                        \
                if(val)                                                                                     \
                    return true;                                                                            \
            }                                                                                               \
            return false;                                                                                   \
        }                                                                                                   \
        TEMPLATES ENABLE_IF_BOOL(bool,NO_DEFAULT_TEMPLATES) TYPENAME::none() const {                        \
            LOOP_OVER_ALL_ELEMENTS(*this) {                                                                 \
                const bool val = DATA_ACCESS(*this);                                                        \
                if(val)                                                                                     \
                    return false;                                                                           \
            }                                                                                               \
            return true;                                                                                    \
        }

/* Reference specific const implementation */
#define CONST_REF_IMPL(TYPENAME,TEMPLATES)                                                                                          \
    TEMPLATES                                                                                                                       \
    ENABLE_IF_FFTW_REAL(const T*,NO_DEFAULT_TEMPLATES) TYPENAME::rdata() const {                                                         \
        return this->data();                                                                                                        \
    }                                                                                                                               \
    TEMPLATES                                                                                                                       \
    ENABLE_IF_FFTW_REAL(const typename fft::fftw_complex_type<T>::std_type*,NO_DEFAULT_TEMPLATES)  TYPENAME::asStdComplexData() const {  \
        return reinterpret_cast<const typename fft::fftw_complex_type<T>::std_type*>(this->data());                                 \
    }                                                                                                                               \
    TEMPLATES                                                                                                                       \
    ENABLE_IF_FFTW_REAL(const typename fft::fftw_complex_type<T>::fftw_type*,NO_DEFAULT_TEMPLATES) TYPENAME::asFftwComplexData() const { \
        return reinterpret_cast<const typename fft::fftw_complex_type<T>::fftw_type*>(this->data());                                \
    }                                                                                                                               \
    TEMPLATES                                                                                                                       \
    ENABLE_IF_FFTW_COMPLEX(const T*,NO_DEFAULT_TEMPLATES) TYPENAME::cdata() const {                                                      \
        return this->data();                                                                                                        \
    }                                                                                                                               \
    TEMPLATES                                                                                                                       \
    ENABLE_IF_FFTW_COMPLEX(const typename fft::fftw_complex_type<T>::std_type*,NO_DEFAULT_TEMPLATES)    TYPENAME::std_cdata() const {    \
        return reinterpret_cast<const typename fft::fftw_complex_type<T>::std_type*>(this->data());                                 \
    }                                                                                                                               \
    TEMPLATES                                                                                                                       \
    ENABLE_IF_FFTW_COMPLEX(const typename fft::fftw_complex_type<T>::fftw_type*,NO_DEFAULT_TEMPLATES)  TYPENAME::fftw_cdata() const {    \
        return reinterpret_cast<const typename fft::fftw_complex_type<T>::fftw_type*>(this->data());                                \
    }                                                                                                                               \
    TEMPLATES                                                                                                                       \
    ENABLE_IF_FFTW_COMPLEX(const typename fft::fftw_complex_type<T>::value_type*,NO_DEFAULT_TEMPLATES) TYPENAME::asRealData() const {    \
        return reinterpret_cast<const typename fft::fftw_complex_type<T>::value_type*>(this->data());                               \
    }                                                                                                                               

/* View specific const implementation */
#define CONST_VIEW_IMPL(TYPENAME,TEMPLATES)

/* All reference const implementation */
#define CONST_REF_IMPLEMENTATION(TYPENAME,TEMPLATES)                                                                             \
    COMMON_CONST_IMPLEMENTATION(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES))                                                      \
    LOOP_DEPENDENT_CONST_IMPLEMENTATION(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES),LOOP_OVER_ALL_REF_ELEMENTS,REF_DATA_ACCESS)   \
    CONST_REF_IMPL(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES))

/* All view const implementation */
#define CONST_VIEW_IMPLEMENTATION(TYPENAME,TEMPLATES)                                                                            \
    COMMON_CONST_IMPLEMENTATION(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES))                                                      \
    LOOP_DEPENDENT_CONST_IMPLEMENTATION(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES),LOOP_OVER_ALL_VIEW_ELEMENTS,VIEW_DATA_ACCESS) \
    CONST_VIEW_IMPL(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES))




/* NON CONST IMPLEMENTATIONS */
/* Common non const implementation */
#define COMMON_NON_CONST_IMPLEMENTATION(TYPENAME,TEMPLATES)

/* Loop dependant non const implementation */
#define LOOP_DEPENDENT_NON_CONST_IMPLEMENTATION(TYPENAME,TEMPLATES,LOOP_OVER_ALL_ELEMENTS,DATA_ACCESS)         \
    /* Apply function to all elements */                                                                       \
    TEMPLATES                                                                                                  \
    TYPENAME& TYPENAME::apply(const std::function<void(T&)>& func) {                                           \
        LOOP_OVER_ALL_ELEMENTS(*this) {                                                                        \
            func( DATA_ACCESS(*this) );                                                                        \
        }                                                                                                      \
        return *this;                                                                                          \
    }                                                                                                          \
    TEMPLATES                                                                                                  \
    TYPENAME& TYPENAME::apply(const std::function<void(T&, const Index<Dim>&)>& func) {                        \
        LOOP_OVER_ALL_VIEW_ELEMENTS(*this) {                                                                   \
            func( VIEW_DATA_ACCESS(*this), LOOP_VARNAME );                                                     \
        }                                                                                                      \
        return *this;                                                                                          \
    }                                                                                                          \
    TEMPLATES                                                                                                  \
    template <typename Functor, typename Arg0, typename... Args>                                               \
    TYPENAME& TYPENAME::apply(const Functor& func, Arg0&& farg0, Args&&... fargs) {                            \
        LOOP_OVER_ALL_VIEW_ELEMENTS(*this) {                                                                   \
            func( VIEW_DATA_ACCESS(*this), LOOP_VARNAME, std::forward<Arg0>(farg0), std::forward<Args>(fargs)... ); \
        }                                                                                                      \
        return *this;                                                                                          \
    }

/* Reference specific non const implementation */
#define NON_CONST_REF_IMPL(TYPENAME,TEMPLATES)                                                                          \
    TEMPLATES                                                                                                           \
    ENABLE_IF_FFTW_REAL(T*,NO_DEFAULT_TEMPLATES) TYPENAME::rdata() {                                                         \
        return this->data();                                                                                            \
    }                                                                                                                   \
    TEMPLATES                                                                                                           \
    ENABLE_IF_FFTW_REAL(typename fft::fftw_complex_type<T>::std_type*,NO_DEFAULT_TEMPLATES)  TYPENAME::asStdComplexData() {  \
        return reinterpret_cast<typename fft::fftw_complex_type<T>::std_type*>(this->data());                           \
    }                                                                                                                   \
    TEMPLATES                                                                                                           \
    ENABLE_IF_FFTW_REAL(typename fft::fftw_complex_type<T>::fftw_type*,NO_DEFAULT_TEMPLATES) TYPENAME::asFftwComplexData() { \
        return reinterpret_cast<typename fft::fftw_complex_type<T>::fftw_type*>(this->data());                          \
    }                                                                                                                   \
    TEMPLATES                                                                                                           \
    ENABLE_IF_FFTW_COMPLEX(T*,NO_DEFAULT_TEMPLATES) TYPENAME::cdata() {                                                      \
        return this->data();                                                                                            \
    }                                                                                                                   \
    TEMPLATES                                                                                                           \
    ENABLE_IF_FFTW_COMPLEX(typename fft::fftw_complex_type<T>::std_type*,NO_DEFAULT_TEMPLATES)    TYPENAME::std_cdata() {    \
        return reinterpret_cast<typename fft::fftw_complex_type<T>::std_type*>(this->data());                           \
    }                                                                                                                   \
    TEMPLATES                                                                                                           \
    ENABLE_IF_FFTW_COMPLEX(typename fft::fftw_complex_type<T>::fftw_type*,NO_DEFAULT_TEMPLATES)  TYPENAME::fftw_cdata() {    \
        return reinterpret_cast<typename fft::fftw_complex_type<T>::fftw_type*>(this->data());                          \
    }                                                                                                                   \
    TEMPLATES                                                                                                           \
    ENABLE_IF_FFTW_COMPLEX(typename fft::fftw_complex_type<T>::value_type*,NO_DEFAULT_TEMPLATES) TYPENAME::asRealData() {    \
        return reinterpret_cast<typename fft::fftw_complex_type<T>::value_type*>(this->data());                         \
    }                                                                                                                               

/* View specific non const implementation */
#define NON_CONST_VIEW_IMPL(TYPENAME,TEMPLATES)

/* Reference specific non const implementation */
#define NON_CONST_REF_IMPLEMENTATION(TYPENAME,TEMPLATES)                                                                             \
    COMMON_NON_CONST_IMPLEMENTATION(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES))                                                      \
    LOOP_DEPENDENT_NON_CONST_IMPLEMENTATION(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES),LOOP_OVER_ALL_REF_ELEMENTS,REF_DATA_ACCESS)   \
    NON_CONST_REF_IMPL(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES))

/* View specific non const implementation */
#define NON_CONST_VIEW_IMPLEMENTATION(TYPENAME,TEMPLATES)                                                                            \
    COMMON_NON_CONST_IMPLEMENTATION(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES))                                                      \
    LOOP_DEPENDENT_NON_CONST_IMPLEMENTATION(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES),LOOP_OVER_ALL_VIEW_ELEMENTS,VIEW_DATA_ACCESS) \
    NON_CONST_VIEW_IMPL(SINGLE_ARG(TYPENAME),SINGLE_ARG(TEMPLATES))


#endif /* end of include guard: HYSOP_MULTI_ARRAY_DEFINES_H */

#endif /* end of MULTI_ARRAY include guard */
