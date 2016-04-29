
#ifndef HYSOP_TYPES_H
#define HYSOP_TYPES_H

#include <complex>
#include <array>

#include "utils/utils.h"
#include "utils/default.h"

namespace hysop {
    
    /* forward declare external types */
    namespace data {
        template <typename T, std::size_t Dim, typename Allocator>
            class multi_array;
        template <typename T, std::size_t Dim>
            class multi_array_ref;
        template <typename T, std::size_t Dim>
            class multi_array_view;
        template <typename T, std::size_t Dim>
            class const_multi_array_view;
        template <typename T, std::size_t Dim>
            class const_multi_array_ref;
    } /* end of namespace data */


    namespace types {
        typedef double             real;  
        typedef std::complex<real> complex;
    } /* end of namespace types */
    
    
/* expose the folowwing types to namespace hysop */

/* swig does not support alias templates... */
    template <std::size_t Dim> 
        struct Shape { 
            typedef std::array<std::size_t, Dim> type; 
        };
    template <std::size_t Dim> 
        struct Offset { 
            typedef std::array<std::ptrdiff_t, Dim> type; 
        };
    
    template <typename T, std::size_t Dim, typename Allocator = hysop::_default::allocator<T>>
    using multi_array = hysop::data::multi_array<T,Dim,Allocator>;

    template <typename T, std::size_t Dim>
    using multi_array_view = hysop::data::multi_array_view<T,Dim>;
    
    template <typename T, std::size_t Dim>
    using const_multi_array_view = hysop::data::const_multi_array_view<T,Dim>;
    
    template <typename T, std::size_t Dim>
    using multi_array_ref = hysop::data::multi_array_ref<T,Dim>;
    
    template <typename T, std::size_t Dim>
    using const_multi_array_ref = hysop::data::const_multi_array_ref<T,Dim>;

} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_TYPES_H */
