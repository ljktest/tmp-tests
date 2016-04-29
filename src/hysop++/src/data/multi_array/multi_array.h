
#ifndef HYSOP_MULTI_ARRAY_H
#define HYSOP_MULTI_ARRAY_H

#include "utils/utils.h"
#include "utils/default.h"
#include <boost/multi_array.hpp>

/****************************************/
/*** Hysop boost multi array wrapper ****/
/****************************************/

namespace hysop {
    namespace data {

        /* forward declaration of types */
        template <typename T, std::size_t Dim, typename Allocator = hysop::_default::allocator<T>>                                           
        class multi_array;                                                                              
        template <typename T, std::size_t Dim>                                                              
        class multi_array_ref;                                                                          
        template <typename T, std::size_t Dim>                                                              
        class const_multi_array_ref;                                                                    
        template <typename T, std::size_t Dim>                                                              
        class multi_array_view;                                                                         
        template <typename T, std::size_t Dim>                                                              
        class const_multi_array_view;                                                                   

        template <typename T, std::size_t Dim, typename Allocator = hysop::_default::allocator<T>>                                          
        using boost_multi_array = boost::multi_array<T,Dim,Allocator>;                                  
        template <typename T, std::size_t Dim>                                                              
        using boost_multi_array_ref = boost::multi_array_ref<T,Dim>;                                    
        template <typename T, std::size_t Dim>                                                              
        using boost_const_multi_array_ref = boost::const_multi_array_ref<T,Dim>;                        
        template <typename T, std::size_t Dim>                                                              
        using boost_multi_array_view = boost::detail::multi_array::multi_array_view<T,Dim>;             
        template <typename T, std::size_t Dim>                                                              
        using boost_const_multi_array_view = boost::detail::multi_array::const_multi_array_view<T,Dim>;
    }
}

#include "data/multi_array/multi_array_defines.h"

#include "data/multi_array/const_multi_array_view.h"
#include "data/multi_array/multi_array_view.h"
#include "data/multi_array/const_multi_array_ref.h"
#include "data/multi_array/multi_array_ref.h"
#include "data/multi_array/multi_array_impl.h"
#include "data/multi_array/multi_array_ext.h"

#include "data/multi_array/multi_array_clean.h"

#endif /* end of include guard: HYSOP_MULTI_ARRAY_H */
