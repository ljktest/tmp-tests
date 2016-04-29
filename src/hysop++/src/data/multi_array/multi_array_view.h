
#ifndef HYSOP_MULTI_ARRAY_H
#include "data/multi_array/multi_array.h"
#else 

#ifndef HYSOP_MULTI_ARRAY_VIEW_H
#define HYSOP_MULTI_ARRAY_VIEW_H

namespace hysop {
    namespace data {

        /* class hysop::data::multi_array */
        template <typename T, std::size_t Dim>
            class multi_array_view : public boost_multi_array_view<T,Dim> {
                static_assert(Dim>0, "Dim cannot be zero !");

                private:
                    using super = boost_multi_array_view<T,Dim>;
                public:
                    PUBLIC_CLASS_TYPES()

                public:
                    multi_array_view(const multi_array_view<T,Dim>& view) = default;
                    multi_array_view& operator=(const multi_array_view<T,Dim>& other) = default;
                    
                    multi_array_view(const boost_multi_array_view<T,Dim>& view);
                    multi_array_view& operator=(const boost_multi_array_view<T,Dim>& other);

                    operator const_array_view() const;

                public:
                    PUBLIC_CONST_VIEW_INTERFACE(SINGLE_ARG(multi_array_view<T,Dim>))
                    PUBLIC_NON_CONST_VIEW_INTERFACE(SINGLE_ARG(multi_array_view<T,Dim>))
                };


        /* Implementation */


// remove boost warning 
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
        template <typename T, std::size_t Dim>
        multi_array_view<T,Dim>::multi_array_view(const boost_multi_array_view<T,Dim>& view) :
            super(view) {
        }
#pragma GCC diagnostic pop
       
        template <typename T, std::size_t Dim>
        multi_array_view<T,Dim>& multi_array_view<T,Dim>::operator=(const boost_multi_array_view<T,Dim>& other) {
            super::operator=(other);
            return *this;
        }
                    
        template <typename T, std::size_t Dim>
        multi_array_view<T,Dim>::operator const_array_view() const {
            return static_cast<boost_const_multi_array_view<T,Dim>>(*this);
        }

        CONST_VIEW_IMPLEMENTATION(SINGLE_ARG(multi_array_view<T,Dim>), SINGLE_ARG(template <typename T, std::size_t Dim>))
        NON_CONST_VIEW_IMPLEMENTATION(SINGLE_ARG(multi_array_view<T,Dim>), SINGLE_ARG(template <typename T, std::size_t Dim>))
        
    } /* end of namespace data */ 
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_MULTI_ARRAY_VIEW_H */

#endif /* end of MULTI_ARRAY include guard */
