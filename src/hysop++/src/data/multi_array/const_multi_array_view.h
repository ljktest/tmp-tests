
#ifndef HYSOP_MULTI_ARRAY_H
#include "data/multi_array/multi_array.h"
#else 

#ifndef HYSOP_CONST_MULTI_ARRAY_VIEW_H
#define HYSOP_CONST_MULTI_ARRAY_VIEW_H

namespace hysop {
    namespace data {
        
        /* class hysop::data::multi_array */
        template <typename T, std::size_t Dim>
            class const_multi_array_view : public boost_const_multi_array_view<T,Dim> {
                static_assert(Dim>0, "Dim cannot be zero !");

                private:
                    using super = boost_const_multi_array_view<T,Dim>;
                public:
                    PUBLIC_CLASS_TYPES()

                public:
                    const_multi_array_view(const const_multi_array_view<T,Dim>& view) = default;
                    const_multi_array_view& operator=(const const_multi_array_view<T,Dim>& other) = default;
                    
                    const_multi_array_view(const boost_const_multi_array_view<T,Dim>& view);
                    const_multi_array_view& operator=(const boost_const_multi_array_view<T,Dim>& other);

                public:
                    PUBLIC_CONST_VIEW_INTERFACE(SINGLE_ARG(const_multi_array_view<T,Dim>))
                };


        /* Implementation */
        template <typename T, std::size_t Dim>
        const_multi_array_view<T,Dim>::const_multi_array_view(const boost_const_multi_array_view<T,Dim>& view) :
            super(view) {
        }

        template <typename T, std::size_t Dim>
        const_multi_array_view<T,Dim>& const_multi_array_view<T,Dim>::operator=(const boost_const_multi_array_view<T,Dim>& other) {
            super::operator=(other);
            return *this;
        }

        CONST_VIEW_IMPLEMENTATION(SINGLE_ARG(const_multi_array_view<T,Dim>), SINGLE_ARG(template <typename T, std::size_t Dim>))

    } /* end of namespace data */ 
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_CONST_MULTI_ARRAY_VIEW_H */

#endif /* end of MULTI_ARRAY include guard */
