
#ifndef HYSOP_MULTI_ARRAY_H
#include "data/multi_array/multi_array.h"
#else 

#ifndef HYSOP_CONST_MULTI_ARRAY_REF_H
#define HYSOP_CONST_MULTI_ARRAY_REF_H

namespace hysop {
    namespace data {
        
        /* class hysop::data::multi_array */
        template <typename T, std::size_t Dim>
            class const_multi_array_ref : public boost_const_multi_array_ref<T,Dim> {
                static_assert(Dim>0, "Dim cannot be zero !");

                private:
                    using super = boost_const_multi_array_ref<T,Dim>;
                public:
                    PUBLIC_CLASS_TYPES()

                public:
                    const_multi_array_ref(const const_multi_array_ref<T,Dim>& ref) = default;
                    const_multi_array_ref& operator=(const const_multi_array_ref<T,Dim>& other) = default;
                    
                    const_multi_array_ref(const boost_const_multi_array_ref<T,Dim>& ref);
                    const_multi_array_ref& operator=(const boost_const_multi_array_ref<T,Dim>& other);

                public:
                    PUBLIC_CONST_REF_INTERFACE(SINGLE_ARG(const_multi_array_ref<T,Dim>))
                };


        /* Implementation */
        template <typename T, std::size_t Dim>
        const_multi_array_ref<T,Dim>::const_multi_array_ref(const boost_const_multi_array_ref<T,Dim>& ref) :
            super(ref) {
        }

        template <typename T, std::size_t Dim>
        const_multi_array_ref<T,Dim>& const_multi_array_ref<T,Dim>::operator=(const boost_const_multi_array_ref<T,Dim>& other) {
            super::operator=(other);
            return *this;
        }

        CONST_REF_IMPLEMENTATION(SINGLE_ARG(const_multi_array_ref<T,Dim>), SINGLE_ARG(template <typename T, std::size_t Dim>))

    } /* end of namespace data */ 
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_CONST_MULTI_ARRAY_REF_H */

#endif /* end of MULTI_ARRAY include guard */
