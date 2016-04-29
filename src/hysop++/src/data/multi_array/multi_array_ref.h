
#ifndef HYSOP_MULTI_ARRAY_H
#include "data/multi_array/multi_array.h"
#else 

#ifndef HYSOP_MULTI_ARRAY_REF_H
#define HYSOP_MULTI_ARRAY_REF_H

namespace hysop {
    namespace data {

        /* class hysop::data::multi_array */
        template <typename T, std::size_t Dim>
            class multi_array_ref : public boost_multi_array_ref<T,Dim> {
                static_assert(Dim>0, "Dim cannot be zero !");

                private:
                    using super = boost_multi_array_ref<T,Dim>;
                public:
                    PUBLIC_CLASS_TYPES()

                public:
                    multi_array_ref(T* base=nullptr, const extents_gen<Dim>& ranges = hysop::utils::buildExtents(std::array<std::size_t,Dim>{0}));
                    multi_array_ref(const multi_array_ref<T,Dim>& ref) = default;
                    multi_array_ref& operator=(const multi_array_ref<T,Dim>& other) = default;
                    
                    multi_array_ref(const boost_multi_array_ref<T,Dim>& ref);
                    multi_array_ref& operator=(const boost_multi_array_ref<T,Dim>& other);
  

                    operator const_array_ref() const;

                public:
                    PUBLIC_CONST_REF_INTERFACE(SINGLE_ARG(multi_array_ref<T,Dim>))
                    PUBLIC_NON_CONST_REF_INTERFACE(SINGLE_ARG(multi_array_ref<T,Dim>))
                };


        /* Implementation */
        template <typename T, std::size_t Dim>
        multi_array_ref<T,Dim>::multi_array_ref(T* base, const extents_gen<Dim>& ranges):
            super(base,ranges) {
        }
                    
        template <typename T, std::size_t Dim>
        multi_array_ref<T,Dim>::multi_array_ref(const boost_multi_array_ref<T,Dim>& ref) :
            super(ref) {
        }
       
        template <typename T, std::size_t Dim>
        multi_array_ref<T,Dim>& multi_array_ref<T,Dim>::operator=(const boost_multi_array_ref<T,Dim>& other) {
            super::operator=(other);
            return *this;
        }
                    
        template <typename T, std::size_t Dim>
        multi_array_ref<T,Dim>::operator const_array_ref() const {
            return static_cast<boost_const_multi_array_ref<T,Dim>>(*this);
        }

        CONST_REF_IMPLEMENTATION(SINGLE_ARG(multi_array_ref<T,Dim>), SINGLE_ARG(template <typename T, std::size_t Dim>))
        NON_CONST_REF_IMPLEMENTATION(SINGLE_ARG(multi_array_ref<T,Dim>), SINGLE_ARG(template <typename T, std::size_t Dim>))
        
    } /* end of namespace data */ 
} /* end of namespace hysop */


#endif /* end of include guard: HYSOP_MULTI_ARRAY_REF_H */

#endif /* end of MULTI_ARRAY include guard */
