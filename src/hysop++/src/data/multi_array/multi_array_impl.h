
#ifndef HYSOP_MULTI_ARRAY_H
#include "data/multi_array/multi_array.h"
#else 

#ifndef HYSOP_MULTI_ARRAY_IMPL_H
#define HYSOP_MULTI_ARRAY_IMPL_H

namespace hysop {
    namespace data {

        /* class hysop::data::multi_array */
        template <typename T, std::size_t Dim, typename Allocator>
            class multi_array : public boost_multi_array<T,Dim,Allocator> {
                static_assert(Dim>0, "Dim cannot be zero !");

                private:
                    using super = boost_multi_array<T,Dim,Allocator>;
                public:
                    PUBLIC_CLASS_TYPES()

                public:
                    multi_array(const extents_gen<Dim>& extents = extents_gen<Dim>());
                    multi_array(const typename Shape<Dim>::type& shape);

                    multi_array(const multi_array& other);
                    multi_array(multi_array&& other);

                    explicit multi_array(const array_ref& other);
                    explicit multi_array(const array_view& other);
                    explicit multi_array(const const_array_ref& other);
                    explicit multi_array(const const_array_view& other);
                   
                    explicit multi_array(const boost_multi_array<T,Dim,Allocator>& other);
                    explicit multi_array(const boost_multi_array_ref<T,Dim>& other);
                    explicit multi_array(const boost_multi_array_view<T,Dim>& other);
                    explicit multi_array(const boost_const_multi_array_ref<T,Dim>& other);
                    explicit multi_array(const boost_const_multi_array_view<T,Dim>& other);
                    explicit multi_array(boost_multi_array<T,Dim,Allocator>&& other);

                    multi_array& operator=(const multi_array& other);
                    multi_array& operator=(const array_ref& ref);
                    multi_array& operator=(const array_view& view);
                    multi_array& operator=(const const_array_ref& ref);
                    multi_array& operator=(const const_array_view& view);
                    multi_array& operator=(multi_array&& other);

                    operator array_ref();
                    operator const_array_ref() const;

                public:
                    PUBLIC_CONST_REF_INTERFACE(SINGLE_ARG(multi_array<T,Dim,Allocator>))
                    PUBLIC_NON_CONST_REF_INTERFACE(SINGLE_ARG(multi_array<T,Dim,Allocator>))
        
                    multi_array& reshape(const typename Shape<Dim>::type& shape);

                protected:
                    static extents_gen<Dim> shapeToExtents(const typename Shape<Dim>::type &shape);
            };


        /* Implementation */
        
        /* constructors */
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const extents_gen<Dim>& extents):
            boost_multi_array<T,Dim,Allocator>(extents) {}
                    
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const typename Shape<Dim>::type& shape):
            boost_multi_array<T,Dim,Allocator>(shapeToExtents(shape)) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const multi_array& other):
            boost_multi_array<T,Dim,Allocator>(static_cast<const boost_multi_array<T,Dim,Allocator>&>(other)) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const array_view& other):
            boost_multi_array<T,Dim,Allocator>(static_cast<const boost_multi_array_view<T,Dim>&>(other)) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const const_array_view& other):
            boost_multi_array<T,Dim,Allocator>(static_cast<const boost_const_multi_array_view<T,Dim>&>(other)) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const array_ref& other):
            boost_multi_array<T,Dim,Allocator>(static_cast<const boost_multi_array_ref<T,Dim>&>(other)) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const const_array_ref& other):
            boost_multi_array<T,Dim,Allocator>(static_cast<const boost_const_multi_array_ref<T,Dim>&>(other)) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(multi_array&& other):
            boost_multi_array<T,Dim,Allocator>(static_cast<boost_multi_array<T,Dim,Allocator>&&>(other)) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const boost_multi_array_view<T,Dim>& other):
            boost_multi_array<T,Dim,Allocator>(other) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const boost_const_multi_array_view<T,Dim>& other):
            boost_multi_array<T,Dim,Allocator>(other) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const boost_multi_array_ref<T,Dim>& other):
            boost_multi_array<T,Dim,Allocator>(other) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(const boost_const_multi_array_ref<T,Dim>& other):
            boost_multi_array<T,Dim,Allocator>(other) {}
        
        template <typename T, std::size_t Dim, typename Allocator> 
        multi_array<T,Dim,Allocator>::multi_array(boost_multi_array<T,Dim,Allocator>&& other):
            boost_multi_array<T,Dim,Allocator>(other) {}

        
        /* operator = */
        /* cast obligatory to avoid shape() function aliasing */
        template <typename T, std::size_t Dim, typename Allocator> 
            multi_array<T,Dim,Allocator>& multi_array<T,Dim,Allocator>::operator=(const multi_array<T,Dim,Allocator>& other) {
                this->reshape(other.shape());
                super::operator=(dynamic_cast<const boost_multi_array<T,Dim,Allocator>&>(other));
                return *this;
            }
        template <typename T, std::size_t Dim, typename Allocator> 
            multi_array<T,Dim,Allocator>& multi_array<T,Dim,Allocator>::operator=(const array_view& other) {
                this->reshape(other.shape());
                super::operator=(dynamic_cast<const boost_multi_array_view<T,Dim>&>(other));
                return *this;
            }
        template <typename T, std::size_t Dim, typename Allocator> 
            multi_array<T,Dim,Allocator>& multi_array<T,Dim,Allocator>::operator=(const const_array_view& other) {
                this->reshape(other.shape());
                super::operator=(dynamic_cast<const boost_const_multi_array_view<T,Dim>&>(other));
                return *this;
            }
        template <typename T, std::size_t Dim, typename Allocator> 
            multi_array<T,Dim,Allocator>& multi_array<T,Dim,Allocator>::operator=(const array_ref& other) {
                this->reshape(other.shape());
                super::operator=(dynamic_cast<const boost_multi_array_ref<T,Dim>&>(other));
                return *this;
            }
        template <typename T, std::size_t Dim, typename Allocator> 
            multi_array<T,Dim,Allocator>& multi_array<T,Dim,Allocator>::operator=(const const_array_ref& other) {
                this->reshape(other.shape());
                super::operator=(dynamic_cast<const boost_const_multi_array_ref<T,Dim>&>(other));
                return *this;
            }
        template <typename T, std::size_t Dim, typename Allocator> 
            multi_array<T,Dim,Allocator>& multi_array<T,Dim,Allocator>::operator=(multi_array&& other) {
                super::operator=(other);
                return *this;
            }

        /* casting operators */
        template <typename T, std::size_t Dim, typename Allocator> 
            multi_array<T,Dim,Allocator>::operator multi_array_ref<T,Dim>() {
                return static_cast<boost_multi_array_ref<T,Dim>>(*this);
            }
        template <typename T, std::size_t Dim, typename Allocator> 
            multi_array<T,Dim,Allocator>::operator const_multi_array_ref<T,Dim>() const {
                return static_cast<boost_const_multi_array_ref<T,Dim>>(*this);
            }

        /* static members */
        template <typename T, std::size_t Dim, typename Allocator> 
        typename multi_array<T,Dim,Allocator>::template extents_gen<Dim> multi_array<T,Dim,Allocator>::shapeToExtents(const typename Shape<Dim>::type &shape) {
            return utils::buildExtents(shape);
        }

        /* multiarray const & non const reference implementation */
        CONST_REF_IMPLEMENTATION(SINGLE_ARG(multi_array<T,Dim,Allocator>), SINGLE_ARG(template <typename T, std::size_t Dim, typename Allocator>)) 
        NON_CONST_REF_IMPLEMENTATION(SINGLE_ARG(multi_array<T,Dim,Allocator>), SINGLE_ARG(template <typename T, std::size_t Dim, typename Allocator>))
        
        /* multi array specific */
        template <typename T, std::size_t Dim, typename Allocator> 
            multi_array<T,Dim,Allocator>& multi_array<T,Dim,Allocator>::reshape(const typename Shape<Dim>::type& shape) { 
                boost::array<int,Dim> extents;
                for (std::size_t d = 0; d < Dim; d++)
                    extents[d] = static_cast<int>(shape[d]);
                this->resize(extents);
                return *this;
            }


    } /* end of namespace data */ 
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_MULTI_ARRAY_IMPL_H */

#endif /* end of MULTI_ARRAY include guard */
