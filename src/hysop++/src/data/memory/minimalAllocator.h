
#ifndef HYSOP_MINIMAL_ALLOCATOR_H
#define HYSOP_MINIMAL_ALLOCATOR_H

#include <limits>

namespace hysop {
    namespace data {
        namespace memory {
        
        /* Minimal MinimalAllocator for boost and std libs  */
        template <typename T>
            struct MinimalAllocator {
                using value_type = T;
                using const_pointer = const T*;

                MinimalAllocator() = default;

                template <class U>
                    MinimalAllocator(const MinimalAllocator<U>&) {}

                T* allocate(std::size_t n, const_pointer hint=nullptr) {
                    if (n <= std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                        if (auto ptr = std::malloc(n * sizeof(T))) {
                            return static_cast<T*>(ptr);
                        }
                    }
                    throw std::bad_alloc();
                }
                void deallocate(T* ptr, std::size_t n) {
                    std::free(ptr);
                }
                void destroy(T* ptr) {
                    ptr->~T();
                }
            };

        template <typename T, typename U>
            inline bool operator == (const MinimalAllocator<T>&, const MinimalAllocator<U>&) {
                return true;
            }

        template <typename T, typename U>
            inline bool operator != (const MinimalAllocator<T>& a, const MinimalAllocator<U>& b) {
                return !(a == b);
            }

        } /* end of namespace memory */
    } /* end of namespace data */
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_MINIMAL_ALLOCATOR_H */
