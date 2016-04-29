
#ifndef HYSOP_FFTW_ALLOCATOR_H
#define HYSOP_FFTW_ALLOCATOR_H

#include <limits>
#include <fftw3.h>

namespace hysop {
    namespace data {
        namespace memory {

        /* FftwAllocator designed to correctly align data for fftw */
        template <typename T>
            struct FftwAllocator {
                using value_type = T;
                using const_pointer = const T*;

                FftwAllocator() = default;

                template <class U>
                    FftwAllocator(const FftwAllocator<U>&) {}

                T* allocate(std::size_t n, const_pointer hint=nullptr) {
                    if (n <= std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                        if (auto ptr = fftw_malloc(n * sizeof(T))) {
                            return static_cast<T*>(ptr);
                        }
                    }
                    throw std::bad_alloc();
                }
                void deallocate(T* ptr, std::size_t n) {
                    fftw_free(ptr);
                }
                void destroy(T* ptr) {
                    ptr->~T();
                }
            };

        template <typename T, typename U>
            inline bool operator == (const FftwAllocator<T>&, const FftwAllocator<U>&) {
                return true;
            }

        template <typename T, typename U>
            inline bool operator != (const FftwAllocator<T>& a, const FftwAllocator<U>& b) {
                return !(a == b);
            }
    
        } /* end of namespace memory */
    } /* end of namespace data */
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_FFTW_ALLOCATOR_H */
