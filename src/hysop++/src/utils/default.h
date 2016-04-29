
#ifndef HYSOP_DEFAULT_H
#define HYSOP_DEFAULT_H

#include "data/memory/minimalAllocator.h"
#include "data/memory/fftwAllocator.h"

namespace hysop {
    namespace _default {
        
        template <typename T>
        using allocator = hysop::data::memory::MinimalAllocator<T>;

        template <typename T>
        using fft_allocator = hysop::data::memory::FftwAllocator<T>;
 
    } /* end of namespace _default */
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_DEFAULT_H */
