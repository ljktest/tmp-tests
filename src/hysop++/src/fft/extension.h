
#ifndef FFT_EXTENSION_H
#define FFT_EXTENSION_H

#include <ostream>

namespace hysop {
    namespace fft {

        enum class Extension : int {
            NONE=-1,
            EVEN=0,
            ODD=1,
            PERIODIC=2
        };

        const char* toStringExtension(Extension ext);
        std::ostream& operator<<(std::ostream& os, const Extension& ext);
    }
}


#endif /* end of include guard: FFT_EXTENSION_H */
