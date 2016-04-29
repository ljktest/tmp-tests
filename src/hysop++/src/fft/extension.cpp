
#include "fft/extension.h"

namespace hysop {
    namespace fft {

        static const char* extension_strings[4] {
            "NONE",
            "EVEN",
            "ODD",
            "PERIODIC"
        };

        const char* toStringExtension(Extension ext) {
            return extension_strings[static_cast<int>(ext)+1];
        }

        std::ostream& operator<<(std::ostream& os, const Extension& ext) {
            os << toStringExtension(ext);
            return os;
        }

    }
}
