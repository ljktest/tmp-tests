
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include "maths/quad_maths.h"
#include "utils/types.h"

namespace hysop {
    namespace constants {
        static constexpr hysop::types::complex I = hysop::types::complex(0,1);
        static constexpr hysop::types::complex Z = hysop::types::complex(0,0);

#ifdef HAS_QUADMATHS
        static const __float128 pi = acosq(-1.0Q);
#else
        static constexpr long double pi = acosl(-1.0L);
#endif
    }
}

#endif /* end of include guard: CONSTANTS_H */
