
#include "fft/transform.h"

namespace hysop {
    namespace fft {

    std::ostream& operator<<(std::ostream& os, const Transform &tr) {
        os << tr.toString();
        return os;
    }
   
    }
}

