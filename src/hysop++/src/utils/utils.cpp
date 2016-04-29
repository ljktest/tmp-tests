
#include "utils/utils.h"

namespace std {
    std::ostream& operator<<(std::ostream& os, const fftw_iodim& iodim) {
        os << "[n=" << iodim.n << ", is=" << iodim.is << ", os=" << iodim.os << "]";
        return os;
    }
}

namespace hysop {
    namespace utils {

    }
}


