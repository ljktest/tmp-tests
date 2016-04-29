
#include "domain/boundary.h"

namespace hysop {
    namespace domain {

        static const char* boundary_strings[6] = {
            "NONE",
            "HOMOGENEOUS_DIRICHLET",
            "HOMOGENEOUS_NEUMANN",
            "DIRICHLET",
            "NEUMANN",
            "PERIODIC"
        };

        const char* toStringBoundary(Boundary bd) {
            return boundary_strings[static_cast<int>(bd)+1];
        }

        std::ostream& operator<<(std::ostream& os, const Boundary& bd) {
            os << toStringBoundary(bd);
            return os;
        }

    }
}
