#ifndef HYSOP_BOUNDARY_H
#define HYSOP_BOUNDARY_H

#include <iostream>

namespace hysop {
    namespace domain {

        enum class Boundary : int {
            NONE                  = -1,
            HOMOGENEOUS_DIRICHLET = 0,
            HOMOGENEOUS_NEUMANN   = 1,
            DIRICHLET             = 2,
            NEUMANN               = 3,
            PERIODIC              = 4
        };

        const char* toStringBoundary(Boundary bd);
        std::ostream& operator<<(std::ostream& os, const Boundary& bd);
    }
}

#endif /* end of include guard: HYSOP_BOUNDARY_H */
