
#ifndef HYSOP_POISSONSOLVER_H
#define HYSOP_POISSONSOLVER_H

#include "data/multi_array/multi_array.h"

namespace hysop {
    namespace solver {
        template <typename T, std::size_t Dim>
            class PoissonSolver {

                public:
                    virtual void apply(hysop::const_multi_array_ref<T,Dim> input,
                            hysop::multi_array_ref<T,Dim> output) const = 0;

                    void operator()(hysop::const_multi_array_ref<T,Dim> input, 
                                    hysop::multi_array_ref<T,Dim> output) {
                        this->apply(input,output);
                    }

            };

    } /* end of namespace solver */
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_POISSONSOLVER_H */

