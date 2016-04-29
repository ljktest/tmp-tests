
#ifndef HYSOP_DIFFSOLVER_H
#define HYSOP_DIFFSOLVER_H

#include "data/multi_array/multi_array.h"

namespace hysop {
    namespace solver {
        template <typename T, std::size_t Dim>
            class DiffSolver {

                public:
                    virtual void apply(hysop::const_multi_array_ref<T,Dim> input,
                            hysop::multi_array_ref<T,Dim> output,
                            const std::array<int,Dim> &order) const = 0;

                    void operator()(hysop::const_multi_array_ref<T,Dim> input, 
                                    hysop::multi_array_ref<T,Dim> output, 
                            const std::array<int,Dim> &order) {
                        this->apply(input,output,order);
                    }

            };

    } /* end of namespace solver */
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_DIFFSOLVER_H */

