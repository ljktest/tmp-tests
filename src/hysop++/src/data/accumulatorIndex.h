
#ifndef HYSOP_ACCUMULATORINDEX_H
#define HYSOP_ACCUMULATORINDEX_H

#include <array>
#include <cassert>
#include <vector>
#include <functional>

#include "data/index.h"

namespace hysop {

    template <typename T, std::size_t Dim, typename Source = std::array<std::vector<T>,Dim>, typename Functor = std::function<T(const T&, const T&)>>
        struct AccumulatorIndex : public Index<Dim> {
    
            private:
                using super = Index<Dim>;
            public:
                using Indices   = typename super::Indices;
                using Dimension = typename super::Dimension;
          
            public:
                AccumulatorIndex(const AccumulatorIndex& idx) = default;
                ~AccumulatorIndex() = default;
                
                template <typename DimArray=Dimension, typename AccumulatorIndexArray=Indices>
                AccumulatorIndex(const DimArray& p_dim = DimArray{0}, 
                                 const Indices &p_ids  = AccumulatorIndexArray{0}):
                     super(p_dim, p_ids), m_accumulatedData{0}, m_sourceData(nullptr), m_functor(std::plus<T>()) {
                            } 

                AccumulatorIndex& setAccumulatorSource(const Source& p_source)     { m_sourceData = &p_source; init(); return *this; }
                AccumulatorIndex& setAccumulatorFunction(const Functor& p_functor) { m_functor    = p_functor; init(); return *this; }

                const T& accumulatedVal() const { return m_accumulatedData[Dim-1]; }
    
            protected:
                void init() {
                    if(m_sourceData != nullptr) {
                        m_accumulatedData[0] = (*m_sourceData)[0][this->operator[](0)];
                        for (std::size_t d=0; d<Dim-1; d++)
                            m_accumulatedData[d+1] = m_functor(m_accumulatedData[d],(*m_sourceData)[d+1][this->operator[](d+1)]);
                    }
                }

                virtual void onIndexChange(std::size_t pos, std::ptrdiff_t offset) final override {
                    if(m_sourceData != nullptr) {
                        assert(pos < Dim);
                        m_accumulatedData[pos] = (pos==0 ? (*m_sourceData)[0][this->operator[](0)] : m_functor(m_accumulatedData[pos-1],(*m_sourceData)[pos][this->operator[](pos)]));
                        for (std::size_t d=pos; d<Dim-1; d++)
                            m_accumulatedData[d+1] = m_functor(m_accumulatedData[d],(*m_sourceData)[d+1][this->operator[](d+1)]);
                    }
                };

            protected:
                std::array<T,Dim> m_accumulatedData;
                const Source* m_sourceData;
                Functor m_functor;
        };

}

#endif /* end of include guard: HYSOP_ACCUMULATORINDEX_H */

