
#ifndef HYSOP_DOMAIN_H
#define HYSOP_DOMAIN_H

#include "data/multi_array/multi_array.h"
#include "utils/types.h"
#include "utils/utils.h"
#include "fft/fftDomainConfiguration.h"

namespace hysop {
    namespace domain {
        
        template <typename T, std::size_t Dim>
            class Domain;
        
        template <typename T, std::size_t Dim>
            std::ostream& operator<< (std::ostream& os, const Domain<T,Dim>& domain);

        template <typename T, std::size_t Dim>
            class Domain {
                public:
                    using DomainSize    = std::array<T, Dim>;
                    using SpaceStep     = std::array<T, Dim>;
                    using SpaceVariable = std::array<T, Dim>;

                public:
                    Domain() :
                        m_shape{0}, m_dataShape{0}, m_leftDataOffset{0},
                        m_domainConfig(), m_domainSize{0}, m_spaceStep{0}, m_data() {}

                    Domain(const typename Shape<Dim>::type& p_shape, const DomainConfiguration<Dim>& p_domainConfig, const DomainSize& p_domainSize) :
                        m_shape{0}, m_dataShape{0}, m_leftDataOffset{0},
                        m_domainConfig(p_domainConfig), m_domainSize(p_domainSize), 
                        m_spaceStep{0}, m_data() {
                            this->reshape(p_shape);
                        }
    
                    Domain(Domain<T,Dim>&& other)                        = default;
                    Domain<T,Dim>& operator=(const Domain<T,Dim>& other) = default; 
                    Domain<T,Dim>& operator=(Domain<T,Dim>&& other)      = default;

                    Domain<T,Dim>& reshape(const typename Shape<Dim>::type& p_domainShape) {
                        m_shape = p_domainShape;
                        m_domainConfig.getShapeConfiguration(m_shape, m_dataShape, m_leftDataOffset);
                        m_data.reshape(m_dataShape);
                        this->computeSpaceStep();
                        return *this;
                    }

                    Domain<T,Dim>& resize(const DomainSize& p_size) {
                        m_domainSize = p_size;
                        this->computeSpaceStep();
                        return *this;
                    }

                    Domain<T,Dim>& resetDomainConfiguration(const DomainConfiguration<Dim>& p_domainConfig) {
                        m_domainConfig = p_domainConfig;
                        this->reshape(m_shape);
                        return *this;
                    }

                    Domain<T,Dim>& print(const std::string &p_name) {
                        m_data.print(p_name);
                        return *this;
                    }
                    
                    fft::FftDomainConfiguration<Dim> fftDomainConfiguration() const { 
                        return fft::FftDomainConfiguration<Dim>(m_domainConfig); 
                    }

                    const typename Shape<Dim>::type&              shape()          const { return m_shape; }
                    const typename Shape<Dim>::type&              dataShape()      const { return m_dataShape; }
                    const SpaceStep&               spaceStep()      const { return m_spaceStep; }
                    const DomainSize&              domainSize()     const { return m_domainSize; }
                    const typename Offset<Dim>::type&             leftDataOffset() const { return m_leftDataOffset; }
                    const DomainConfiguration<Dim> boundaryConfig() const { return m_domainConfig; }

                    const hysop::multi_array<T,Dim>&    data() const { return m_data; }
                          hysop::multi_array_ref<T,Dim> data()       { return m_data; } 

                    
                    /* Apply f(X, fargs...) on the whole domain where X = [x_0, x_1, ..., x_{Dim-1}] is the space variable */
                    /* The result of the Functor f should be convertible to domain real data type T                        */
                    template <typename Functor, typename... Args>
                    Domain<T,Dim>& apply(const Functor& f,  Args&&... fargs);
    
                    T distance_L1(const Domain<T,Dim> &other) {
                        assert(this->dataShape() == other.dataShape());
                        return hysop::data::distance_L1<T,Dim>(this->data(), other.data());
                    }
                    T distance_L2(const Domain<T,Dim> &other) {
                        assert(this->dataShape() == other.dataShape());
                        return hysop::data::distance_L2<T,Dim>(this->data(), other.data());
                    }
                    T distance_Linf(const Domain<T,Dim> &other) {
                        assert(this->dataShape() == other.dataShape());
                        return hysop::data::distance_Linf<T,Dim>(this->data(), other.data());
                    }
                    std::tuple<T,T,T> distance(const Domain<T,Dim>& other) {
                        return std::tuple<T,T,T>(distance_L1(other), distance_L2(other), distance_Linf(other));
                    }

                protected:
                    void computeSpaceStep() {
                        for (std::size_t d = 0; d < Dim; d++) {
                            //std::size_t N = ((m_domainConfig[d].first==domain::Boundary::PERIODIC && !m_domainConfig.includePeriodicBoundaries()) ? m_shape[d]-1 : m_shape[d]-1);
                            m_spaceStep[d] = m_domainSize[d] / (m_shape[d]-1);
                        }
                    }

                protected:
                    typename Shape<Dim>::type  m_shape, m_dataShape;
                    typename Offset<Dim>::type m_leftDataOffset;

                    DomainConfiguration<Dim> m_domainConfig;
                    DomainSize m_domainSize;
                    SpaceStep  m_spaceStep;

                    hysop::multi_array<T, Dim> m_data;
            };
                    
        /* Apply f(X, args...) on the whole domain where X = [x_0, x_1, ..., x_{Dim-1}] is the space variable */
        template <typename T, std::size_t Dim>
        template <typename Functor, typename... Args>
            Domain<T,Dim>& Domain<T,Dim>::apply(const Functor& f,  Args&&... fargs) {
                hysop::Index<Dim> idx(m_dataShape);
                std::array<T,Dim> X{0};
                T* data = m_data.origin();
                for (std::size_t k = 0; k < m_data.num_elements(); k++) {
                    data[k] = static_cast<T>(f(X, std::forward<Args>(fargs)...));
                    ++idx;
                    for (std::size_t d = 0; d < Dim; d++)
                        X[d] = (idx[d]+m_leftDataOffset[d])*m_spaceStep[d];
                }
                return *this;
            }
        
        template <typename T, std::size_t Dim>
            std::ostream& operator<< (std::ostream& os, const Domain<T,Dim>& domain) {
                os << "== Domain ==" << std::endl;
                os << "\tShape     : " << domain.shape() << std::endl;
                os << "\tSize      : " << domain.domainSize() << std::endl;
                os << "\tSpaceStep : " << domain.spaceStep() << std::endl;
                os << domain.boundaryConfig();
                os << "\tLeftDataOffset: " << domain.leftDataOffset() << std::endl;
                os << "\tDataShape     : " << domain.dataShape() << std::endl;
                return os;
            }

    }
}

#endif /* end of include guard: HYSOP_DOMAIN_H */

