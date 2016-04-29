
#ifndef HYSOP_DOMAIN_CONFIGURATION_H
#define HYSOP_DOMAIN_CONFIGURATION_H

#include <array>
#include <stdexcept>

#include "utils/types.h"
#include "domain/boundary.h"
#include "detail/index_seq.h"

namespace hysop {
    namespace domain {
        
        template <std::size_t Dim>
            class DomainConfiguration;

        template <std::size_t Dim>
            std::ostream& operator<<(std::ostream& os, const DomainConfiguration<Dim>& config);
        
        template <std::size_t Dim>
            class DomainConfiguration {

                public:
                    typedef std::pair<Boundary,Boundary> BoundaryPair;
                    typedef std::array<BoundaryPair,Dim> BoundaryArray;
                public: 
                    DomainConfiguration(const BoundaryArray& p_boundaries = defaultDomainBoundaries(), 
                            //bool p_includeDirichletBoundaries = true,
                            bool p_includePeriodicBoundaries  = true) :
                        m_boundaries(p_boundaries),
                        //m_includeDirichletBoundaries(p_includeDirichletBoundaries),
                        m_includePeriodicBoundaries(p_includePeriodicBoundaries) {
                            checkBoundaries();
                        }
                
                    const BoundaryArray& boundaries() const { return m_boundaries;                 }
                    //bool includeDirichletBoundaries() const { return m_includeDirichletBoundaries; }
                    bool includePeriodicBoundaries()  const { return m_includePeriodicBoundaries;  }

                    const BoundaryPair& operator[](std::size_t k) const { return m_boundaries[k]; }

                    void getShapeConfiguration(const typename Shape<Dim>::type &p_fullShape, typename Shape<Dim>::type &p_realShape, typename Offset<Dim>::type &p_leftOffset) const {
                        for (std::size_t d = 0; d < Dim; d++) {
                            const BoundaryPair& pair = m_boundaries[d];
                            //bool hasDirichletLeftOffset  = (pair.first  == DIRICHLET || pair.first  == HOMOGENEOUS_DIRICHLET);
                            //bool hasDirichletRightOffset = (pair.second == DIRICHLET || pair.second == HOMOGENEOUS_DIRICHLET);
                            //std::size_t dirichletLeftOffset  = hasDirichletLeftOffset  && !this->includeDirichletBoundaries();
                            //std::size_t dirichletRightOffset = hasDirichletRightOffset && !this->includeDirichletBoundaries();
                            bool hasPeriodicRightOffset  = (pair.second == Boundary::PERIODIC);
                            std::size_t periodicRightOffset  = hasPeriodicRightOffset  && !this->includePeriodicBoundaries();
                            std::size_t leftOffset  = 0;
                            std::size_t rightOffset = periodicRightOffset;
                            if(p_fullShape[d] <= (leftOffset + rightOffset))
                                throw std::runtime_error("Domain shape is to small on axe " + std::to_string(d) + " for prescribed boundaries !");
                            p_leftOffset[d] = std::ptrdiff_t(leftOffset);
                            p_realShape[d]  = p_fullShape[d] - leftOffset - rightOffset;
                        }
                    }

                protected:
                    void checkBoundaries() const {
                        for (std::size_t d = 0; d < Dim; d++) {
                            const BoundaryPair& pair = m_boundaries[d];
                            if((pair.first == Boundary::PERIODIC) ^ (pair.second == Boundary::PERIODIC))
                                throw std::runtime_error("Bad boundaries configuration on axe " + std::to_string(d) + " !");
                        }
                    }

                    #ifndef SWIG
                        template <int... I>
                        static const std::array<BoundaryPair, Dim> defaultDomainBoundariesImpl(hysop::detail::index_seq<I...>) {
                            const BoundaryPair defaultVal[1] = { std::make_pair(Boundary::NONE,Boundary::NONE) };
                            return { defaultVal[I]..., };
                        }
                        static const std::array<BoundaryPair, Dim> defaultDomainBoundaries() {
                            return defaultDomainBoundariesImpl(hysop::detail::constant_seq_gen<0,Dim>());
                        }
                    #endif 

                protected:
                    BoundaryArray  m_boundaries;
                    bool m_includePeriodicBoundaries;
                    //bool m_includeDirichletBoundaries;
            };
        
        template <std::size_t Dim>
            std::ostream& operator<<(std::ostream& os, const DomainConfiguration<Dim>& config) {
                os << "== DomainConfiguration<Dim=" << std::to_string(Dim) << "> ==" << std::endl;
                for (std::size_t d = 0; d < Dim; d++)
                    os << "\taxe[" << d << "]: " << config[d].first << "/" << config[d].second << std::endl;
                //os << "\tDirichlet boundaries included ? " << std::boolalpha << config.includeDirichletBoundaries() << std::endl;
                os << "\tPeriodic  boundaries included ? " << std::boolalpha << config.includePeriodicBoundaries()  << std::endl;
                return os;
            }

    }
}

#endif /* end of include guard: HYSOP_DOMAIN_CONFIGURATION_H */
