
#ifndef HYSOP_FFTDOMAINCONFIGURATION_H
#define HYSOP_FFTDOMAINCONFIGURATION_H

#include "utils/defines.h"
#include "domain/domainConfiguration.h"
#include "fft/extension.h"

namespace hysop {
    namespace fft {

        template <std::size_t Dim>
            class FftDomainConfiguration;

        template <std::size_t Dim>
            std::ostream& operator<<(std::ostream& os, const FftDomainConfiguration<Dim>& config);

        template <std::size_t Dim>
        class FftDomainConfiguration {

                public:
                    typedef typename domain::DomainConfiguration<Dim>::BoundaryPair  BoundaryPair;
                    typedef typename domain::DomainConfiguration<Dim>::BoundaryArray BoundaryArray;
                    typedef std::pair<fft::Extension,fft::Extension>                 ExtensionPair;
                    typedef std::array<ExtensionPair,Dim>                            ExtensionArray;
                public: 
                    FftDomainConfiguration(const ExtensionArray& p_extensions, bool p_includePeriodicBoundaries) :
                        m_extensions(p_extensions), m_includePeriodicBoundaries(p_includePeriodicBoundaries) {
                        }

                    FftDomainConfiguration(const domain::DomainConfiguration<Dim>& p_domain):
                        m_extensions(boundariesToExtensions(p_domain.boundaries())), 
                        m_includePeriodicBoundaries(p_domain.includePeriodicBoundaries()) {
                        }
                    
                    const ExtensionArray& extensions() const { return m_extensions; }
                    bool   includePeriodicBoundaries() const { return m_includePeriodicBoundaries; }

                    ExtensionPair operator[](std::size_t k) const {
                        return m_extensions[k];
                    }

                    domain::DomainConfiguration<Dim> boundariesConfiguration() const {
                        return domain::DomainConfiguration<Dim>(fftExtensionsToBoundaries(m_extensions), m_includePeriodicBoundaries);
                    }

                    static BoundaryArray fftExtensionsToBoundaries(const ExtensionArray& extArray) {
                        BoundaryArray bdArray;
                        for(std::size_t d=0; d<Dim; d++) {
                            const ExtensionPair& extPair = extArray[d];
                            bdArray[d] = std::make_pair(fftExtensionToBoundary(extPair.first), fftExtensionToBoundary(extPair.second));
                        }
                        return bdArray;
                    }
                    
                    static ExtensionArray boundariesToExtensions(const BoundaryArray& bdArray) {
                        ExtensionArray extArray;
                        for(std::size_t d=0; d<Dim; d++) {
                            const BoundaryPair& bdPair = bdArray[d];
                            extArray[d] = std::make_pair(boundaryToExtension(bdPair.first), boundaryToExtension(bdPair.second));
                        }
                        return extArray;
                    }

                    static domain::Boundary fftExtensionToBoundary(fft::Extension ext) {
                        switch(ext) {
                            case(fft::Extension::PERIODIC):
                                return domain::Boundary::PERIODIC;
                            case(fft::Extension::EVEN):
                                return domain::Boundary::HOMOGENEOUS_DIRICHLET;
                            case(fft::Extension::ODD):
                                return domain::Boundary::HOMOGENEOUS_NEUMANN;
                            case(fft::Extension::NONE):
                                return domain::Boundary::NONE;
                            default:
                                NOT_IMPLEMENTED_YET;
                        }
                    }
                    
                    static fft::Extension boundaryToExtension(domain::Boundary bd) {
                        switch(bd) {
                            case(domain::Boundary::PERIODIC):
                                return fft::Extension::PERIODIC;
                            case(domain::Boundary::HOMOGENEOUS_DIRICHLET):
                                return fft::Extension::EVEN;
                            case(domain::Boundary::HOMOGENEOUS_NEUMANN):
                                return fft::Extension::ODD;
                            case(domain::Boundary::NONE):
                                return fft::Extension::NONE;
                                //throw std::runtime_error("Cannot build a FftDomainConfiguration based on a boundary of type 'NONE' !");
                            default:
                                NOT_IMPLEMENTED_YET;
                        }
                    }
                
                protected:
                    ExtensionArray  m_extensions;
                    bool m_includePeriodicBoundaries;
            };

        template <std::size_t Dim>
            std::ostream& operator<<(std::ostream& os, const FftDomainConfiguration<Dim>& config) {
                os << "== FftDomainConfiguration<Dim=" << std::to_string(Dim) << "> ==" << std::endl;
                for (std::size_t d = 0; d < Dim; d++)
                    os << "\taxe[" << d << "]: " << config[d].first << "/" << config[d].second;
                os << "\tPeriodic  boundaries included ?" << std::boolalpha << config.includePeriodicBoundaries()  << std::endl;
                return os;
            }

    }
}

#endif /* end of include guard: HYSOP_FFTDOMAINCONFIGURATION_H */

