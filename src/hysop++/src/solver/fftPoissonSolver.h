
#ifndef HYSOP_FFTPOISSONSOLVER_H
#define HYSOP_FFTPOISSONSOLVER_H

#include <cmath>

#include "maths/quad_maths.h"
#include "solver/poissonSolver.h"
#include "fft/planner.h"
#include "data/accumulatorIndex.h"

namespace hysop {
    namespace solver {

        template <typename T, std::size_t Dim>
            class FftPoissonSolver : public PoissonSolver<T,Dim>  {

                private:
                    using super = PoissonSolver<T,Dim>;
                public:
                    FftPoissonSolver()                                 = default;
                    FftPoissonSolver(const FftPoissonSolver&  p_other) = default;
                    FftPoissonSolver(      FftPoissonSolver&& p_other) = default;
                    ~FftPoissonSolver()                                = default;

                    FftPoissonSolver& operator=(const FftPoissonSolver&  p_other) = default;
                    FftPoissonSolver& operator=(      FftPoissonSolver&& p_other) = default;

                    FftPoissonSolver(const std::array<T,Dim> &p_domainSize, const domain::DomainConfiguration<Dim> &p_domainConfig,
                            unsigned int p_fftFlags = FFTW_MEASURE,
                            bool p_includeOutputPeriodicBds=false, bool p_mirrorOutputPeriodicBds=false);

                    /* Mutators */
                    FftPoissonSolver& setDomainSize(const std::array<T,Dim>& p_domainSize);
                    FftPoissonSolver& setInputDomainConfiguration(const domain::DomainConfiguration<Dim>& p_domainConfig);

                    FftPoissonSolver& setFftFlags(unsigned int p_flags);
                    FftPoissonSolver& appendFftFlags(unsigned int p_flags);

                    FftPoissonSolver& includeOutputPeriodicBoundaries(bool p_val = true);
                    FftPoissonSolver& excludeOutputPeriodicBoundaries();

                    FftPoissonSolver& enableOutputPeriodicBoundariesMirroring(bool p_val = true);
                    FftPoissonSolver& disableOutputPeriodicBoundariesMirroring();

                    /* Accessors */
                    std::array<T,Dim>                domainSize() const;
                    domain::DomainConfiguration<Dim> inputDomainConfig() const;

                    unsigned int fftFlags() const;
                    bool includeOutputPeriodicBds() const;
                    bool mirrorOutputPeriodicBds()  const;

                    /* Apply operator */
                    virtual void apply(hysop::const_multi_array_ref<T,Dim> p_input,
                                       hysop::multi_array_ref<T,Dim> p_output) const final override;

                protected:
                    std::array<T,Dim>           m_domainSize;
                    domain::DomainConfiguration<Dim> m_inputDomainConfig;
                    unsigned int m_fftFlags;
                    bool m_includeOutputPeriodicBds, m_mirrorOutputPeriodicBds;
            };
                    
        

        /* Implementation */

        template <typename T, std::size_t Dim>
        FftPoissonSolver<T,Dim>::FftPoissonSolver(const std::array<T,Dim> &p_domainSize, const domain::DomainConfiguration<Dim> &p_inputDomainConfig,
                unsigned int p_fftFlags,
                bool p_includeOutputPeriodicBds, bool p_mirrorOutputPeriodicBds):
                m_domainSize(p_domainSize), m_inputDomainConfig(p_inputDomainConfig), 
                m_fftFlags(p_fftFlags),
                m_includeOutputPeriodicBds(p_includeOutputPeriodicBds), m_mirrorOutputPeriodicBds(p_mirrorOutputPeriodicBds) {
        }


        /* Mutators */
        template <typename T, std::size_t Dim>
            FftPoissonSolver<T,Dim>& FftPoissonSolver<T,Dim>::setDomainSize(const std::array<T,Dim>& p_domainSize) {
                m_domainSize = p_domainSize;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftPoissonSolver<T,Dim>& FftPoissonSolver<T,Dim>::setInputDomainConfiguration(const domain::DomainConfiguration<Dim>& p_inputDomainConfig) {
                m_inputDomainConfig = p_inputDomainConfig;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftPoissonSolver<T,Dim>& FftPoissonSolver<T,Dim>::includeOutputPeriodicBoundaries(bool p_val) {
                m_includeOutputPeriodicBds = p_val;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftPoissonSolver<T,Dim>& FftPoissonSolver<T,Dim>::excludeOutputPeriodicBoundaries() {
                return this->includeOutputPeriodicBoundaries(false);
            }
        template <typename T, std::size_t Dim>
            FftPoissonSolver<T,Dim>& FftPoissonSolver<T,Dim>::setFftFlags(unsigned int p_flags) {
                m_fftFlags = p_flags;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftPoissonSolver<T,Dim>& FftPoissonSolver<T,Dim>::appendFftFlags(unsigned int p_flags) {
                m_fftFlags |= p_flags;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftPoissonSolver<T,Dim>& FftPoissonSolver<T,Dim>::enableOutputPeriodicBoundariesMirroring(bool p_val) {
                m_mirrorOutputPeriodicBds = p_val;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftPoissonSolver<T,Dim>& FftPoissonSolver<T,Dim>::disableOutputPeriodicBoundariesMirroring() {
                return this->enableOutputPeriodicBoundariesMirroring(false);
            }

        /* Accessors */
        template <typename T, std::size_t Dim>
            std::array<T,Dim> FftPoissonSolver<T,Dim>::domainSize() const {
                return m_domainSize;
            }
        template <typename T, std::size_t Dim>
            domain::DomainConfiguration<Dim> FftPoissonSolver<T,Dim>::inputDomainConfig() const {
                return m_inputDomainConfig;
            }
        template <typename T, std::size_t Dim>
            unsigned int FftPoissonSolver<T,Dim>::fftFlags() const {
                return m_fftFlags;
            }
        template <typename T, std::size_t Dim>
            bool FftPoissonSolver<T,Dim>::includeOutputPeriodicBds() const {
                return m_includeOutputPeriodicBds;
            }
        template <typename T, std::size_t Dim>
            bool FftPoissonSolver<T,Dim>::mirrorOutputPeriodicBds()  const {
                return m_mirrorOutputPeriodicBds;
            }

        /* Apply operator */
        template <typename T, std::size_t Dim>
        void FftPoissonSolver<T,Dim>::apply(
                hysop::const_multi_array_ref<T,Dim> p_input,
                hysop::multi_array_ref<T,Dim> p_output) const {


            fft::Planner<T,Dim> planner;
            planner.setFixedAxeWaveNumberPows(std::complex<T>(0,0));
            {
                std::array<int, Dim> order;
                for (std::size_t d=0; d<Dim; d++)
                    order[d] = (m_inputDomainConfig[d].first == domain::Boundary::NONE ? 0 : 2);
                bool success = planner.plan(p_input, p_output, m_inputDomainConfig, order, m_domainSize, m_fftFlags, 
                                    m_includeOutputPeriodicBds, m_mirrorOutputPeriodicBds);
                if(!success) 
                    throw std::runtime_error("Failed to plan transforms !");
            }
            const T normalisationFactor = planner.normalisationFactor();
            //std::cout << planner << std::endl;

            planner.executeForwardTransform();
            {
                AccumulatorIndex<std::complex<T>,Dim> idx;
                idx.setAccumulatorSource(planner.waveNumbersPows()).setAccumulatorFunction(std::plus<std::complex<T>>());

                if(planner.transformType() == fft::FftTransformType::FFT_R2R) {
                    multi_array_view<T,Dim> view = planner.transformedRealData();
                    //view.print("PRE-RDATA");

                    idx.reset(view.shape());
                    while(!idx.atMaxId()) {
                        T filter = idx.accumulatedVal().real(); 
                        filter = (std::fpclassify(filter)==FP_ZERO ? T(0) : (T(1)/filter)*(T(1)/normalisationFactor));  
                        view(idx.ids()) *= filter;
                        ++idx;
                    }
                    
                    //view.print("POST-RDATA");
                }
                else if(planner.transformType() == fft::FftTransformType::FFT_R2C) {
                    multi_array_ref<std::complex<T>,Dim> ref = planner.transformedComplexData();
                    std::complex<T> *data = ref.data();
                    //ref.print("PRE-CDATA");

                    idx.reset(ref.shape());
                    while(!idx.atMaxId()) {
                        std::complex<T> filter = idx.accumulatedVal(); 
                        filter = ((std::fpclassify(filter.real())==FP_ZERO) && (std::fpclassify(filter.imag())==FP_ZERO) ? 
                                std::complex<T>(0,0) : (T(1)/filter)*(T(1)/normalisationFactor));
                        data[idx()] *= filter;
                        ++idx;
                    }
                    
                    //ref.print("POST-CDATA");
                }
            }
            planner.executeBackwardTransform();
        }

    } /* end of namespace solver */
} /* end of namespace hysop */



#endif /* end of include guard: HYSOP_FFTPOISSONSOLVER_H */
