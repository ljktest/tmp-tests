
#ifndef HYSOP_FFTDIFFSOLVER_H
#define HYSOP_FFTDIFFSOLVER_H

#include "solver/diffSolver.h"
#include "fft/planner.h"
#include "data/accumulatorIndex.h"

namespace hysop {
    namespace solver {

        template <typename T, std::size_t Dim>
            class FftDiffSolver : public DiffSolver<T,Dim>  {

                private:
                    using super = DiffSolver<T,Dim>;
                public:
                    FftDiffSolver()                              = default;
                    FftDiffSolver(const FftDiffSolver&  p_other) = default;
                    FftDiffSolver(      FftDiffSolver&& p_other) = default;
                    ~FftDiffSolver()                             = default;

                    FftDiffSolver& operator=(const FftDiffSolver&  p_other) = default;
                    FftDiffSolver& operator=(      FftDiffSolver&& p_other) = default;

                    FftDiffSolver(const std::array<T,Dim> &p_domainSize, const fft::FftDomainConfiguration<Dim> &p_inputFftConfig,
                            unsigned int p_fftFlags = FFTW_MEASURE,
                            bool p_includeOutputPeriodicBds=false, bool p_mirrorOutputPeriodicBds=false);

                    /* Mutators */
                    FftDiffSolver& setDomainSize(const std::array<T,Dim>& p_domainSize);
                    FftDiffSolver& setFftDomainConfiguration(const fft::FftDomainConfiguration<Dim>& p_fftConfig);

                    FftDiffSolver& setFftFlags(unsigned int p_flags);
                    FftDiffSolver& appendFftFlags(unsigned int p_flags);

                    FftDiffSolver& includeOutputPeriodicBoundaries(bool p_val = true);
                    FftDiffSolver& excludeOutputPeriodicBoundaries();

                    FftDiffSolver& enableOutputPeriodicBoundariesMirroring(bool p_val = true);
                    FftDiffSolver& disableOutputPeriodicBoundariesMirroring();

                    /* Accessors */
                    std::array<T,Dim>                domainSize() const;
                    fft::FftDomainConfiguration<Dim> inputFftConfig() const;

                    unsigned int fftFlags() const;
                    bool includeOutputPeriodicBds() const;
                    bool mirrorOutputPeriodicBds()  const;

                    /* Apply operator */
                    virtual void apply(hysop::const_multi_array_ref<T,Dim> p_input,
                                       hysop::multi_array_ref<T,Dim> p_output,
                                       const std::array<int,Dim> &p_order) const final override;

                protected:
                    std::array<T,Dim>           m_domainSize;
                    fft::FftDomainConfiguration<Dim> m_inputFftConfig;
                    unsigned int m_fftFlags;
                    bool m_includeOutputPeriodicBds, m_mirrorOutputPeriodicBds;
            };
                    
        

        /* Implementation */

        template <typename T, std::size_t Dim>
        FftDiffSolver<T,Dim>::FftDiffSolver(const std::array<T,Dim> &p_domainSize, const fft::FftDomainConfiguration<Dim> &p_inputFftConfig,
                unsigned int p_fftFlags,
                bool p_includeOutputPeriodicBds, bool p_mirrorOutputPeriodicBds):
                m_domainSize(p_domainSize), m_inputFftConfig(p_inputFftConfig), 
                m_fftFlags(p_fftFlags),
                m_includeOutputPeriodicBds(p_includeOutputPeriodicBds), m_mirrorOutputPeriodicBds(p_mirrorOutputPeriodicBds) {
        }


        /* Mutators */
        template <typename T, std::size_t Dim>
            FftDiffSolver<T,Dim>& FftDiffSolver<T,Dim>::setDomainSize(const std::array<T,Dim>& p_domainSize) {
                m_domainSize = p_domainSize;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftDiffSolver<T,Dim>& FftDiffSolver<T,Dim>::setFftDomainConfiguration(const fft::FftDomainConfiguration<Dim>& p_fftConfig) {
                m_inputFftConfig = p_fftConfig;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftDiffSolver<T,Dim>& FftDiffSolver<T,Dim>::includeOutputPeriodicBoundaries(bool p_val) {
                m_includeOutputPeriodicBds = p_val;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftDiffSolver<T,Dim>& FftDiffSolver<T,Dim>::excludeOutputPeriodicBoundaries() {
                return this->includeOutputPeriodicBoundaries(false);
            }
        template <typename T, std::size_t Dim>
            FftDiffSolver<T,Dim>& FftDiffSolver<T,Dim>::setFftFlags(unsigned int p_flags) {
                m_fftFlags = p_flags;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftDiffSolver<T,Dim>& FftDiffSolver<T,Dim>::appendFftFlags(unsigned int p_flags) {
                m_fftFlags |= p_flags;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftDiffSolver<T,Dim>& FftDiffSolver<T,Dim>::enableOutputPeriodicBoundariesMirroring(bool p_val) {
                m_mirrorOutputPeriodicBds = p_val;
                return *this;
            }
        template <typename T, std::size_t Dim>
            FftDiffSolver<T,Dim>& FftDiffSolver<T,Dim>::disableOutputPeriodicBoundariesMirroring() {
                return this->enableOutputPeriodicBoundariesMirroring(false);
            }

        /* Accessors */
        template <typename T, std::size_t Dim>
            std::array<T,Dim> FftDiffSolver<T,Dim>::domainSize() const {
                return m_domainSize;
            }
        template <typename T, std::size_t Dim>
            fft::FftDomainConfiguration<Dim> FftDiffSolver<T,Dim>::inputFftConfig() const {
                return m_inputFftConfig;
            }
        template <typename T, std::size_t Dim>
            unsigned int FftDiffSolver<T,Dim>::fftFlags() const {
                return m_fftFlags;
            }
        template <typename T, std::size_t Dim>
            bool FftDiffSolver<T,Dim>::includeOutputPeriodicBds() const {
                return m_includeOutputPeriodicBds;
            }
        template <typename T, std::size_t Dim>
            bool FftDiffSolver<T,Dim>::mirrorOutputPeriodicBds()  const {
                return m_mirrorOutputPeriodicBds;
            }

        /* Apply operator */
        template <typename T, std::size_t Dim>
        void FftDiffSolver<T,Dim>::apply(
                hysop::const_multi_array_ref<T,Dim> p_input,
                hysop::multi_array_ref<T,Dim> p_output,
                const std::array<int,Dim> &p_order) const {

            fft::Planner<T,Dim> planner;
            bool success = planner.plan(p_input, p_output, m_inputFftConfig, p_order, m_domainSize, m_fftFlags, 
                                m_includeOutputPeriodicBds, m_mirrorOutputPeriodicBds);
            if(!success) 
                throw std::runtime_error("Failed to plan transforms !");
            
            planner.executeForwardTransform();
            {
                AccumulatorIndex<std::complex<T>,Dim> idx;
                idx.setAccumulatorSource(planner.waveNumbersPows()).setAccumulatorFunction(std::multiplies<std::complex<T>>());

                if(planner.transformType() == fft::FftTransformType::FFT_R2R) {
                    multi_array_view<T,Dim> view = planner.transformedRealData();
                    //view.print("PRE-RDATA");

                    idx.reset(view.shape());
                    while(!idx.atMaxId()) {
                        view(idx.ids()) *= idx.accumulatedVal().real()/planner.normalisationFactor();
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
                        data[idx()] *= idx.accumulatedVal()/planner.normalisationFactor();
                        ++idx;
                    }
                    
                    //ref.print("POST-CDATA");
                }
            }
            planner.executeBackwardTransform();
        }

    } /* end of namespace solver */
} /* end of namespace hysop */



#endif /* end of include guard: HYSOP_FFTDIFFSOLVER_H */
