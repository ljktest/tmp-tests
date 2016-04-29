
#ifndef HYSOP_PLANNER_H
#define HYSOP_PLANNER_H

#include <list>
#include <cassert>
#include <functional>

#include "fft/fftw3.h"
#include "fft/transform.h"
#include "fft/fftDomainConfiguration.h"
#include "domain/domain.h"
#include "utils/default.h"

namespace hysop {
    namespace fft {
                    
        enum FftTransformType : int {
            FFT_NONE=-1,
            FFT_R2R,
            FFT_R2C
        };
        
        template <typename T, std::size_t Dim>
            class Planner;

        template <typename T, std::size_t Dim>
            std::ostream& operator<<(std::ostream& os, const Planner<T,Dim>& planner);

        template <typename T, std::size_t Dim>
            class Planner : private Fftw3<T> {
                static_assert(hysop::fft::is_fftw_supported_type<T>::value, 
                        "Planner data type is not currently supported by fftw !");

                public:
                    using         real = typename Fftw3<T>::R;
                    using fftw_complex = typename Fftw3<T>::C;
                    using  std_complex = typename Fftw3<T>::stdC;
                    using    fftw_plan = typename Fftw3<T>::plan;
                    using   fftw_iodim = typename Fftw3<T>::iodim;
                    
                protected:
                    using RealArray = hysop::multi_array<T,Dim>;
                    using TransformArray = std::array<fft::Transform,Dim>;

                public:
                    Planner(std::complex<T> p_fixedAxeWaveNumbers    = std::complex<T>(1,0),
                            std::complex<T> p_fixedAxeWaveNumberPows = std::complex<T>(1,0));
                    Planner(const Planner& other) = delete; /* cannot copy fftw plans to prevent inter instance plan destroying */
                    Planner(Planner&& other)      = default;
                    virtual ~Planner();

                    Planner& operator=(const Planner& planner) = delete;
                    Planner& operator=(Planner&& planner)      = default;
                
                    /* Plans a forward and a backward DCT/DST/FFT on each axe of the domain.                                */
                    /* An axe with a NONE/NONE fft fomain configuration or order=0 is not transformed.                      */
                    /* Input and output can be the same but in case of inplace transform, a temporary buffer is created     */
                    /* during planning.                                                                                     */
                    /* If transforms include a complex transform (FFT) or if the transform is inplace, additional data may  */
                    /* be allocated inside the class.                                                                       */ 
                    /* fftw_flags are used to pass flags to FFTW.                                                           */  
                    /* Return true if planning was successfull AND if there is any transform, else return false.            */
                    /* Note: To get max performance use hysop::_default::fft_allocator<T> with hysop::multi_array           */
                    bool plan(hysop::const_multi_array_ref<T,Dim> input_rdata, 
                              hysop::multi_array_ref<T,Dim> output_rdata,
                              const fft::FftDomainConfiguration<Dim>& inputFftDomainConfig,
                              const std::array<int,Dim>& order,
                              const std::array<T  ,Dim>& domainSize,
                              unsigned int fftw_flags,
                              bool includeOutputPeriodicBoundaries=false, bool mirrorOutputPeriodicBoundaries=false);

                    void executeForwardTransform();
                    void executeBackwardTransform();

                    FftTransformType transformType() const;

                    T normalisationFactor() const;
                    const std::array<T,Dim> signs() const;
                    const std::array<std::vector<std::complex<T>>,Dim>& waveNumbers() const;
                    const std::array<std::vector<std::complex<T>>,Dim>& waveNumbersPows() const;

                    hysop::multi_array_view<T,Dim>              transformedRealData();    /* view because possibly non contiguous */
                    hysop::multi_array_ref<std::complex<T>,Dim> transformedComplexData(); /* ref  because contiguous (allocated inside class) */
                    
                    /* Get planned transform description */
                    const std::string& toString() const;

                    /* Modify untransformed axe generated wave numbers to simplify algorithm implementations based on the planner */
                    Planner<T,Dim>& setFixedAxeWaveNumbers(std::complex<T> p_fixedAxeWaveNumber); 
                    Planner<T,Dim>& setFixedAxeWaveNumberPows(std::complex<T> p_fixedAxeWaveNumberPow); 

                protected:
                    void checkDomainCompatibility(const fft::FftDomainConfiguration<Dim>& domainConfig) const;
                    void checkExtensionCompatibility(const fft::Extension &lhs, const fft::Extension &rhs) const;

                    fft::Transform findTransform(const std::pair<Extension,Extension>& ed) const;

                    void reset();
                        
                    template <typename Transfo>
                    void toStreamTransform(std::ostream& os, const Transfo& tr, 
                            int rank, int howmany_rank, 
                            const fftw_iodim* dims, const fftw_iodim *howmany_dims,
                            int input_data_offset=0, int output_data_offset=0);

                protected:
                    std::complex<T> m_fixedAxeWaveNumbers, m_fixedAxeWaveNumberPows;

                    hysop::multi_array_ref<T,Dim> m_realBuffer;
                    hysop::multi_array<std_complex,Dim,hysop::_default::fft_allocator<std_complex>>   m_complexBuffer;
                    typename hysop::multi_array<T,Dim>::template index_gen<Dim,Dim>  m_transformedRealBufferView;

                    std::list<fftw_plan> m_forward_R2R_plans,  m_R2C_plans;
                    std::list<fftw_plan> m_backward_R2R_plans, m_C2R_plans;

                    FftTransformType m_transformType;
                    std::vector<std::size_t> m_periodicAxes;

                    std::array<T,Dim> m_signs;
                    std::array<std::vector<std_complex>,Dim> m_waveNumbers;
                    std::array<std::vector<std_complex>,Dim> m_waveNumbersPows;
                    T m_normalisationFactor;

                    bool m_mirrorOutputPeriodicBoundaries;
                    std::string m_plannedTransformStr;
            };
        
        template <typename T, std::size_t Dim>
            Planner<T,Dim>::Planner(std::complex<T> p_fixedAxeWaveNumbers, std::complex<T> p_fixedAxeWaveNumberPows):
                m_fixedAxeWaveNumbers(p_fixedAxeWaveNumbers),
                m_fixedAxeWaveNumberPows(p_fixedAxeWaveNumberPows),
                m_realBuffer(), m_complexBuffer(),
                m_forward_R2R_plans(), m_R2C_plans(),
                m_backward_R2R_plans(), m_C2R_plans(),
                m_transformType(), 
                m_signs(), m_waveNumbers(), m_waveNumbersPows(), m_normalisationFactor(),
                m_mirrorOutputPeriodicBoundaries(),
                m_plannedTransformStr() {
                    reset();
            }
        
        template <typename T, std::size_t Dim>
            Planner<T,Dim>::~Planner() {
                reset();
            }

        template <typename T, std::size_t Dim>
            void Planner<T,Dim>::reset() {
                new (&m_realBuffer) multi_array_ref<T,Dim>();
                m_complexBuffer.reshape(typename Shape<Dim>::type{0});
                for(fftw_plan plan : m_forward_R2R_plans)
                    this->fftw_destroy_plan(plan);
                for(fftw_plan plan : m_backward_R2R_plans)
                    this->fftw_destroy_plan(plan);
                for(fftw_plan plan : m_C2R_plans)
                    this->fftw_destroy_plan(plan);
                for(fftw_plan plan : m_R2C_plans)
                    this->fftw_destroy_plan(plan);
                m_forward_R2R_plans.clear();
                m_R2C_plans.clear();
                m_backward_R2R_plans.clear();
                m_C2R_plans.clear();
                m_periodicAxes.clear();
                m_transformType = FftTransformType::FFT_NONE;
                for (std::size_t d = 0; d < Dim; d++) {
                    m_waveNumbers[d].clear(); 
                    m_waveNumbersPows[d].clear();
                }
                m_signs.fill(0);
                m_normalisationFactor = T(0);
                m_mirrorOutputPeriodicBoundaries = false;
                m_plannedTransformStr = "** No planned transforms **";
        }
                    
        template <typename T, std::size_t Dim>
            void Planner<T,Dim>::checkDomainCompatibility(const fft::FftDomainConfiguration<Dim>& domainConfig) const {
                for (std::size_t d = 0; d < Dim; d++)
                    checkExtensionCompatibility(domainConfig[d].first, domainConfig[d].second);
            }

        template <typename T, std::size_t Dim>
            void Planner<T,Dim>::checkExtensionCompatibility(const fft::Extension &lhs, const fft::Extension &rhs) const {
                if ((lhs==fft::Extension::PERIODIC) ^ (rhs==fft::Extension::PERIODIC)) 
                    throw std::runtime_error("Planner error: Periodic domain extensions are not compatible !");
                if ((lhs==fft::Extension::NONE) ^ (rhs==fft::Extension::NONE))
                    throw std::runtime_error("Planner error: None domain extensions are not compatible !");
            }

        template <typename T, std::size_t Dim>
            bool Planner<T,Dim>::plan(hysop::const_multi_array_ref<T,Dim> input_rdata, 
                                      hysop::multi_array_ref<T,Dim> output_rdata,
                    const fft::FftDomainConfiguration<Dim>& inputFftDomainConfig,
                    const std::array<int,Dim>& order,
                    const std::array<T  ,Dim>& domainSize,
                                  unsigned int fftw_flags, 
                    const bool includeOutputPeriodicBoundaries,
                    const bool mirrorOutputPeriodicBoundaries) {

                this->checkDomainCompatibility(inputFftDomainConfig);
                this->reset();

                bool inplace = (input_rdata.data() == output_rdata.data());
                hysop::multi_array<T,Dim> input_rdata_buffer;
                if(inplace)
                    input_rdata_buffer = input_rdata;

                m_mirrorOutputPeriodicBoundaries = includeOutputPeriodicBoundaries && mirrorOutputPeriodicBoundaries;
                
                TransformArray forwardTransforms, backwardTransforms;
                typename Shape<Dim>::type  realBufferShape, complexBufferShape;
                std::array<int,Dim> forward_transform_size, backward_transform_size, complex_transform_size;
                std::array<int,Dim> forward_input_offset,  forward_output_offset;
                std::array<int,Dim> backward_input_offset, backward_output_offset;
                std::array<bool,Dim> oddOrder, axeTransformed;
                
                bool hasRealTransforms=false, hasComplexTransforms=false;
                int lastComplexTransformAxe=-1;
                std::stringstream ss;

                ss << "=== Planner ===" << std::endl;
                ss << "** input **" << std::endl;
                ss << "\tInputDataDim: " << input_rdata.shape() << std::endl;
                ss << "\tDomainConfig: " << inputFftDomainConfig;
                ss << "\tOrder       : " << order << std::endl;
                ss << "\tDomainSize  : " << domainSize << std::endl;
                ss << "\tipb/mpb     : " << std::boolalpha << includeOutputPeriodicBoundaries << "/" 
                    << std::boolalpha << mirrorOutputPeriodicBoundaries << std::endl;

                for (std::size_t d = 0; d < Dim; d++) {
                    const      FftDomainConfiguration<Dim> &inputConfig = inputFftDomainConfig;
                    const auto &inputExtensions = inputConfig.extensions()[d]; 
                    const bool axeIsPeriodic = (inputExtensions.first==Extension::PERIODIC);
                    const bool axeIsOddOrder = (order[d]%2!=0);
                    const bool axeIsTransformed = !(order[d]==0 || inputExtensions.first == fft::Extension::NONE);
                    const std::size_t inputSize = input_rdata.shape()[d] 
                        - std::size_t(axeIsPeriodic && inputConfig.includePeriodicBoundaries());
                    
                    const fft::Transform tr = this->findTransform(inputExtensions);

                    forwardTransforms[d] = tr;
                    if(axeIsOddOrder)
                        backwardTransforms[d] = tr.conjugateInverseTransform();
                    else
                        backwardTransforms[d] = tr.inverseTransform();
                       
                    realBufferShape[d]         = inputSize + 
                        std::size_t((inputExtensions.second==Extension::PERIODIC) && includeOutputPeriodicBoundaries);
                    complexBufferShape[d]      = inputSize;
                    forward_transform_size[d]  = inputSize;
                    backward_transform_size[d] = inputSize;
                    complex_transform_size[d]  = inputSize;

                    forward_input_offset[d]   = 0;
                    forward_output_offset[d]  = 0;
                    backward_input_offset[d]  = 0;
                    backward_output_offset[d] = 0;

                    oddOrder[d]       = axeIsOddOrder;
                    axeTransformed[d] = axeIsTransformed;

                    if(axeIsPeriodic)
                        m_periodicAxes.push_back(d);
                    
                    if(!axeIsTransformed) {
                        continue;
                    }
                    else {
                        if(tr.isR2R()) {
                            hasRealTransforms = true;
                            const std::size_t firstExtOdd   = (inputExtensions.first  == Extension::ODD);
                            const std::size_t secondExtOdd  = (inputExtensions.second == Extension::ODD);
                            const std::size_t firstExtEven  = (inputExtensions.first  == Extension::EVEN);
                            const std::size_t secondExtEven = (inputExtensions.second == Extension::EVEN);

                            forward_transform_size[d]  -= (firstExtOdd + secondExtOdd);
                            forward_input_offset[d]   = std::ptrdiff_t(firstExtOdd);
                            forward_output_offset[d]  = std::ptrdiff_t(firstExtOdd);
                            
                            complexBufferShape[d]     = forward_transform_size[d];
                            complex_transform_size[d] = forward_transform_size[d];
                            
                            if(axeIsOddOrder) {
                                backward_transform_size[d] -= (firstExtEven + secondExtEven);
                                backward_input_offset[d] = std::ptrdiff_t( (tr.kind == FFTW_RODFT01) 
                                        || ((tr.kind != FFTW_REDFT01) && firstExtEven) );
                                backward_output_offset[d] = std::ptrdiff_t(firstExtEven);
                            }
                            else {
                                backward_transform_size[d] = forward_transform_size[d];
                                backward_input_offset[d]   = forward_output_offset[d]; 
                                backward_output_offset[d]  = forward_input_offset[d];
                            }
                        }
                        else {
                            hasComplexTransforms = true;
                            lastComplexTransformAxe = d;
                        }
                    }
                }
                
                
                /* Allocate complex buffer only if necessary, return if no transforms */
                if(hasComplexTransforms) {
                    m_transformType = FFT_R2C;
                    
                    assert(lastComplexTransformAxe >= 0);
                    complexBufferShape[lastComplexTransformAxe] = complexBufferShape[lastComplexTransformAxe]/2 + 1;
                    m_complexBuffer.reshape(complexBufferShape);
                }
                else if(hasRealTransforms) {
                    m_transformType = FFT_R2R;
                }
                else {
                    m_transformType = FFT_NONE;
                    if(!inplace)
                        std::copy(input_rdata.begin(), input_rdata.end(), output_rdata.begin());
                    this->reset();
                    ss << m_plannedTransformStr << std::endl;
                        m_plannedTransformStr = ss.str();
                    return false;
                }
                
                /* Check if output buffer shape match */
                {
                    if(output_rdata.shape() != realBufferShape)
                        throw std::runtime_error("Output buffer shape should match the planned one !");
                    new (&m_realBuffer) multi_array_ref<T,Dim>(output_rdata);
                }
                
                /* Compute normalisation factor and wave numbers */
                m_normalisationFactor = T(1);
                for(std::size_t d=0; d<Dim; d++) {
                    const fft::Transform tr = forwardTransforms[d];
                    const std::size_t N = (tr.isR2R() ? realBufferShape[d] : complexBufferShape[d]);
                    const std::size_t K = (tr.isR2R() ? forward_transform_size[d] : complexBufferShape[d]);
                    const std::size_t S = (tr.isR2R() ? forward_transform_size[d] : complex_transform_size[d]);
                    const real L = domainSize[d];
                    T sign = T(1);
                    if(!axeTransformed[d]) {
                        m_waveNumbers[d].resize(N);
                        m_waveNumbersPows[d].resize(N);
                        m_normalisationFactor*=T(1);
                        for (std::size_t k = 0; k < N; k++) {
                            m_waveNumbers[d][k]     = m_fixedAxeWaveNumbers;
                            m_waveNumbersPows[d][k] = m_fixedAxeWaveNumberPows;
                        }
                    }
                    else {
                        m_waveNumbers[d].resize(K);
                        m_waveNumbersPows[d].resize(K);
                        m_normalisationFactor*= tr.normalisation<T>(S);
                        if(tr.isR2R()) {
                            const std::size_t sign_offset = (tr.basefunc()==fft::BaseFunc::COS ? 1 : 0);
                            sign *= std::pow(T(-1),(order[d]+sign_offset)/2);
                        }
                        for (std::size_t k = 0; k < K; k++) {
                            m_waveNumbers[d][k] = tr.omega<T>(k,K,L,d==std::size_t(lastComplexTransformAxe));
                            m_waveNumbersPows[d][k] = sign*std::pow(m_waveNumbers[d][k], order[d]);
                        }
                    }
                    m_signs[d] = sign;
                }
            
                /* Build planned transforms description string */
                ss << "** configuration **"                                                  << std::endl;
                ss << "\thasRealTransforms    : " << std::boolalpha << hasRealTransforms     << std::endl;
                ss << "\thasComplexTransforms : " << std::boolalpha << hasComplexTransforms  << std::endl;
                ss << "\tTransformed axes     : " << std::noboolalpha << axeTransformed      << std::endl;
                ss << "\tOddOrder             : " << std::noboolalpha << oddOrder            << std::endl;
                ss << "\tForward  transforms  : " << forwardTransforms                       << std::endl;
                ss << "\tBackward transforms  : " << backwardTransforms                      << std::endl;
                ss << "\tLast cplx trans. axe : " << lastComplexTransformAxe                 << std::endl;

                ss << "** buffer shape **"                                                   << std::endl;
                ss << "\tReal    buffer shape  : " << realBufferShape                        << std::endl;
                ss << "\tComplex buffer shape  : " << complexBufferShape                     << std::endl;

                ss << "** transform size and offsets **"                                     << std::endl;
                ss << "\tForward  transf. size : " << forward_transform_size                 << std::endl;
                ss << "\tForward  input  offset: " << forward_input_offset                   << std::endl;
                ss << "\tForward  output offset: " << forward_output_offset                  << std::endl;
                ss << "\tComplex  transf. size : " << complex_transform_size                 << std::endl;
                ss << "\tBackward transf. size : " << backward_transform_size                << std::endl;
                ss << "\tBackward input  offset: " << backward_input_offset                  << std::endl;
                ss << "\tBackward output offset: " << backward_output_offset                 << std::endl;

                ss << "** normalisation & wave numbers **" << std::endl;
                ss << "\tNormalisation: " << m_normalisationFactor << std::endl;
                ss << "\tSigns: " << m_signs << std::endl;
                ss << "\t--Wave numbers--"<< std::endl;
                for (std::size_t k = 0; k < Dim; k++)
                    ss << "\t\taxe" << k << ": " << m_waveNumbers[k] << std::endl;
                ss << "\t--wave numbers powers--"<< std::endl;
                for (std::size_t k = 0; k < Dim; k++)
                    ss << "\t\taxe" << k << ": " << m_waveNumbersPows[k] << std::endl;
                
                
                /* Compute complex plans */
                if(hasComplexTransforms) {
                    ss << "** complex transforms detail **" << std::endl;
                    fftw_plan R2C_plan=NULL, C2R_plan=NULL;
                    const int rank = lastComplexTransformAxe+1;
                    const int howmany_rank = Dim;
                    fftw_iodim dims[Dim];
                    fftw_iodim howmany_dims[Dim];
                    int forwardInputOffset=0, backwardOutputOffset=0;
                    {
                        int local_io_stride = 1;
                        for(int d=Dim-1; d>=0; d--) {
                            forwardInputOffset   += forward_output_offset[d]*local_io_stride;
                            backwardOutputOffset += forward_output_offset[d]*local_io_stride;
                            local_io_stride *= realBufferShape[d];
                        }
                    }
                    {
                        int input_stride=1, output_stride=1;
                        for(int d=Dim-1, dd=rank; d>=0; d--) {
                            const bool isR2C            = forwardTransforms[d].isR2C();
                            const bool isAxeTransformed = axeTransformed[d];
                            if(isR2C && isAxeTransformed) {
                                dims[d]         = fftw_iodim{complex_transform_size[d], input_stride, output_stride};
                                howmany_dims[d] = fftw_iodim{                        1, input_stride, output_stride};
                            }
                            else {
                                dims[d]         = fftw_iodim{                        1, input_stride, output_stride};
                                howmany_dims[d] = fftw_iodim{forward_transform_size[d], input_stride, output_stride};
                            }
                            input_stride  *= realBufferShape[d];
                            output_stride *= complexBufferShape[d];
                        }
                        R2C_plan = this->fftw_plan_guru_dft_r2c(
                                rank, dims, 
                                howmany_rank, howmany_dims,
                                m_realBuffer.rdata()+forwardInputOffset, m_complexBuffer.fftw_cdata(),
                                fftw_flags);
                        this->toStreamTransform(ss, "FFTW_FORWARD", rank, howmany_rank, dims, howmany_dims,
                                forwardInputOffset, 0);
                        if(!R2C_plan) {
                            ss << "=> R2C plan creation FAILED !" << std::endl;
                            m_plannedTransformStr = ss.str();
                            return false;
                        }
                    }
                    {
                        int input_stride=1, output_stride=1;
                        for(int d=Dim-1, dd=rank; d>=0; d--) {
                            const bool isR2C            = forwardTransforms[d].isR2C();
                            const bool isAxeTransformed = axeTransformed[d];
                            if(isR2C && isAxeTransformed) {
                                dims[d]         = fftw_iodim{complex_transform_size[d], input_stride, output_stride};
                                howmany_dims[d] = fftw_iodim{                        1, input_stride, output_stride};
                            }
                            else {
                                dims[d]         = fftw_iodim{                         1, input_stride, output_stride};
                                howmany_dims[d] = fftw_iodim{ forward_transform_size[d], input_stride, output_stride};
                            }
                            input_stride  *= complexBufferShape[d];
                            output_stride *= realBufferShape[d];
                        }
                        C2R_plan = this->fftw_plan_guru_dft_c2r(
                                rank, dims, 
                                howmany_rank, howmany_dims,
                                m_complexBuffer.fftw_cdata(), m_realBuffer.rdata()+backwardOutputOffset,
                                fftw_flags);
                        this->toStreamTransform(ss, "FFTW_BACKWARD", rank, howmany_rank, dims, howmany_dims,
                                0, backwardOutputOffset);
                        if(!C2R_plan) {
                            ss << "=> C2R plan creation FAILED !" << std::endl;
                            m_plannedTransformStr = ss.str();
                            return false;
                        }
                    }
                    m_R2C_plans.push_back(R2C_plan);
                    m_C2R_plans.push_back(C2R_plan);
                }
                
                /* Compute real plans */
                if(hasRealTransforms) {
                    ss << "** real transforms detail **" << std::endl;
                    const int rank = 1;
                    const int howmany_rank = Dim;
                    fftw_r2r_kind kind[rank];
                    fftw_iodim    dims[rank];
                    fftw_iodim howmany_dims[howmany_rank];
                    int io_stride = 1;
                    for(int k=Dim-1; k>=0; k--) {
                        const fft::Transform& ftr  = forwardTransforms[k];
                        const fft::Transform& btr  = backwardTransforms[k];
                        const bool isR2R            = ftr.isR2R() && btr.isR2R();
                        const bool isAxeTransformed = axeTransformed[k];
                        //int forwardInputOffset=0, forwardOutputOffset=0, backwardInputOffset=0, backwardOutputOffset=0;
                        if(isR2R && isAxeTransformed) {
                            ss << "\tTRANSFORM (" << std::to_string(k) << "):" << std::endl;
                            fftw_plan forward_plan=NULL, backward_plan=NULL;
                            int local_io_stride = 1;
                            for(int d=Dim-1; d>=0; d--) {
                                //howmany_dims[d] = fftw_iodim{ (d==k ? 1 : forward_transform_size[d]), local_io_stride, local_io_stride };
                                //forwardInputOffset   += forward_input_offset[d]  *local_io_stride;
                                //forwardOutputOffset  += forward_output_offset[d] *local_io_stride;
                                //backwardInputOffset  += backward_input_offset[d] *local_io_stride;
                                //backwardOutputOffset += backward_output_offset[d]*local_io_stride;
                                howmany_dims[d] = fftw_iodim{ (d==k ? 1 : int(realBufferShape[d])), local_io_stride, local_io_stride };
                                local_io_stride *= realBufferShape[d];
                            }
                            {
                                kind[0] = static_cast<fftw_r2r_kind>(ftr.kind);
                                dims[0] = { fftw_iodim{forward_transform_size[k], io_stride, io_stride} };
                                forward_plan = this->fftw_plan_guru_r2r(
                                        rank, dims, 
                                        howmany_rank, howmany_dims,
                                        //m_realBuffer.rdata()+forwardInputOffset,
                                        //m_realBuffer.rdata()+forwardOutputOffset,
                                        m_realBuffer.rdata()+forward_input_offset[k]*io_stride,
                                        m_realBuffer.rdata()+forward_output_offset[k]*io_stride,
                                        kind, fftw_flags);
                                //this->toStreamTransform(ss, ftr, rank, howmany_rank, dims, howmany_dims, 
                                        //forwardInputOffset, forwardOutputOffset);
                                this->toStreamTransform(ss, ftr, rank, howmany_rank, dims, howmany_dims, 
                                        forward_input_offset[k]*io_stride, forward_output_offset[k]*io_stride);
                                if(!forward_plan) {
                                    ss << "=> Forward R2R plan creation FAILED !" << std::endl;
                                    m_plannedTransformStr = ss.str();
                                    return false;
                                }
                            }
                            {
                                kind[0] = static_cast<fftw_r2r_kind>(btr.kind);
                                dims[0] = { fftw_iodim{backward_transform_size[k], io_stride, io_stride} };
                                backward_plan = this->fftw_plan_guru_r2r(
                                        rank, dims, 
                                        howmany_rank, howmany_dims,
                                        //m_realBuffer.rdata()+backwardInputOffset,
                                        //m_realBuffer.rdata()+backwardOutputOffset,
                                        m_realBuffer.rdata()+backward_input_offset[k]*io_stride,
                                        m_realBuffer.rdata()+backward_output_offset[k]*io_stride,
                                        kind, fftw_flags);
                                //this->toStreamTransform(ss, btr, rank, howmany_rank, dims, howmany_dims, 
                                        //backwardInputOffset, backwardOutputOffset);
                                this->toStreamTransform(ss, btr, rank, howmany_rank, dims, howmany_dims, 
                                        backward_input_offset[k]*io_stride, backward_output_offset[k]*io_stride);
                                if(!backward_plan) {
                                    ss << "=> Backward R2R plan creation FAILED !" << std::endl;
                                    m_plannedTransformStr = ss.str();
                                    return false;
                                }
                            }
                            m_forward_R2R_plans.push_back(forward_plan);
                            m_backward_R2R_plans.push_back(backward_plan);
                        }
                        io_stride *= realBufferShape[k];
                    }
                }
                   
                /* Copy input data into the buffer */
                if(inputFftDomainConfig.includePeriodicBoundaries() ^ includeOutputPeriodicBoundaries) {
                    NOT_IMPLEMENTED_YET;
                }
                else {
                    if(inplace) 
                        std::copy(input_rdata_buffer.begin(), input_rdata_buffer.end(), m_realBuffer.begin());
                    else
                        std::copy(input_rdata.begin()       , input_rdata.end()       , m_realBuffer.begin());
                }
               
                /* Create real buffer subview */
                ss << "** real buffer view **" << std::endl;
                {
                    ss << "\tview = index_gen";
                    std::array<boost::multi_array_types::index_range,Dim> ranges;
                    for (std::size_t d = 0; d < Dim; d++) {
                        const int offset = forward_output_offset[d];
                        ranges[d] = boost::multi_array_types::index_range(
                                offset,
                                offset+forward_transform_size[d]);
                                
                        ss << "[range(" << ranges[d].start() << "," << ranges[d].finish() << ")]";
                    }
                    ss << std::endl;
                    m_transformedRealBufferView = hysop::utils::buildIndices<Dim>(ranges);
                }
                
                ss << "===============" << std::endl;
                m_plannedTransformStr = ss.str();

                return true;
            }

        template <typename T, std::size_t Dim>
            void Planner<T,Dim>::executeForwardTransform() {
                for(auto& plan : m_forward_R2R_plans) 
                    this->fftw_execute(plan);
                for(auto& plan : m_R2C_plans) 
                    this->fftw_execute(plan);
            }

        template <typename T, std::size_t Dim>
            void Planner<T,Dim>::executeBackwardTransform() {
                for(auto& plan : m_C2R_plans) 
                    this->fftw_execute(plan);
                for(auto plan = m_backward_R2R_plans.rbegin(); plan!=m_backward_R2R_plans.rend(); ++plan) 
                    this->fftw_execute(*plan);
    
                if(m_mirrorOutputPeriodicBoundaries) {
                    const typename Shape<Dim>::type rshape = m_realBuffer.shape();
                    T *data = m_realBuffer.rdata();
                    for(const int axe : m_periodicAxes) {
                        const int N = rshape[axe];
                        const int num_elem = m_realBuffer.num_elements()/N;
                        std::array<int,Dim> ids{0};
                        int id = 0;
                        int offset=1;
                        {
                            for(int d=Dim-1; d>axe; d--) offset*= rshape[d];
                            offset *= (N-1);
                        }
                        for(int i=0; i<num_elem; i++) {
                            data[id+offset]=data[id];
                            for (int d=Dim-1; d>=0; d--) {
                                if(d==axe)
                                    continue;
                                ids[d]++;
                                if(ids[d]==int(rshape[d]))
                                    ids[d]=0;
                                else
                                    break;
                            }
                            id = ids[0];
                            for (std::size_t d=1; d < Dim; d++)
                                id = id*rshape[d] + ids[d];
                        }
                    }
                }
            }
                    
        template <typename T, std::size_t Dim>
        FftTransformType Planner<T,Dim>::transformType() const {
            return m_transformType;
        }

        template <typename T, std::size_t Dim>
            T Planner<T,Dim>::normalisationFactor() const {
                return m_normalisationFactor;
            }

        template <typename T, std::size_t Dim>
        const std::array<T,Dim> Planner<T,Dim>::signs() const {
            return m_signs;
        }

        template <typename T, std::size_t Dim>
            const std::array<std::vector<std::complex<T>>,Dim>& Planner<T,Dim>::waveNumbers() const {
                return m_waveNumbers;
            }
                    
        template <typename T, std::size_t Dim>
        const std::array<std::vector<std::complex<T>>,Dim>& Planner<T,Dim>::waveNumbersPows() const {
            return m_waveNumbersPows;
        }

        template <typename T, std::size_t Dim>
            hysop::multi_array_view<T,Dim> Planner<T,Dim>::transformedRealData() { 
                if(m_transformType==FFT_R2C)
                    throw std::runtime_error("Requesting planner real data but planned transform is real to complex !");
                else if(m_transformType==FFT_NONE)
                    throw std::runtime_error("Requesting planner real data but there was no successfull planned transforms !");
                return m_realBuffer[m_transformedRealBufferView]; 
            }

        template <typename T, std::size_t Dim>
            hysop::multi_array_ref<std::complex<T>,Dim> Planner<T,Dim>::transformedComplexData() {
                if(m_transformType==FFT_R2R)
                    throw std::runtime_error("Requesting planner complex data but planned transform is real to real !");
                else if(m_transformType==FFT_NONE)
                    throw std::runtime_error("Requesting planner real data but there was no successfull planned transforms !");
                return m_complexBuffer;
            }

        template <typename T, std::size_t Dim>
            fft::Transform Planner<T,Dim>::findTransform(const std::pair<Extension,Extension>& ed) const {
                if(ed.first == Extension::EVEN) {
                    if(ed.second == Extension::EVEN)
                        return fft::Transform(FFTW_REDFT00);
                    else
                        return fft::Transform(FFTW_REDFT01);
                }
                else if(ed.first == Extension::ODD) {
                    if(ed.second == Extension::EVEN)
                        return fft::Transform(FFTW_RODFT01);
                    else
                        return fft::Transform(FFTW_RODFT00);
                }
                else {
                    return fft::Transform(FFTW_FORWARD);
                }
            }
                    
        template <typename T, std::size_t Dim>
        template <typename Transfo>
        void Planner<T,Dim>::toStreamTransform(std::ostream& os, const Transfo& tr, 
                int rank, int howmany_rank, 
                const fftw_iodim* dims, const fftw_iodim* howmany_dims,
                const int input_data_offset, const int output_data_offset) {
            os << "\t  --" << tr << "--" << std::endl;
            os << "\t\tdims[" << rank << "] = {" << std::endl;
            for (int i = 0; i < rank-1; i++) 
                os << "\t\t  " << dims[i] << "," << std::endl;
            os << "\t\t  " << dims[rank-1] << std::endl;
            os << "\t\t};" << std::endl;
            os << "\t\thowmany[" << howmany_rank << "] = {" << std::endl;
            for (int i = 0; i < howmany_rank-1; i++) 
                os << "\t\t  " << howmany_dims[i] << "," << std::endl;
            os << "\t\t  " << howmany_dims[howmany_rank-1] << std::endl;
            os << "\t\t};" << std::endl;
            os << "\t\tinput  data offset: " << input_data_offset << std::endl;
            os << "\t\toutput data offset: " << output_data_offset << std::endl;
        }
                    
        /* Get planned transform description */
        template <typename T, std::size_t Dim>
            const std::string& Planner<T,Dim>::toString() const {
                return m_plannedTransformStr;
        }
        
        template <typename T, std::size_t Dim>
            std::ostream& operator<<(std::ostream& os, const Planner<T,Dim>& planner) {
                os << planner.toString();
                return os;
        }
                    
        template <typename T, std::size_t Dim>
        Planner<T,Dim>& Planner<T,Dim>::setFixedAxeWaveNumbers(std::complex<T> p_fixedAxeWaveNumber) {
            m_fixedAxeWaveNumbers = p_fixedAxeWaveNumber;
            return *this;
        }
        
        template <typename T, std::size_t Dim>
        Planner<T,Dim>& Planner<T,Dim>::setFixedAxeWaveNumberPows(std::complex<T> p_fixedAxeWaveNumberPow) {
            m_fixedAxeWaveNumberPows = p_fixedAxeWaveNumberPow;
            return *this;
        }

    } /* end of namespace fft */
} /* end of namespace hysop */


#endif /* end of include guard: HYSOP_PLANNER_H */
