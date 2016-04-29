
#include "maths/quad_maths.h"
#include "testPlanner.h"

#include "data/multi_array/multi_array.h"
#include "domain/domain.h"
#include "utils/constants.h"
#include "fft/planner.h"
#include "fft/extension.h"


using namespace hysop;
using namespace hysop::domain;

static constexpr std::size_t nExtensions = 4 ;
static constexpr std::size_t nExtensionsPair = 6 ;
static constexpr fft::Extension ext[nExtensions] = 
{ fft::Extension::NONE, fft::Extension::ODD, fft::Extension::EVEN, fft::Extension::PERIODIC };
static constexpr std::pair<fft::Extension,fft::Extension> pext[nExtensionsPair] {
    std::make_pair(ext[0],ext[0]), //none-none
        std::make_pair(ext[1],ext[1]), //odd-odd
        std::make_pair(ext[1],ext[2]), //odd-even
        std::make_pair(ext[1],ext[2]), //even-odd
        std::make_pair(ext[2],ext[2]), //even-even
        std::make_pair(ext[3],ext[3]), //periodic-periodic
};

template <typename T, std::size_t Dim, bool verbose=false> 
void test(bool inplace, bool includePeriodicBds);

#ifdef FFTW_HAS_FFTW3F
TEST_F(PlannerTest, InplaceFloatTransforms) {
    std::cout << std::endl; 
    std::cout << "== TEST 1D - float       ==\t";
    test<float,1>(false,true);
    std::cout << "== TEST 2D - float       ==\t";
    test<float,2>(false,true);
    std::cout << "== TEST 3D - float       ==\t";
    test<float,3>(false,true);
}
#endif

#ifdef FFTW_HAS_FFTW3D
TEST_F(PlannerTest, InplaceDoubleTransforms) {
    std::cout << std::endl; 
    std::cout << "== TEST 1D - double      ==\t";
    test<double,1>(true,true);
    std::cout << "== TEST 2D - double      ==\t";
    test<double,2>(true,true);
    std::cout << "== TEST 3D - double      ==\t";
    test<double,3>(true,true);
}
#endif
    
#ifdef FFTW_HAS_FFTW3L
TEST_F(PlannerTest, InplaceLongDoubleTransforms) {
    std::cout << std::endl; 
    std::cout << "== TEST 1D - long double ==\t";
    test<long double,1>(false,false);
    std::cout << "== TEST 2D - long double ==\t";
    test<long double,2>(false,false);
    std::cout << "== TEST 3D - long double ==\t";
    test<long double,3>(false,false);
}
#endif

#ifdef FFTW_HAS_FFTW3Q
TEST_F(PlannerTest, InplaceQuadDoubleTransforms) {
    std::cout << std::endl; 
    std::cout << "== TEST 1D - __float128 ==\t";
    test<__float128,1>(false,false);
    std::cout << "== TEST 2D - __float128 ==\t";
    test<__float128,2>(false,false);
    std::cout << "== TEST 3D - __float128 ==\t";
    test<__float128,3>(false,false);
}
#endif

template <typename T, std::size_t Dim, bool verbose> 
void test(bool inplace, bool includePeriodicBds) {
    typename Shape<Dim>::type shape;
    typename Domain<T,Dim>::DomainSize domainSize;
    Domain<T,Dim> ref, inBuffer, outBuffer;

    Domain<T,Dim>& in  = inBuffer;
    Domain<T,Dim>& out = (inplace ? inBuffer : outBuffer);

    fft::Planner<T,Dim> planner;
    std::array<int,Dim> order;

    const std::size_t nPoints = 16;
    shape.fill(nPoints);
    domainSize.fill(1.0);
    order.fill(2);

    const T eps = std::numeric_limits<T>::epsilon();
    const std::size_t N = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());
    const auto criteria = std::make_tuple(50*eps*N,sqrt(50)*eps*N,700*eps);
    
    ref.resize(domainSize).reshape(shape);
    in  = ref;
    out = ref;
   
    typename Shape<Dim>::type testCases;
    testCases.fill(nExtensionsPair);
    Index<Dim> testCaseId(testCases);
    std::array<T,3> meanDists{0};
    while(!testCaseId.atMaxId()) { 
        /* generate transform configuration */
        std::array<std::pair<fft::Extension,fft::Extension>, Dim> extConfig;
        for (std::size_t k=0; k<Dim; k++) {
            std::size_t id = testCaseId[k];
            extConfig[k] = pext[id];
        }
        fft::FftDomainConfiguration<Dim> domainConfig(extConfig, includePeriodicBds);
    
        const auto f = [&](T &val, const hysop::Index<Dim>& idx) { 
            val = static_cast<T>(rand())/static_cast<T>(RAND_MAX);
            for (std::size_t d=0; d<Dim; d++) {
                if(idx[d]==0) {
                    if(extConfig[d].first == fft::Extension::ODD) {
                        val=T(0); 
                        return;
                    }
                    else if(extConfig[d].first == fft::Extension::PERIODIC)
                        val=T(0.42);
                }
                else if(std::size_t(idx[d]) == idx.dim()[d]-1) {
                    if(extConfig[d].second == fft::Extension::ODD) {
                        val=T(0);
                        return;
                    }
                    else if(extConfig[d].second == fft::Extension::PERIODIC && includePeriodicBds)
                        val=T(0.42);
                }
            }
        };
        
        if(includePeriodicBds)
            ref.resetDomainConfiguration(domainConfig.boundariesConfiguration()); 
        
        /* fill reference and copy into input buffer */
        ref.data().apply(f);
        in = ref;
       
        /* plan transforms and check if planning succeeded */
        bool status = planner.plan(in.data(), out.data(), domainConfig, order, domainSize, FFTW_MEASURE, 
                    includePeriodicBds, includePeriodicBds);
        assert(status || testCaseId()==0);
    
        /* execute forward and backward inplace transforms */
        planner.executeForwardTransform();
        {
            if(planner.transformType() == fft::FftTransformType::FFT_R2C)
                planner.transformedComplexData().apply([&](std::complex<T>& val) { val /= planner.normalisationFactor(); }); 
            else if(planner.transformType() == fft::FftTransformType::FFT_R2R)
                planner.transformedRealData().apply([&](T& val) { val /= planner.normalisationFactor(); }); 
        }
        planner.executeBackwardTransform();
        
        std::stringstream ss;
        ss << "[";
        for (std::size_t k=0; k<Dim-1; k++) 
            ss << extConfig[k].first << "/" << extConfig[k].second << ",";
        ss << extConfig[Dim-1].first << "/" << extConfig[Dim-1].second;
        ss << "]";

        const auto dist = out.distance(ref);
        const bool pass =      (std::get<0>(dist) < std::get<0>(criteria)) 
                            && (std::get<1>(dist) < std::get<1>(criteria))
                            && (std::get<2>(dist) < std::get<2>(criteria));

        if((pass && verbose) || !pass) {
            std::cout << (pass ? GREEN : RED);
            std::cout << "\t" << std::setw(Dim*15) << ss.str() << " => " << (pass ? "OK" : "KO") 
                << "  " << RESET << std::scientific << std::setprecision(2) << dist << std::endl;
        }
        if(!pass) {
            if(!inplace)
                in.print("IN");
            ref.print("REF");
            out.print("OUT");
            std::cout << planner << std::endl;
            exit(EXIT_FAILURE);
        }

        meanDists[0] += std::get<0>(dist);
        meanDists[1] += std::get<1>(dist);
        meanDists[2] += std::get<2>(dist);

        ++testCaseId;
    }
    for (std::size_t k = 0; k < 3; k++)
        meanDists[k] /= T(testCaseId.maxId());
    std::cout << "Mean distances over " << std::scientific << std::setprecision(1) << std::setw(4)
         << testCaseId.maxId() << " testcases: " << meanDists;
    for (std::size_t k = 0; k < 3; k++)
        meanDists[k] = std::round(meanDists[k]/eps);
    std::cout << " ~= " << std::fixed << std::setprecision(0) << meanDists << " eps" << std::endl; 
}
