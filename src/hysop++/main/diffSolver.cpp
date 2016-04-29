
#include <cstdlib>

#include "domain/domain.h"
#include "solver/fftDiffSolver.h"
#include "data/multi_array/multi_array.h"
#include "utils/constants.h"
#include "fft/extension.h"
#include "maths/quad_maths.h"

using namespace hysop;
using namespace hysop::domain;

static constexpr std::size_t nExtensions = 4 ;
static constexpr std::size_t nExtensionsPair = 7 ;
static constexpr fft::Extension ext[nExtensions] = 
{ fft::Extension::NONE, fft::Extension::ODD, fft::Extension::EVEN, fft::Extension::PERIODIC };
static constexpr std::pair<fft::Extension,fft::Extension> pext[nExtensionsPair] {
        std::make_pair(ext[3],ext[3]), //periodic-periodic
        std::make_pair(ext[3],ext[3]), //periodic-periodic
        std::make_pair(ext[2],ext[1]), //even-odd
        std::make_pair(ext[1],ext[2]), //odd-even
        std::make_pair(ext[2],ext[2]), //even-even
        std::make_pair(ext[1],ext[1]), //odd-odd
        std::make_pair(ext[0],ext[0]), //none-none
};

#ifdef HAS_QUADMATHS
    static constexpr __float128  freqs[6] = { 1.0Q, 1.0Q, 0.75Q, 0.75Q, 0.50Q, 0.50Q };
#else
    static constexpr long double freqs[6] = { 1.0L, 1.0L, 0.75L, 0.75L, 0.50L, 0.50L };
#endif

template <typename T>
std::function<T(T)> func(std::size_t k) {
    switch(k) {
        case 0: return [=](T x) {return std::cos(T(freqs[0])*x);};
        case 1: return [=](T x) {return std::sin(T(freqs[1])*x);};
        case 2: return [=](T x) {return std::cos(T(freqs[2])*x);};
        case 3: return [=](T x) {return std::sin(T(freqs[3])*x);};
        case 4: return [=](T x) {return std::cos(T(freqs[4])*x);};
        case 5: return [=](T x) {return std::sin(T(freqs[5])*x);};
        default: return[=](T x) { return T(1); };
    }
}

template <typename T>
std::function<T(T)> derivative(std::size_t k, int order) {
    bool even = (k%2==0);
    std::size_t p, offset;
    T sign, coeff;
    if(k>5) {
        if(order != 0)
            throw std::runtime_error("Non zero order !");
        return func<T>(k);
    }
    else if(even) { /* cos func */
        p      = (order%2==0 ? k : k+1);
        sign  = std::pow(T(-1),(order+1)/2);
        coeff = std::pow(freqs[k], order);
    }
    else { /* sin func */
        p     = (order%2==0 ? k : k-1); 
        sign  = std::pow(T(-1),order/2);
        coeff = std::pow(freqs[k], order);
    }
    return [=](T x) { return sign*coeff*(func<T>(p)(x)); };
}
    
template <typename T, std::size_t Dim, bool verbose=false> 
void test(std::size_t p_maxOrder, bool includePeriodicBds=false) {
    typename Shape<Dim>::type shape;
    typename Domain<T,Dim>::DomainSize domainSize;
    Domain<T,Dim> ref, inBuffer, outBuffer;

    Domain<T,Dim>& in  = inBuffer;
    Domain<T,Dim>& out = outBuffer;

    std::array<int,Dim> order;

    shape.fill(8);
    domainSize.fill(2*hysop::constants::pi);

    T eps = std::numeric_limits<T>::epsilon();
    const std::size_t N = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());
    
    ref.resize(domainSize).reshape(shape);
    in  = ref;
    out = ref;
   
    typename Shape<Dim>::type maxOrder, testCases;
    maxOrder.fill(p_maxOrder+1);
    testCases.fill(nExtensionsPair);
    Index<Dim> orderId(maxOrder);
    Index<Dim> testCaseId;
    std::size_t testCaseCount;
    while(!(++orderId).atMaxId()) {
        std::cout << "  ::Order::" << orderId.ids() << (verbose ? "\n" : "");
    
        std::array<T,3> meanDists;
        meanDists.fill(0);
        testCaseId.reset(testCases);
        testCaseCount = testCaseId.maxId();
        while(!testCaseId.atMaxId()) { 
            std::copy(orderId.ids().begin(),orderId.ids().end(), order.begin());

            /* generate transform configuration */
            std::array<std::pair<fft::Extension,fft::Extension>, Dim> extConfig;
            for (std::size_t k=0; k<Dim; k++) {
                std::size_t id = testCaseId[k];
                extConfig[k] = pext[id];
                if(pext[id].first==fft::Extension::NONE)
                    order[k] = 0;
            }
            fft::FftDomainConfiguration<Dim> domainConfig(extConfig, includePeriodicBds);
            
            const std::size_t orderSum = std::accumulate(order.begin(), order.end(), 0);
            if(orderSum == 0) {
                testCaseCount--;
                ++testCaseId;
                continue;
            }
            T orderPow = std::pow(T(10),T(orderSum));
            if(std::is_same<T,long double>::value) /* just in case long doubles are not hardware supported... */
                orderPow *= 1e3;
            const auto criteria = std::make_tuple(orderPow*eps*N,orderPow*eps*sqrt(N),2*orderPow*eps);

            const auto f = [&](const typename Domain<T,Dim>::SpaceVariable &x) { 
                T val = func<T>(testCaseId[0])(x[0]);
                for (std::size_t d=1; d < Dim; d++)
                    val *= func<T>(testCaseId[d])(x[d]);
                return val;
            };
            const auto d = [&](const typename Domain<T,Dim>::SpaceVariable &x) { 
                T val = derivative<T>(testCaseId[0],order[0])(x[0]);
                for (std::size_t d=1; d < Dim; d++)
                    val *= derivative<T>(testCaseId[d],order[d])(x[d]);
                return val;
            };
            {
                ref.resetDomainConfiguration(domainConfig.boundariesConfiguration());
                in  = ref;
                out = ref;

                in.apply(f);
                ref.apply(d);
                out.data().apply([](T& v){ v=T(0);});
            }

            solver::FftDiffSolver<T,Dim> solver(domainSize, domainConfig, FFTW_MEASURE, includePeriodicBds, includePeriodicBds);
            solver.apply(in.data(), out.data(), order);

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
                //in.print("IN");
                //ref.print("REF");
                //out.print("OUT");
                std::cout << criteria << std::endl;
                exit(EXIT_FAILURE);
            }

            meanDists[0] += std::get<0>(dist);
            meanDists[1] += std::get<1>(dist);
            meanDists[2] += std::get<2>(dist);

            ++testCaseId;
        }
        for (std::size_t k = 0; k < 3; k++)
            meanDists[k] /= T(testCaseCount);
        std::cout << "=> mean distances over " << std::scientific << std::setprecision(1) << std::setw(4)
            << testCaseCount << " testcases: " << meanDists;
        for (std::size_t k = 0; k < 3; k++)
            meanDists[k] = std::round(meanDists[k]/eps);
        std::cout << " ~= " <<  std::fixed << std::setprecision(0) << meanDists << " eps" << std::endl; 
    }
}
    
int main(int argc, const char *argv[]) {

#ifdef FFTW_HAS_FFTW3F
    std::cout << "== TEST 1D - float       ==" << std::endl;
    test<float,1,false>(5);
    std::cout << "== TEST 2D - float       ==" << std::endl;
    test<float,2,false>(3);
    std::cout << "== TEST 3D - float       ==" << std::endl;
    test<float,3,false>(1);
    std::cout << std::endl;
#endif
    
#ifdef FFTW_HAS_FFTW3D
    std::cout << "== TEST 1D - double       ==" << std::endl;
    test<double,1,false>(5);
    std::cout << "== TEST 2D - double       ==" << std::endl;
    test<double,2,false>(3);
    std::cout << "== TEST 3D - double       ==" << std::endl;
    test<double,3,false>(1);
    std::cout << std::endl;
#endif
    
#ifdef FFTW_HAS_FFTW3L
    std::cout << "== TEST 1D - long double       ==" << std::endl;
    test<long double,1,false>(5);
    std::cout << "== TEST 2D - long double       ==" << std::endl;
    test<long double,2,false>(3);
    std::cout << "== TEST 3D - long double       ==" << std::endl;
    test<long double,3,false>(1);
    std::cout << std::endl;
#endif

#ifdef FFTW_HAS_FFTW3Q
    std::cout << "== TEST 1D - __float128       ==" << std::endl;
    test<__float128,1,false>(5);
    std::cout << "== TEST 2D - __float128       ==" << std::endl;
    test<__float128,2,false>(3);
    std::cout << "== TEST 3D - __float128       ==" << std::endl;
    test<__float128,3,false>(1);
    std::cout << std::endl;
#endif

    return EXIT_SUCCESS;
}
