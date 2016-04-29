
#include <cstdlib>

#include "domain/domain.h"
#include "solver/fftPoissonSolver.h"
#include "data/multi_array/multi_array.h"
#include "utils/constants.h"
#include "domain/boundary.h"

using namespace hysop;
using namespace hysop::domain;

static constexpr std::size_t nBoundaries = 4;
static constexpr std::size_t nBoundaryPairs = 7;
static constexpr domain::Boundary bds[nBoundaries] = 
{ domain::Boundary::NONE, domain::Boundary::HOMOGENEOUS_NEUMANN, domain::Boundary::HOMOGENEOUS_DIRICHLET, domain::Boundary::PERIODIC };
static constexpr std::pair<domain::Boundary,domain::Boundary> pbds[nBoundaryPairs] {
        std::make_pair(bds[3],bds[3]), //periodic-periodic
        std::make_pair(bds[3],bds[3]), //periodic-periodic
        std::make_pair(bds[2],bds[1]), //even-odd
        std::make_pair(bds[1],bds[2]), //odd-even
        std::make_pair(bds[2],bds[2]), //even-even
        std::make_pair(bds[1],bds[1]), //odd-odd
        std::make_pair(bds[0],bds[0]), //none-none
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

std::string bdsToStr(domain::Boundary bd) {
    switch(bd) {
        case(Boundary::NONE):                    return "None      ";
        case(Boundary::PERIODIC) :               return "Periodic  ";
        case(Boundary::HOMOGENEOUS_NEUMANN):     return "Hom_Neum. ";
        case(Boundary::HOMOGENEOUS_DIRICHLET):   return "Hom_Diric.";
        case(Boundary::NEUMANN):                 return "Neumann   ";
        case(Boundary::DIRICHLET):               return "Dirichlet ";
    }
    return "";
}

template <typename T, std::size_t Dim, bool verbose=false> 
void test(bool includePeriodicBds=false) {
    typename Shape<Dim>::type shape;
    typename Domain<T,Dim>::DomainSize domainSize;
    Domain<T,Dim> ref, inBuffer, outBuffer;

    Domain<T,Dim>& in  = inBuffer;
    Domain<T,Dim>& out = outBuffer;

    shape.fill(16);
    domainSize.fill(2*hysop::constants::pi);

    const T eps = std::numeric_limits<T>::epsilon();
    const std::size_t N = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());
    
    ref.resize(domainSize).reshape(shape);
    in  = ref;
    out = ref;
   
    typename Shape<Dim>::type testCases;
    testCases.fill(nBoundaryPairs);
    Index<Dim> testCaseId(testCases);
    std::array<T,3> meanDists{0};
    std::size_t testCaseCount = testCaseId.maxId()-1;

    if(verbose)
        std::cout << std::endl;
    
    while(testCaseId() != testCaseId.maxId()-1) { 

        /* generate transform configuration */
        std::size_t orderSum = 0;
        std::array<std::pair<domain::Boundary,domain::Boundary>, Dim> bdsConfig;
        T W2sum = T(0);
        for (std::size_t k=0; k<Dim; k++) {
            std::size_t id = testCaseId[k];
            bdsConfig[k] = pbds[id];
            if(bdsConfig[k].first != domain::Boundary::NONE) {
                W2sum += freqs[id]*freqs[id];
                orderSum+=2;
            }
        }
        domain::DomainConfiguration<Dim> domainConfig(bdsConfig, includePeriodicBds);
        
        T orderPow = std::pow(T(10),T(orderSum));
        if(std::is_same<T,long double>::value) /* just in case long doubles are not hardware supported... */
            orderPow *= 1e3;
        const auto criteria = std::make_tuple(orderPow*eps*N,orderPow*eps*sqrt(N),2*orderPow*eps);

        const auto phi = [&](const typename Domain<T,Dim>::SpaceVariable &x) { 
            T val = func<T>(testCaseId[0])(x[0]);
            for (std::size_t d=1; d < Dim; d++)
                val *= func<T>(testCaseId[d])(x[d]);
            return val;
        };
        const auto f = [&](const typename Domain<T,Dim>::SpaceVariable &x) { 
            return -W2sum*phi(x);
        };

        {
            ref.resetDomainConfiguration(domainConfig);
            in  = ref;
            out = ref;

            in.apply(f);
            ref.apply(phi);
            out.data().apply([](T& v){ v=T(0);});
        }

        solver::FftPoissonSolver<T,Dim> solver(domainSize, domainConfig, FFTW_MEASURE, includePeriodicBds, includePeriodicBds);
        solver.apply(in.data(), out.data());

        std::stringstream ss;
        ss << "[";
        for (std::size_t k=0; k<Dim-1; k++) 
            ss << bdsToStr(bdsConfig[k].first) << "/" << bdsToStr(bdsConfig[k].second) << ",";
        ss << bdsToStr(bdsConfig[Dim-1].first) << "/" << bdsToStr(bdsConfig[Dim-1].second);
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
            std::cout << "\t\tTest Failed... Criteria was " << criteria << "." << std::endl;
            exit(EXIT_FAILURE);
        }

        meanDists[0] += std::get<0>(dist);
        meanDists[1] += std::get<1>(dist);
        meanDists[2] += std::get<2>(dist);

        ++testCaseId;
    }

        for (std::size_t k = 0; k < 3; k++)
            meanDists[k] /= T(testCaseCount);
        std::cout << "\t=> mean distances over " << std::scientific << std::setprecision(1) << std::setw(4)
            << testCaseCount << " testcases: " << meanDists;
        for (std::size_t k = 0; k < 3; k++)
            meanDists[k] = std::round(meanDists[k]/eps);
        std::cout << " ~= " <<  std::fixed << std::setprecision(0) << meanDists << " eps" << std::endl; 
}
    
int main(int argc, const char *argv[]) {

#ifdef FFTW_HAS_FFTW3F
    std::cout << "== TEST 1D - float       ==";
    test<float,1,true>();
    std::cout << "== TEST 2D - float       ==";
    test<float,2,true>();
    std::cout << "== TEST 3D - float       ==";
    test<float,3,false>();
    std::cout << std::endl;
#endif
    
#ifdef FFTW_HAS_FFTW3D
    std::cout << "== TEST 1D - double      ==";
    test<double,1,false>();
    std::cout << "== TEST 2D - double      ==";
    test<double,2,false>();
    std::cout << "== TEST 3D - double      ==";
    test<double,3,false>();
    std::cout << std::endl;
#endif
    
#ifdef FFTW_HAS_FFTW3L
    std::cout << "== TEST 1D - long double ==";
    test<long double,1,false>();
    std::cout << "== TEST 2D - long double ==";
    test<long double,2,false>();
    std::cout << "== TEST 3D - long double ==";
    test<long double,3,false>();
    std::cout << std::endl;
#endif

#ifdef FFTW_HAS_FFTW3Q
    std::cout << "== TEST 1D - __float128 ==";
    test<__float128,1,false>();
    std::cout << "== TEST 2D - __float128 ==";
    test<__float128,2,false>();
    std::cout << "== TEST 3D - __float128 ==";
    test<__float128,3,false>();
    std::cout << std::endl;
#endif
    return EXIT_SUCCESS;
}
