
#include "testPolynoms.h"

using namespace hysop::maths;

template <typename T, std::size_t Dim>
void evalTest(std::size_t p_size, std::size_t p_samples);
    
TEST_F(PolynomialTest, EvalTest1D) {
    const std::size_t order=10;
    const std::size_t samples=4096;

    evalTest<bool,1>(order,samples); 

    evalTest<char,1>(order,samples); 
    evalTest<short,1>(order,samples); 
    evalTest<int,1>(order,samples); 
    evalTest<long int,1>(order,samples); 
    evalTest<long long int,1>(order,samples); 

    evalTest<unsigned char,1>(order,samples); 
    evalTest<unsigned short,1>(order,samples); 
    evalTest<unsigned int,1>(order,samples); 
    evalTest<unsigned long int,1>(order,samples); 
    evalTest<unsigned long long int,1>(order,samples); 

    evalTest<float,1>(order,samples); 
    evalTest<double,1>(order,samples); 
    evalTest<long double,1>(order,samples); 
#ifdef HAS_QUADMATHS
    evalTest<__float128,1>(order,samples); 
#endif

    evalTest<std::size_t,1>(order,samples);
    evalTest<std::ptrdiff_t,1>(order,samples);
}

TEST_F(PolynomialTest, EvalTest2D) {
    const std::size_t order=10;
    const std::size_t samples=32;

    evalTest<float,2>(order,samples); 
    evalTest<double,2>(order,samples); 
    evalTest<long double,2>(order,samples); 
#ifdef HAS_QUADMATHS
    evalTest<__float128,2>(order,samples); 
#endif
}

TEST_F(PolynomialTest, EvalTest3D) {
    const std::size_t order=10;
    const std::size_t samples=4;

    evalTest<float,3>(order,samples); 
    evalTest<double,3>(order,samples); 
    evalTest<long double,3>(order,samples); 
#ifdef HAS_QUADMATHS
    evalTest<__float128,3>(order,samples); 
#endif
}

TEST_F(PolynomialTest, EvalTest4D) {
    const std::size_t order=10;
    const std::size_t samples=2;

    evalTest<float,4>(order,samples); 
    evalTest<double,4>(order,samples); 
    evalTest<long double,4>(order,samples); 
#ifdef HAS_QUADMATHS
    evalTest<__float128,4>(order,samples); 
#endif
}

template <typename T, std::size_t Dim>
void evalTest(const std::size_t p_size, const std::size_t p_samples) {
    Polynomial<T,Dim> P;
    Index<Dim> polyIdx;
    { 
        typename Shape<Dim>::type polyShape;
        polyShape.fill(p_size);
        P.reshape(polyShape).applyToCoefficients([](T& ak, const Index<Dim>& idx){ 
                ak = T(idx())/T(idx.maxId());
        });
    }

    std::array<T,Dim> X;
    T dX;
    {
        const T a = T(0);
        const T b = T(1);
        dX = (b-a)/(p_samples-1);   
    }
    
    typename Shape<Dim>::type sampleShape;
    sampleShape.fill(p_samples);
    Index<Dim> sampleIdx(sampleShape);
    while(!sampleIdx.atMaxId()) {
        for (std::size_t d=0; d < Dim; d++)
            X[d] = sampleIdx[d]*dX;
        T lhs, rhs;
        lhs = P(X); 
        rhs = T(0);
        polyIdx.reset(P.shape());
        while(!polyIdx.atMaxId()) {
            T val = T(1);
            for (std::size_t d=0; d<Dim; d++)
                val *= std::pow(X[d],polyIdx[d]);
            rhs += T(polyIdx())/T(polyIdx.maxId())*val;
            ++polyIdx;
        }
        ASSERT_LE(std::abs(rhs-lhs),std::pow(10,Dim)*std::numeric_limits<T>::epsilon());
        ++sampleIdx;
    }
}

