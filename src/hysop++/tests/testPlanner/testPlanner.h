
#include "gtest/gtest.h"
#include "fft/planner.h"
#include "fft/extension.h"

using T = double;
constexpr std::size_t Dim = 1;
    
using namespace hysop;

class PlannerTest : public ::testing::Test {
    protected:
        PlannerTest() {}
        void SetUp()  {}
        void TearDown() {}
        virtual ~PlannerTest() {}

    public:
        fft::Planner<T,Dim> planner;
};
