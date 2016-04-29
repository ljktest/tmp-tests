/** \file main.cxx Main program, only for tests.

 */
#include<iostream>
//#include"ppm_wrapper.hpp"
#include <string>
#include <cstring>
#include <Grid.hpp>
#include<HySoPDef.hpp>
#include<Domain.hpp>
#ifdef USE_MPI
#include<mpi.h>
#endif
#include "WrapC.hpp"

using namespace std ;
using HySoP::Def::real_t;

extern "C" void createTopoG(int*, int*, int*, double*, double*, int*, double*);
extern "C" void plouhmans();
extern "C" void NavierStokes3D();
extern "C" void NavierStokes2D();
extern "C" void testPoisson2D();
extern "C" void testMain();
//!extern "C" void Scalar3D();

//   // =====  Physical domain definition =====
//   // Problem dimension
//   int pbDim = 3;
//   // dimensions of the domain 
//   HySoP::Def::vector3D dimsD = { { 1.0, 3.1, 4.3} };
//   // "Lowest" point
//   HySoP::Def::vector3D startPoint = { { 0., 1., 2.} };
//   // The domain
//   HySoP::Model::Domain<3> domain(dimsD, startPoint);
  
//   // =====  Grid definition =====
//   // Number of points in each dir ...
//   boost::array<size_t, 3>  nbSteps = { { 3, 4, 5} };
//   HySoP::Discr::Grid<3> grid(domain, nbSteps);
	
//   std::cout << grid << std::endl;

//   std::string msg = "Main Programm in c++";
  
//   MPI::Init();
//   assert(MPI::Is_initialized());
  
//   MPI::Intracomm Comm = MPI::COMM_WORLD;
  
//   // ==== Initialize  ppm ====
//   PPM::wrapper::init(pbDim, 8, -15, Comm, 2, &info, 99, 98,97);
//   PPM::wrapper::substart(msg, &t0, &info);
  
//   HySoP::Def::vector3D minPhys = {{0.0,0.0,0.0}}, maxPhys ={{ 3.14, 3.14, 6.28}};
//   HySoP::Def::vector3D minSub, maxSub;
//   boost::array<int,6> bc = {{ 1, 1 ,1 ,1 ,1,1}}; 
//   boost::array<int,3>  nx ={{ 65, 65, 129}};
//   HySoP::Def::real_t ghostsize = 1.0;
//   int topoid = -1, decomp, dim = 3;
//   int meshid = -1;
//   // Time step
//   real_t dt = 0.96;
//   real_t finalT = 1000;
//   real_t nu = 0.001;
  
//   HySoP::Def::real_t * costPerProc;
//   createTopoG(&dim, &topoid, &decomp,&minPhys[0], &maxPhys[0], &bc[0], &ghostsize);
  
  
//   cout << topoid << endl;

//   PPM::wrapper::finalize(info);
  
  
//   // ==== Finalize everything ====
//   MPI::Finalize();
//   PPM::wrapper::substop(msg, &t0, &info);
  
// }

int main(int argc, char* argv[])
{
#ifdef USE_MPI
  MPI::Init();
  assert(MPI::Is_initialized());
#endif
//  NavierStokes3D();
  std::cout << "C++ calls Fortran ..." << std::endl;
  testPoisson2D();
  //NavierStokes2D();
  //Scalar3D();
  //testMain();
#ifdef USE_MPI
  MPI::Finalize();
#endif
}

