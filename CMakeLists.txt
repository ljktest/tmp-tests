#===============================================================================
# cmake utility to compile and install hysop python modules and libraries
#
# It includes :
#  - high level python interface to HySoP routines
#  - HySoP fortran library (with fftw solver and scales interface)
#  - HySoP C++     library (poisson solver)
#
#  HySoP depends (optionally) on :
#   - scales (WITH_SCALES=ON, default)
#
# LJK-CNRS, F. Pérignon, june 2012
#
#===============================================================================
# ============= Global cmake Settings =============
# Set minimum version for cmake
cmake_minimum_required(VERSION 3.0.2)

if(POLICY CMP0053)
  cmake_policy(SET CMP0053 NEW)
endif()

# Set cmake modules directory (i.e. the one which contains all user-defined FindXXX.cmake files among other things)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
# Force out-of-source build
include(OutOfSourceBuild)
# Some usefull macros
include(MyTools)

# User defined options
option(VERBOSE_MODE "enable verbose mode for cmake exec. Default = on" ON)
option(DOUBLEPREC "set precision for real numbers to double precision when this mode is enable. Default = on." ON)
option(USE_MPI "compile and link HySoP with mpi when this mode is enable. Default = on." ON)
option(WITH_TESTS "Enable testing. Default = off" ON)
option(BUILD_SHARED_LIBS "Enable dynamic library build, default = ON." ON)
option(WITH_LIB_CXX "Generate libhysop from fortran files in src, wrapped into hysop.cpp2hysop module. Default = ON." ON)
option(WITH_SCALES "compile/create scales lib and link it with HySoP. Default = ON." ON)
option(WITH_FFTW "Link with fftw library (required for some HySoP solvers), default = ON" ON)
option(WITH_EXTRAS "Link with some extra fortran libraries (like arnoldi solver), default = OFF" OFF)
option(WITH_GPU "Use of GPU (required for some HySoP solvers), default = ON" ON)
option(WITH_MAIN_FORTRAN "Create an executable (test purpose) from fortran sources in src/main, linked with libhysop, default = ON" ON)
option(WITH_MAIN_CXX "Create an executable (test purpose) from cxx sources in src/hysop++/main, linked with libhysop, default = ON" ON)
option(DEBUG "Enable debug mode for HySoP (0:disabled, 1:verbose, 2:trace, 3:verbose+trace). Default = 0" 0)
option(FULL_TEST "Enable all test options (pep8, mpi ...) - Default = OFF" OFF)
option(PROFILE "Enable profiling mode for HySoP. 0:disabled, 1: enabled. Default = 0" 0)
option(OPTIM "To allow python -OO run, some packages must be deactivated. Set this option to 'ON' to do so. Default = OFF" OFF)
option(WITH_MPI_TESTS "Enable mpi tests. Default = ON if USE_MPI is ON." ON)
option(WITH_GOOGLE_TESTS "Enable google tests (c++). Default = OFF." OFF)
option(FORTRAN_LAYOUT "Choose default data layout ('fortran', column-major or 'C' order, row-major) for arrays. Default = column-major." ON)
option(WITH_DOCUMENTATION "Build Documentation. Default = OFF" ON)

# Set python install mode:
# - user --> behave as 'python setup.py install --user'
# - standard --> install in python site-package (ie behave as python setup.py install)
# - prefix --> install in python CMAKE_INSTALL_PREFIX (ie behave as python setup.py install --prefix=CMAKE_INSTALL_PREFIX)
set(HYSOP_INSTALL "user" CACHE STRING "Install mode for hysop python package")

# Install lib directory 32, 64 etc. on Fedora, Debian
# http://public.kitware.com/Bug/view.php?id=11964
# See also http://www.cmake.org/cmake/help/v3.0/module/GNUInstallDirs.html?highlight=gnuinstalldirs
include(GNUInstallDirs)
# Set prefix path for libraries installation
# --> means that any library target will be installed
# in CMAKE_INSTALL_PREFIX/_install_lib
if(${PROJECT_NAME}_INSTALL_LIB_DIR)
  set(_install_lib ${${PROJECT_NAME}_INSTALL_LIB_DIR})
else()
  ASSERT(CMAKE_INSTALL_LIBDIR)
  set(_install_lib ${CMAKE_INSTALL_LIBDIR})
  set(${PROJECT_NAME}_INSTALL_LIB_DIR ${_install_lib})
endif()

if(NOT USE_MPI)
  message(FATAL_ERROR "No-mpi version of hysop is broken, please enable mpi.")
endif()

if(WITH_FFTW OR WITH_SCALES OR WITH_EXTRAS)
  set(USE_FORTRAN "ON")
endif()

if(NOT WITH_LIB_CXX)
    set(WITH_MAIN_CXX "OFF")
    set(WITH_GOOGLE_TESTS "OFF")
endif()

# true if hysop used Fortran and/or c++ sources
# We can not run scales or fftw without mpi ...
if(USE_FORTRAN OR WITH_LIB_CXX)
  set(WITH_COMPILED_LIB "ON")
  #set(WITH_FFTW "ON")
else()
  set(WITH_COMPILED_LIB "OFF")
  #set(WITH_FFTW "OFF")
endif()

# Force a default build type if not provided by user
# CMAKE_BUILD_TYPE = empty, Debug, Release, RelWithDebInfo or MinSizeRel.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Choose the type of build, options are: None Debug Release."
    FORCE)
endif()

# cmake project name
set(PROJECT_NAME hysop)
# --- Name for the package ---
# This name will be used as the Python Package name
set(PACKAGE_NAME "hysop")
# --- The name (without extension) of the lib to be created ---
set(PROJECT_LIBRARY_NAME ${PROJECT_NAME})
# ============= The project =============
# Set project name and project languages
# => this automatically defines:
#   - ${PROJECT_NAME}_BINARY_DIR : where you have run cmake, i.e. the place for compilation
#   - ${PROJECT_NAME}_SOURCE_DIR : where sources (.f and .h and this CMakeLists.txt) are located
# Note that because of OutOfSourceBuild, binary_dir and source_dir must be different.

set(LANGLIST)
if(WITH_LIB_CXX)
  set(LANGLIST ${LANGLIST} C CXX)
endif()
if(USE_FORTRAN)
  set(LANGLIST ${LANGLIST} Fortran)
endif()
include(HysopVersion)
project(${PROJECT_NAME} ${LANGLIST})

set(HYSOP_LIBRARY_NAME hysop)
set(HYSOP_INCLUDE_DIRECTORIES
  CACHE INTERNAL "Include directories for external dependencies.")
set(HYSOP_LINK_LIBRARIES CACHE INTERNAL "List of external libraries.")

# ============= Python and its packages =============
# - Global setup (interp and lib) -
find_package(PythonFull REQUIRED)
include(FindPythonModule)
# - python packages -
find_python_module(numpy REQUIRED)
find_python_module(scipy)
find_python_module(matplotlib)
if(NOT matplotlib_FOUND)
  find_python_module(Gnuplot)
endif()
find_python_module(scitools)
find_python_module(h5py REQUIRED)
find_python_module(sympy)
# --- OpenCL ---
find_python_module(pyopencl REQUIRED)
# --- MPI ---
if(USE_MPI)
  find_package(MPI REQUIRED)
  find_python_module(mpi4py REQUIRED)
endif()
# --- Wheel, required for a proper build/install process ---
find_python_module(wheel REQUIRED)
if(WITH_LIB_CXX)
  find_package(SWIG 3.0.2 REQUIRED)
  # WARNING FP : for cmake < 3.0 UseSWIG.cmake
  # does not work properly (bug for swig outdir)
  if(CMAKE_VERSION VERSION_LESS 3.0.0)
    set(SWIG_USE_FILE ${CMAKE_SOURCE_DIR}/cmake/UseSWIG.cmake)
  endif()
  include(${SWIG_USE_FILE})
endif()

# Find python build dir name --> needed for tests and doc
if(WITH_COMPILED_LIB)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.util as ut ; import distutils.sysconfig as sy; print 'lib.'+ut.get_platform()+'-'+sy.get_python_version()"
    OUTPUT_VARIABLE ${PROJECT_NAME}_PYTHON_BUILD_DIR)
  string(STRIP ${${PROJECT_NAME}_PYTHON_BUILD_DIR} ${PROJECT_NAME}_PYTHON_BUILD_DIR)
  set(HYSOP_BUILD_PYTHONPATH ${CMAKE_BINARY_DIR}/build/${${PROJECT_NAME}_PYTHON_BUILD_DIR} CACHE INTERNAL "Python package build path")
else()
  set(HYSOP_BUILD_PYTHONPATH ${CMAKE_BINARY_DIR}/build/lib CACHE INTERNAL "Python package build path")
endif()

# ============= Other dependencies =============

# --- FFTW ---
if(WITH_FFTW)
    set(FIND_FFTW_VERBOSE OFF)
    set(FIND_FFTW_DEBUG OFF)
    compile_with(FFTW 
      REQUIRED COMPONENTS Fftw3d Fftw3f Fftw3d-mpi Fftw3f-mpi
      OPTIONAL_COMPONENTS Fftw3l)# Fftw3q)
    set(dirlist)

    foreach(_file ${FFTW_LIBRARIES})
      get_filename_component(_name ${_file} DIRECTORY)
      list(FIND dirlist ${_name} isin)
      if(isin EQUAL -1)
	list(APPEND dirlist ${_name})
      endif()
    endforeach()
    set(FFTWLIB ${dirlist} CACHE PATH "fftw libraries dir")
    add_definitions(${FFTW_DEFINES})
endif()

if(WITH_EXTRAS)
  # Arnoldi solver needs zgeev, which means lapack
  compile_with(LAPACK)
endif()

# ========= Check which opencl devices are available on the system =========
if(WITH_GPU)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/opencl_explore.py "EXPLORE")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/opencl_explore.py
    OUTPUT_VARIABLE OPENCL_DEFAULT_OPENCL_ID)
else()
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/opencl_explore.py "EXPLORE" CPU)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/opencl_explore.py CPU
    OUTPUT_VARIABLE OPENCL_DEFAULT_OPENCL_ID)
endif()

string(REPLACE " " ";" MY_LIST ${OPENCL_DEFAULT_OPENCL_ID})
list(GET MY_LIST 0 OPENCL_DEFAULT_OPENCL_PLATFORM_ID)
list(GET MY_LIST 1 OPENCL_DEFAULT_OPENCL_DEVICE_ID)
display(OPENCL_DEFAULT_OPENCL_PLATFORM_ID)
display(OPENCL_DEFAULT_OPENCL_DEVICE_ID)
# =========== RPATH stuff ===========
# Doc :
#  - https://cmake.org/Wiki/CMake_RPATH_handling
#  - https://cmake.org/cmake/help/git-master/manual/cmake-packages.7.html#creating-packages
#  - http://matthew-brett.github.io/docosx/mac_runtime_link.html

# Force MACOSX_RPATH, whatever the cmake version is.
# RPATH management on macosx with cmake is quite a mess, so please
# think hard before changing the following lines ...
set(CMAKE_MACOSX_RPATH 1)

# don't skip the full RPATH for the build tree
# except if explicitely asked by user
if(FORCE_SKIP_RPATH)
  set(CMAKE_SKIP_BUILD_RPATH TRUE)
else()
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
endif()

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${HYSOP_PYTHON_INSTALL_DIR}/${PACKAGE_NAME}")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ====== Create (and setup) build target ======
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/build)

set(PYTHON_SETUP "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
if(USE_FORTRAN)
  # add_custom_target(python-build ALL 
  #     COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_SETUP} build config_fc --f90exec=${CMAKE_Fortran_COMPILER}
  #     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMENT "build hysop package"
  #     DEPENDS ${HYSOP_LIBRARY_NAME}) #fortran module files have to be generated before
    add_custom_target(wheel ALL
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel build config_fc --f90exec=${CMAKE_Fortran_COMPILER}#-d ${CMAKE_BINARY_DIR}/wheel/
      COMMENT "create wheel file for ${PACKAGE_NAME} package"
      DEPENDS ${HYSOP_LIBRARY_NAME}
      )#  DEPENDS python-build)
else()
  # add_custom_target(python-build ALL 
  #   COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_SETUP} build
  #   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMENT "build hysop package")
  add_custom_target(wheel ALL
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel #-d ${CMAKE_BINARY_DIR}/wheel/
    COMMENT "create wheel file for ${PACKAGE_NAME} package"
    )#  DEPENDS python-build)
endif()
# add_custom_target(wheel
#   COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel #-d ${CMAKE_BINARY_DIR}/wheel/
#   COMMENT "create wheel file for ${PACKAGE_NAME} package"
#   )#  DEPENDS python-build)

# if(USE_FORTRAN)
#   add_dependencies(wheel ${HYSOP_LIBRARY_NAME})
# #  add_dependencies(python-install ${HYSOP_LIBRARY_NAME})
# endif()


#  ====== Create (and setup) install/uninstall targets ======
#
# --> set installation dir
# --> set options for python install
# --> create install/uninstall targets
include(HySoPInstallSetup)
# Remark : this must be done before add_subdir below, since install process in src needs CMAKE_INSTALL_PREFIX
# to be properly set.

if(USE_FORTRAN)
  if(EXISTS ${CMAKE_SOURCE_DIR}/${PACKAGE_NAME}/.f2py_f2cmap)
	message(STATUS "Generate f2py map file ...")
	configure_file(${CMAKE_SOURCE_DIR}/${PACKAGE_NAME}/.f2py_f2cmap
      ${CMAKE_BINARY_DIR}/.f2py_f2cmap)
  endif()
  include(fortran_utils)
  # Create f2hysop.pyf file
  write_main_pyf_file(f2hysop)

  # ====== Create non-python (fortran) libraries (fftw and scales interfaces), if required =====

  # Set module files directory (i.e. where .mod will be created)
  set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/Modules)
  #  Add compilation flags:
  #append_Fortran_FLAGS("-Wall -fPIC -ffree-line-length-none -DBLOCKING_SEND_PLUS -DBLOCKING_SEND")
  append_Fortran_FLAGS("-Wall -fPIC -ffree-line-length-none -cpp")
  
  if(USE_MPI)
    # -I
    include_directories(${MPI_Fortran_INCLUDE_PATH})
    # Add compilation/link flags
    list(APPEND ${HYSOP_LIBRARY_NAME}_LINK_FLAGS ${MPI_Fortran_LINK_FLAGS})
    append_Fortran_flags(${MPI_Fortran_COMPILE_FLAGS})
    # Append mpi libraries to the list of libraries linked with libhysop.
    list(APPEND HYSOP_LINK_LIBRARIES ${MPI_Fortran_LIBRARIES} )
  endif(USE_MPI)

  #set(Fortran_FLAGS ${CMAKE_Fortran_FLAGS})
  #append_flags(Fortran_FLAGS ${CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE}})
endif()

if(WITH_COMPILED_LIB)
  add_subdirectory(src)
  #get_directory_property(FORTRAN_INCLUDE_DIRS
    #DIRECTORY ${CMAKE_SOURCE_DIR}/src
    #DEFINITION FORTRAN_INCLUDE_DIRS)
endif()

if(WITH_LIB_CXX)
  #C++ variables used by setup.py.in for swig
  set(CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-parameter ${FFTW_COMPILE_FLAGS} -fPIC -std=c++11")
  set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_EXE_LINKER_FLAGS         "${CMAKE_EXE_LINKER_FLAGS}")
  
  set(CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CXX_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  set(CXX_EXT_INCLUDES ${PYTHON_INCLUDE_DIR} ${FFTW_INCLUDE_DIRS})
  set(CXX_EXT_LIBS ${PYTHON_LIBRARIES} ${FFTW_LIBRARIES})
  set(CXX_EXT_LIB_DIRS ${FFTW_LIBRARY_DIRS})
  set(CXX_EXTRA_DEFINES ${FFTW_DEFINES} -DHAS_EXTERN_TEMPLATES)

  #swig package name (lib name generated by swig)
  set(CPP_2_HYSOP "cpp2hysop")
endif()
# ====== Generates python files required for build/install process ======
# The file setup.py will be generated from setup.py.in.
if(EXISTS ${CMAKE_SOURCE_DIR}/setup.py.in)
  message(STATUS "Generate setup.py file ...")
  configure_file(setup.py.in setup.py)
endif()

# The file __init__.py will be generated from __init__.py.in.
if(EXISTS ${CMAKE_SOURCE_DIR}/hysop/__init__.py.in)
  message(STATUS "Generate __init__.py file ...")
  file(REMOVE ${CMAKE_SOURCE_DIR}/hysop/__init__.py)
  configure_file(hysop/__init__.py.in ${CMAKE_SOURCE_DIR}/hysop/__init__.py)
endif()

# Hysop C++ library is generated in setup.py by swig
# --- C++ main and tests  ---
if(WITH_LIB_CXX)
        
    get_filename_component(CXX_DIR  "${CMAKE_SOURCE_DIR}/src/hysop++" ABSOLUTE)
    get_filename_component(CXX_MAIN_DIR "${CXX_DIR}/main"             ABSOLUTE)
    get_filename_component(CXX_TEST_DIR "${CXX_DIR}/tests"            ABSOLUTE)
    get_filename_component(CXX_SOURCE_DIR "${CXX_DIR}/src"            ABSOLUTE)
    
    include_directories(${CXX_SOURCE_DIR})
    include_directories(SYSTEM ${CXX_EXT_INCLUDES})
    link_directories(${CXX_EXT_LIB_DIRS})
    add_definitions(${CXX_EXTRA_DEFINES})

    if(WITH_MAIN_CXX OR WITH_GOOGLE_TESTS)
        if(APPLE) #swig only generates a bundle, need to generate another static library...
            set(HYSOP_CXX_LIBRARY_DYLIB "cpp2hysop_dylib")

            file(GLOB_RECURSE source_files ${CXX_SOURCE_DIR}/*.cpp)
            add_library(${HYSOP_CXX_LIBRARY_DYLIB} STATIC ${source_files})
            target_link_libraries(${HYSOP_CXX_LIBRARY_DYLIB} ${EXT_LIBRARIES})

            set(HYSOP_CXX_LIBRARY ${HYSOP_CXX_LIBRARY_DYLIB})
            set(HYSOP_CXX_LIBRARY_DEP cpp2hysop_dylib)
        else() #nothing to do on other platforms bundle <=> dynamic libraries, so just copy the swig generated one
            set(HYSOP_CXX_LIBRARY_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/libcpp2hysop_bundle.so")
            add_custom_target(cpp2hysop_bundle
                DEPENDS wheel
                COMMAND cp `find ${CMAKE_CURRENT_BINARY_DIR}/build -name _${CPP_2_HYSOP}.so` ${HYSOP_CXX_LIBRARY_BUNDLE}
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                COMMENT "Copy swig c++ library to link")
            set(HYSOP_CXX_LIBRARY ${HYSOP_CXX_LIBRARY_BUNDLE})
            set(HYSOP_CXX_LIBRARY_DEP cpp2hysop_bundle)
        endif()
    endif()
  
    if(WITH_MAIN_CXX)
        list(APPEND cxx_executable_sources "${CXX_MAIN_DIR}/planner.cpp")
        list(APPEND cxx_executable_sources "${CXX_MAIN_DIR}/diffSolver.cpp")
        list(APPEND cxx_executable_sources "${CXX_MAIN_DIR}/poissonSolver.cpp")
        foreach(cxx_main_source ${cxx_executable_sources})
            get_filename_component(cxx_exec_name "${cxx_main_source}" NAME_WE)
            add_executable(${cxx_exec_name} ${cxx_main_source})
            add_dependencies(${cxx_exec_name} ${HYSOP_CXX_LIBRARY_DEP})
            target_link_libraries(${cxx_exec_name} ${HYSOP_CXX_LIBRARY} ${CXX_EXT_LIBS})
        endforeach()
    endif()

    if(WITH_GOOGLE_TESTS)
        add_subdirectory(${CXX_TEST_DIR})
    endif()
endif()

# =========== RPATH stuff ===========
# Doc :
#  - https://cmake.org/Wiki/CMake_RPATH_handling
#  - https://cmake.org/cmake/help/git-master/manual/cmake-packages.7.html#creating-packages
#  - http://matthew-brett.github.io/docosx/mac_runtime_link.html

# Force MACOSX_RPATH, whatever the cmake version is.
# RPATH management on macosx with cmake is quite a mess, so please
# think hard before changing the following lines ...
set(CMAKE_MACOSX_RPATH 1)

# don't skip the full RPATH for the build tree
# except if explicitely asked by user
if(FORCE_SKIP_RPATH)
  set(CMAKE_SKIP_BUILD_RPATH TRUE)
else()
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
endif()

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${PACKAGE_NAME}")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${PACKAGE_NAME}" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${PACKAGE_NAME}")
endif()

# ====== Create a Target to clean sources (remove .pyc files) and build dir ======

file(GLOB_RECURSE PYCFILES "${CMAKE_SOURCE_DIR}/*.pyc")
add_custom_target(pyclean COMMAND rm -f ${PYCFILES}
  COMMAND make clean
  COMMAND rm -rf ${CMAKE_BINARY_DIR}/build  ${CMAKE_BINARY_DIR}/DoxygenGeneratedDoc
  COMMAND rm ${CMAKE_SOURCE_DIR}/hysop/__init__.py
  COMMENT "clean hysop sources and build.")

# ====== Create a Target to generate the documentation ======
find_package(Doxygen)
if(DOXYGEN_FOUND)
  find_file(DOXY name doxypy.py PATH ENV{PATH})
  if(DOXY-NOTFOUND)
    message(STATUS "Warning, doxypy seems to be missing on your system. You may not be able to properly generate the documentation.")
  endif()
  configure_file(${CMAKE_SOURCE_DIR}/DoxyConf/hysop.doxyfile.in ${CMAKE_BINARY_DIR}/DoxyConf/hysop.doxyfile)
  add_custom_target(doc COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/DoxyConf/hysop.doxyfile
  COMMENT "Generate hysop documentation using doxygen.")
else()
  message(STATUS "Warning : cmake cannot find doxygen on your system. It means you will not be able to generate documentation for hysop.")
  add_custom_target(doc COMMAND echo "Doxygen was not found on your system. Documentation generation is not possible."
  COMMENT "Generate hysop documentation using doxygen.")
endif()

# ============= Tests =============
if(WITH_TESTS)
  include(CTest)
  # Number of mpi processes used to run tests.
  set(NBPROCS_FOR_TESTS "8" CACHE STRING "Number of mpi processes used to run tests")
  if(NOT USE_MPI)
    set(WITH_MPI_TESTS "OFF")
  endif()
  include(HySoPTests)
endif(WITH_TESTS)


# ============= Summary =============
if(VERBOSE_MODE)
  message("\n====================== End of configuration process ======================")
  message("\n Summary: ")
  message(STATUS " Python libraries : ${PYTHON_LIBRARIES}")
  message(STATUS " Python include : ${PYTHON_INCLUDE_DIRS}")
  message(STATUS " Python version : ${PYTHON_VERSION_STRING}")
  message(STATUS " Python executable : ${PYTHON_EXECUTABLE}")
  message(STATUS " Install mode is `${HYSOP_INSTALL}` and ${PACKAGE_NAME} will be installed in : ${HYSOP_PYTHON_INSTALL_DIR}")
  message(STATUS " ${PACKAGE_NAME} will be built in ${HYSOP_BUILD_PYTHONPATH}")
  if(USE_FORTRAN)
    message(STATUS " Fortran compiler : ${CMAKE_Fortran_COMPILER}")
  else()
    message(WARNING "You deactivate libhysop (fortran) generation. This will disable the fortran interface, including fftw and scales fonctionnalities.")
  endif()
  if(WITH_LIB_CXX)
    message(STATUS " CXX compiler : ${CMAKE_CXX_COMPILER}")
  else()
    message(WARNING "You deactivate libhysop (cxx) generation. This will disable the Aitken-Schwarz Poisson solver.")    
  endif()
  message(STATUS " Sources are in : ${CMAKE_SOURCE_DIR}")
  message(STATUS " Project uses MPI : ${USE_MPI}")
  message(STATUS " Project uses Scales : ${WITH_SCALES}")
  message(STATUS " Project uses FFTW : ${WITH_FFTW}")
  message(STATUS " Project uses GPU : ${WITH_GPU}")
  message(STATUS " ${PROJECT_NAME} debug mode : ${DEBUG}")
  message(STATUS " Enable -OO run? : ${OPTIM}")
  message(STATUS "====================== ======= ======================")
  message(STATUS " ")
  message(STATUS "Try :")
  message(STATUS " 'make -jN' to build the project, N being the number of available processes.")
  message(STATUS " 'make install' to install python modules and their dependencies. ")
  message(STATUS " 'make doc' to generate doxygen documentation for hysop.")
  message(STATUS " 'make test' to run some test (after the build! Do not use -j with this target).")
  message(STATUS " 'make clean' to clean build directory.")
  message(STATUS " 'make uninstall' to clean install directory. Dry-run (make -n uninstall) is advisable to check what will really be deleted.")
  message(STATUS "\n\n/!\\ Warning /!\\ : depending on your python environment configuration, you may need to set PYTHONPATH.")
  message("Try to run python -c 'import hysop'. If it fails, add ${HYSOP_PYTHON_INSTALL_DIR} to PYTHONPATH environment variable.")
  message("Example : \n export PYTHONPATH=${HYSOP_PYTHON_INSTALL_DIR}:\${PYTHONPATH}\n")
endif()
