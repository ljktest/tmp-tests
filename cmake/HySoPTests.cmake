# === Configuration for tests in HySoP ===
#
# --> collect test directories/files
# --> create tests (ctest)
#
# For each function 'test_something' in all files of directories
# 'tests' a new test is created.
#
# Those tests will be run after a call to 'make test'
# or a call to ctest.

enable_testing()
find_python_module(pytest REQUIRED)

# Declaration of python test
# Usage:
# add_python_test(name file)
# with 'name' is the name of the test and file the source
# file for the test.
macro(add_python_test test_name test_file)
  add_test(${test_name} py.test "${pytest_opt}" ${test_file})
  set_tests_properties(${test_name} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILURE;Exception;[^x]failed;ERROR;Assertion")
  set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/dataForTests)
  set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "PYTHONPATH=$ENV{PYTHONPATH}:${HYSOP_BUILD_PYTHONPATH}")
  set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "PYTHONPATH=$ENV{PYTHONPATH}:${HYSOP_BUILD_PYTHONPATH},LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${HYSOP_BUILD_PYTHONPATH}")
  if(WITH_MPI_TESTS)
    # Run the same test using mpi multi process run.
    # The number of processes used is set with NBPROCS_FOR_TESTS variable (user option for cmake, default=8)
    add_test(${test_name}_mpi ${MPIEXEC} -np ${NBPROCS_FOR_TESTS} ${PYTHON_EXECUTABLE} -m pytest "${pytest_opt}" ${test_file})

    #mpirun -np 1 python -m pytest -v

    set_tests_properties(${test_name}_mpi PROPERTIES FAIL_REGULAR_EXPRESSION "FAILURE;Exception;[^x]failed;ERROR;Assertion")
    set_tests_properties(${test_name}_mpi PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/dataForTests)
    set_tests_properties(${test_name}_mpi PROPERTIES ENVIRONMENT "PYTHONPATH=$ENV{PYTHONPATH}:${HYSOP_BUILD_PYTHONPATH},LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${HYSOP_BUILD_PYTHONPATH}")
  endif()
  
endmacro()

# Choose python build dir as directory where tests will be run.
# --> set test dir
set(testDir ${HYSOP_BUILD_PYTHONPATH})

# === Tests options ===
if(FULL_TEST)
  set(pytest_opt "-s -v -pep8" CACHE INTERNAL "extra options for py.test")
else()
  set(pytest_opt "-v" CACHE INTERNAL "extra options for py.test")
endif()


# === Set the list of all directories which may contain tests ===
set(py_src_dirs
  fields
  domain
  operator
  numerics
  problem
  tools
  mpi
  )

# If GPU is on, we add test_XXX.py files of hysop/gpu directory
if(WITH_GPU)
  list(APPEND py_src_dirs gpu)
endif()

# Copy the OpenCL sources files to build dir (required since only python files are copied by setup.py)
set(clfiles)
file(GLOB clfilestmp RELATIVE ${CMAKE_SOURCE_DIR} hysop/gpu/cl_src/[a-z]*.cl)
set(clfiles ${clfiles} ${clfilestmp})
file(GLOB clfilestmp RELATIVE ${CMAKE_SOURCE_DIR} hysop/gpu/cl_src/kernels/[a-z]*.cl)
set(clfiles ${clfiles} ${clfilestmp})
file(GLOB clfilestmp RELATIVE ${CMAKE_SOURCE_DIR} hysop/gpu/cl_src/advection/[a-z]*.cl)
set(clfiles ${clfiles} ${clfilestmp})
file(GLOB clfilestmp RELATIVE ${CMAKE_SOURCE_DIR} hysop/gpu/cl_src/remeshing/[a-z]*.cl)
set(clfiles ${clfiles} ${clfilestmp})
foreach(_F ${clfiles})
  configure_file(${_F} ${testDir}/${_F} COPYONLY)
endforeach()

# === Create the files list from all directories in py_src_dirs ===

# Build a list of test_*.py files for each directory of hysop/${py_src_dirs}
set(py_test_files)
foreach(testdir ${py_src_dirs})
  file(GLOB testfiles RELATIVE ${CMAKE_SOURCE_DIR} hysop/${testdir}/tests/test_*.py)
  set(py_test_files ${py_test_files} ${testfiles})
  # copy data files
  file(GLOB reffiles hysop/${testdir}/tests/ref_files/*)
  file(COPY ${reffiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/dataForTests)
endforeach()

# Build doctests
# Handling doctest in *.py files recursively for each directory of hysop/${py_src_dirs}
# excluding  __init__ or test_ files. 
# Doctest are run for every line which contains '>>>'
set(py_doctest_files)
foreach(testdir ${py_src_dirs})
  file(GLOB testfiles hysop/${testdir}/[a-zA-Z]*.py)
  foreach(testfile ${testfiles})
    file(STRINGS ${testfile} test_doctest REGEX ">>>")
    if(NOT "${test_doctest}" STREQUAL "")
      set(py_doctest_files ${py_doctest_files} ${testfile})
    endif()
  endforeach()
endforeach()

# === Create tests from py_test_files ===
foreach(testfile ${py_test_files})
  get_filename_component(testName ${testfile} NAME_WE)
  set(exename ${testDir}/${testfile})
  #message(STATUS "Add test ${exename} ...")
  add_python_test(${testName} ${exename})
endforeach()

# Add files containing doctests
foreach(testfile ${py_doctest_files})
  get_filename_component(testName ${testfile} NAME_WE)
  add_test("doctest_${testName}" py.test -v --doctest-modules ${testfile})
endforeach()

