# - Try to find FFTW3 libraries using components
#
# == Usage ==
#   Call 
#        find_package(FFTW 
#             REQUIRED COMPONENTS <required components here>
#            [OPTIONAL_COMPONENTS <optional components here>]
#            [QUIET]
#        )
#   in your CmakeList.txt (note the missing underscore for required components)
#   
#   Possible component names: fftw3[fdlq](-(mpi|threads|omp))\?, where case doesn't matter
#     and [fdlq] stands for f=float, d=double, l=long double, q=gcc quad long
#   Note that the only exeption is fftw3q-mpi which doesn't exist at this day.
#
# == Full example == 
#  set(FIND_FFTW_VERBOSE ON) (optional)
#  set(FIND_FFTW_DEBUG OFF)  (optional)
#  find_package(FFTW 
#      REQUIRED COMPONENTS Fftw3f Fftw3d 
#      OPTIONAL_COMPONENTS Fftw3l Fftw3q 
#      OPTIONAL_COMPONENTS Fftw3f-mpi Fftw3d-mpi Fftw3l-mpi 
#      OPTIONAL_COMPONENTS Fttw3f-omp Fftw3d-omp Fftw3l-omp Fftw3q-omp
#      OPTIONAL_COMPONENTS Fttw3f-threads Fftw3d-threads Fftw3l-threads Fftw3q-threads)
#
# == Defined variables ==
#
#   FFTW_FOUND - will be set to true if and only if all required components and their dependencies were found
#   FFTW_INCLUDE_DIRS - all FFTW include directories
#   FFTW_LIBRARY_DIRS - all FFTW library directories
#   FFTW_LIBRARIES - all FFTW libraries
#   FFTW_DEFINES - all FFTW defines
#   FTTW_COMPILE_FLAGS - all FFTW compile flags (currently only used for quadfloat fftw)
#
#   For each required or optional component, this package defines:
#  		COMPONENT_FOUND   - will be set to true the component and its dependencies were found, else false
#  		COMPONENT_DEFINES - all COMPONENT defines (-DFFTW_HAS_COMPONENT)
#
#  		COMPONENT_INCLUDE_DIRS - COMPONENT and its dependancies include directories
#  		COMPONENT_LIBRARY_DIRS - COMPONENT and its dependancies library directories
#  		COMPONENT_LIBRARIES    - COMPONENT and its dependancies libraries
#
#  		COMPONENT_INCLUDE_DIR - COMPONENT include directory
#  		COMPONENT_LIBRARY_DIR - COMPONENT library directory
#  		COMPONENT_LIBRARY     - COMPONENT library
#
#   where COMPONENT is equal to <input component name> transformed with the following rules:
#       * uppercased
#       * hyphens (-) replaced by underscores (_)
#   Examples: fFtW3Q => FFTW3Q, fftw3f-mpi => FFTW3F_MPI
# 
# == Using a specific FFTW ==
#   Set the variable ${FFTW_DIR} to your desired search paths if it's not in a standard place or if you want a specific version. 
#
# == Checking against a specific version or the library ==
#   Not supported yet.
#
# == Debug and verbose mode ==
# Set ${FIND_FFTW_VERBOSE} to ON before find_package call to enable verbose mode
# Set ${FIND_FFTW_DEBUG}   to ON before find_package call to enable debug mode
#
# Written by F. Pérignon, nov/2009
# Updated by J-B. Keck, feb/2016
# inspired from http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries

find_package(PkgConfig)

# --version check not supported yet
set(FFTW_VERSION_STRING ${FFTW_FIND_VERSION})

if(FIND_FFTW_VERBOSE)
    message(STATUS "Entering FindFFTW.cmake in verbose mode:")
endif()

foreach(fftw_comp ${FFTW_FIND_COMPONENTS})
    string(REPLACE "-" "_" fftw_comp_no_dash "${fftw_comp}")
    string(TOLOWER ${fftw_comp_no_dash} component)
    string(TOUPPER ${fftw_comp_no_dash} COMPONENT)

    # -- check if component has not already been found
    if(${COMPONENT}_FOUND)
        continue()
    endif()

    # -- find header name given the component name
    string(REGEX REPLACE "_(omp|threads)" "" component_without_ext "${component}")
    string(REGEX REPLACE "_mpi" "-mpi" component_without_ext "${component_without_ext}")
    string(REGEX REPLACE "fftw3[fdlq]" "fftw3" header "${component_without_ext}.h")

    # -- find library name given the component name
    string(REPLACE "fftw3d" "fftw3" library "${component}")
   
    if(FIND_FFTW_DEBUG)
        message("\tFFTW::${fftw_comp}:${COMPONENT}:${component}, LIB=${library} HEADER=${header}")
    endif()
    
    # -- use pkg-config to get hints about paths
    pkg_check_modules(${COMPONENT}_PKGCONF ${library} ${header} QUIET)

    # -- find include dir
    find_path(
        ${COMPONENT}_INCLUDE_DIR
        NAMES ${header}
        PATHS ${FFTW_INCLUDE_DIRS}
        NO_DEFAULT_PATH
    )
    find_path(
        ${COMPONENT}_INCLUDE_DIR
        NAMES ${header}
        PATHS ${fftw_DIR} 
        PATHS ${${COMPONENT}_PKGCONF_INCLUDE_DIRS}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
    )
    # -- search in default locations only if last search failed
    find_path(${COMPONENT}_INCLUDE_DIR NAMES ${header}
            PATHS ENV INCLUDE 
                  ENV PATH 
                  ENV C_INCLUDE_PATH 
                  ENV CXX_INCLUDE_PATH 
        )

    if(${${COMPONENT}_INCLUDE_DIR} STREQUAL "${COMPONENT}_INCLUDE_DIR-NOTFOUND")
        set(INCLUDE_DIR_FOUND FALSE)
    else()
        set(INCLUDE_DIR_FOUND TRUE)
    endif()

    # -- find library
    find_library(
      ${COMPONENT}_LIBRARY
      NAMES ${library}
      PATHS ${FFTW_LIBRARY_DIRS}
      NO_DEFAULT_PATH
    )
    find_library(
      ${COMPONENT}_LIBRARY
      NAMES ${library}
      PATHS ${fftw_DIR} 
      PATHS ${${COMPONENT}_INCLUDE_DIR}/.. 
      PATHS ${${COMPONENT}_PKGCONF_LIBRARY_DIRS}}
      PATH_SUFFIXES lib
      NO_DEFAULT_PATH
    )
    # -- default locations
    find_library(${COMPONENT}_LIBRARY 
        NAMES ${library}
        PATHS ENV LIBRARY_PATH 
              ENV LD_LIBRARY_PATH  
              ENV DYLD_LIBRARY_PATH)
    
    # -- if component is required append it to required vars
    if(FFTW_FIND_REQUIRED_${fftw_comp})
        list(APPEND FFTW_REQUIRED_INCLUDE_DIRS ${COMPONENT}_INCLUDE_DIR)
        list(APPEND FFTW_REQUIRED_LIBRARIES    ${COMPONENT}_LIBRARY)
    endif()

    # --extract library dir and library name
    if(${${COMPONENT}_LIBRARY} STREQUAL "${COMPONENT}_LIBRARY-NOTFOUND")
        set(LIBRARY_DIR_FOUND FALSE)
    else()
        get_filename_component(${COMPONENT}_LIBRARY_DIR "${${COMPONENT}_LIBRARY}" DIRECTORY)
        #set(${COMPONENT}_LIBRARY "${library}")
        set(LIBRARY_DIR_FOUND TRUE)
    endif()
    
    # -- find quadmath library if required
    string(FIND ${component} "fftw3q" FFTWQ_POS)
    if(FFTWQ_POS EQUAL 0)
        if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(DEPENDENCIES_FOUND FALSE) # -- only gcc supports quadmaths
          else()
			find_library(QUADMATHLIB
			  NAMES quadmath
			  )
			if(QUADMATHLIB_FOUND)
			  list(APPEND ${COMPONENT}_LIBRARIES ${QUADMATHLIB})
			  #list(APPEND ${COMPONENT}_LIBRARIES "quadmath")
			  list(APPEND ${COMPONENT}_DEFINES "-DHAS_QUADMATHS")
			  list(APPEND FFTW_COMPILE_FLAGS "-fext-numeric-literals")
			  set(DEPENDENCIES_FOUND TRUE)
			endif()
		  endif()
    else()
        set(DEPENDENCIES_FOUND TRUE)
    endif()
    
    if(LIBRARY_DIR_FOUND AND INCLUDE_DIR_FOUND AND DEPENDENCIES_FOUND)
        set(${COMPONENT}_FOUND TRUE)
        list(APPEND ${COMPONENT}_INCLUDE_DIRS ${${COMPONENT}_INCLUDE_DIR})
        list(APPEND ${COMPONENT}_LIBRARY_DIRS ${${COMPONENT}_LIBRARY_DIR})
        list(APPEND ${COMPONENT}_LIBRARIES    ${${COMPONENT}_LIBRARY})
        list(APPEND ${COMPONENT}_DEFINES "-DFFTW_HAS_${COMPONENT}")

        list(APPEND FFTW_INCLUDE_DIRS ${${COMPONENT}_INCLUDE_DIRS})
        list(APPEND FFTW_LIBRARY_DIRS ${${COMPONENT}_LIBRARY_DIRS})
        list(APPEND FFTW_LIBRARIES    ${${COMPONENT}_LIBRARIES})
        list(APPEND FFTW_DEFINES      ${${COMPONENT}_DEFINES})
        
        if(FIND_FFTW_VERBOSE)
            message("\tFound FFTW::${fftw_comp} with parameters '-I${${COMPONENT}_INCLUDE_DIR} -L${${COMPONENT}_LIBRARY_DIR}  -l${${COMPONENT}_LIBRARY}'.")
        endif()
    else()
        set(${COMPONENT}_FOUND FALSE)
        if(FFTW_FIND_REQUIRED_${fftw_comp})
            if(NOT FFTW_FIND_QUIETLY)
                message(FATAL_ERROR "Error: Could not find required component FFTW::${fftw_comp} (${COMPONENT}_INCLUDE_DIR='${${COMPONENT}_INCLUDE_DIR}' and ${COMPONENT}_LIBRARY='${${COMPONENT}_LIBRARY}'.)")
            endif()
        else()
            if(FIND_FFTW_VERBOSE)
                message("\tCould not find optional FFTW::${fftw_comp}.")
            endif()
            if(FIND_FFTW_DEBUG)
                message(STATUS "\t\t${COMPONENT}_INCLUDE_DIR='${${COMPONENT}_INCLUDE_DIR}' and ${COMPONENT}_LIBRARY='${${COMPONENT}_LIBRARY}'")
            endif()
        endif()
    endif()
        
    unset(FFTWQ_POS)
    unset(library)
    unset(header)
    unset(fftw_comp_no_dash)
    unset(component)
    unset(COMPONENT)
endforeach()
list(REMOVE_DUPLICATES FFTW_INCLUDE_DIRS)
list(REMOVE_DUPLICATES FFTW_LIBRARY_DIRS)
list(REMOVE_DUPLICATES FFTW_LIBRARIES)

# -- check required variables, version and set FFTW_FOUND to TRUE if ok
find_package_handle_standard_args(FFTW FOUND_VAR FFTW_FOUND
                                  REQUIRED_VARS ${FFTW_REQUIRED_LIBRARIES} ${FFTW_REQUIRED_INCLUDE_DIRS} 
                                  VERSION_VAR FFTW_VERSION_STRING)
    
if(FIND_FFTW_DEBUG)
    message(STATUS "FFTW_FOUND='${FFTW_FOUND}'")
    message(STATUS "FFTW_INCLUDE_DIRS='${FFTW_INCLUDE_DIRS}'")
    message(STATUS "FFTW_LIBRARY_DIRS='${FFTW_LIBRARY_DIRS}'")
    message(STATUS "FFTW_LIBRARIES='${FFTW_LIBRARIES}'")
    message(STATUS "FFTW_DEFINES='${FFTW_DEFINES}'")
endif()

unset(FFTW_REQUIRED_LIBRARIES)
unset(FFTW_REQUIRED_INCLUDE_DIRS)
