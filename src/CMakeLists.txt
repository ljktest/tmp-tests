#=======================================================
# cmake utility to compile,link and install :
#  - hysop library (libhysop...)
#  - library particular solver from scales
#
#=======================================================

# The list of all dirs containing sources to be compiled for the HySoP lib
# Any file in those dirs will be used to create libhysop
set(${HYSOP_LIBRARY_NAME}_SRCDIRS
    .)

# --- fftw interface ---
if(WITH_FFTW)
  list(APPEND ${HYSOP_LIBRARY_NAME}_SRCDIRS fftw)
endif()

if(WITH_LIB_CXX)
  list(APPEND ${HYSOP_LIBRARY_NAME}_SRCDIRS hysop++/src)
endif()

# --- Choose scales sources location ---
# --> cache var to allow user input
set(SCALES_DIR scalesInterface CACHE FILEPATH "Scales sources location, default=src/scalesInterface")

# --- scales ---
if(WITH_SCALES)
  set(${HYSOP_LIBRARY_NAME}_SRCDIRS
    ${${HYSOP_LIBRARY_NAME}_SRCDIRS}
    ${SCALES_DIR}/.
    ${SCALES_DIR}/layout
    ${SCALES_DIR}/particles
    ${SCALES_DIR}/particles/advec_line
    )
endif()
if(WITH_EXTRAS)
  list(APPEND ${HYSOP_LIBRARY_NAME}_SRCDIRS} extras)
endif()
# extensions of source files that must be taken into account
get_standard_ext()
set(SRC_EXTS ${ALL_EXTS})
set(HDR_EXTS *.hpp *.h)

#============= Collect source and header files and create the library =============

get_sources("${${HYSOP_LIBRARY_NAME}_SRCDIRS}")
set(${HYSOP_LIBRARY_NAME}_SRCS ${SOURCES_FILES})
# --- Collect headers ---
# --> set HYSOP_HDRS (for installation)
set(HDRS_DIRS ${${HYSOP_LIBRARY_NAME}_SRCDIRS})
get_headers("${HDRS_DIRS}")
set(${HYSOP_LIBRARY_NAME}_HDRS ${HDRS_FILES})

# ============= Generates HySoPConfig.hpp =============
# The file HYSOP_LIBRARY_NAME_defines.hpp will be generated from config.hpp.cmake;
if(EXISTS ${CMAKE_SOURCE_DIR}/config.hpp.cmake)
  configure_file(${CMAKE_SOURCE_DIR}/config.hpp.cmake ${HYSOP_LIBRARY_NAME}_defines.hpp)
  include_directories(${CMAKE_BINARY_DIR})
endif()

list(APPEND ${HYSOP_LIBRARY_NAME}_HDRS "hysop_defines.hpp")

# We add headers to source files list
list(APPEND ${HYSOP_LIBRARY_NAME}_SRCS ${${HYSOP_LIBRARY_NAME}_HDRS})

# --- include dirs for library ---
# for local headers ...
include_directories(${HDRS_DIRS})
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})
# and for headers of external libraries
include_directories(${HYSOP_INCLUDE_DIRECTORIES})

add_library(${HYSOP_LIBRARY_NAME} STATIC ${${HYSOP_LIBRARY_NAME}_SRCS})
target_link_libraries(${HYSOP_LIBRARY_NAME} PRIVATE ${HYSOP_LINK_LIBRARIES})

#install(TARGETS ${HYSOP_LIBRARY_NAME}
#  LIBRARY DESTINATION ${HYSOP_PYTHON_INSTALL_DIR}/${PACKAGE_NAME})

# ============= Create an executable linked with libhysop =============
# This part is optional and only useful to test libhysop in a
# way that does not depends on python.
# At the time it only includes fftw tests.
if(WITH_FORTRAN_TESTS)
  if(WITH_FFTW)
    # Set the name of a executable file that will be linked with libHYSOP_LIBRARY_NAME.
    # Useful to test libhysop in a way that does not depend on python.
    set(EXE_NAME ${HYSOP_LIBRARY_NAME}Run)
    # A main file to create an executable (test purpose)
    # Any files in these dirs will be used to create HySoP exec (linked with libhysop)
    set(${EXE_NAME}_SRCDIRS main)
    # Source and header files list, to generate a working executable based on libhysop.
    get_sources("${${EXE_NAME}_SRCDIRS}")
    get_headers("${${EXE_NAME}_SRCDIRS}")
    set(${EXE_NAME}_SRCS ${SOURCES_FILES})
    set(${EXE_NAME}_HDRS ${HDRS_FILES})
    list(APPEND ${EXE_NAME}_SRC ${${EXE_NAME}_HDRS})
    include_directories(${${EXE_NAME}_HDRS})
    add_executable(${EXE_NAME} ${${EXE_NAME}_SRCS})
    add_dependencies(${EXE_NAME} ${HYSOP_LIBRARY_NAME})
    
    # libs to link with EXE_NAME
    target_link_libraries(${EXE_NAME} ${HYSOP_LIBRARY_NAME})
    target_link_libraries(${EXE_NAME} ${HYSOP_LINK_LIBRARIES})
  endif()
endif()