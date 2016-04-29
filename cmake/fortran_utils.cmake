# --- CMake utilities, to deal with f2py interface ---


# Create a pyf file that include
# all 'fortran' signature files (sub-pyf files)
# required by f2py.
#
# Usage:
# write_main_pyf_file(fname)
# where fname must be the name of the generated module
#
# For example to create hysop.f2hysop python module,
# call (in CMakeLists.txt)
# write_main_pyf_file(f2hysop)
# --> create f2hysop.pyf that will be used to generate hysop.f2hysop module.
#
function(write_main_pyf_file filename)
  set(_file ${CMAKE_SOURCE_DIR}/${PACKAGE_NAME}/${filename}.pyf.in)
  file(WRITE ${_file}
"!    -*- f90 -*-\n
! Generated file - Do not edit.\n
! Note: the context of this file is case sensitive.\n
python module f2hysop ! in\n
  interface\n")
 file(APPEND ${_file} 
      "      ! Example
      include '@CMAKE_SOURCE_DIR@/hysop/fortran/template.pyf'\n")
 file(APPEND ${_file}
      "      ! precision
      include '@CMAKE_SOURCE_DIR@/hysop/f2py/parameters.pyf'\n")
	if(WITH_FFTW)
	  file(APPEND ${_file}
      "      ! fftw
      include '@CMAKE_SOURCE_DIR@/hysop/f2py/fftw2py.pyf'\n")
    endif()
    if(WITH_SCALES)
      file(APPEND ${_file}
      "      ! scales
      include '@CMAKE_SOURCE_DIR@/hysop/f2py/scales2py.pyf'\n")
    endif()
    if(WITH_EXTRAS)
      file(APPEND ${_file}
      "      ! arnoldi
      include '@CMAKE_SOURCE_DIR@/hysop/fortran/arnoldi2py.pyf'\n")
	endif()
 file(APPEND ${_file} "  end interface\n
end python module f2hysop")

message(STATUS "Generate pyf file ...")
configure_file(${_file} ${CMAKE_SOURCE_DIR}/${PACKAGE_NAME}/${filename}.pyf @ONLY)

endfunction()
