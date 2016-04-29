#
# Some convenience macros
#
include(CMakeParseArguments)

# -- Basic list manipulation --
# Get first element of list var
macro(CAR var)
  set(${var} ${ARGV1})
endmacro(CAR)

# get elements in list var minus the first one.
macro(CDR var junk)
  set(${var} ${ARGN})
endmacro(CDR)

# LIST(APPEND ...) is not correct on <COMPILER>_FLAGS 
macro(append_flags)
  CAR(_V ${ARGV})
  CDR(_F ${ARGV})
  set(${_V} "${${_V}} ${_F}")
endmacro(append_flags)

# The use of ADD_DEFINITION results in a warning with Fortran compiler
macro(APPEND_C_FLAGS)
  append_flags(CMAKE_C_FLAGS ${ARGV})
endmacro(APPEND_C_FLAGS)

macro(APPEND_CXX_FLAGS)
  append_flags(CMAKE_CXX_FLAGS ${ARGV})
endmacro(APPEND_CXX_FLAGS)

macro(APPEND_Fortran_FLAGS)
  append_flags(CMAKE_Fortran_FLAGS ${ARGV})
endmacro(APPEND_Fortran_FLAGS)

# Scans DIRS (list of directories) and returns a list of all files in those dirs
# matching extensions defined in SRC_EXTS list.
# Results are saved in SOURCES_FILES
#
# Usage:
# set(src_dirs dir1 dir2)
# get_sources(src_dirs)
macro(get_sources)
  set(SOURCES_FILES)
  foreach(DIR ${ARGV})
    foreach(_EXT ${SRC_EXTS})
      file(GLOB FILES_LIST ${DIR}/*.${_EXT})
      if(FILES_LIST)
	list(APPEND SOURCES_FILES ${FILES_LIST})
      endif()
    endforeach()
  endforeach()
  if(SOURCES_FILES)
    list(REMOVE_DUPLICATES SOURCES_FILES)
  endif()
endmacro()

# Scans DIRS (list of directories) and returns a list of all files in those dirs
# matching extensions defined in HDR_EXTS list.
# Results are saved in HDRS_FILES
#
# Usage:
# set(src_dirs dir1 dir2)
# get_headers(src_dirs)
macro(get_headers DIRS)
  set(HDRS_FILES)
  foreach(DIR ${ARGV})
    foreach(_EXT ${HDR_EXTS})
      file(GLOB FILES_LIST ${DIR}/*.${_EXT})
      if(FILES_LIST)
	list(APPEND HDRS_FILES ${FILES_LIST})
      endif()
    endforeach()
  endforeach()
  if(HDRS_FILES)
    list(REMOVE_DUPLICATES HDRS_FILES)
  endif()
endmacro()


# Return all files matching ext in directories of list dirs
function(get_files ext dirs)
  set(files_list)
  foreach(_dir IN LISTS ${dirs})
    file(GLOB files ${_DIR}/*.${ext})
  endforeach()
  list(APPEND files_list ${files})
  if(files_list)
    list(REMOVE_DUPLICATES files_list)
  endif()
  set(files_list ${files_list} PARENT_SCOPE)
endfunction()

# -- returns a list of source files extension --
# Results in var ALL_EXTS
macro(get_standard_ext)
  set(ALL_EXTS)
  foreach(_EXT
      ${CMAKE_CXX_SOURCE_FILE_EXTENSIONS}
      ${CMAKE_C_SOURCE_FILE_EXTENSIONS}
      ${CMAKE_Fortran_SOURCE_FILE_EXTENSIONS}
      ${CMAKE_Java_SOURCE_FILE_EXTENSIONS}
      ${CMAKE_RC_SOURCE_FILE_EXTENSIONS})
    list(APPEND ALL_EXTS ${_EXT})
  endforeach()
  list(REMOVE_DUPLICATES ALL_EXTS)
endmacro()

# debug
macro(display V)
  message(STATUS "${V} = ${${V}}")
endmacro(display V)

macro(ASSERT VAR)
  if (NOT DEFINED ${VAR})
	message( FATAL_ERROR "ASSERTION ERROR : ${VAR} UNSET" )
  endif()
endmacro()

# =======================================
# For a given package name, try to find
# corresponding headers and libraries and
# add them to the include directories
# and list of linked libraries.
#
# It sets (if found):
# - HYSOP_INCLUDE_DIRECTORIES with the list
# of directories of headers required for hysop to work with
# - HYSOP_LINK_LIBRARIES with the list of external libraries
# (full path!) needed by hysop project.
#
# Usage :
#  compile_with(Packagename options)
#
# with the same 'options' as find_package
# (see http://www.cmake.org/cmake/help/v3.0/command/find_package.html?highlight=find_package)
macro(COMPILE_WITH)

  set(options REQUIRED)
  set(oneValueArgs ONLY)
  set(multiValueArgs COMPONENTS)
  
  cmake_parse_arguments(COMPILE_WITH "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  set(_NAME)
  set(_NAME_VERSION)

  # Get package name and extra args ...
  CAR(_NAME ${COMPILE_WITH_UNPARSED_ARGUMENTS})
  CDR(_NAME_VERSION ${COMPILE_WITH_UNPARSED_ARGUMENTS})

  set(_NAMES)
  STRING(TOUPPER ${_NAME} _UNAME)
  list(APPEND _NAMES ${_NAME})
  list(APPEND _NAMES ${_UNAME})
  set(_FOUND)

  if(COMPILE_WITH_COMPONENTS)
    set(_COMPONENTS COMPONENTS ${COMPILE_WITH_COMPONENTS})
    set(_COMPONENTS_STR "components ${COMPILE_WITH_COMPONENTS} of the package")
  else()
    set(_COMPONENTS)
    set(_COMPONENTS_STR "package")
  endif()

  if(${COMPILE_WITH_REQUIRED})
    set(_REQUIRED REQUIRED)
    set(_REQUIRED_STR "required")
  else()
    set(_REQUIRED)
    set(_REQUIRED_STR "optional")
  endif()

  if(_NAME_VERSION)
    set(_NAME_VERSION_STR "version ${_NAME_VERSION}")
  else()
    set(_NAME_VERSION_STR "")
  endif()

  FIND_PACKAGE(${_NAME} ${_NAME_VERSION} ${_COMPONENTS} ${_REQUIRED})
  set(_LINK_LIBRARIES)
  FOREACH(_N ${_NAMES})
    if(${_N}_FOUND)
      set(_FOUND TRUE)
      set(_NAME_VERSION_STR "version ${${_N}_VERSION}")
      # add headers dirs into 'include' path
      # INCLUDE_DIR var name depends on FindNAME
      # We try to check the standard var names.
      if(DEFINED ${_N}_INCLUDE_DIRS)
	remember_include_directories("${${_N}_INCLUDE_DIRS}")
      endif()
      if(DEFINED ${_N}_INCLUDE_DIR)
	remember_include_directories("${${_N}_INCLUDE_DIR}")
      endif()
      if(DEFINED ${_N}_INCLUDE_PATH)
	remember_include_directories("${${_N}_INCLUDE_DIR}")
      endif()
      # Now we set list of libs that must be linked with.
      if(DEFINED ${_N}_LIBRARIES)
	list(APPEND _LINK_LIBRARIES ${${_N}_LIBRARIES})
      endif()
      # And the compiler flags
      if(DEFINED ${_N}_DEFINITIONS)
       foreach(_DEF ${${_N}_DEFINITIONS})
        append_c_flags(${_DEF})
        append_cxx_flags(${_DEF})
       endforeach()
      endif()
    endif()
  endforeach()
  if(_LINK_LIBRARIES)
    list(REMOVE_DUPLICATES _LINK_LIBRARIES)
    foreach(_lib ${_LINK_LIBRARIES})
      get_filename_component(libpath ${_lib} DIRECTORY)
      list(FIND CMAKE_C_IMPLICIT_LINK_DIRECTORIES "${libpath}" isSystemDir)
      if(${isSystemDir} GREATER -1)
        list(GET CMAKE_C_IMPLICIT_LINK_DIRECTORIES ${isSystemDir} result)
        if("${libpath}" STRGREATER "${result}")
          message("oiaoazoazioazioza ${_lib}")
	  list(APPEND CMAKE_INSTALL_RPATH "${libpath}")
	  list(APPEND CMAKE_BUILD_RPATH "${libpath}")
        endif()
      else()#"${isSystemDir}" STREQUAL "-1")
	#get_property(libpath SOURCE ${_lib} PROPERTY LOCATION)
	list(APPEND CMAKE_INSTALL_RPATH "${libpath}")
	list(APPEND CMAKE_BUILD_RPATH "${libpath}")
      endif()
    endforeach()
    if(CMAKE_INSTALL_RPATH)
      list(REMOVE_DUPLICATES CMAKE_INSTALL_RPATH)
    endif()
    if(CMAKE_BUILD_RPATH)
      list(REMOVE_DUPLICATES CMAKE_BUILD_RPATH)
    endif()
  endif()
  if(COMPILE_WITH_ONLY)
    set(_sico_component ${COMPILE_WITH_ONLY})
    set(${_sico_component}_LINK_LIBRARIES ${${_sico_component}_LINK_LIBRARIES}
      ${_LINK_LIBRARIES} CACHE INTERNAL "List of external libraries for ${_sico_component}.")
  else()
    set(HYSOP_LINK_LIBRARIES ${HYSOP_LINK_LIBRARIES}
      ${_LINK_LIBRARIES} CACHE INTERNAL "List of external libraries.")
  endif()

  IF (_FOUND)
    message(STATUS "Compilation with ${_REQUIRED_STR} ${_COMPONENTS_STR} ${_NAME} ${_NAME_VERSION_STR}")
  else()
    message(STATUS "Compilation without ${_REQUIRED_STR} ${_COMPONENTS_STR} ${_NAME} ${_NAME_VERSION_STR}")
  endif()

  set(_N)
  set(_NAME) 
  set(_NAME_VERSION)
  set(_NAME_VERSION_STR)
  set(_UNAME)
  set(_NAMES)
  set(_FOUND)
  set(_REQUIRED)
  set(_REQUIRED_STR)
  set(_COMPONENTS)
  set(_COMPONENTS_STR)
  set(_VERSION_STR)

endmacro(COMPILE_WITH)

# ==== Save directories required for include_directory ===
# 
# Set variable HYSOP_INCLUDE_DIRECTORIES with the list
# of directories of headers required for hysop to work with
# its dependencies.
# Usage :
# set(dirs d1 d2 d3)
# remember_include_directories(${dirs})
#  --> save d1, d2, d3 into HYSOP_INCLUDE_DIRECTORIES
# 
macro(REMEMBER_INCLUDE_DIRECTORIES _DIRS)
  foreach(_D ${_DIRS})
    list(APPEND HYSOP_INCLUDE_DIRECTORIES ${_D})
  endforeach()
  list(REMOVE_DUPLICATES HYSOP_INCLUDE_DIRECTORIES)
  set(HYSOP_INCLUDE_DIRECTORIES ${HYSOP_INCLUDE_DIRECTORIES}
    CACHE INTERNAL "Include directories for external dependencies.")

endmacro()

# ==== Save directories required for include_directory ===
# 
# Set variable ${PROJECT_NAME}_LOCAL_INCLUDE_DIRECTORIES with the list
# of directories of headers for hysop
#
# Usage :
# set(dirs d1 d2 d3)
# remember_local_include(${dirs})
#  --> save d1, d2, d3 into ${PROJECT_NAME}_LOCAL_INCLUDE_DIRECTORIES
#
# mind the ${CMAKE_CURRENT_SOURCE_DIR} below!
macro(remember_local_include_directories _DIRS)
  foreach(_D ${_DIRS})
    list(APPEND ${PROJECT_NAME}_LOCAL_INCLUDE_DIRECTORIES
      ${CMAKE_CURRENT_SOURCE_DIR}/${_D})
  endforeach()
  list(REMOVE_DUPLICATES ${PROJECT_NAME}_LOCAL_INCLUDE_DIRECTORIES)
  set(${PROJECT_NAME}_LOCAL_INCLUDE_DIRECTORIES
    ${${PROJECT_NAME}_LOCAL_INCLUDE_DIRECTORIES}
    CACHE INTERNAL "Include directories for external dependencies.")
endmacro()


# Usage: list_subdirectories(the_list_is_returned_here dir 1)
# 1 if you want relative directories as output.
macro(list_subdirectories retval curdir return_relative)
  file(GLOB subdir RELATIVE ${curdir} *)
  display(subdir)
  set(list_of_dirs "")
  foreach(dir ${subdir})
    if(IS_DIRECTORY ${curdir}/${dir})
      if (${return_relative})
        set(list_of_dirs ${list_of_dirs} ${dir})
      else()
        set(list_of_dirs ${list_of_dirs} ${curdir}/${dir})
      endif()
    endif()
  endforeach()
  set(${retval} ${list_of_dirs})
endmacro()

# Get the list of all directories in a directory
macro(get_subdirectories result curdir)
  file(GLOB subdir RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(dir ${subdir})
    if(IS_DIRECTORY ${curdir}/${dir})
      set(dirlist ${dirlist} ${dir})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

# Find the name (platform dependant) of the python build
# (result of python setup.py build)
# Note : this macro obviously works only if build exists i.e.
# after a call to python build
macro(get_python_builddir where result) 
  get_subdirectories(listofdirs ${where})
  foreach(dir ${listofdirs})
    find_file(hysopfile 
      NAMES __init__.py
      PATHS ${where}/${dir}/hysop
      PATH_SUFFIXES hysop
      NO_DEFAULT_PATH)
  endforeach()
  get_filename_component(builddir ${hysopfile} PATH)
  get_filename_component(builddir ${builddir} PATH)
  set(${result} ${builddir})
endmacro()


# ------------------------------------
# Get the list of subdirectories
# of a given dir
# ------------------------------------
macro(get_subdirectories result current_dir)
  file(GLOB subdirs RELATIVE ${current_dir} ${current_dir}/*)
  set(dirs "")
  foreach(_dir ${subdirs})
    if(IS_DIRECTORY ${current_dir}/${_dir})
      list(APPEND dirs ${_dir})
    endif()
  endforeach()
  set(${result} ${dirs})
endmacro()
