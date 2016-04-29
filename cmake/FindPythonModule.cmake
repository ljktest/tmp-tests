# Search for a particular python module.
# Sources : see http://www.cmake.org/pipermail/cmake/2011-January/041666.html
#
# Usage : find_python_module(mpi4py REQUIRED)
#
function(find_python_module module)
	string(TOUPPER ${module} module_upper)
	if(ARGC GREATER 1 AND ARGV1 STREQUAL "REQUIRED")
	  set(${module}_FIND_REQUIRED TRUE)
	endif()
	# A module's location is usually a directory, but for binary modules
	# it's a .so file.
	execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
	  "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
	  RESULT_VARIABLE _${module}_status
	  OUTPUT_VARIABLE _${module}_location
	  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
	if(_${module}_location)
	  # Sometimes the output of the command above is multi-lines.
	  # We need to keep only the last one.
	  string(FIND ${_${module}_location} "\n" matches REVERSE)
	  if(${matches} GREATER 0)
	    MATH(EXPR matches "${matches}+1")
	    string(SUBSTRING ${_${module}_location} ${matches} -1 _${module}_location)
  	  endif()
	endif()
    	if(NOT _${module}_status)
	  set(python_${module_upper} ${_${module}_location} CACHE STRING
	    "Location of Python module ${module}")
	endif(NOT _${module}_status)

	find_package_handle_standard_args(${module} DEFAULT_MSG _${module}_location)
	set(${module}_FOUND ${${module_upper}_FOUND} PARENT_SCOPE)
endfunction(find_python_module)
