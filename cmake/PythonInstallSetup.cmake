# --- Function to compute the install path/options ---
#
# Usage :
# set_install_options()
#
# Read HYSOP_INSTALL option (from user input)
# and set python install dir according to its value.
#
# Summary :
# cmake path-to-your-sources -Dpython_install_dir=standard
# make install
#
# ---> install in 'system' python site-package
#
# cmake path-to-your-sources -Dpython_install_dir=user
# make install
#
# ---> install in USER_SITE (no virtualenv case)
# ---> install in site-packages of your virtualenv
#
# cmake path-to-your-sources -Dpython_install_dir=prefix -DCMAKE_INSTALL_PREFIX=/some/install/path
# make install
#
# ---> install in CMAKE_INSTALL_PREFIX
#
# If /some/install/path is not a standard path of your system,
# you'll probably need something like :
# export PYTHONPATH=${PYTHONPATH}:/some/install/path
#
#
function(set_python_install_path)
  set(python_install_options "" CACHE INTERNAL "")
  set(python_install_options)# "--record;${CMAKE_BINARY_DIR}/python_install_manifest.txt")
  if(HYSOP_INSTALL STREQUAL user)
    # --- Case 1 : HYSOP_INSTALL=user ---
    # In that case, we need to find the user path. It depends on the operating system
    # and on which python is used (virtualenv or not)
    # First, we need to check if '--user' option works in the current environment.
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
      "import site; print site.ENABLE_USER_SITE" OUTPUT_VARIABLE ENABLE_USER)
    string(STRIP ${ENABLE_USER} ENABLE_USER)
    
    if(ENABLE_USER) # --user works ...
      # Find install path for --user (site.USER_SITE)
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
	"import site; print site.USER_BASE" OUTPUT_VARIABLE USER_BASE)
      string(STRIP ${USER_BASE} USER_BASE)
      list(APPEND python_install_options --user)#prefix=${USER_BASE})
      # Get python user site and install path = USER_SITE + project_name
      set(PYTHON_COMMAND_GET_INSTALL_DIR
       "import site, os, sys ; print os.path.join(site.USER_BASE, os.path.join(\"lib\", os.path.join(\"python\" + str(sys.version_info.major) + '.' + str(sys.version_info.minor),
 \"site-packages\")))")
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "${PYTHON_COMMAND_GET_INSTALL_DIR}"
      OUTPUT_VARIABLE PY_INSTALL_DIR)

    else()
      # user site not included in the path,
      # which probably means that python is run using virtualenv
      # Command to find 'global' site-packages
      # default path will probably be ok --> no options
      set(GET_SITE_PACKAGE
       "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
	"${GET_SITE_PACKAGE}" OUTPUT_VARIABLE GLOBAL_SITE_PACKAGE)
      string(STRIP ${GLOBAL_SITE_PACKAGE} GLOBAL_SITE_PACKAGE)
      set(PYTHON_COMMAND_GET_INSTALL_DIR ${GET_SITE_PACKAGE})
      execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "${PYTHON_COMMAND_GET_INSTALL_DIR}"
	OUTPUT_VARIABLE PY_INSTALL_DIR)
    endif()

  elseif(HYSOP_INSTALL STREQUAL standard)
    # install in python standard installation location
    # (may need to be root)
    # Depends on OS/platform type, check for example https://docs.python.org/2/install/.
    #set(PYTHON_COMMAND_GET_INSTALL_DIR   
    #  "import site; print(site.getsitepackages()[0])")
    # --> this does not work properly: the order in resulting
    # list depends on the OS, the python version ...
    configure_file(CMake/fake/setup.py tmp/setup.py)
    configure_file(CMake/fake/__init__.py tmp/fake/__init__.py)
    configure_file(CMake/find_python_install.py tmp/find_python_install.py)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} find_python_install.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tmp/
      OUTPUT_VARIABLE PY_INSTALL_DIR)
    
  else()#HYSOP_INSTALL STREQUAL prefix)
    # Last case : HYSOP_INSTALL="some_prefix", user-defined prefix
    # we use CMAKE_INSTALL_PREFIX as the path for python install
    list(APPEND python_install_options --user)
    set(PY_INSTALL_DIR ${HYSOP_INSTALL})
    set(HYSOP_INSTALL prefix PARENT_SCOPE)
  endif()

  string(STRIP ${PY_INSTALL_DIR} PY_INSTALL_DIR)
  if(NOT CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(NOT CMAKE_INSTALL_PREFIX STREQUAL PY_INSTALL_DIR)
      message(STATUS "!!! Warning !!! Use HYSOP_INSTALL option rather than CMAKE_INSTALL_PREFIX that will be overwritten if not complient with HYSOP_INSTALL value.")
    endif()
  endif()

  # Set the HYSOP_PYTHON_INSTALL_DIR to the proper path
  set(HYSOP_PYTHON_INSTALL_DIR ${PY_INSTALL_DIR}
    CACHE PATH "Install directory for hysop python package" FORCE)
  set(CMAKE_INSTALL_PREFIX ${PY_INSTALL_DIR}
    CACHE PATH "Install directory for hysop python package" FORCE)
  #list(APPEND python_install_options config_fc)
  #list(APPEND python_install_options --f90exec=${CMAKE_Fortran_COMPILER})
  set(python_install_options ${python_install_options} CACHE INTERNAL "")
endfunction()

