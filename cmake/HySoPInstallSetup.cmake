# --- HySoP Install Process ---
#
# By default, hysop libraries and python modules will be installed
# in the usual 'user' path of python (see https://docs.python.org/2/install/)
# as if you run "python setup.py install --user".
# This directory corresponds to site.USER_SITE variable.
#
# But if you are using virualenv, USER_SITE is not enable.
# In that case install will be done in site-packages of the virtualenv.
#
# You can also set your own install path using CMAKE_INSTALL_PREFIX option.
# In that case a proper set of PYTHONPATH environment variable will
# be required for hysop to work.
#
# Summary :
#
# cmake path-to-hysop-sources -DHYSOP_INSTALL=user,standard,'some_path'
# make install
#
# * HYSOP_INSTALL=user --> install in USER_SITE (no virtualenv case)
# * HYSOP_INSTALL=standard --> install in python standard installation location
#   (may need to be root)
# * HYSOP_INSTALL='some_path' --> install in 'some_path/...

# If HYSOP_INSTALL is not set, default behavior is 'user'.
#
#
# Notes
# -----
# * If /some/install/path is not a standard path of your system,
# to import hysop package, you'll probably need something like :
# export PYTHONPATH=${PYTHONPATH}:/some/install/path
#
# * all pip command are supposed to work with hysop:
# --> pip install, pip show, pip uninstall ...
# 

include(PythonInstallSetup)

# =========== Find python install prefix, determined by HYSOP_INSTALL value ===========
set_python_install_path()
# ---> set HYSOP_PYTHON_INSTALL_DIR and python_install_options

# =========== for uninstall target ===========
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

# =========== install target ===========
# To avoid wheel file name prediction, just run install on dist/*.whl ... Ugly but works fine.
# When using prefix option (i.e. install in a user-defined directory not equal to 'python userbase)
# we need to set PYTHONUSERBASE to this new directory for pip.
if(HYSOP_INSTALL STREQUAL prefix)
  add_custom_target(python-install
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONUSERBASE=${HYSOP_PYTHON_INSTALL_DIR}" pip install --upgrade dist/*.whl ${python_install_options}
    DEPENDS wheel
    COMMENT "build/install hysop package")
  add_custom_target(uninstall
    echo >> ${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONUSERBASE=${HYSOP_PYTHON_INSTALL_DIR}" pip uninstall ${PACKAGE_NAME}
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
else()
  add_custom_target(python-install
    COMMAND pip install --upgrade dist/*.whl ${python_install_options}
    DEPENDS wheel
    COMMENT "build/install hysop package")
  add_custom_target(uninstall
    echo >> ${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt
    #COMMAND cat ${CMAKE_CURRENT_BINARY_DIR}/python_install_manifest.txt >> ${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt
    COMMAND pip uninstall ${PACKAGE_NAME}
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

install(CODE "execute_process(COMMAND ${CMAKE_BUILD_TOOL} python-install WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")")

