
file(GLOB CPP_SRCS *.cpp)
set(SRCS ${CPP_SRCS})

get_filename_component(test_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_definitions(${CXX_EXTRA_DEFINES})
add_executable(${test_name} ${SRCS})
add_dependencies(${test_name} ${HYSOP_CXX_LIBRARY_DEP})

target_link_libraries(${test_name} ${HYSOP_CXX_LIBRARY})
target_link_libraries(${test_name} ${GTEST_LIBRARIES} ${CXX_EXT_LIBS})

add_test("${test_name}" "${test_name}")

