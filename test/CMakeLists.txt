include_directories(${gtest_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

file(GLOB TEST_SOURCES *.cc)

file(COPY data DESTINATION ${EXECUTABLE_OUTPUT_PATH})

add_executable(test_all ${TEST_SOURCES})
target_link_libraries(test_all gtest_main pgbm ${MANDATORY_LIBS})
